"""Budget optimiser: turning uplift into money, under a spending constraint.

Feature 9 asked which campaign each customer should receive and answered it in
visits. A business does not have a visits budget. It has a marketing budget in
dollars, a cost per send, and a margin on whatever revenue comes back -- and
those three numbers change the decision, because they introduce something the
visit analysis never had: **a reason not to send**.

Three questions, in order:

1. **Does emailing pay for itself?** An email costs `COST_PER_EMAIL` and returns
   `GROSS_MARGIN` on whatever incremental spend it causes. That gives a
   break-even incremental spend of `cost / margin`, and an arm only earns its
   keep if its spend effect clears it.
2. **Who should be dropped when the budget is short?** With a fixed cost per
   send, a budget buys a fixed *number* of emails. Choosing whom to keep is a
   ranking problem, and Feature 8 already established which rankings can be
   trusted.
3. **How sensitive is any of this to the two numbers nobody measured?** Margin
   and send cost are assumptions, not data. A conclusion that flips between a
   25% and a 35% margin is not a conclusion.

Two structural points about how profit is computed here.

**Profit is a column, not a separate estimator.** Realised profit per customer
is `margin * spend - cost * (they were emailed)`. Because control customers
carry no cost, the arm-minus-control difference in that column *is* the
incremental profit per email, with the send cost already netted out. So the
whole of Feature 4's inference machinery -- Welch intervals -- and Feature 9's
inverse propensity weighting apply unchanged, and profit needs no new theory.

**Greedy allocation is exactly optimal, not a heuristic.** Every email costs the
same, so "spend a budget to maximise profit" reduces to "choose at most k items
to maximise total value", and taking the k highest-valued items solves that
exactly. The usual knapsack caveats apply only when item costs differ.

Run as a script for the full analysis:

    python -m src.models.budget
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import StratifiedKFold

from src import config
from src.analysis.ab_test import welch_t_test
from src.data.load import load_processed
from src.models.policy import (
    DEFAULT_LEARNER,
    NO_EMAIL,
    cross_fitted_arm_uplifts,
    policy_always,
    policy_difference,
    policy_value,
)
from src.models.uplift import build_design_matrix

logger = logging.getLogger(__name__)

N_FOLDS = 5

# Realised profit is attached to the frame under this name rather than added to
# `config.OUTCOMES`, because it is a derived quantity that depends on two
# business assumptions. The measured outcomes do not.
PROFIT_COLUMN = "profit"

# Number of budget levels on the curve, from zero to emailing everyone.
N_BUDGET_STEPS = 21


# ==========================================================================
# The profit model
# ==========================================================================
def break_even_uplift(
    margin: float = config.GROSS_MARGIN, cost: float = config.COST_PER_EMAIL
) -> float:
    """Incremental spend an email must cause before it pays for itself.

    At the configured 30% margin and $0.10 send cost this is $0.33 -- an email
    has to generate a third of a dollar of extra spend just to break even.
    Worth stating in those terms, because it is a far more demanding bar than
    "the campaign has a positive effect".
    """
    if margin <= 0:
        raise ValueError(f"margin must be positive, got {margin}")
    return cost / margin


def realised_profit(
    df: pd.DataFrame,
    margin: float = config.GROSS_MARGIN,
    cost: float = config.COST_PER_EMAIL,
) -> pd.Series:
    """Profit actually earned from each customer, given the arm they received.

    Control customers were not emailed, so they carry revenue and no cost. That
    asymmetry is the whole point: differencing this column against control
    yields incremental profit *net of the send cost*, with no further
    bookkeeping.
    """
    emailed = (df[config.TREATMENT_COL] != config.CONTROL_ARM).astype(float)
    return margin * df["spend"] - cost * emailed


def with_profit(
    df: pd.DataFrame,
    margin: float = config.GROSS_MARGIN,
    cost: float = config.COST_PER_EMAIL,
) -> pd.DataFrame:
    """Return a copy of the frame carrying a realised-profit column."""
    return df.assign(**{PROFIT_COLUMN: realised_profit(df, margin, cost)})


def incremental_profit(
    spend_uplift: np.ndarray | pd.Series,
    margin: float = config.GROSS_MARGIN,
    cost: float = config.COST_PER_EMAIL,
) -> np.ndarray:
    """Predicted profit from emailing a customer, per unit of predicted spend."""
    return margin * np.asarray(spend_uplift, dtype=float) - cost


# ==========================================================================
# Campaign economics
# ==========================================================================
@dataclass(frozen=True)
class CampaignEconomics:
    """Whether one campaign pays for itself, and how much slack it has."""

    treatment_arm: str
    spend_effect: float
    spend_ci_low: float
    spend_ci_high: float
    profit_per_email: float
    profit_ci_low: float
    profit_ci_high: float
    p_value: float
    break_even_spend: float
    break_even_margin: float
    break_even_margin_low: float
    break_even_margin_high: float
    margin: float
    cost: float

    @property
    def profitable(self) -> bool:
        """Whether profitability is *demonstrated* rather than merely estimated.

        The test is that the whole interval sits above zero. A positive point
        estimate with an interval straddling zero means the campaign might be
        losing money, which is not a basis for committing a budget to it.
        """
        return self.profit_ci_low > 0

    @property
    def clears_break_even(self) -> bool:
        """Whether the spend interval sits entirely above the break-even spend.

        Equivalent to `profitable` by construction -- the two are the same
        statement in different units -- and kept because the spend framing is
        the one a marketer can act on.
        """
        return self.spend_ci_low > self.break_even_spend

    @property
    def margin_headroom(self) -> float:
        """How far the assumed margin can fall before the campaign stops paying.

        Measured against the *worst end* of the spend interval, so it is the
        margin at which profitability would fail even on pessimistic
        assumptions. Negative means the assumed margin is already inside the
        range where the answer could go either way.
        """
        return self.margin - self.break_even_margin_high


def campaign_economics(
    df: pd.DataFrame,
    margin: float = config.GROSS_MARGIN,
    cost: float = config.COST_PER_EMAIL,
    alpha: float = config.ALPHA,
) -> list[CampaignEconomics]:
    """Profit per email for each campaign, with a confidence interval.

    Runs Welch's t-test on the profit column rather than deriving intervals
    from the spend result, so the interval accounts for the arms' unequal
    variances directly. The two agree, since the cost term is a constant shift;
    doing it on the profit column keeps the units honest.
    """
    frame = with_profit(df, margin, cost)
    threshold = break_even_uplift(margin, cost)
    results = []

    for treatment_arm, control_arm in config.COMPARISONS:
        arms = frame[config.TREATMENT_COL]
        profit = welch_t_test(
            frame.loc[arms == treatment_arm, PROFIT_COLUMN],
            frame.loc[arms == control_arm, PROFIT_COLUMN],
            alpha,
        )
        spend = welch_t_test(
            frame.loc[arms == treatment_arm, "spend"],
            frame.loc[arms == control_arm, "spend"],
            alpha,
        )
        effect = spend["absolute_effect"]

        def required_margin(spend_effect: float) -> float:
            """Margin at which this spend effect exactly covers the send cost."""
            return cost / spend_effect if spend_effect > 0 else float("inf")

        results.append(
            CampaignEconomics(
                treatment_arm=treatment_arm,
                spend_effect=effect,
                spend_ci_low=spend["ci_low"],
                spend_ci_high=spend["ci_high"],
                profit_per_email=profit["absolute_effect"],
                profit_ci_low=profit["ci_low"],
                profit_ci_high=profit["ci_high"],
                p_value=profit["p_value"],
                break_even_spend=threshold,
                # The margin at which this campaign would exactly break even,
                # holding the send cost fixed, and the same computed at each end
                # of the spend interval. The high end is the decision-relevant
                # one: the margin needed even if the effect is at its worst
                # plausible value.
                break_even_margin=required_margin(effect),
                break_even_margin_low=required_margin(spend["ci_high"]),
                break_even_margin_high=required_margin(spend["ci_low"]),
                margin=margin,
                cost=cost,
            )
        )

    return results


# ==========================================================================
# Predicted spend uplift for every arm
# ==========================================================================
def _spend_models(
    design: pd.DataFrame,
    spend: np.ndarray,
    treated: np.ndarray,
    seed: int,
) -> tuple[HistGradientBoostingRegressor, HistGradientBoostingRegressor]:
    """Fit one spend regressor per arm. Hyperparameters match Feature 8's."""

    def fit(mask):
        return HistGradientBoostingRegressor(
            max_iter=150, max_depth=4, learning_rate=0.08, random_state=seed
        ).fit(design[mask], spend[mask])

    return fit(treated == 1), fit(treated == 0)


def cross_fitted_arm_spend_uplifts(
    df: pd.DataFrame,
    n_folds: int = N_FOLDS,
    seed: int = config.RANDOM_SEED,
    features: list[str] | None = None,
) -> pd.DataFrame:
    """Predicted spend uplift of each campaign, for *every* customer.

    Feature 9's `cross_fitted_spend_uplifts` scores each customer only for the
    treatment/control pair they belong to, which is enough to value an arm but
    not to make a decision: a customer in the Womens arm still needs a Mens
    spend estimate, because the optimiser may want to give them Mens. This
    scores both arms for everyone, using the same fold structure as
    `cross_fitted_arm_uplifts` so the two are directly comparable.

    Folds are stratified on arm crossed with whether the customer spent
    anything, because 99% of customers spend exactly zero and an unstratified
    split can hand a fold almost no spenders to learn from.
    """
    design = build_design_matrix(df, features)
    spend = df["spend"].to_numpy(dtype=float)
    arm = df[config.TREATMENT_COL].astype(str).to_numpy()

    predictions = {a: np.zeros(len(df)) for a, _ in config.COMPARISONS}
    strata = arm + "_" + (spend > 0).astype(int).astype(str)

    folds = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    for train_index, test_index in folds.split(design, strata):
        train_arm = arm[train_index]
        test_design = design.iloc[test_index]

        for treatment_arm, control_arm in config.COMPARISONS:
            in_pair = np.isin(train_arm, [treatment_arm, control_arm])
            pair_index = train_index[in_pair]

            treated_model, control_model = _spend_models(
                design.iloc[pair_index],
                spend[pair_index],
                (arm[pair_index] == treatment_arm).astype(int),
                seed,
            )
            predictions[treatment_arm][test_index] = treated_model.predict(
                test_design
            ) - control_model.predict(test_design)

    return pd.DataFrame(
        {f"spend_uplift_{a}": v for a, v in predictions.items()}, index=df.index
    )


# ==========================================================================
# Allocation
# ==========================================================================
def best_arm_by_score(scores: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Per customer: the arm with the highest score, and that score.

    Expects one column per arm, in `config.COMPARISONS` order.
    """
    arms = np.array([arm for arm, _ in config.COMPARISONS], dtype=object)
    values = scores.to_numpy(dtype=float)
    best = values.argmax(axis=1)
    return arms[best], values[np.arange(len(values)), best]


def max_sends(budget: float, cost: float = config.COST_PER_EMAIL) -> int:
    """How many emails a budget buys.

    Floor division is wrong here. Neither $0.10 nor $6,400.00 is exactly
    representable in binary floating point, and `6400.0 // 0.1` evaluates to
    63,999 -- so a budget sized to email everyone would quietly leave one
    customer out, at every budget level. The tolerance absorbs that
    representation error without ever buying an email the budget cannot cover.
    """
    if cost <= 0:
        raise ValueError(f"cost must be positive, got {cost}")
    return int(np.floor(budget / cost + 1e-9))


def greedy_allocation(
    arms: np.ndarray,
    scores: np.ndarray,
    n_sends: int,
    eligible: np.ndarray | None = None,
) -> np.ndarray:
    """Send to the highest-ranked eligible customers until the budget runs out.

    Two decisions are kept deliberately separate, because they rest on
    different evidence:

    - **Eligibility** -- is this customer worth emailing *at all*? That is an
      economic question, answered by whether their predicted incremental profit
      clears zero.
    - **Priority** -- when the budget cannot cover everyone eligible, who goes
      first? That is a ranking question, and any score can supply it.

    Collapsing the two would tie the ranking to the spend model, which
    Feature 8 showed is the least trustworthy thing available.

    Ties are broken by a stable sort, so the ordering is reproducible.
    """
    n = len(scores)
    policy = np.full(n, NO_EMAIL, dtype=object)

    n_sends = int(np.clip(n_sends, 0, n))
    if n_sends == 0:
        return policy

    scores = np.asarray(scores, dtype=float)
    mask = np.ones(n, dtype=bool) if eligible is None else np.asarray(eligible, dtype=bool)

    candidates = np.flatnonzero(mask)
    order = candidates[np.argsort(-scores[candidates], kind="stable")]
    chosen = order[:n_sends]

    policy[chosen] = np.asarray(arms, dtype=object)[chosen]
    return policy


@dataclass(frozen=True)
class Allocation:
    """What one budget level, spent under one ranking, is worth."""

    ranking: str
    budget: float
    n_sends: int
    cost_spent: float
    profit_per_customer: float
    standard_error: float
    ci_low: float
    ci_high: float
    total_profit: float
    baseline_profit: float
    n_customers: int

    @property
    def budget_used(self) -> float:
        """Share of the offered budget actually spent.

        Below 1 once every eligible customer has been emailed -- at which point
        more budget buys nothing, which is itself a finding.
        """
        return self.cost_spent / self.budget if self.budget > 0 else 0.0

    @property
    def net_gain(self) -> float:
        """Profit above what emailing nobody would have earned.

        The distinction matters more than it looks. Customers spend money
        whether or not they are emailed, so `total_profit` is dominated by
        revenue the campaign did not cause. Judging a budget by total profit
        credits the campaign with all of it and produces returns in the tens of
        multiples; only the increment over the do-nothing baseline is
        attributable to the send.
        """
        return self.total_profit - self.baseline_profit

    @property
    def return_on_spend(self) -> float:
        """Net gain per dollar of email budget. Undefined when nothing is sent."""
        return self.net_gain / self.cost_spent if self.cost_spent > 0 else float("nan")


def baseline_profit(
    profit_frame: pd.DataFrame, alpha: float = config.ALPHA
) -> float:
    """Total profit from emailing nobody -- the reference every budget beats.

    Estimated by the same inverse propensity weighting as every other policy,
    rather than read off the control arm directly, so that the baseline and the
    allocations are measured on a common footing.
    """
    nobody = policy_always(NO_EMAIL, len(profit_frame))
    value = policy_value(profit_frame, nobody, PROFIT_COLUMN, "email nobody", alpha)
    return value.value * value.n_total


def evaluate_allocation(
    profit_frame: pd.DataFrame,
    policy: np.ndarray,
    ranking: str,
    budget: float,
    baseline: float,
    cost: float = config.COST_PER_EMAIL,
    alpha: float = config.ALPHA,
) -> Allocation:
    """Value an allocation by inverse propensity weighting, as in Feature 9.

    The interval is normal-approximate. Profit inherits spend's heavy tail --
    99% of customers contribute exactly zero and a handful contribute hundreds
    of dollars -- so it should be read as indicative rather than exact. It is
    reported because the *width* is the load-bearing part of this analysis, and
    the width is not sensitive to the approximation.
    """
    value = policy_value(profit_frame, policy, PROFIT_COLUMN, name=ranking, alpha=alpha)
    n_sends = int((policy != NO_EMAIL).sum())

    return Allocation(
        ranking=ranking,
        budget=float(budget),
        n_sends=n_sends,
        cost_spent=n_sends * cost,
        profit_per_customer=value.value,
        standard_error=value.standard_error,
        ci_low=value.ci_low,
        ci_high=value.ci_high,
        total_profit=value.value * value.n_total,
        baseline_profit=baseline,
        n_customers=value.n_total,
    )


def budget_curve(
    profit_frame: pd.DataFrame,
    arms: np.ndarray,
    scores: np.ndarray,
    ranking: str,
    budgets: np.ndarray,
    eligible: np.ndarray | None = None,
    cost: float = config.COST_PER_EMAIL,
    alpha: float = config.ALPHA,
    baseline: float | None = None,
) -> pd.DataFrame:
    """Profit as a function of budget, for one ranking."""
    reference = baseline_profit(profit_frame, alpha) if baseline is None else baseline

    rows = []
    for budget in budgets:
        policy = greedy_allocation(arms, scores, max_sends(budget, cost), eligible)
        allocation = evaluate_allocation(
            profit_frame, policy, ranking, budget, reference, cost, alpha
        )
        rows.append(
            asdict(allocation)
            | {
                "budget_used": allocation.budget_used,
                "net_gain": allocation.net_gain,
                "return_on_spend": allocation.return_on_spend,
            }
        )
    return pd.DataFrame(rows)


def compare_rankings(
    profit_frame: pd.DataFrame,
    rankings: dict[str, tuple[np.ndarray, np.ndarray]],
    budget: float,
    versus: str = "always Mens",
    eligible: np.ndarray | None = None,
    cost: float = config.COST_PER_EMAIL,
    alpha: float = config.ALPHA,
) -> pd.DataFrame:
    """Paired comparison of every ranking against the no-model baseline.

    Feature 9's lesson applies unchanged: rankings that agree on most customers
    have correlated errors, and differencing per customer before aggregating is
    what makes the comparison sensitive enough to be worth running. On profit
    it matters even more, because the marginal intervals are dominated by
    spend's heavy tail -- noise that is common to both policies and cancels
    exactly wherever they agree.
    """
    n_sends = max_sends(budget, cost)
    policies = {
        name: greedy_allocation(arms, scores, n_sends, eligible)
        for name, (arms, scores) in rankings.items()
    }

    rows = []
    for name, policy in policies.items():
        if name == versus:
            continue
        rows.append(
            {"ranking": name, "versus": versus, "budget": float(budget)}
            | policy_difference(profit_frame, policy, policies[versus],
                                PROFIT_COLUMN, alpha)
        )
    return pd.DataFrame(rows)


# ==========================================================================
# Rankings
# ==========================================================================
def build_rankings(
    df: pd.DataFrame,
    spend_uplifts: pd.DataFrame,
    visit_uplifts: pd.DataFrame,
    margin: float = config.GROSS_MARGIN,
    cost: float = config.COST_PER_EMAIL,
    seed: int = config.RANDOM_SEED,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """The candidate ways of deciding whom to email first.

    Each entry is `(arm per customer, priority score per customer)`. They differ
    in what evidence they lean on:

    - **predicted profit** uses the spend model for both the arm and the
      ordering. The most direct answer to the question, and the one resting on
      the weakest model.
    - **predicted visit uplift** uses the visit model, which Feature 8 showed is
      the only individual-level signal in this data that survives a null test.
      It ranks on a proxy, on the argument that a reliable proxy beats an
      unreliable target.
    - **always Mens** is the no-model baseline: Feature 9's winner, with
      customers dropped at random when the budget binds.
    - **random ranking** is the null. It keeps the profit model's *arm* choice
      and randomises only the priority order, so any gap between it and
      "predicted profit" is attributable to the ranking alone.

    That last pairing carries a built-in check. Priority only matters while the
    budget binds, so at a budget large enough to email everyone eligible the
    two must produce byte-identical results. If they ever diverge there, the
    allocator is not doing what it claims.
    """
    rng = np.random.default_rng(seed)
    n = len(df)

    spend_columns = [f"spend_uplift_{arm}" for arm, _ in config.COMPARISONS]
    visit_columns = [f"uplift_{arm}" for arm, _ in config.COMPARISONS]

    profit_scores = incremental_profit(spend_uplifts[spend_columns], margin, cost)
    profit_arms, profit_best = best_arm_by_score(
        pd.DataFrame(profit_scores, columns=spend_columns, index=df.index)
    )
    visit_arms, visit_best = best_arm_by_score(visit_uplifts[visit_columns])

    return {
        "predicted profit": (profit_arms, profit_best),
        "predicted visit uplift": (visit_arms, visit_best),
        "always Mens": (policy_always(config.MENS_ARM, n), rng.random(n)),
        "random ranking": (profit_arms, rng.random(n)),
    }


# ==========================================================================
# Sensitivity
# ==========================================================================
def sensitivity_grid(
    df: pd.DataFrame,
    margins: np.ndarray,
    costs: np.ndarray,
    alpha: float = config.ALPHA,
) -> pd.DataFrame:
    """Profit per email across the two assumptions nobody measured.

    Margin and send cost are stated in `config.py` as assumptions. Any profit
    conclusion is conditional on them, so the honest presentation is the grid
    rather than the single cell.
    """
    rows = []
    for margin in margins:
        for cost in costs:
            for economics in campaign_economics(df, margin, cost, alpha):
                rows.append(asdict(economics) | {"profitable": economics.profitable})
    return pd.DataFrame(rows)


# ==========================================================================
# Orchestration
# ==========================================================================
def run_budget_analysis(
    df: pd.DataFrame | None = None,
    margin: float = config.GROSS_MARGIN,
    cost: float = config.COST_PER_EMAIL,
    learner: str = DEFAULT_LEARNER,
    n_folds: int = N_FOLDS,
    save: bool = True,
    seed: int = config.RANDOM_SEED,
) -> dict[str, pd.DataFrame]:
    """Campaign economics, the budget curve for each ranking, and sensitivity."""
    frame = load_processed() if df is None else df
    profit_frame = with_profit(frame, margin, cost)

    economics = pd.DataFrame(
        [
            asdict(e) | {"profitable": e.profitable}
            for e in campaign_economics(frame, margin, cost)
        ]
    )

    logger.info("Cross-fitting spend and visit uplift models")
    spend_uplifts = cross_fitted_arm_spend_uplifts(frame, n_folds, seed)
    visit_uplifts = cross_fitted_arm_uplifts(
        frame, config.HETEROGENEITY_PRIMARY_OUTCOME, learner, n_folds, seed
    )

    rankings = build_rankings(frame, spend_uplifts, visit_uplifts, margin, cost, seed)

    # Eligibility is one economic question, answered once, and shared by every
    # ranking -- so the curves differ only in the order they spend the budget.
    spend_columns = [f"spend_uplift_{arm}" for arm, _ in config.COMPARISONS]
    _, best_profit = best_arm_by_score(
        pd.DataFrame(
            incremental_profit(spend_uplifts[spend_columns], margin, cost),
            columns=spend_columns,
            index=frame.index,
        )
    )
    eligible = best_profit > 0

    full_send_cost = len(frame) * cost
    budgets = np.linspace(0, full_send_cost, N_BUDGET_STEPS)
    reference = baseline_profit(profit_frame)

    curves = pd.concat(
        [
            budget_curve(
                profit_frame, arms, scores, name, budgets, eligible, cost,
                baseline=reference,
            )
            for name, (arms, scores) in rankings.items()
        ],
        ignore_index=True,
    )

    comparisons = compare_rankings(
        profit_frame, rankings, full_send_cost, eligible=eligible, cost=cost
    )

    sensitivity = sensitivity_grid(
        frame,
        margins=np.arange(0.10, 0.51, 0.05),
        costs=np.array([0.05, 0.10, 0.20, 0.30]),
    )

    if save:
        config.ensure_dirs()
        economics.to_csv(
            config.RESULTS_DIR / "10_campaign_economics.csv", index=False
        )
        curves.to_csv(config.RESULTS_DIR / "10_budget_curve.csv", index=False)
        comparisons.to_csv(
            config.RESULTS_DIR / "10_ranking_comparison.csv", index=False
        )
        sensitivity.to_csv(config.RESULTS_DIR / "10_sensitivity.csv", index=False)
        logger.info("Wrote budget analysis to %s", config.RESULTS_DIR)

    return {
        "economics": economics,
        "curves": curves,
        "comparisons": comparisons,
        "sensitivity": sensitivity,
        "spend_uplifts": spend_uplifts,
        "visit_uplifts": visit_uplifts,
        "eligible": eligible,
        "baseline_profit": reference,
    }


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    results = run_budget_analysis()
    economics, curves = results["economics"], results["curves"]

    threshold = break_even_uplift()
    print("\n" + "=" * 96)
    print(
        f"CAMPAIGN ECONOMICS  (margin = {config.GROSS_MARGIN:.0%}, "
        f"cost = ${config.COST_PER_EMAIL:.2f}, "
        f"break-even incremental spend = ${threshold:.4f})"
    )
    print("=" * 96 + "\n")

    print(
        economics[
            [
                "treatment_arm", "spend_effect", "spend_ci_low", "spend_ci_high",
                "profit_per_email", "profit_ci_low", "profit_ci_high",
                "break_even_margin", "break_even_margin_high", "profitable",
            ]
        ].to_string(
            index=False,
            formatters={
                "spend_effect": "${:.4f}".format,
                "spend_ci_low": "${:.4f}".format,
                "spend_ci_high": "${:.4f}".format,
                "profit_per_email": "${:+.4f}".format,
                "profit_ci_low": "${:+.4f}".format,
                "profit_ci_high": "${:+.4f}".format,
                "break_even_margin": "{:.1%}".format,
                "break_even_margin_high": "{:.1%}".format,
            },
        )
    )

    print("\n" + "-" * 96)
    print(
        "Net gain over emailing nobody, by budget  "
        f"(baseline = ${results['baseline_profit']:,.0f})"
    )
    print("-" * 96 + "\n")

    best = curves.loc[curves.groupby("budget")["net_gain"].idxmax()]
    print(
        best[
            [
                "budget", "ranking", "n_sends", "cost_spent", "net_gain",
                "return_on_spend",
            ]
        ].to_string(
            index=False,
            formatters={
                "budget": "${:,.0f}".format,
                "n_sends": "{:,}".format,
                "cost_spent": "${:,.0f}".format,
                "net_gain": "${:+,.0f}".format,
                "return_on_spend": "{:.2f}x".format,
            },
        )
    )

    print("\n" + "-" * 96)
    print("At a full-send budget, paired against the no-model baseline")
    print("-" * 96 + "\n")

    full = curves[curves["budget"] == curves["budget"].max()]
    print(
        full[["ranking", "n_sends", "net_gain", "return_on_spend"]].to_string(
            index=False,
            formatters={
                "n_sends": "{:,}".format,
                "net_gain": "${:+,.0f}".format,
                "return_on_spend": "{:.2f}x".format,
            },
        )
    )
    print()
    comparisons = results["comparisons"]
    n = len(results["eligible"])
    print(
        comparisons.assign(total=comparisons["difference"] * n)[
            ["ranking", "versus", "total", "difference", "ci_low", "ci_high",
             "p_value", "n_disagree"]
        ].to_string(
            index=False,
            formatters={
                "total": "${:+,.0f}".format,
                "difference": "{:+.4f}".format,
                "ci_low": "{:+.4f}".format,
                "ci_high": "{:+.4f}".format,
                "p_value": "{:.3f}".format,
                "n_disagree": "{:,}".format,
            },
        )
    )

    eligible = results["eligible"]
    print(
        f"\nCustomers with positive predicted incremental profit: "
        f"{eligible.sum():,} of {len(eligible):,} ({eligible.mean():.1%})"
    )


if __name__ == "__main__":
    main()
