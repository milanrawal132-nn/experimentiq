"""Targeting policy: which campaign each customer should receive.

Feature 8 produced a predicted uplift per customer for each campaign. A policy
turns those numbers into a decision -- Mens, Womens, or nothing -- and the
question this module answers is what that decision is actually worth.

Evaluating a policy is harder than building one, for a reason specific to
causal problems. A customer received one arm; the outcomes under the other two
were never observed. So the value of a policy that would have assigned them
differently cannot be read off the data directly.

The way through is that assignment was **randomised with known probabilities**.
Each customer had a known chance of landing in the arm the policy would have
chosen. Keeping only the customers whose actual arm matches the policy, and
weighting them by the inverse of that probability, gives an unbiased estimate
of what the policy would have earned across everyone. That is inverse
propensity weighting, and randomisation is what makes it valid here -- in
observational data the propensities would have to be modelled and the estimate
would inherit that model's errors.

Two things this module is careful about:

**The policy is cross-fitted.** A policy chosen using a customer's own outcome
and then evaluated on that same customer scores its own hindsight. Uplift
models are fitted on training folds and the policy is evaluated only on held-
out customers.

**Policy comparisons are paired.** Two policies evaluated on the same customers
have correlated errors, so the standard error of their *difference* is much
smaller than the two individual standard errors suggest. Differencing per
customer before aggregating captures that.

Run as a script for the policy comparison:

    python -m src.models.policy
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.model_selection import StratifiedKFold

from src import config
from src.data.load import load_processed
from src.models.uplift import (
    build_design_matrix,
    make_learner,
    regression_t_learner_uplift,
)

logger = logging.getLogger(__name__)

# Feature 8's winner: highest Qini, simplest to explain, and its ranking maps
# onto a mechanism Feature 7 independently established.
DEFAULT_LEARNER = "logistic_t_learner"

N_FOLDS = 5

# The policy's action space. "none" is a real action, not the absence of one:
# an email costs money and may be worth withholding.
NO_EMAIL = "none"


@dataclass(frozen=True)
class PolicyValue:
    """Estimated value of one policy, and how much of the data supports it."""

    policy: str
    outcome: str
    value: float
    standard_error: float
    ci_low: float
    ci_high: float
    n_matched: int
    n_total: int

    @property
    def coverage(self) -> float:
        """Share of customers whose actual arm matched the policy.

        Only these contribute to the estimate. A policy that disagrees with the
        experiment almost everywhere is estimated from a small slice of the
        data, however many customers were enrolled.
        """
        return self.n_matched / self.n_total


# ==========================================================================
# Cross-fitted uplift for every arm
# ==========================================================================
def cross_fitted_arm_uplifts(
    df: pd.DataFrame,
    outcome: str = config.HETEROGENEITY_PRIMARY_OUTCOME,
    learner: str = DEFAULT_LEARNER,
    n_folds: int = N_FOLDS,
    seed: int = config.RANDOM_SEED,
    features: list[str] | None = None,
) -> pd.DataFrame:
    """Predicted uplift of each campaign over control, for every customer.

    For each fold, two uplift models are fitted on the training rows -- one on
    (Mens, Control), one on (Womens, Control) -- and both are used to score
    *every* held-out customer, whatever arm they actually landed in. A customer
    in the Womens arm still needs a Mens uplift estimate, because the policy
    may want to assign them Mens.

    Folds are stratified on arm-by-outcome so each fold keeps all three arms.
    """
    design = build_design_matrix(df, features)
    y = df[outcome].to_numpy()
    arm = df[config.TREATMENT_COL].astype(str).to_numpy()

    predictions = {a: np.zeros(len(df)) for a, _ in config.COMPARISONS}
    strata = arm + "_" + y.astype(str)

    folds = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    for train_index, test_index in folds.split(design, strata):
        train_arm = arm[train_index]

        for treatment_arm, control_arm in config.COMPARISONS:
            in_pair = np.isin(train_arm, [treatment_arm, control_arm])
            pair_index = train_index[in_pair]

            model = make_learner(learner, seed)
            model.fit(
                design.iloc[pair_index],
                y[pair_index],
                (arm[pair_index] == treatment_arm).astype(int),
            )
            predictions[treatment_arm][test_index] = model.predict(
                design.iloc[test_index]
            )

    return pd.DataFrame(
        {f"uplift_{a}": v for a, v in predictions.items()}, index=df.index
    )


def cross_fitted_spend_uplifts(
    df: pd.DataFrame,
    n_folds: int = N_FOLDS,
    seed: int = config.RANDOM_SEED,
) -> pd.DataFrame:
    """Predicted spend uplift per campaign, for Feature 10's profit model.

    Uses the regression two-model path, since spend is continuous. Feature 8
    found these rankings are far noisier than the visit ones -- they are
    produced here for valuation, not for deciding who to target.
    """
    design = build_design_matrix(df)
    spend = df["spend"].to_numpy()
    arm = df[config.TREATMENT_COL].astype(str).to_numpy()

    predictions = {}
    for treatment_arm, control_arm in config.COMPARISONS:
        in_pair = np.isin(arm, [treatment_arm, control_arm])
        pair_uplift = regression_t_learner_uplift(
            design[in_pair].reset_index(drop=True),
            spend[in_pair],
            (arm[in_pair] == treatment_arm).astype(int),
            n_folds=n_folds,
            seed=seed,
        )
        full = np.full(len(df), np.nan)
        full[in_pair] = pair_uplift
        predictions[f"spend_uplift_{treatment_arm}"] = full

    return pd.DataFrame(predictions, index=df.index)


# ==========================================================================
# Policies
# ==========================================================================
def policy_always(arm: str, n: int) -> np.ndarray:
    """Send the same campaign to everyone."""
    return np.full(n, arm, dtype=object)


def policy_random(n: int, seed: int = config.RANDOM_SEED) -> np.ndarray:
    """Assign arms uniformly at random -- what the experiment itself did."""
    rng = np.random.default_rng(seed)
    return rng.choice([config.MENS_ARM, config.WOMENS_ARM, NO_EMAIL], size=n)


def policy_best_campaign(uplifts: pd.DataFrame) -> np.ndarray:
    """Send whichever campaign has the higher predicted uplift. Always send."""
    columns = [f"uplift_{arm}" for arm, _ in config.COMPARISONS]
    best = uplifts[columns].to_numpy().argmax(axis=1)
    arms = np.array([arm for arm, _ in config.COMPARISONS], dtype=object)
    return arms[best]


def policy_best_or_none(
    uplifts: pd.DataFrame, threshold: float = 0.0
) -> np.ndarray:
    """Send the better campaign, but only when its uplift clears a threshold.

    With `threshold = 0` this withholds email from customers predicted to be
    unaffected or harmed. Feature 10 sets the threshold from the economics --
    the break-even uplift at which incremental profit covers the send cost.
    """
    columns = [f"uplift_{arm}" for arm, _ in config.COMPARISONS]
    values = uplifts[columns].to_numpy()
    assignment = policy_best_campaign(uplifts)
    return np.where(values.max(axis=1) > threshold, assignment, NO_EMAIL)


# ==========================================================================
# Off-policy evaluation
# ==========================================================================
def _propensities(df: pd.DataFrame) -> dict[str, float]:
    """Probability of each action under the experiment's own assignment.

    Known by design rather than estimated: the experiment randomised into three
    arms, so these are just the realised shares. `none` maps to the control
    arm, which is what "was not emailed" means in this data.
    """
    shares = df[config.TREATMENT_COL].value_counts(normalize=True)
    propensity = {str(arm): float(share) for arm, share in shares.items()}
    propensity[NO_EMAIL] = propensity[config.CONTROL_ARM]
    return propensity


def _influence(
    df: pd.DataFrame, policy: np.ndarray, outcome: str
) -> tuple[np.ndarray, np.ndarray]:
    """Per-customer contribution to the IPW estimate, and the match indicator.

    Returns `(psi, matched)` where psi_i is the customer's weighted outcome and
    is zero when their actual arm differs from the policy's choice. Keeping the
    per-customer values rather than only their mean is what lets two policies
    be compared as a paired difference later.
    """
    propensity = _propensities(df)
    actual = df[config.TREATMENT_COL].astype(str).to_numpy()
    y = df[outcome].to_numpy(dtype=float)

    # "none" and the control arm are the same observed state.
    chosen = np.where(policy == NO_EMAIL, config.CONTROL_ARM, policy)
    matched = actual == chosen

    weights = np.array([propensity[a] for a in chosen])
    return np.where(matched, y / weights, 0.0), matched


def policy_value(
    df: pd.DataFrame,
    policy: np.ndarray,
    outcome: str,
    name: str,
    alpha: float = config.ALPHA,
) -> PolicyValue:
    """Inverse-propensity estimate of what a policy would have earned.

    Unbiased because assignment was randomised: every customer had a known,
    non-zero probability of receiving whatever the policy would have chosen, so
    reweighting the matched customers reconstructs the full population.

    The standard error comes from the spread of the per-customer contributions,
    which is wide -- two thirds of customers contribute exactly zero, and the
    rest are inflated threefold to compensate.
    """
    psi, matched = _influence(df, policy, outcome)

    value = float(psi.mean())
    standard_error = float(psi.std(ddof=1) / np.sqrt(len(psi)))
    critical = stats.norm.ppf(1 - alpha / 2)

    return PolicyValue(
        policy=name,
        outcome=outcome,
        value=value,
        standard_error=standard_error,
        ci_low=value - critical * standard_error,
        ci_high=value + critical * standard_error,
        n_matched=int(matched.sum()),
        n_total=len(df),
    )


def policy_difference(
    df: pd.DataFrame,
    policy_a: np.ndarray,
    policy_b: np.ndarray,
    outcome: str,
    alpha: float = config.ALPHA,
) -> dict[str, float]:
    """Paired comparison of two policies on the same customers.

    Differencing per customer before averaging is what makes this useful. The
    two policies agree on most customers, and wherever they agree the
    difference is exactly zero and contributes no noise. Comparing the two
    marginal standard errors instead would treat those shared customers as two
    independent measurements and badly overstate the uncertainty.
    """
    psi_a, _ = _influence(df, policy_a, outcome)
    psi_b, _ = _influence(df, policy_b, outcome)

    paired = psi_a - psi_b
    difference = float(paired.mean())
    standard_error = float(paired.std(ddof=1) / np.sqrt(len(paired)))
    critical = stats.norm.ppf(1 - alpha / 2)

    z = difference / standard_error if standard_error > 0 else 0.0
    return {
        "difference": difference,
        "standard_error": standard_error,
        "ci_low": difference - critical * standard_error,
        "ci_high": difference + critical * standard_error,
        "p_value": float(2 * stats.norm.sf(abs(z))),
        "n_disagree": int((policy_a != policy_b).sum()),
    }


# ==========================================================================
# Orchestration
# ==========================================================================
def build_policies(
    df: pd.DataFrame, uplifts: pd.DataFrame, seed: int = config.RANDOM_SEED
) -> dict[str, np.ndarray]:
    """The policies compared against one another."""
    n = len(df)
    return {
        "email nobody": policy_always(NO_EMAIL, n),
        "everyone gets Mens": policy_always(config.MENS_ARM, n),
        "everyone gets Womens": policy_always(config.WOMENS_ARM, n),
        "random assignment": policy_random(n, seed),
        "best campaign per customer": policy_best_campaign(uplifts),
        "best campaign, or none if uplift <= 0": policy_best_or_none(uplifts),
    }


def run_policy_analysis(
    df: pd.DataFrame | None = None,
    outcome: str = config.HETEROGENEITY_PRIMARY_OUTCOME,
    learner: str = DEFAULT_LEARNER,
    save: bool = True,
    seed: int = config.RANDOM_SEED,
) -> dict[str, pd.DataFrame]:
    """Estimate and compare the value of every candidate policy."""
    frame = load_processed() if df is None else df

    logger.info("Cross-fitting uplift models")
    uplifts = cross_fitted_arm_uplifts(frame, outcome, learner, seed=seed)
    policies = build_policies(frame, uplifts, seed)

    values = pd.DataFrame(
        [asdict(policy_value(frame, p, outcome, name)) for name, p in policies.items()]
    )
    values["coverage"] = values["n_matched"] / values["n_total"]

    # Every policy against the strongest fixed alternative, which is the
    # comparison a business actually faces: does personalising beat simply
    # sending the better campaign to everyone?
    baseline_name = "everyone gets Mens"
    comparisons = []
    for name, policy in policies.items():
        if name == baseline_name:
            continue
        result = policy_difference(
            frame, policy, policies[baseline_name], outcome
        )
        comparisons.append({"policy": name, "versus": baseline_name, **result})

    differences = pd.DataFrame(comparisons)

    if save:
        config.ensure_dirs()
        values.to_csv(config.RESULTS_DIR / "09_policy_values.csv", index=False)
        differences.to_csv(
            config.RESULTS_DIR / "09_policy_differences.csv", index=False
        )
        uplifts.assign(**{
            "assignment": policies["best campaign, or none if uplift <= 0"]
        }).to_csv(config.RESULTS_DIR / "09_customer_assignments.csv", index=False)
        logger.info("Wrote policy analysis to %s", config.RESULTS_DIR)

    return {"values": values, "differences": differences, "uplifts": uplifts,
            "policies": policies}


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    results = run_policy_analysis()
    values, differences = results["values"], results["differences"]

    print("\n" + "=" * 96)
    print(f"POLICY VALUE  (outcome = {config.HETEROGENEITY_PRIMARY_OUTCOME}, "
          "inverse propensity weighting)")
    print("=" * 96 + "\n")

    print(values[[
        "policy", "value", "standard_error", "ci_low", "ci_high",
        "n_matched", "coverage",
    ]].sort_values("value", ascending=False).to_string(index=False, formatters={
        "value": "{:.4f}".format,
        "standard_error": "{:.4f}".format,
        "ci_low": "{:.4f}".format,
        "ci_high": "{:.4f}".format,
        "n_matched": "{:,}".format,
        "coverage": "{:.1%}".format,
    }))

    print("\n" + "-" * 96)
    print("Paired comparison against 'everyone gets Mens'")
    print("-" * 96 + "\n")
    print(differences[[
        "policy", "difference", "ci_low", "ci_high", "p_value", "n_disagree",
    ]].to_string(index=False, formatters={
        "difference": "{:+.4f}".format,
        "ci_low": "{:+.4f}".format,
        "ci_high": "{:+.4f}".format,
        "p_value": "{:.4f}".format,
        "n_disagree": "{:,}".format,
    }))

    print("\n" + "-" * 96)
    print("What the learned policy actually assigns")
    print("-" * 96 + "\n")
    assignment = pd.Series(results["policies"]["best campaign, or none if uplift <= 0"])
    print((assignment.value_counts(normalize=True) * 100).round(1).to_string())


if __name__ == "__main__":
    main()
