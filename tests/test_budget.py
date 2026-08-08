"""Tests for the budget optimiser and the profit model underneath it.

Profit is the first quantity in this project that depends on assumptions rather
than measurements, so the tests split into two kinds. Some pin down arithmetic
that must hold exactly -- the break-even identity, the fact that greedy
allocation is optimal and not merely good, the fact that differencing the
profit column against control nets out the send cost. Others check that the
optimiser actually finds profit when profit is there to be found, using
synthetic data with a known responder segment, because the real dataset's
answer is largely "there is nothing here" and a broken optimiser would say the
same thing.
"""

import itertools

import numpy as np
import pandas as pd
import pytest

from src import config
from src.analysis.ab_test import welch_t_test
from src.models.budget import (
    NO_EMAIL,
    PROFIT_COLUMN,
    Allocation,
    baseline_profit,
    best_arm_by_score,
    break_even_uplift,
    budget_curve,
    campaign_economics,
    compare_rankings,
    cross_fitted_arm_spend_uplifts,
    evaluate_allocation,
    greedy_allocation,
    incremental_profit,
    max_sends,
    realised_profit,
    run_budget_analysis,
    sensitivity_grid,
    with_profit,
)
from src.models.policy import policy_always, policy_value


@pytest.fixture(scope="module")
def economics(processed_df):
    return {e.treatment_arm: e for e in campaign_economics(processed_df)}


@pytest.fixture(scope="module")
def budget_results(processed_df):
    return run_budget_analysis(processed_df, save=False)


def synthetic_profit_experiment(n: int = 30_000, seed: int = 0) -> pd.DataFrame:
    """A three-arm experiment where only one segment is worth emailing.

    30% of customers are responders whose purchase probability jumps when
    emailed; the rest are unaffected. Spend is gamma-distributed among buyers,
    so the outcome carries the same zero-inflated heavy tail the real data has.
    A budget that cannot cover everyone should therefore go to the responders,
    and a ranking that finds them must beat a random one.
    """
    rng = np.random.default_rng(seed)
    responder = rng.random(n) < 0.3
    arm = rng.choice(
        [config.CONTROL_ARM, config.MENS_ARM, config.WOMENS_ARM], size=n
    )

    emailed = arm != config.CONTROL_ARM
    bought = rng.binomial(1, np.where(responder & emailed, 0.25, 0.05))
    spend = bought * rng.gamma(shape=2.0, scale=30.0, size=n)

    return pd.DataFrame({
        "spend": spend,
        "visit": bought,
        "responder": responder.astype(float),
        config.TREATMENT_COL: pd.Categorical(
            arm,
            categories=[config.CONTROL_ARM, config.MENS_ARM, config.WOMENS_ARM],
        ),
    })


# ==========================================================================
# The profit model
# ==========================================================================
class TestProfitModel:
    def test_break_even_is_cost_over_margin(self):
        assert break_even_uplift(margin=0.30, cost=0.10) == pytest.approx(1 / 3)
        assert break_even_uplift(margin=0.50, cost=0.10) == pytest.approx(0.20)

    def test_free_email_breaks_even_at_zero_uplift(self):
        assert break_even_uplift(margin=0.30, cost=0.0) == 0.0

    @pytest.mark.parametrize("margin", [0.0, -0.1])
    def test_non_positive_margin_is_rejected(self, margin):
        """A zero margin makes the break-even uplift infinite, which is a
        modelling error rather than a number worth returning."""
        with pytest.raises(ValueError):
            break_even_uplift(margin=margin, cost=0.10)

    def test_control_customers_carry_revenue_and_no_cost(self, processed_df):
        profit = realised_profit(processed_df, margin=0.30, cost=0.10)
        control = processed_df[config.TREATMENT_COL] == config.CONTROL_ARM

        assert profit[control].to_numpy() == pytest.approx(
            0.30 * processed_df.loc[control, "spend"].to_numpy()
        )

    def test_emailed_customers_carry_the_send_cost(self, processed_df):
        profit = realised_profit(processed_df, margin=0.30, cost=0.10)
        emailed = processed_df[config.TREATMENT_COL] != config.CONTROL_ARM

        assert profit[emailed].to_numpy() == pytest.approx(
            0.30 * processed_df.loc[emailed, "spend"].to_numpy() - 0.10
        )

    def test_profit_difference_nets_out_the_send_cost(self, processed_df):
        """The identity the whole module rests on.

        Because control carries no cost, the arm-minus-control difference in
        realised profit equals `margin * spend_effect - cost` exactly. That is
        what lets Feature 4's t-test and Feature 9's IPW be reused on profit
        without any new theory.
        """
        margin, cost = 0.30, 0.10
        frame = with_profit(processed_df, margin, cost)
        arms = frame[config.TREATMENT_COL]

        profit_effect = welch_t_test(
            frame.loc[arms == config.MENS_ARM, PROFIT_COLUMN],
            frame.loc[arms == config.CONTROL_ARM, PROFIT_COLUMN],
        )["absolute_effect"]
        spend_effect = welch_t_test(
            frame.loc[arms == config.MENS_ARM, "spend"],
            frame.loc[arms == config.CONTROL_ARM, "spend"],
        )["absolute_effect"]

        assert profit_effect == pytest.approx(margin * spend_effect - cost)

    def test_incremental_profit_is_linear_in_uplift(self):
        uplift = np.array([0.0, 1.0, 2.0])
        assert incremental_profit(uplift, margin=0.5, cost=0.10) == pytest.approx(
            [-0.10, 0.40, 0.90]
        )

    def test_with_profit_leaves_the_original_frame_untouched(self, processed_df):
        columns_before = list(processed_df.columns)
        with_profit(processed_df)
        assert list(processed_df.columns) == columns_before


# ==========================================================================
# Campaign economics
# ==========================================================================
class TestCampaignEconomics:
    def test_break_even_margin_zeroes_the_profit(self, economics):
        """At the break-even margin, profit per email is exactly zero."""
        for arm, result in economics.items():
            assert result.break_even_margin * result.spend_effect == pytest.approx(
                result.cost
            ), arm

    def test_the_two_framings_agree(self, economics):
        """`profitable` (profit interval above zero) and `clears_break_even`
        (spend interval above the break-even spend) are the same statement in
        different units, so they can never disagree."""
        for arm, result in economics.items():
            assert result.profitable == result.clears_break_even, arm

    def test_break_even_margin_interval_brackets_the_point_estimate(self, economics):
        for arm, result in economics.items():
            assert (
                result.break_even_margin_low
                <= result.break_even_margin
                <= result.break_even_margin_high
            ), arm

    def test_headroom_is_positive_exactly_when_profitable(self, economics):
        for arm, result in economics.items():
            assert (result.margin_headroom > 0) == result.profitable, arm

    def test_a_costless_email_is_profitable_whenever_the_effect_is(
        self, processed_df
    ):
        """With no send cost, profitability reduces to "the campaign works",
        which Feature 4 already established for both arms."""
        for result in campaign_economics(processed_df, margin=0.30, cost=0.0):
            assert result.break_even_spend == 0.0
            assert result.profitable

    def test_an_expensive_email_is_profitable_for_neither_arm(self, processed_df):
        for result in campaign_economics(processed_df, margin=0.30, cost=5.0):
            assert not result.profitable

    def test_mens_is_demonstrably_profitable(self, economics):
        """The one decision-grade finding in this feature."""
        mens = economics[config.MENS_ARM]
        assert mens.profit_per_email > 0
        assert mens.profit_ci_low > 0
        assert mens.break_even_margin_high < config.GROSS_MARGIN

    def test_womens_profitability_is_not_established(self, economics):
        """Positive point estimate, interval straddling zero. Emailing Womens
        may well lose money at a 30% margin, and this experiment cannot say."""
        womens = economics[config.WOMENS_ARM]
        assert womens.profit_per_email > 0
        assert womens.profit_ci_low < 0 < womens.profit_ci_high
        assert not womens.profitable
        assert womens.break_even_margin_high > config.GROSS_MARGIN


# ==========================================================================
# Allocation
# ==========================================================================
class TestGreedyAllocation:
    def test_budget_buys_whole_emails_only(self):
        assert max_sends(1.00, cost=0.10) == 10
        assert max_sends(0.95, cost=0.10) == 9
        assert max_sends(0.0, cost=0.10) == 0

    def test_a_budget_sized_to_the_file_covers_the_whole_file(self):
        """Guards a floating-point trap: `6400.0 // 0.1` is 63,999, so plain
        floor division drops the last customer at every budget level."""
        assert max_sends(64_000 * 0.10, cost=0.10) == 64_000
        assert max_sends(21_307 * 0.10, cost=0.10) == 21_307

    def test_a_budget_one_cent_short_buys_one_fewer_email(self):
        """The tolerance must absorb representation error without ever buying
        an email the budget cannot cover."""
        assert max_sends(64_000 * 0.10 - 0.01, cost=0.10) == 63_999

    def test_zero_cost_is_rejected(self):
        with pytest.raises(ValueError):
            max_sends(100.0, cost=0.0)

    def test_zero_budget_emails_nobody(self):
        arms = np.array([config.MENS_ARM] * 10, dtype=object)
        policy = greedy_allocation(arms, np.arange(10.0), n_sends=0)
        assert (policy == NO_EMAIL).all()

    def test_sends_exactly_the_budgeted_number(self):
        arms = np.array([config.MENS_ARM] * 100, dtype=object)
        policy = greedy_allocation(arms, np.arange(100.0), n_sends=30)
        assert (policy != NO_EMAIL).sum() == 30

    def test_a_budget_larger_than_the_file_is_clipped(self):
        arms = np.array([config.MENS_ARM] * 10, dtype=object)
        policy = greedy_allocation(arms, np.arange(10.0), n_sends=999)
        assert (policy != NO_EMAIL).sum() == 10

    def test_the_highest_scores_are_chosen(self):
        arms = np.array([config.MENS_ARM] * 10, dtype=object)
        scores = np.arange(10.0)
        policy = greedy_allocation(arms, scores, n_sends=3)
        assert set(np.flatnonzero(policy != NO_EMAIL)) == {7, 8, 9}

    def test_each_chosen_customer_gets_their_own_arm(self):
        arms = np.array(
            [config.MENS_ARM, config.WOMENS_ARM] * 5, dtype=object
        )
        policy = greedy_allocation(arms, np.arange(10.0), n_sends=4)
        chosen = policy != NO_EMAIL
        assert (policy[chosen] == arms[chosen]).all()

    def test_ineligible_customers_are_never_emailed(self):
        arms = np.array([config.MENS_ARM] * 10, dtype=object)
        eligible = np.zeros(10, dtype=bool)
        eligible[:2] = True

        policy = greedy_allocation(arms, np.arange(10.0), n_sends=8, eligible=eligible)
        assert (policy != NO_EMAIL).sum() == 2
        assert set(np.flatnonzero(policy != NO_EMAIL)) == {0, 1}

    def test_eligibility_can_leave_budget_unspent(self):
        """More budget than eligible customers means the budget cannot be used
        up -- which is a finding about the file, not a failure to allocate."""
        arms = np.array([config.MENS_ARM] * 10, dtype=object)
        eligible = np.arange(10) < 3

        policy = greedy_allocation(arms, np.arange(10.0), n_sends=10, eligible=eligible)
        assert (policy != NO_EMAIL).sum() == 3

    def test_greedy_is_exactly_optimal_not_merely_good(self):
        """Equal send costs turn the knapsack into a top-k selection.

        Brute-forced against every subset of the right size. This is the
        justification for not reaching for an LP solver, so it is worth pinning
        down rather than asserting in a comment.
        """
        rng = np.random.default_rng(0)
        n, k = 14, 6
        scores = rng.normal(size=n)
        arms = np.array([config.MENS_ARM] * n, dtype=object)

        policy = greedy_allocation(arms, scores, n_sends=k)
        chosen = scores[policy != NO_EMAIL].sum()
        best = max(
            scores[list(subset)].sum()
            for subset in itertools.combinations(range(n), k)
        )
        assert chosen == pytest.approx(best)

    def test_ranking_is_deterministic_under_ties(self):
        arms = np.array([config.MENS_ARM] * 8, dtype=object)
        scores = np.ones(8)
        first = greedy_allocation(arms, scores, n_sends=4)
        second = greedy_allocation(arms, scores, n_sends=4)
        assert (first == second).all()


# ==========================================================================
# Valuing an allocation
# ==========================================================================
class TestAllocationValue:
    def test_baseline_is_the_control_arms_profit(self, processed_df):
        """IPW on the profit column must reproduce the control arm's mean
        profit, scaled to the whole file -- the Feature 9 identity again."""
        frame = with_profit(processed_df, margin=0.30, cost=0.10)
        observed = frame.loc[
            frame[config.TREATMENT_COL] == config.CONTROL_ARM, PROFIT_COLUMN
        ].mean()

        assert baseline_profit(frame) == pytest.approx(
            observed * len(frame), rel=1e-12
        )

    def test_full_send_matches_that_arms_profit(self, processed_df):
        frame = with_profit(processed_df, margin=0.30, cost=0.10)
        policy = policy_always(config.MENS_ARM, len(frame))
        observed = frame.loc[
            frame[config.TREATMENT_COL] == config.MENS_ARM, PROFIT_COLUMN
        ].mean()

        value = policy_value(frame, policy, PROFIT_COLUMN, "mens")
        assert value.value == pytest.approx(observed, rel=1e-12)

    def test_zero_budget_gains_nothing_over_the_baseline(self, processed_df):
        frame = with_profit(processed_df, margin=0.30, cost=0.10)
        reference = baseline_profit(frame)
        arms = np.array([config.MENS_ARM] * len(frame), dtype=object)

        allocation = evaluate_allocation(
            frame, greedy_allocation(arms, np.zeros(len(frame)), 0),
            "none", budget=0.0, baseline=reference,
        )
        assert allocation.n_sends == 0
        assert allocation.cost_spent == 0.0
        assert allocation.net_gain == pytest.approx(0.0, abs=1e-6)

    def test_return_on_spend_uses_the_gain_not_the_total(self):
        """A regression test with teeth.

        Customers spend money whether or not they are emailed, so dividing
        *total* profit by the send cost credits the campaign with revenue it did
        not cause and inflates the return by an order of magnitude.
        """
        allocation = Allocation(
            ranking="x", budget=1000.0, n_sends=1000, cost_spent=100.0,
            profit_per_customer=0.0, standard_error=0.0, ci_low=0.0, ci_high=0.0,
            total_profit=1200.0, baseline_profit=1000.0, n_customers=1000,
        )
        assert allocation.net_gain == pytest.approx(200.0)
        assert allocation.return_on_spend == pytest.approx(2.0)

    def test_return_on_spend_is_undefined_when_nothing_is_sent(self):
        allocation = Allocation(
            ranking="x", budget=0.0, n_sends=0, cost_spent=0.0,
            profit_per_customer=0.0, standard_error=0.0, ci_low=0.0, ci_high=0.0,
            total_profit=1000.0, baseline_profit=1000.0, n_customers=1000,
        )
        assert np.isnan(allocation.return_on_spend)
        assert allocation.budget_used == 0.0

    def test_the_curve_never_overspends_its_budget(self, budget_results):
        curves = budget_results["curves"]
        assert (curves["cost_spent"] <= curves["budget"] + 1e-9).all()

    def test_the_curve_covers_every_ranking_at_every_budget(self, budget_results):
        curves = budget_results["curves"]
        counts = curves.groupby("ranking")["budget"].nunique()
        assert (counts == counts.iloc[0]).all()


# ==========================================================================
# Predicted spend uplift
# ==========================================================================
class TestSpendUplifts:
    def test_scores_every_customer_for_every_arm(self, budget_results, processed_df):
        uplifts = budget_results["spend_uplifts"]
        assert len(uplifts) == len(processed_df)
        assert uplifts.notna().all().all()

        for arm, _ in config.COMPARISONS:
            assert f"spend_uplift_{arm}" in uplifts.columns

    def test_scores_customers_outside_the_arm_being_modelled(
        self, budget_results, processed_df
    ):
        """Feature 9's pairwise version leaves these NaN, which is enough to
        value an arm but not to choose one."""
        womens_rows = (
            processed_df[config.TREATMENT_COL] == config.WOMENS_ARM
        ).to_numpy()
        mens = budget_results["spend_uplifts"].loc[
            womens_rows, f"spend_uplift_{config.MENS_ARM}"
        ]

        assert len(mens) == 21_387
        assert mens.notna().all()

    def test_best_arm_picks_the_higher_score(self):
        scores = pd.DataFrame({
            f"s_{config.MENS_ARM}": [1.0, 0.0],
            f"s_{config.WOMENS_ARM}": [0.0, 1.0],
        })
        arms, best = best_arm_by_score(scores)

        assert list(arms) == [config.MENS_ARM, config.WOMENS_ARM]
        assert best == pytest.approx([1.0, 1.0])


# ==========================================================================
# The optimiser works when there is profit to find
# ==========================================================================
def test_profit_ranking_beats_random_when_a_responder_segment_exists():
    """Validation on data built so that targeting must pay.

    30% of customers respond; the rest do not. At a budget covering only a
    third of the file, a ranking that finds the responders should beat a random
    one by a wide margin. Without this the real data's null result would be
    indistinguishable from a broken optimiser.
    """
    data = synthetic_profit_experiment(n=30_000, seed=1)
    frame = with_profit(data, margin=0.30, cost=0.10)

    uplifts = cross_fitted_arm_spend_uplifts(data, features=["responder"])
    columns = [f"spend_uplift_{arm}" for arm, _ in config.COMPARISONS]
    arms, scores = best_arm_by_score(uplifts[columns])

    rng = np.random.default_rng(0)
    n_sends = len(data) // 3
    reference = baseline_profit(frame)

    targeted = evaluate_allocation(
        frame, greedy_allocation(arms, scores, n_sends), "targeted",
        budget=n_sends * 0.10, baseline=reference,
    )
    random_order = evaluate_allocation(
        frame, greedy_allocation(arms, rng.random(len(data)), n_sends), "random",
        budget=n_sends * 0.10, baseline=reference,
    )

    assert targeted.net_gain > random_order.net_gain
    assert targeted.return_on_spend > random_order.return_on_spend


def test_targeting_concentrates_the_budget_on_responders():
    data = synthetic_profit_experiment(n=30_000, seed=2)
    uplifts = cross_fitted_arm_spend_uplifts(data, features=["responder"])
    columns = [f"spend_uplift_{arm}" for arm, _ in config.COMPARISONS]
    arms, scores = best_arm_by_score(uplifts[columns])

    policy = greedy_allocation(arms, scores, n_sends=len(data) // 3)
    responder = data["responder"].to_numpy().astype(bool)

    assert responder[policy != NO_EMAIL].mean() > 0.8


# ==========================================================================
# Sensitivity to the assumptions
# ==========================================================================
class TestSensitivity:
    def test_a_higher_margin_never_reduces_profit(self, processed_df):
        grid = sensitivity_grid(
            processed_df, margins=np.array([0.20, 0.40]), costs=np.array([0.10])
        )
        for arm in [config.MENS_ARM, config.WOMENS_ARM]:
            rows = grid[grid["treatment_arm"] == arm].sort_values("margin")
            assert rows["profit_per_email"].is_monotonic_increasing, arm

    def test_a_higher_cost_never_increases_profit(self, processed_df):
        grid = sensitivity_grid(
            processed_df, margins=np.array([0.30]), costs=np.array([0.05, 0.50])
        )
        for arm in [config.MENS_ARM, config.WOMENS_ARM]:
            rows = grid[grid["treatment_arm"] == arm].sort_values("cost")
            assert rows["profit_per_email"].is_monotonic_decreasing, arm

    def test_break_even_margin_does_not_depend_on_the_assumed_margin(
        self, processed_df
    ):
        """The break-even margin is a property of the spend effect and the send
        cost. Changing the margin you assume cannot move it."""
        grid = sensitivity_grid(
            processed_df, margins=np.array([0.20, 0.30, 0.40]),
            costs=np.array([0.10]),
        )
        for arm in [config.MENS_ARM, config.WOMENS_ARM]:
            values = grid.loc[grid["treatment_arm"] == arm, "break_even_margin"]
            assert values.nunique() == 1, arm


# ==========================================================================
# The real dataset
# ==========================================================================
class TestRealData:
    def test_ranking_does_not_matter_once_the_budget_covers_everyone(
        self, budget_results
    ):
        """A built-in check on the allocator.

        Priority order only bites while the budget binds. At a full-send budget
        the profit ranking and the random ranking select the same customers and
        assign the same arms, so their values must coincide exactly.
        """
        curves = budget_results["curves"]
        full = curves[curves["budget"] == curves["budget"].max()].set_index("ranking")

        assert full.loc["predicted profit", "total_profit"] == pytest.approx(
            full.loc["random ranking", "total_profit"]
        )

    def test_ranking_on_the_spend_model_does_not_beat_no_model_at_all(
        self, budget_results
    ):
        """Feature 8 found spend uplift too noisy to rank on. Acting on it
        anyway is measurably worse than simply sending Mens to everyone."""
        comparisons = budget_results["comparisons"].set_index("ranking")
        assert comparisons.loc["predicted profit", "difference"] < 0

    def test_ranking_on_visit_uplift_is_indistinguishable_from_no_model(
        self, budget_results
    ):
        """Consistent with Feature 9: the visit-based policy is directionally
        fine and its advantage is far below what this experiment can resolve."""
        comparisons = budget_results["comparisons"].set_index("ranking")
        visit = comparisons.loc["predicted visit uplift"]

        assert visit["p_value"] > config.ALPHA
        assert visit["ci_low"] < 0 < visit["ci_high"]

    def test_most_customers_clear_the_break_even_bar(self, budget_results):
        eligible = budget_results["eligible"]
        assert 0.5 < eligible.mean() < 1.0

    def test_spending_the_full_budget_still_pays(self, budget_results):
        curves = budget_results["curves"]
        full = curves[curves["budget"] == curves["budget"].max()]
        assert (full["net_gain"] > 0).all()
        assert (full["return_on_spend"] > 1.0).all()

    def test_writes_results_files(self, processed_df, tmp_path, monkeypatch):
        monkeypatch.setattr(config, "RESULTS_DIR", tmp_path)
        monkeypatch.setattr(config, "ensure_dirs", lambda: None)
        run_budget_analysis(processed_df, save=True)

        for name in [
            "10_campaign_economics.csv",
            "10_budget_curve.csv",
            "10_ranking_comparison.csv",
            "10_sensitivity.csv",
        ]:
            assert (tmp_path / name).exists()


def test_comparison_against_itself_is_exactly_zero(processed_df):
    frame = with_profit(processed_df)
    arms = np.array([config.MENS_ARM] * len(frame), dtype=object)
    scores = np.arange(len(frame), dtype=float)

    rankings = {"a": (arms, scores), "b": (arms, scores)}
    result = compare_rankings(frame, rankings, budget=1_000.0, versus="a")

    assert result.loc[0, "difference"] == pytest.approx(0.0)
    assert result.loc[0, "n_disagree"] == 0


def test_curve_reports_a_baseline_shared_by_every_row(processed_df):
    frame = with_profit(processed_df)
    arms = np.array([config.MENS_ARM] * len(frame), dtype=object)

    curve = budget_curve(
        frame, arms, np.arange(len(frame), dtype=float), "test",
        budgets=np.array([0.0, 500.0, 1_000.0]),
    )
    assert curve["baseline_profit"].nunique() == 1
    assert curve["total_profit"].iloc[0] == pytest.approx(
        curve["baseline_profit"].iloc[0]
    )
