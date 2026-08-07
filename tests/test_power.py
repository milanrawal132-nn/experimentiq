"""Tests for the power and minimum detectable effect analysis.

Power calculations are easy to get subtly wrong and impossible to spot-check by
eye, so these lean on three things: cross-validation against statsmodels,
round-trip identities (the MDE must be exactly the effect that yields the
target power), and the scaling laws the formulae imply.
"""

import numpy as np
import pytest
from statsmodels.stats.power import zt_ind_solve_power

from src import config
from src.analysis.power import (
    mde_for_mean,
    mde_for_proportion,
    observed_power,
    power_curve,
    power_for_means,
    power_for_proportions,
    required_n_for_mean,
    required_n_for_proportion,
    run_power_analysis,
)


@pytest.fixture(scope="module")
def power_table(processed_df):
    return run_power_analysis(processed_df, save=False)


# ==========================================================================
# Cross-validation against statsmodels
# ==========================================================================
class TestAgreesWithStatsmodels:
    @pytest.mark.parametrize("n_per_arm", [500, 5_000, 21_306])
    def test_mde_for_mean_matches(self, n_per_arm):
        """MDE for a difference in means, against statsmodels' solver.

        statsmodels works in standardised units, so its effect size is
        multiplied back by the standard deviation.

        The tolerance is 1e-4 rather than 1e-6 because statsmodels' solve-for-
        effect-size path converges loosely: its answer lands ~7e-6 away from
        the requested power, while ours lands ~1e-13 away. The next test
        asserts that directly, since "agrees with statsmodels" is the weaker
        claim of the two.
        """
        sd = 11.5
        mine = mde_for_mean(sd, n_per_arm, alpha=config.ALPHA, power=0.80)

        standardised = zt_ind_solve_power(
            nobs1=n_per_arm, alpha=config.ALPHA, power=0.80, ratio=1.0
        )
        assert mine == pytest.approx(standardised * sd, rel=1e-4)

    @pytest.mark.parametrize("n_per_arm", [500, 5_000, 21_306])
    def test_mde_hits_the_requested_power_exactly(self, n_per_arm):
        """The defining property of the MDE, asserted to machine precision."""
        sd = 11.5
        mde = mde_for_mean(sd, n_per_arm, alpha=config.ALPHA, power=0.80)
        assert power_for_means(sd, mde, n_per_arm, config.ALPHA) == pytest.approx(
            0.80, abs=1e-12
        )

    @pytest.mark.parametrize("effect", [0.5, 1.0, 2.0])
    def test_required_n_for_mean_matches(self, effect):
        sd = 11.5
        mine = required_n_for_mean(sd, effect, alpha=config.ALPHA, power=0.80)

        expected = zt_ind_solve_power(
            effect_size=effect / sd, alpha=config.ALPHA, power=0.80, ratio=1.0
        )
        assert mine == pytest.approx(expected, rel=1e-6)


# ==========================================================================
# Round-trip identities
# ==========================================================================
class TestRoundTrips:
    def test_power_at_the_mde_is_the_target(self):
        """By definition, the MDE is the effect achieving the target power."""
        mde = mde_for_mean(11.5, 21_306, power=0.80)
        assert power_for_means(11.5, mde, 21_306) == pytest.approx(0.80, abs=1e-6)

    def test_power_at_the_proportion_mde_is_the_target(self):
        mde = mde_for_proportion(0.1062, 21_306, power=0.80)
        assert power_for_proportions(0.1062, mde, 21_306) == pytest.approx(
            0.80, abs=1e-6
        )

    def test_required_n_reproduces_the_mde(self):
        sd, effect = 11.5, 0.4
        n = required_n_for_mean(sd, effect, power=0.80)
        assert mde_for_mean(sd, int(round(n)), power=0.80) == pytest.approx(
            effect, rel=1e-3
        )

    def test_required_n_for_proportion_reproduces_the_mde(self):
        p_control, effect = 0.1062, 0.01
        n = required_n_for_proportion(p_control, effect, power=0.80)
        assert mde_for_proportion(p_control, int(round(n)), power=0.80) == pytest.approx(
            effect, rel=1e-3
        )


# ==========================================================================
# Monotonicity and scaling laws
# ==========================================================================
class TestScaling:
    def test_power_increases_with_effect(self):
        powers = [power_for_means(11.5, e, 21_306) for e in [0.1, 0.3, 0.5, 1.0]]
        assert powers == sorted(powers)

    def test_power_increases_with_sample_size(self):
        powers = [power_for_means(11.5, 0.4, n) for n in [1_000, 10_000, 50_000]]
        assert powers == sorted(powers)

    def test_power_at_no_effect_equals_alpha(self):
        """With no true effect, the rejection rate is the false positive rate."""
        assert power_for_means(11.5, 0.0, 21_306, alpha=0.05) == pytest.approx(0.05)

    def test_mde_shrinks_with_the_square_root_of_n(self):
        """Quadrupling the sample halves the detectable effect."""
        small = mde_for_mean(11.5, 10_000)
        large = mde_for_mean(11.5, 40_000)
        assert large == pytest.approx(small / 2, rel=1e-9)

    def test_required_n_grows_with_the_inverse_square_of_the_effect(self):
        """Halving the effect you want to detect quadruples the sample."""
        big_effect = required_n_for_mean(11.5, 1.0)
        small_effect = required_n_for_mean(11.5, 0.5)
        assert small_effect == pytest.approx(4 * big_effect, rel=1e-9)

    def test_mde_grows_with_variance(self):
        assert mde_for_mean(20.0, 21_306) > mde_for_mean(10.0, 21_306)

    def test_zero_effect_needs_infinite_sample(self):
        assert required_n_for_mean(11.5, 0.0) == float("inf")
        assert required_n_for_proportion(0.1, 0.0) == float("inf")


# ==========================================================================
# Proportion variance handling
# ==========================================================================
class TestProportionVariance:
    def test_accounts_for_the_treated_arm_variance_increase(self):
        """Raising a low base rate raises the binomial variance with it.

        The naive formula evaluates both arms at the control rate and so
        understates the standard error, giving an over-optimistic MDE. Ours
        must be the more conservative of the two.
        """
        p_control, n = 0.005726, 21_306
        mine = mde_for_proportion(p_control, n, power=0.80)

        z = 2.8015993  # z_{0.975} + z_{0.80}
        naive = z * np.sqrt(2 * p_control * (1 - p_control) / n)

        assert mine > naive
        assert mine == pytest.approx(naive, rel=0.15)

    def test_difference_is_negligible_for_a_mid_range_rate(self):
        """Near p = 0.5 the variance barely moves, so the two agree."""
        p_control, n = 0.50, 21_306
        mine = mde_for_proportion(p_control, n, power=0.80)

        z = 2.8015993
        naive = z * np.sqrt(2 * p_control * (1 - p_control) / n)
        assert mine == pytest.approx(naive, rel=0.02)


# ==========================================================================
# The observed-power fallacy
# ==========================================================================
class TestObservedPowerFallacy:
    def test_is_a_monotone_function_of_the_p_value(self):
        """The core of why observed power is uninformative.

        Holding the standard error fixed and varying the observed effect,
        observed power rises exactly as the p-value falls. It is a relabelling
        of the p-value, so it cannot corroborate it.
        """
        from scipy import stats

        se = 0.1
        effects = np.linspace(0.01, 0.6, 40)
        p_values = 2 * stats.norm.sf(np.abs(effects) / se)
        powers = np.array([observed_power(e, se, alpha=0.05) for e in effects])

        # Sorting by p-value must reverse-sort power, exactly.
        order = np.argsort(p_values)
        assert np.all(np.diff(powers[order]) <= 1e-12)

    def test_a_result_at_exactly_alpha_has_the_conventional_power(self):
        """A p-value of exactly 0.05 always yields ~50% observed power,
        whatever the effect or sample size -- the tell that it carries no
        independent information."""
        from scipy import stats

        for se in [0.01, 0.1, 10.0]:
            effect = stats.norm.ppf(0.975) * se  # p = 0.05 exactly
            assert observed_power(effect, se, alpha=0.05) == pytest.approx(
                0.50, abs=0.01
            )


# ==========================================================================
# Full analysis
# ==========================================================================
class TestRunPowerAnalysis:
    def test_covers_every_comparison(self, power_table):
        assert len(power_table) == len(config.COMPARISONS) * len(config.OUTCOMES)

    def test_mde_is_a_property_of_the_design_not_the_arm(self, power_table):
        """Both comparisons share the same control arm and arm size, so their
        MDEs must be identical -- the MDE depends on the design, not on which
        treatment happened to be applied."""
        for outcome in config.OUTCOMES:
            mdes = power_table.loc[
                power_table["outcome"] == outcome, "mde_absolute"
            ].unique()
            assert len(mdes) == 1

    def test_robust_flag_compares_the_interval_low_end_to_the_mde(self, power_table):
        assert (
            power_table["robust"]
            == (power_table["observed_ci_low"] > power_table["mde_absolute"])
        ).all()

    def test_visit_is_the_best_powered_outcome(self, power_table):
        """Visit has the highest base rate and lowest relative variance."""
        relative = power_table.groupby("outcome")["mde_relative"].first()
        assert relative["visit"] < relative["conversion"]
        assert relative["visit"] < relative["spend"]

    def test_womens_arm_results_are_not_robust(self, power_table):
        """The finding this feature exists to surface: the two weakest effects
        clear the threshold on their point estimate but not on their interval."""
        womens = power_table[power_table["treatment_arm"] == config.WOMENS_ARM]
        not_robust = set(womens.loc[~womens["robust"], "outcome"])
        assert not_robust == {"conversion", "spend"}

    def test_mens_arm_results_are_all_robust(self, power_table):
        mens = power_table[power_table["treatment_arm"] == config.MENS_ARM]
        assert mens["robust"].all()

    def test_writes_results_file(self, processed_df, tmp_path, monkeypatch):
        monkeypatch.setattr(config, "RESULTS_DIR", tmp_path)
        monkeypatch.setattr(config, "ensure_dirs", lambda: None)
        run_power_analysis(processed_df, save=True)
        assert (tmp_path / "05_power_analysis.csv").exists()


# ==========================================================================
# Curves
# ==========================================================================
class TestPowerCurve:
    def test_is_monotone_in_effect_size(self):
        curve = power_curve(
            "spend", control_mean=0.65, control_sd=11.5,
            n_per_arm=21_306, effects=np.linspace(0.01, 2.0, 50),
        )
        assert curve["power"].is_monotonic_increasing

    def test_stays_within_zero_and_one(self):
        curve = power_curve(
            "visit", control_mean=0.1062, control_sd=0.308,
            n_per_arm=21_306, effects=np.linspace(0.0, 0.2, 50),
        )
        assert curve["power"].between(0, 1).all()

    def test_uses_proportion_maths_for_binary_outcomes(self):
        """A binary outcome must not be routed through the means formula."""
        curve = power_curve(
            "visit", control_mean=0.1062, control_sd=0.308,
            n_per_arm=21_306, effects=np.array([0.02]),
        )
        assert curve["power"].iloc[0] == pytest.approx(
            power_for_proportions(0.1062, 0.02, 21_306)
        )
