"""Tests for the treatment effect estimation.

The statistical tests here are hand-rolled so their internals can be shown and
explained, which means they need independent verification. The most valuable
tests below cross-check them against scipy and statsmodels: agreement with a
mature implementation is what distinguishes a correct formula from a plausible
one.
"""

import numpy as np
import pandas as pd
import pytest
from scipy import stats
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.proportion import proportions_ztest

from src import config
from src.analysis.ab_test import (
    EffectEstimate,
    _bootstrap_means,
    bootstrap_effect,
    estimate_effect,
    run_ab_tests,
    two_proportion_test,
    welch_t_test,
)


@pytest.fixture(scope="module")
def ab_results(processed_df):
    return run_ab_tests(processed_df, save=False)


@pytest.fixture(scope="module")
def proportion_sample():
    """A 30% vs 25% conversion comparison with clearly unequal rates."""
    return np.array([1.0] * 300 + [0.0] * 700), np.array([1.0] * 250 + [0.0] * 750)


@pytest.fixture(scope="module")
def welch_sample():
    """Deliberately unequal variances (sd 2 vs sd 5), which is why Welch."""
    rng = np.random.default_rng(config.RANDOM_SEED)
    return rng.normal(1.0, 2.0, 500), rng.normal(0.0, 5.0, 700)


# ==========================================================================
# Two-proportion z-test
# ==========================================================================
class TestTwoProportionTest:
    def test_matches_statsmodels(self, proportion_sample):
        """Cross-check against a mature implementation of the same test."""
        treated, control = proportion_sample
        result = two_proportion_test(treated, control)

        expected_z, expected_p = proportions_ztest(
            count=[treated.sum(), control.sum()],
            nobs=[len(treated), len(control)],
        )
        assert result["test_statistic"] == pytest.approx(expected_z, rel=1e-9)
        assert result["p_value"] == pytest.approx(expected_p, rel=1e-9)

    def test_effect_is_the_difference_in_rates(self, proportion_sample):
        treated, control = proportion_sample
        result = two_proportion_test(treated, control)
        assert result["absolute_effect"] == pytest.approx(0.30 - 0.25)

    def test_interval_uses_the_unpooled_standard_error(self, proportion_sample):
        """The CI must not assume the null it is meant to be testing.

        The test statistic pools the arms (valid only under H0); the interval
        must not, or it would assume the very thing being estimated.
        """
        treated, control = proportion_sample
        result = two_proportion_test(treated, control)

        p_t, p_c = treated.mean(), control.mean()
        unpooled = np.sqrt(
            p_t * (1 - p_t) / len(treated) + p_c * (1 - p_c) / len(control)
        )
        p_pooled = (treated.sum() + control.sum()) / (len(treated) + len(control))
        pooled = np.sqrt(
            p_pooled * (1 - p_pooled) * (1 / len(treated) + 1 / len(control))
        )

        assert result["standard_error"] == pytest.approx(unpooled)
        assert not np.isclose(unpooled, pooled), "sample must distinguish the two"

    def test_interval_is_centred_on_the_effect(self, proportion_sample):
        treated, control = proportion_sample
        result = two_proportion_test(treated, control)
        midpoint = (result["ci_low"] + result["ci_high"]) / 2
        assert midpoint == pytest.approx(result["absolute_effect"])

    def test_identical_arms_give_no_effect(self):
        values = np.array([1.0] * 100 + [0.0] * 900)
        result = two_proportion_test(values, values.copy())

        assert result["absolute_effect"] == 0.0
        assert result["p_value"] == pytest.approx(1.0)
        assert result["ci_low"] < 0 < result["ci_high"]

    def test_lower_alpha_widens_the_interval(self, proportion_sample):
        treated, control = proportion_sample
        wide = two_proportion_test(treated, control, alpha=0.01)
        narrow = two_proportion_test(treated, control, alpha=0.10)

        assert (wide["ci_high"] - wide["ci_low"]) > (
            narrow["ci_high"] - narrow["ci_low"]
        )

    def test_handles_a_zero_rate_arm(self):
        """A never-converting control must not produce a divide-by-zero."""
        result = two_proportion_test(
            np.array([1.0] * 10 + [0.0] * 990), np.zeros(1_000)
        )
        assert np.isfinite(result["test_statistic"])
        assert np.isfinite(result["p_value"])


# ==========================================================================
# Welch's t-test
# ==========================================================================
class TestWelchTTest:
    def test_matches_scipy(self, welch_sample):
        treated, control = welch_sample
        result = welch_t_test(treated, control)

        expected = stats.ttest_ind(treated, control, equal_var=False)
        assert result["test_statistic"] == pytest.approx(
            expected.statistic, rel=1e-9
        )
        assert result["p_value"] == pytest.approx(expected.pvalue, rel=1e-9)

    def test_interval_matches_scipy(self, welch_sample):
        treated, control = welch_sample
        result = welch_t_test(treated, control)

        expected = stats.ttest_ind(
            treated, control, equal_var=False
        ).confidence_interval(confidence_level=0.95)
        assert result["ci_low"] == pytest.approx(expected.low, rel=1e-9)
        assert result["ci_high"] == pytest.approx(expected.high, rel=1e-9)

    def test_differs_from_student_under_unequal_variance(self, welch_sample):
        """Welch is chosen precisely because it does not assume equal variance;
        on these samples the two must therefore disagree."""
        treated, control = welch_sample
        welch = welch_t_test(treated, control)
        student = stats.ttest_ind(treated, control, equal_var=True)

        assert not np.isclose(welch["p_value"], student.pvalue, rtol=1e-3)

    def test_identical_arms_give_no_effect(self):
        rng = np.random.default_rng(config.RANDOM_SEED)
        values = rng.normal(0, 1, 500)
        result = welch_t_test(values, values.copy())

        assert result["absolute_effect"] == pytest.approx(0.0)
        assert result["p_value"] == pytest.approx(1.0)


# ==========================================================================
# Bootstrap
# ==========================================================================
class TestBootstrap:
    def test_binary_shortcut_matches_direct_resampling(self):
        """Resampling 0/1 data with replacement IS a binomial draw.

        The shortcut is an identity, not an approximation, so its bootstrap
        distribution must match brute-force resampling to sampling error.
        """
        rng = np.random.default_rng(config.RANDOM_SEED)
        values = np.array([1.0] * 300 + [0.0] * 700)

        shortcut = _bootstrap_means(values, 20_000, np.random.default_rng(1))

        n = len(values)
        brute = np.array(
            [rng.choice(values, size=n, replace=True).mean() for _ in range(2_000)]
        )

        assert shortcut.mean() == pytest.approx(brute.mean(), abs=0.005)
        assert shortcut.std() == pytest.approx(brute.std(), rel=0.10)

    def test_continuous_path_is_used_for_non_binary_data(self):
        """Values outside {0, 1} must not take the binomial shortcut."""
        values = np.array([0.0, 2.0] * 500)
        means = _bootstrap_means(values, 500, np.random.default_rng(0))

        assert means.min() >= 0.0
        assert means.max() <= 2.0
        assert means.mean() == pytest.approx(1.0, abs=0.05)

    def test_constant_data_gives_a_constant_mean(self):
        means = _bootstrap_means(np.full(100, 7.0), 200, np.random.default_rng(0))
        assert np.allclose(means, 7.0)

    def test_agrees_with_the_analytic_interval_on_well_behaved_data(self):
        """On near-normal data the bootstrap should reproduce Welch's CI."""
        rng = np.random.default_rng(config.RANDOM_SEED)
        treated, control = rng.normal(1, 1, 5_000), rng.normal(0, 1, 5_000)

        analytic = welch_t_test(treated, control)
        boot = bootstrap_effect(treated, control, n_boot=4_000)

        assert boot["absolute_ci_low"] == pytest.approx(analytic["ci_low"], abs=0.03)
        assert boot["absolute_ci_high"] == pytest.approx(analytic["ci_high"], abs=0.03)

    def test_is_reproducible(self):
        rng_values = np.array([1.0] * 100 + [0.0] * 400)
        first = bootstrap_effect(rng_values, rng_values.copy(), n_boot=500, seed=7)
        second = bootstrap_effect(rng_values, rng_values.copy(), n_boot=500, seed=7)
        assert first == second

    def test_interval_is_ordered(self):
        rng = np.random.default_rng(config.RANDOM_SEED)
        boot = bootstrap_effect(
            rng.normal(1, 1, 2_000), rng.normal(0, 1, 2_000), n_boot=1_000
        )
        assert boot["absolute_ci_low"] < boot["absolute_ci_high"]
        assert boot["relative_ci_low"] < boot["relative_ci_high"]


# ==========================================================================
# Effect estimation
# ==========================================================================
class TestEstimateEffect:
    def test_rejects_unknown_outcome(self, processed_df):
        with pytest.raises(ValueError, match="Unknown outcome"):
            estimate_effect(processed_df, "profit", config.MENS_ARM)

    @pytest.mark.parametrize(
        ("outcome", "expected_test"),
        [
            ("visit", "two-proportion z-test"),
            ("conversion", "two-proportion z-test"),
            ("spend", "Welch's t-test"),
        ],
    )
    def test_selects_the_test_by_outcome_type(
        self, processed_df, outcome, expected_test
    ):
        estimate = estimate_effect(processed_df, outcome, config.MENS_ARM)
        assert estimate.test_name == expected_test

    def test_uses_the_full_arms(self, processed_df):
        estimate = estimate_effect(processed_df, "visit", config.MENS_ARM)
        assert estimate.n_treated == 21_307
        assert estimate.n_control == 21_306

    def test_effect_matches_the_arm_means(self, processed_df):
        estimate = estimate_effect(processed_df, "visit", config.MENS_ARM)
        assert estimate.absolute_effect == pytest.approx(
            estimate.treated_mean - estimate.control_mean
        )
        assert estimate.relative_effect == pytest.approx(
            estimate.absolute_effect / estimate.control_mean
        )

    def test_means_match_pandas(self, processed_df):
        estimate = estimate_effect(processed_df, "spend", config.WOMENS_ARM)
        expected = processed_df.groupby(config.TREATMENT_COL, observed=True)["spend"].mean()

        assert estimate.treated_mean == pytest.approx(expected[config.WOMENS_ARM])
        assert estimate.control_mean == pytest.approx(expected[config.CONTROL_ARM])

    def test_finds_no_effect_when_treatment_is_relabelled_noise(self, processed_df):
        """Randomly relabelling control customers as treated must produce a
        null result -- a test that cannot fail to find an effect is useless."""
        rng = np.random.default_rng(config.RANDOM_SEED)
        control = processed_df[
            processed_df[config.TREATMENT_COL] == config.CONTROL_ARM
        ].copy()

        half = rng.random(len(control)) < 0.5
        control[config.TREATMENT_COL] = pd.Categorical(
            np.where(half, config.MENS_ARM, config.CONTROL_ARM),
            categories=[config.CONTROL_ARM, config.MENS_ARM, config.WOMENS_ARM],
        )

        estimate = estimate_effect(control, "visit", config.MENS_ARM)
        assert estimate.p_value > 0.05
        assert estimate.ci_low < 0 < estimate.ci_high


# ==========================================================================
# Full run and multiple-testing correction
# ==========================================================================
class TestRunABTests:
    def test_covers_every_comparison(self, ab_results):
        assert len(ab_results) == len(config.COMPARISONS) * len(config.OUTCOMES)
        assert set(ab_results["outcome"]) == set(config.OUTCOMES)
        assert set(ab_results["treatment_arm"]) == {
            config.MENS_ARM,
            config.WOMENS_ARM,
        }

    def test_adjusted_p_values_are_never_smaller(self, ab_results):
        """Correcting for multiplicity can only make evidence weaker."""
        assert (ab_results["p_value_adjusted"] >= ab_results["p_value"] - 1e-12).all()

    def test_correction_matches_statsmodels(self, ab_results):
        expected = multipletests(
            ab_results["p_value"], alpha=config.ALPHA, method="holm"
        )[1]
        assert np.allclose(ab_results["p_value_adjusted"], expected)

    def test_holm_is_at_least_as_powerful_as_bonferroni(self, ab_results):
        """Holm never rejects less than Bonferroni; that is why it is used."""
        bonferroni = multipletests(
            ab_results["p_value"], alpha=config.ALPHA, method="bonferroni"
        )[1]
        assert (ab_results["p_value_adjusted"] <= bonferroni + 1e-12).all()

    def test_significance_flag_uses_the_adjusted_values(self, ab_results):
        assert (
            ab_results["significant"]
            == (ab_results["p_value_adjusted"] < config.ALPHA)
        ).all()

    def test_all_effects_survive_correction(self, ab_results):
        """The headline result: every effect is significant after Holm."""
        assert ab_results["significant"].all()

    def test_effects_are_positive(self, ab_results):
        """Both emails helped on every outcome, and every CI excludes zero."""
        assert (ab_results["absolute_effect"] > 0).all()
        assert (ab_results["ci_low"] > 0).all()

    def test_mens_beats_womens_on_every_outcome(self, ab_results):
        indexed = ab_results.set_index(["treatment_arm", "outcome"])
        for outcome in config.OUTCOMES:
            assert (
                indexed.loc[(config.MENS_ARM, outcome), "absolute_effect"]
                > indexed.loc[(config.WOMENS_ARM, outcome), "absolute_effect"]
            )

    def test_writes_results_file(self, processed_df, tmp_path, monkeypatch):
        monkeypatch.setattr(config, "RESULTS_DIR", tmp_path)
        monkeypatch.setattr(config, "ensure_dirs", lambda: None)
        run_ab_tests(processed_df, save=True)
        assert (tmp_path / "04_ab_test_results.csv").exists()


# ==========================================================================
# Spend: does the normal approximation hold?
# ==========================================================================
def test_bootstrap_confirms_the_welch_interval_on_spend(processed_df):
    """Spend is ~99% zeros with a std dev ~14x its mean, so the normal
    approximation behind Welch's interval is worth verifying rather than
    assuming. The two intervals should agree at n > 21,000."""
    estimate = estimate_effect(processed_df, "spend", config.MENS_ARM)

    frame = processed_df[
        processed_df[config.TREATMENT_COL].isin([config.MENS_ARM, config.CONTROL_ARM])
    ]
    treated = frame.loc[frame[config.TREATMENT_COL] == config.MENS_ARM, "spend"]
    control = frame.loc[frame[config.TREATMENT_COL] == config.CONTROL_ARM, "spend"]

    boot = bootstrap_effect(treated, control, n_boot=5_000)

    assert boot["absolute_ci_low"] == pytest.approx(estimate.ci_low, abs=0.05)
    assert boot["absolute_ci_high"] == pytest.approx(estimate.ci_high, abs=0.05)


def test_effect_estimate_is_immutable():
    """Results are a record of what was computed, not a mutable scratchpad."""
    estimate = EffectEstimate(
        outcome="visit", treatment_arm="a", control_arm="b",
        n_treated=1, n_control=1, treated_mean=0.5, control_mean=0.4,
        absolute_effect=0.1, relative_effect=0.25, standard_error=0.01,
        ci_low=0.08, ci_high=0.12, test_statistic=10.0, p_value=0.001,
        test_name="test",
    )
    with pytest.raises((AttributeError, TypeError)):
        estimate.p_value = 0.5
