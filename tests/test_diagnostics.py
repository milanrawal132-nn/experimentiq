"""Tests for the randomisation diagnostics.

A diagnostic that always passes is worthless, so most of these tests build a
deliberately broken experiment and assert the diagnostic catches it. The
`confounded_df` fixture reassigns treatment as a function of `recency`, which
is exactly the failure randomisation exists to prevent.
"""

import numpy as np
import pandas as pd
import pytest

from src import config
from src.analysis.diagnostics import (
    covariate_balance,
    omnibus_balance_test,
    run_diagnostics,
    srm_test,
    standardised_mean_difference,
)


def _reassign(frame: pd.DataFrame, treated_mask: np.ndarray) -> pd.DataFrame:
    """Rebuild the treatment column from a boolean mask, preserving dtype."""
    out = frame.copy()
    out[config.TREATMENT_COL] = pd.Categorical(
        np.where(treated_mask, config.MENS_ARM, config.CONTROL_ARM),
        categories=[config.CONTROL_ARM, config.MENS_ARM, config.WOMENS_ARM],
    )
    return out


@pytest.fixture(scope="module")
def confounded_df(processed_df):
    """An experiment where assignment is *biased* by a pre-treatment covariate.

    Recent purchasers are assigned to the Mens arm with probability 0.8 and
    everyone else with probability 0.2. This is the realistic failure mode --
    a skewed assignment mechanism rather than a deterministic one -- and it
    leaves enough overlap for the logistic model to fit, so the omnibus test
    is exercised on its normal path rather than its separation path.
    """
    rng = np.random.default_rng(config.RANDOM_SEED)
    probability = np.where(processed_df["recency"] <= 4, 0.8, 0.2)
    return _reassign(processed_df, rng.random(len(processed_df)) < probability)


@pytest.fixture(scope="module")
def separated_df(processed_df):
    """An experiment where a covariate predicts assignment *exactly*."""
    return _reassign(processed_df, (processed_df["recency"] <= 4).to_numpy())


@pytest.fixture(scope="module")
def balance(processed_df):
    return covariate_balance(processed_df, config.MENS_ARM)


@pytest.fixture(scope="module")
def diagnostics(processed_df):
    return run_diagnostics(processed_df, save=False)


# ==========================================================================
# Sample ratio mismatch
# ==========================================================================
class TestSRM:
    def test_real_experiment_passes(self, processed_df):
        result = srm_test(processed_df)
        assert result.passed
        assert result.p_value > 0.05
        assert result.dof == 2
        assert "No sample ratio mismatch" in result.verdict

    def test_counts_sum_to_dataset_size(self, processed_df):
        result = srm_test(processed_df)
        assert result.counts.sum() == 64_000
        assert result.expected.sum() == pytest.approx(64_000)

    def test_detects_a_dropped_arm(self, processed_df):
        """Silently losing a fifth of one arm is the classic SRM signature."""
        mens = processed_df[processed_df[config.TREATMENT_COL] == config.MENS_ARM]
        broken = pd.concat(
            [
                processed_df[processed_df[config.TREATMENT_COL] != config.MENS_ARM],
                mens.head(int(len(mens) * 0.8)),
            ]
        )
        result = srm_test(broken)
        assert not result.passed
        assert result.p_value < config.SRM_ALPHA
        assert "SAMPLE RATIO MISMATCH" in result.verdict

    def test_honours_an_unequal_intended_split(self, processed_df):
        """A 50/25/25 design should fail against this equal-split data."""
        result = srm_test(
            processed_df,
            expected_shares={
                config.CONTROL_ARM: 0.5,
                config.MENS_ARM: 0.25,
                config.WOMENS_ARM: 0.25,
            },
        )
        assert not result.passed

    def test_rejects_shares_that_do_not_sum_to_one(self, processed_df):
        with pytest.raises(ValueError, match="sum to 1"):
            srm_test(
                processed_df,
                expected_shares={
                    config.CONTROL_ARM: 0.5,
                    config.MENS_ARM: 0.2,
                    config.WOMENS_ARM: 0.2,
                },
            )

    def test_rejects_incomplete_shares(self, processed_df):
        with pytest.raises(ValueError, match="missing arms"):
            srm_test(
                processed_df,
                expected_shares={config.CONTROL_ARM: 0.5, config.MENS_ARM: 0.5},
            )


# ==========================================================================
# Standardised mean difference
# ==========================================================================
class TestStandardisedMeanDifference:
    def test_identical_arms_give_zero(self):
        values = np.array([1.0, 2.0, 3.0, 4.0])
        assert standardised_mean_difference(values, values) == 0.0

    def test_recovers_a_known_shift(self):
        rng = np.random.default_rng(config.RANDOM_SEED)
        control = rng.normal(0, 1, 200_000)
        treated = rng.normal(0.5, 1, 200_000)
        assert standardised_mean_difference(treated, control) == pytest.approx(
            0.5, abs=0.01
        )

    def test_is_antisymmetric(self):
        rng = np.random.default_rng(config.RANDOM_SEED)
        a, b = rng.normal(1, 1, 1_000), rng.normal(0, 1, 1_000)
        assert standardised_mean_difference(a, b) == pytest.approx(
            -standardised_mean_difference(b, a)
        )

    def test_is_scale_invariant(self):
        """The point of standardising: dollars and flags land on one scale."""
        rng = np.random.default_rng(config.RANDOM_SEED)
        a, b = rng.normal(1, 1, 1_000), rng.normal(0, 1, 1_000)
        assert standardised_mean_difference(a * 1_000, b * 1_000) == pytest.approx(
            standardised_mean_difference(a, b)
        )

    def test_binary_case_matches_the_proportion_formula(self):
        """For 0/1 data the general formula must reduce to the proportion one."""
        treated = np.array([1.0] * 300 + [0.0] * 700)
        control = np.array([1.0] * 250 + [0.0] * 750)

        p_t, p_c = 0.3, 0.25
        expected = (p_t - p_c) / np.sqrt(
            (p_t * (1 - p_t) + p_c * (1 - p_c)) / 2
        )
        assert standardised_mean_difference(treated, control) == pytest.approx(
            expected, rel=1e-3
        )

    def test_constant_arms_at_the_same_value_are_balanced(self):
        ones = np.ones(100)
        assert standardised_mean_difference(ones, ones) == 0.0

    def test_perfectly_separated_arms_are_maximally_imbalanced(self):
        """Zero variance with different means is total separation, not balance."""
        assert standardised_mean_difference(np.ones(100), np.zeros(100)) == np.inf
        assert standardised_mean_difference(np.zeros(100), np.ones(100)) == -np.inf


# ==========================================================================
# Covariate balance
# ==========================================================================
class TestCovariateBalance:
    def test_covers_every_covariate(self, balance):
        covered = set(balance["covariate"])
        expected = set(config.NUMERIC_COVARIATES) | set(config.CATEGORICAL_COVARIATES)
        assert covered == expected

    def test_expands_categoricals_into_levels(self, balance):
        levels = balance[balance["covariate"] == "history_segment"]
        assert len(levels) == 7

    def test_never_includes_an_outcome(self, balance):
        """Balance is a pre-treatment property. An outcome appearing here would
        mean the check had been silently inverted into a results table."""
        assert set(balance["covariate"]).isdisjoint(set(config.OUTCOMES))

    def test_excludes_recency_bucket(self, balance):
        """It is a deterministic function of recency; including both would
        double-count the same imbalance."""
        assert "recency_bucket" not in set(balance["covariate"])

    def test_real_experiment_is_balanced(self, balance):
        assert balance["balanced"].all()
        assert balance["abs_smd"].max() < config.SMD_THRESHOLD

    def test_balanced_flag_matches_threshold(self, balance):
        assert (
            balance["balanced"] == (balance["abs_smd"] < config.SMD_THRESHOLD)
        ).all()

    def test_difference_matches_the_arm_means(self, balance):
        assert np.allclose(
            balance["difference"], balance["treated_mean"] - balance["control_mean"]
        )

    def test_detects_confounded_assignment(self, confounded_df):
        """Recency drove assignment, so recency must show a large imbalance."""
        balance = covariate_balance(confounded_df, config.MENS_ARM)
        recency = balance[balance["covariate"] == "recency"].iloc[0]

        assert abs(recency["smd"]) > 1.0
        assert recency["p_value"] < 1e-10
        assert not recency["balanced"]


# ==========================================================================
# Omnibus balance
# ==========================================================================
class TestOmnibusBalance:
    def test_real_experiment_passes(self, processed_df):
        result = omnibus_balance_test(processed_df, config.MENS_ARM)
        assert result.passed
        assert result.p_value > config.ALPHA
        assert "do not jointly predict" in result.verdict

    def test_covariates_explain_almost_nothing(self, processed_df):
        """Under valid randomisation, assignment should be near-unpredictable."""
        result = omnibus_balance_test(processed_df, config.MENS_ARM)
        assert result.pseudo_r2 < 0.001

    def test_uses_both_arms_entirely(self, processed_df):
        result = omnibus_balance_test(processed_df, config.MENS_ARM)
        assert result.n == 21_307 + 21_306

    def test_detects_confounded_assignment(self, confounded_df):
        result = omnibus_balance_test(confounded_df, config.MENS_ARM)
        assert not result.passed
        assert result.p_value < 1e-10
        assert result.pseudo_r2 > 0.1
        assert "DO jointly predict" in result.verdict

    # The solver overflows on its way to failing; that is the condition under
    # test, not a defect worth surfacing.
    @pytest.mark.filterwarnings("ignore::RuntimeWarning")
    def test_reports_perfect_separation_instead_of_crashing(self, separated_df):
        """A covariate that predicts assignment exactly makes the logistic fit
        singular. That is the most extreme possible answer to the question, so
        it must be reported, not raised as a LinAlgError from the solver."""
        result = omnibus_balance_test(separated_df, config.MENS_ARM)

        assert result.separated
        assert not result.passed
        assert result.p_value == 0.0
        assert result.pseudo_r2 == 1.0
        assert "PERFECT SEPARATION" in result.verdict


# ==========================================================================
# Orchestration
# ==========================================================================
class TestRunDiagnostics:
    def test_returns_all_three_diagnostics(self, diagnostics):
        assert set(diagnostics) == {"srm", "balance", "omnibus"}

    def test_balance_covers_both_comparisons(self, diagnostics):
        assert set(diagnostics["balance"]["treatment_arm"]) == {
            config.MENS_ARM,
            config.WOMENS_ARM,
        }

    def test_omnibus_covers_both_comparisons(self, diagnostics):
        assert len(diagnostics["omnibus"]) == len(config.COMPARISONS)

    def test_everything_passes_on_the_real_experiment(self, diagnostics):
        assert diagnostics["srm"].passed
        assert diagnostics["balance"]["balanced"].all()
        assert diagnostics["omnibus"]["passed"].all()

    def test_writes_expected_files(self, processed_df, tmp_path, monkeypatch):
        monkeypatch.setattr(config, "RESULTS_DIR", tmp_path)
        monkeypatch.setattr(config, "ensure_dirs", lambda: None)
        run_diagnostics(processed_df, save=True)

        for name in [
            "03_srm.csv",
            "03_covariate_balance.csv",
            "03_omnibus_balance.csv",
        ]:
            assert (tmp_path / name).exists()
