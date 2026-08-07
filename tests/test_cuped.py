"""Tests for CUPED variance reduction.

On this dataset CUPED delivers almost nothing. That makes these tests unusually
important: a null result is only informative if the implementation is known to
be correct, otherwise "the technique did not help" is indistinguishable from
"the code is wrong."

So the suite is built around synthetic data with a *known* correlation, where
the right answer can be computed in advance. If CUPED recovers a 49% variance
reduction at rho = 0.7 and visibly narrows confidence intervals there, then its
0.04% on real spend is a fact about the data, not a bug.
"""

import numpy as np
import pandas as pd
import pytest

from src import config
from src.analysis.cuped import (
    ancova_effect,
    apply_cuped,
    cupac_covariate,
    cuped_effect,
    cuped_theta,
    run_cuped_analysis,
    variance_reduction,
)


def synthetic_experiment(
    rho: float, n: int = 20_000, effect: float = 0.5, seed: int = 0
) -> pd.DataFrame:
    """A randomised experiment where outcome-covariate correlation is known.

    The covariate is pre-treatment: it is drawn independently of assignment
    and the treatment effect is added afterwards, exactly as in a real
    experiment.
    """
    rng = np.random.default_rng(seed)
    covariate = rng.normal(0, 1, n)
    baseline = rho * covariate + np.sqrt(1 - rho**2) * rng.normal(0, 1, n)
    treated = rng.random(n) < 0.5

    return pd.DataFrame(
        {
            "x": covariate,
            "y": baseline + effect * treated,
            "treated": treated.astype(int),
            config.TREATMENT_COL: pd.Categorical(
                np.where(treated, config.MENS_ARM, config.CONTROL_ARM),
                categories=[config.CONTROL_ARM, config.MENS_ARM, config.WOMENS_ARM],
            ),
        }
    )


@pytest.fixture(scope="module")
def cuped_table(processed_df):
    return run_cuped_analysis(processed_df, save=False, include_cupac=False)


# ==========================================================================
# The estimator itself
# ==========================================================================
class TestCupedEstimator:
    def test_theta_is_the_ols_slope(self):
        rng = np.random.default_rng(0)
        x = rng.normal(0, 2, 5_000)
        y = 3.0 + 1.7 * x + rng.normal(0, 1, 5_000)

        slope = np.polyfit(x, y, 1)[0]
        assert cuped_theta(y, x) == pytest.approx(slope, rel=1e-9)

    def test_adjustment_preserves_the_mean(self):
        """Centring the covariate is what keeps the estimator unbiased."""
        rng = np.random.default_rng(0)
        x = rng.normal(5, 2, 5_000)
        y = 2 * x + rng.normal(0, 1, 5_000)

        assert apply_cuped(y, x).mean() == pytest.approx(y.mean(), abs=1e-10)

    def test_theta_is_zero_for_a_constant_covariate(self):
        y = np.array([1.0, 2.0, 3.0, 4.0])
        assert cuped_theta(y, np.ones(4)) == 0.0

    def test_constant_covariate_leaves_the_outcome_untouched(self):
        y = np.array([1.0, 2.0, 3.0, 4.0])
        assert np.allclose(apply_cuped(y, np.ones(4)), y)

    @pytest.mark.parametrize("rho", [0.1, 0.3, 0.5, 0.7, 0.9])
    def test_variance_reduction_equals_rho_squared(self, rho):
        """The identity the whole technique rests on.

        This is the test that makes the null result on real data believable.

        `effect=0` here so the correlation is exactly rho. A treatment effect
        adds variance to the outcome but none to the pre-treatment covariate,
        which dilutes their correlation below its nominal value -- true of any
        real experiment, and worth isolating away from the identity itself.
        """
        data = synthetic_experiment(rho, n=200_000, effect=0.0, seed=1)
        adjusted = apply_cuped(data["y"].to_numpy(), data["x"].to_numpy())

        realised = variance_reduction(data["y"].to_numpy(), adjusted)
        assert realised == pytest.approx(rho**2, abs=0.01)

    def test_a_treatment_effect_dilutes_the_correlation(self):
        """The flip side of the above, asserted rather than assumed."""
        without = cuped_effect(
            synthetic_experiment(0.7, n=100_000, effect=0.0, seed=11),
            "y", config.MENS_ARM, covariate="x",
        )
        with_effect = cuped_effect(
            synthetic_experiment(0.7, n=100_000, effect=1.0, seed=11),
            "y", config.MENS_ARM, covariate="x",
        )
        assert with_effect.correlation < without.correlation

    def test_uncorrelated_covariate_achieves_nothing(self):
        data = synthetic_experiment(0.0, n=50_000, seed=2)
        adjusted = apply_cuped(data["y"].to_numpy(), data["x"].to_numpy())
        assert variance_reduction(data["y"].to_numpy(), adjusted) == pytest.approx(
            0.0, abs=0.005
        )


# ==========================================================================
# Behaviour on a strong covariate
# ==========================================================================
class TestWorksWhenTheCovariateIsGood:
    def test_narrows_the_interval_substantially(self):
        """A large variance cut must translate into a proportionally narrower
        interval: precision scales with the square root of the variance."""
        data = synthetic_experiment(0.7, n=50_000, seed=3)
        result = cuped_effect(data, "y", config.MENS_ARM, covariate="x")

        assert result.realised_variance_reduction > 0.40
        expected_narrowing = 1 - np.sqrt(1 - result.realised_variance_reduction)
        assert result.ci_narrowing == pytest.approx(expected_narrowing, abs=0.01)

    def test_recovers_the_true_effect(self):
        """Variance reduction must not come at the cost of bias."""
        data = synthetic_experiment(0.8, n=50_000, effect=0.5, seed=4)
        result = cuped_effect(data, "y", config.MENS_ARM, covariate="x")

        assert result.effect_cuped == pytest.approx(0.5, abs=0.02)
        assert result.ci_low_cuped < 0.5 < result.ci_high_cuped

    def test_agrees_with_the_unadjusted_estimate(self):
        """CUPED changes precision, not the answer."""
        data = synthetic_experiment(0.8, n=50_000, seed=5)
        result = cuped_effect(data, "y", config.MENS_ARM, covariate="x")
        assert result.effect_cuped == pytest.approx(result.effect_raw, abs=0.02)

    def test_equivalent_sample_multiplier_matches_the_variance_cut(self):
        data = synthetic_experiment(0.7, n=50_000, seed=6)
        result = cuped_effect(data, "y", config.MENS_ARM, covariate="x")
        assert result.equivalent_sample_multiplier == pytest.approx(
            1 / (1 - result.realised_variance_reduction), rel=1e-9
        )

    def test_stronger_covariates_reduce_more(self):
        reductions = [
            cuped_effect(
                synthetic_experiment(rho, n=50_000, seed=7),
                "y", config.MENS_ARM, covariate="x",
            ).realised_variance_reduction
            for rho in [0.2, 0.5, 0.8]
        ]
        assert reductions == sorted(reductions)


# ==========================================================================
# ANCOVA equivalence
# ==========================================================================
class TestAncovaEquivalence:
    def test_matches_cuped_with_one_covariate(self):
        """CUPED and regression adjustment are the same idea in two notations."""
        data = synthetic_experiment(0.7, n=50_000, seed=8)

        cuped = cuped_effect(data, "y", config.MENS_ARM, covariate="x")
        ancova = ancova_effect(data, "y", config.MENS_ARM, covariates=["x"])

        assert ancova["effect"] == pytest.approx(cuped.effect_cuped, abs=0.005)
        assert ancova["se"] == pytest.approx(cuped.se_cuped, rel=0.02)

    def test_matches_on_real_data(self, processed_df):
        cuped = cuped_effect(processed_df, "spend", config.MENS_ARM)
        ancova = ancova_effect(processed_df, "spend", config.MENS_ARM)

        assert ancova["effect"] == pytest.approx(cuped.effect_cuped, rel=0.01)
        assert ancova["se"] == pytest.approx(cuped.se_cuped, rel=0.02)


# ==========================================================================
# CUPAC cross-fitting
# ==========================================================================
class TestCupac:
    def test_returns_one_prediction_per_row(self, processed_df):
        predictions = cupac_covariate(
            processed_df, "visit", config.MENS_ARM, n_splits=3
        )
        assert len(predictions) == 21_307 + 21_306
        assert predictions.notna().all()

    def test_beats_a_single_covariate_on_visit(self, processed_df):
        """Where signal exists, pooling covariates should find more of it."""
        single = cuped_effect(processed_df, "visit", config.MENS_ARM)
        learned = cupac_covariate(processed_df, "visit", config.MENS_ARM)
        combined = cuped_effect(
            processed_df, "visit", config.MENS_ARM,
            covariate_values=learned, covariate_name="cupac",
        )
        assert (
            combined.realised_variance_reduction
            > single.realised_variance_reduction
        )

    def test_cross_fitting_prevents_overstated_reduction(self, processed_df):
        """Without cross-fitting the model has seen each row's own outcome, so
        its prediction tracks that row's noise and the variance reduction is
        inflated. The out-of-fold version must be the more conservative one."""
        from sklearn.ensemble import HistGradientBoostingRegressor

        from src.data.load import make_comparison_frame

        frame = make_comparison_frame(processed_df, config.MENS_ARM)
        design = pd.get_dummies(
            frame[config.PRE_TREATMENT_COVARIATES], drop_first=False
        ).astype(float)
        y = frame["visit"].to_numpy(dtype=float)

        in_sample_model = HistGradientBoostingRegressor(
            max_iter=200, max_depth=4, learning_rate=0.08,
            random_state=config.RANDOM_SEED,
        ).fit(design, y)
        in_sample = variance_reduction(
            y, apply_cuped(y, in_sample_model.predict(design))
        )

        out_of_fold = variance_reduction(
            y,
            apply_cuped(
                y, cupac_covariate(processed_df, "visit", config.MENS_ARM).to_numpy()
            ),
        )
        assert in_sample > out_of_fold

    def test_rejects_a_mismatched_covariate_length(self, processed_df):
        with pytest.raises(ValueError, match="length"):
            cuped_effect(
                processed_df, "visit", config.MENS_ARM,
                covariate_values=pd.Series([1.0, 2.0, 3.0]),
            )


# ==========================================================================
# The real dataset
# ==========================================================================
class TestRealData:
    def test_covers_every_comparison(self, cuped_table):
        assert len(cuped_table) == len(config.COMPARISONS) * len(config.OUTCOMES)

    def test_variance_reduction_is_negligible(self, cuped_table):
        """The headline finding: prior spend is too weak a predictor to help.

        If a future dataset ever breaks this, that is good news worth noticing
        rather than a broken test.
        """
        assert cuped_table["realised_variance_reduction"].max() < 0.01

    def test_spend_reduction_is_essentially_zero(self, cuped_table):
        spend = cuped_table[cuped_table["outcome"] == "spend"]
        assert spend["realised_variance_reduction"].max() < 0.001

    def test_total_reduction_matches_the_predicted_rho_squared(self, cuped_table):
        """Confirms the identity holds on real data too, not just synthetic."""
        assert np.allclose(
            cuped_table["total_variance_reduction"],
            cuped_table["predicted_variance_reduction"],
            atol=1e-9,
        )

    def test_the_two_variance_definitions_agree_here(self, cuped_table):
        """Within-arm and total variance reduction differ only by the treatment
        effect's contribution to total variance. Every effect in this dataset
        is tiny next to the outcome's spread, so the two coincide -- which is
        why the distinction is safe to gloss over in most write-ups, and worth
        measuring rather than assuming."""
        assert np.allclose(
            cuped_table["realised_variance_reduction"],
            cuped_table["total_variance_reduction"],
            atol=1e-4,
        )

    def test_effect_estimates_are_preserved(self, cuped_table):
        """The adjustment corrects for residual covariate imbalance, which
        Feature 3 showed is tiny, so the estimates should barely move."""
        relative_shift = (
            (cuped_table["effect_cuped"] - cuped_table["effect_raw"]).abs()
            / cuped_table["effect_raw"].abs()
        )
        assert relative_shift.max() < 0.01

    def test_conclusions_are_unchanged(self, cuped_table):
        """Every effect significant before adjustment is significant after."""
        assert (
            (cuped_table["p_value_raw"] < config.ALPHA)
            == (cuped_table["p_value_cuped"] < config.ALPHA)
        ).all()

    def test_writes_results_file(self, processed_df, tmp_path, monkeypatch):
        monkeypatch.setattr(config, "RESULTS_DIR", tmp_path)
        monkeypatch.setattr(config, "ensure_dirs", lambda: None)
        run_cuped_analysis(processed_df, save=True, include_cupac=False)
        assert (tmp_path / "06_cuped.csv").exists()
