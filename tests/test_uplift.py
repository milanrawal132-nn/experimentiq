"""Tests for the uplift models.

An uplift model cannot be scored row by row, which makes it unusually easy to
ship one that has learned nothing. The tests here are built around that risk:

- Synthetic data with a *known* responder segment, where the model must find it.
- Synthetic data with *no* heterogeneity, where the model must find nothing.
- A leakage check that in-sample scoring inflates Qini, which is why every
  prediction in this module is out-of-fold.
"""

import numpy as np
import pandas as pd
import pytest

from src import config
from src.models.uplift import (
    LEARNERS,
    UpliftResult,
    build_design_matrix,
    evaluate_uplift,
    make_learner,
    out_of_fold_uplift,
    qini_null_distribution,
    regression_t_learner_uplift,
    run_uplift_analysis,
)

# Keeping the test suite quick: two contrasting learners rather than all five.
TEST_LEARNERS = ["t_learner", "logistic_t_learner"]
TEST_NULL_DRAWS = 200


def synthetic_uplift(
    n: int = 20_000,
    responder_uplift: float = 0.15,
    base_rate: float = 0.10,
    seed: int = 0,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
    """An experiment where exactly one known segment responds to treatment.

    Half the customers are 'responders'; treatment lifts only them. Two pure
    noise features are included so the model has to find the real one rather
    than being handed a single column.

    Returns (design, outcome, treatment, is_responder).
    """
    rng = np.random.default_rng(seed)

    responder = rng.random(n) < 0.5
    treated = rng.random(n) < 0.5
    noise_a, noise_b = rng.normal(size=n), rng.normal(size=n)

    probability = np.clip(base_rate + treated * responder * responder_uplift, 0, 1)
    outcome = rng.binomial(1, probability)

    design = pd.DataFrame(
        {"responder": responder.astype(float), "noise_a": noise_a, "noise_b": noise_b}
    )
    return design, outcome, treated.astype(int), responder


@pytest.fixture(scope="module")
def uplift_table(processed_df):
    return run_uplift_analysis(
        processed_df,
        learners=TEST_LEARNERS,
        n_null_draws=TEST_NULL_DRAWS,
        save=False,
    )


# ==========================================================================
# Features
# ==========================================================================
class TestDesignMatrix:
    def test_contains_no_outcome(self, processed_df):
        """The model must never see the outcome it is estimating the effect on."""
        design = build_design_matrix(processed_df.head(1_000))
        for outcome in config.OUTCOMES:
            assert not any(outcome in column for column in design.columns)

    def test_contains_no_treatment(self, processed_df):
        """Treatment is passed separately; leaking it into the features would
        let a T-learner's two models see which arm they were fit on."""
        design = build_design_matrix(processed_df.head(1_000))
        assert config.TREATMENT_COL not in design.columns

    def test_expands_categoricals(self, processed_df):
        design = build_design_matrix(processed_df.head(1_000))
        assert any("channel" in column for column in design.columns)
        assert design.dtypes.eq(float).all()


# ==========================================================================
# Learners
# ==========================================================================
class TestLearners:
    @pytest.mark.parametrize("name", LEARNERS)
    def test_every_learner_constructs(self, name):
        assert make_learner(name) is not None

    def test_rejects_unknown_learner(self):
        with pytest.raises(ValueError, match="Unknown learner"):
            make_learner("magic_learner")


# ==========================================================================
# The model finds a known segment
# ==========================================================================
class TestRecoversKnownUplift:
    @pytest.mark.parametrize("learner", TEST_LEARNERS)
    def test_ranks_the_responder_segment_highest(self, learner):
        """The core validation: a segment built to respond must be ranked up."""
        design, outcome, treatment, responder = synthetic_uplift(seed=1)
        uplift = out_of_fold_uplift(design, outcome, treatment, learner)

        assert uplift[responder].mean() > uplift[~responder].mean()

    def test_qini_far_exceeds_the_null(self):
        design, outcome, treatment, _ = synthetic_uplift(seed=2)
        uplift = out_of_fold_uplift(design, outcome, treatment, "t_learner")
        null = qini_null_distribution(outcome, treatment, TEST_NULL_DRAWS)

        result = evaluate_uplift(
            outcome, uplift, treatment, "synthetic", "visit", "t_learner", null
        )
        assert result.beats_random
        assert result.qini_z_score > 5

    def test_top_decile_uplift_exceeds_the_average_effect(self):
        """The practical claim a targeting policy depends on."""
        design, outcome, treatment, _ = synthetic_uplift(seed=3)
        uplift = out_of_fold_uplift(design, outcome, treatment, "t_learner")
        null = qini_null_distribution(outcome, treatment, TEST_NULL_DRAWS)

        result = evaluate_uplift(
            outcome, uplift, treatment, "synthetic", "visit", "t_learner", null
        )
        average_effect = (
            outcome[treatment == 1].mean() - outcome[treatment == 0].mean()
        )
        assert result.uplift_at_k["top_10pct"] > average_effect


# ==========================================================================
# The model finds nothing when there is nothing
# ==========================================================================
class TestFindsNothingInNoise:
    def test_uniform_effect_produces_no_ranking_signal(self):
        """Treatment helps everyone equally, so no ranking can beat random."""
        rng = np.random.default_rng(4)
        n = 20_000
        treated = rng.random(n) < 0.5
        outcome = rng.binomial(1, np.clip(0.10 + treated * 0.08, 0, 1))
        design = pd.DataFrame(rng.normal(size=(n, 3)), columns=list("abc"))

        uplift = out_of_fold_uplift(design, outcome, treated.astype(int), "t_learner")
        null = qini_null_distribution(outcome, treated.astype(int), TEST_NULL_DRAWS)
        result = evaluate_uplift(
            outcome, uplift, treated.astype(int), "synthetic", "visit",
            "t_learner", null,
        )
        assert result.qini_z_score < 3

    def test_no_treatment_effect_at_all(self):
        rng = np.random.default_rng(5)
        n = 20_000
        treated = rng.random(n) < 0.5
        outcome = rng.binomial(1, 0.10, n)
        design = pd.DataFrame(rng.normal(size=(n, 3)), columns=list("abc"))

        uplift = out_of_fold_uplift(design, outcome, treated.astype(int), "t_learner")
        null = qini_null_distribution(outcome, treated.astype(int), TEST_NULL_DRAWS)
        result = evaluate_uplift(
            outcome, uplift, treated.astype(int), "synthetic", "visit",
            "t_learner", null,
        )
        assert result.qini_z_score < 3


# ==========================================================================
# Leakage
# ==========================================================================
def test_in_sample_scoring_inflates_qini():
    """Why every prediction in this module is out-of-fold.

    A model scored on its own training rows ranks them using outcomes it has
    already seen. On data with no heterogeneity at all, that alone manufactures
    a Qini well above the honest one.
    """
    rng = np.random.default_rng(6)
    n = 12_000
    treated = rng.random(n) < 0.5
    outcome = rng.binomial(1, np.clip(0.10 + treated * 0.08, 0, 1))
    design = pd.DataFrame(rng.normal(size=(n, 8)), columns=[f"x{i}" for i in range(8)])

    model = make_learner("t_learner")
    model.fit(design, outcome, treated.astype(int))
    in_sample = model.predict(design)

    out_of_sample = out_of_fold_uplift(design, outcome, treated.astype(int), "t_learner")

    null = qini_null_distribution(outcome, treated.astype(int), TEST_NULL_DRAWS)
    leaked = evaluate_uplift(
        outcome, in_sample, treated.astype(int), "s", "visit", "t", null
    )
    honest = evaluate_uplift(
        outcome, out_of_sample, treated.astype(int), "s", "visit", "t", null
    )
    assert leaked.qini > honest.qini


# ==========================================================================
# The null distribution
# ==========================================================================
class TestNullDistribution:
    def test_is_centred_near_zero(self, processed_df):
        from src.data.load import make_comparison_frame

        frame = make_comparison_frame(processed_df, config.MENS_ARM)
        null = qini_null_distribution(
            frame["visit"].to_numpy(), frame["treated"].to_numpy(), TEST_NULL_DRAWS
        )
        assert abs(null.mean()) < 3 * null.std(ddof=1) / np.sqrt(len(null))

    def test_has_the_requested_length(self, processed_df):
        from src.data.load import make_comparison_frame

        frame = make_comparison_frame(processed_df, config.MENS_ARM)
        null = qini_null_distribution(
            frame["visit"].to_numpy(), frame["treated"].to_numpy(), 50
        )
        assert len(null) == 50

    def test_is_reproducible(self, processed_df):
        from src.data.load import make_comparison_frame

        frame = make_comparison_frame(processed_df, config.MENS_ARM)
        args = (frame["visit"].to_numpy(), frame["treated"].to_numpy(), 30)
        assert np.allclose(
            qini_null_distribution(*args, seed=1),
            qini_null_distribution(*args, seed=1),
        )


# ==========================================================================
# Reported statistics
# ==========================================================================
class TestReportedStatistics:
    def _result(self, qini, null):
        return UpliftResult(
            treatment_arm="a", outcome="visit", learner="t", n=100,
            qini=qini, qini_null_mean=float(null.mean()),
            qini_null_std=float(null.std(ddof=1)),
            qini_z_score=0.0, qini_percentile=0.0,
            p_value=max(float((null >= qini).mean()), 1 / len(null)),
            p_value_normal=0.01, uplift_auc=0.0,
        )

    def test_empirical_p_is_floored_at_the_simulation_resolution(self):
        """With 200 draws the smallest observable p is 1/200; reporting 0 would
        claim more certainty than the simulation can support."""
        null = np.random.default_rng(7).normal(0, 0.01, 200)
        result = self._result(qini=10.0, null=null)
        assert result.p_value == pytest.approx(1 / 200)

    def test_sidak_adjustment_increases_the_p_value(self):
        null = np.random.default_rng(8).normal(0, 0.01, 200)
        result = self._result(qini=0.02, null=null)
        assert result.p_value_adjusted(5) > result.p_value

    def test_sidak_with_one_learner_is_a_no_op(self):
        null = np.random.default_rng(9).normal(0, 0.01, 200)
        result = self._result(qini=0.02, null=null)
        assert result.p_value_adjusted(1) == pytest.approx(result.p_value)


# ==========================================================================
# Continuous outcomes
# ==========================================================================
def test_regression_t_learner_recovers_a_spend_segment():
    """The path Feature 10 needs, since sklift's learners are classifiers."""
    rng = np.random.default_rng(10)
    n = 20_000
    responder = rng.random(n) < 0.5
    treated = rng.random(n) < 0.5
    spend = rng.normal(0, 1, n) + treated * responder * 3.0

    design = pd.DataFrame(
        {"responder": responder.astype(float), "noise": rng.normal(size=n)}
    )
    uplift = regression_t_learner_uplift(design, spend, treated.astype(int))

    assert uplift[responder].mean() > uplift[~responder].mean()


# ==========================================================================
# The real dataset
# ==========================================================================
class TestRealData:
    def test_covers_every_arm_and_learner(self, uplift_table):
        assert len(uplift_table) == len(config.COMPARISONS) * len(TEST_LEARNERS)

    def test_womens_campaign_shows_strong_signal(self, uplift_table):
        """Feature 7 predicted this: the Womens effect varies 6.6x by purchase
        history, so an uplift model should rank customers well."""
        womens = uplift_table[uplift_table["treatment_arm"] == config.WOMENS_ARM]
        assert womens["significant"].all()
        assert womens["qini_z_score"].min() > 5

    def test_mens_campaign_shows_little_signal(self, uplift_table):
        """The counterpart prediction, and the check against overfitting: a
        model claiming strong signal on the uniform campaign would be wrong."""
        mens = uplift_table[uplift_table["treatment_arm"] == config.MENS_ARM]
        assert mens["qini_z_score"].max() < womens_floor()

    def test_womens_qini_far_exceeds_mens(self, uplift_table):
        womens = uplift_table[uplift_table["treatment_arm"] == config.WOMENS_ARM]
        mens = uplift_table[uplift_table["treatment_arm"] == config.MENS_ARM]
        assert womens["qini"].min() > 2 * mens["qini"].max()

    def test_writes_results_file(self, processed_df, tmp_path, monkeypatch):
        monkeypatch.setattr(config, "RESULTS_DIR", tmp_path)
        monkeypatch.setattr(config, "ensure_dirs", lambda: None)
        run_uplift_analysis(
            processed_df, learners=["logistic_t_learner"],
            n_null_draws=50, save=True,
        )
        assert (tmp_path / "08_uplift_models.csv").exists()


def womens_floor() -> float:
    """The z-score separating 'real signal' from 'noise' on this dataset.

    The Womens learners sit at z > 7 and the Mens learners below 3. Any
    threshold in that gap distinguishes them; 5 is used as the midpoint.
    """
    return 5.0
