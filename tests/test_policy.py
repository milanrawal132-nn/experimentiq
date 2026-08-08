"""Tests for the targeting policy and its off-policy evaluation.

Inverse propensity weighting has an exact property that makes it easy to
verify: the estimated value of "send everyone arm A" must equal the observed
mean outcome in arm A, to floating-point precision. Several tests below pin
that down, because an IPW estimator that is subtly wrong still returns
plausible-looking numbers.
"""

import numpy as np
import pandas as pd
import pytest

from src import config
from src.models.policy import (
    NO_EMAIL,
    build_policies,
    cross_fitted_arm_uplifts,
    policy_always,
    policy_best_campaign,
    policy_best_or_none,
    policy_difference,
    policy_random,
    policy_value,
    run_policy_analysis,
)


@pytest.fixture(scope="module")
def uplifts(processed_df):
    return cross_fitted_arm_uplifts(processed_df)


@pytest.fixture(scope="module")
def policy_results(processed_df):
    return run_policy_analysis(processed_df, save=False)


def synthetic_three_arm(n: int = 30_000, seed: int = 0) -> pd.DataFrame:
    """A three-arm experiment where the better campaign depends on a segment.

    Segment A responds to Mens, segment B to Womens. A policy that routes each
    segment to its own campaign must therefore beat any fixed single-arm
    policy -- which is the property the real data turns out not to have.
    """
    rng = np.random.default_rng(seed)
    segment_a = rng.random(n) < 0.5
    arm = rng.choice(
        [config.CONTROL_ARM, config.MENS_ARM, config.WOMENS_ARM], size=n
    )

    base = 0.10
    effect = np.zeros(n)
    effect[(arm == config.MENS_ARM) & segment_a] = 0.15
    effect[(arm == config.MENS_ARM) & ~segment_a] = 0.02
    effect[(arm == config.WOMENS_ARM) & segment_a] = 0.02
    effect[(arm == config.WOMENS_ARM) & ~segment_a] = 0.15

    return pd.DataFrame({
        "visit": rng.binomial(1, np.clip(base + effect, 0, 1)),
        "spend": 0.0,
        "segment_a": segment_a,
        config.TREATMENT_COL: pd.Categorical(
            arm,
            categories=[config.CONTROL_ARM, config.MENS_ARM, config.WOMENS_ARM],
        ),
    })


# ==========================================================================
# IPW recovers what it must
# ==========================================================================
class TestInversePropensityWeighting:
    @pytest.mark.parametrize("arm", [config.MENS_ARM, config.WOMENS_ARM])
    def test_single_arm_policy_equals_that_arms_mean(self, processed_df, arm):
        """The exact identity that validates the whole estimator.

        Weighting arm A's customers by 1/P(A) and averaging over everyone must
        reproduce arm A's own mean outcome. Any error in the propensities or
        the matching logic breaks this.
        """
        result = policy_value(
            processed_df, policy_always(arm, len(processed_df)), "visit", arm
        )
        observed = processed_df.loc[
            processed_df[config.TREATMENT_COL] == arm, "visit"
        ].mean()
        assert result.value == pytest.approx(observed, rel=1e-12)

    def test_email_nobody_equals_the_control_mean(self, processed_df):
        result = policy_value(
            processed_df, policy_always(NO_EMAIL, len(processed_df)), "visit", "none"
        )
        observed = processed_df.loc[
            processed_df[config.TREATMENT_COL] == config.CONTROL_ARM, "visit"
        ].mean()
        assert result.value == pytest.approx(observed, rel=1e-12)

    def test_only_matching_customers_contribute(self, processed_df):
        result = policy_value(
            processed_df,
            policy_always(config.MENS_ARM, len(processed_df)),
            "visit",
            "mens",
        )
        assert result.n_matched == 21_307
        assert result.coverage == pytest.approx(21_307 / 64_000)

    def test_interval_brackets_the_estimate(self, processed_df):
        result = policy_value(
            processed_df,
            policy_always(config.MENS_ARM, len(processed_df)),
            "visit",
            "mens",
        )
        assert result.ci_low < result.value < result.ci_high

    def test_spend_policy_also_recovers_the_arm_mean(self, processed_df):
        """The identity is not specific to a binary outcome."""
        result = policy_value(
            processed_df,
            policy_always(config.WOMENS_ARM, len(processed_df)),
            "spend",
            "womens",
        )
        observed = processed_df.loc[
            processed_df[config.TREATMENT_COL] == config.WOMENS_ARM, "spend"
        ].mean()
        assert result.value == pytest.approx(observed, rel=1e-12)


# ==========================================================================
# Paired comparison
# ==========================================================================
class TestPolicyDifference:
    def test_a_policy_against_itself_is_exactly_zero(self, processed_df):
        policy = policy_always(config.MENS_ARM, len(processed_df))
        result = policy_difference(processed_df, policy, policy, "visit")

        assert result["difference"] == pytest.approx(0.0)
        assert result["standard_error"] == pytest.approx(0.0)
        assert result["n_disagree"] == 0

    def test_pairing_beats_treating_the_policies_as_independent(
        self, processed_df, uplifts
    ):
        """Why the comparison is paired.

        Two policies that agree on most customers contribute exactly zero
        difference wherever they agree. Treating their marginal standard errors
        as independent would count those shared customers twice and overstate
        the uncertainty substantially.
        """
        learned = policy_best_campaign(uplifts)
        baseline = policy_always(config.MENS_ARM, len(processed_df))

        paired = policy_difference(processed_df, learned, baseline, "visit")
        marginal_a = policy_value(processed_df, learned, "visit", "a")
        marginal_b = policy_value(processed_df, baseline, "visit", "b")

        naive = np.sqrt(
            marginal_a.standard_error**2 + marginal_b.standard_error**2
        )
        assert paired["standard_error"] < naive

    def test_difference_matches_the_two_values(self, processed_df, uplifts):
        learned = policy_best_campaign(uplifts)
        baseline = policy_always(config.MENS_ARM, len(processed_df))

        paired = policy_difference(processed_df, learned, baseline, "visit")
        value_a = policy_value(processed_df, learned, "visit", "a").value
        value_b = policy_value(processed_df, baseline, "visit", "b").value

        assert paired["difference"] == pytest.approx(value_a - value_b, abs=1e-12)


# ==========================================================================
# Policy construction
# ==========================================================================
class TestPolicies:
    def test_best_campaign_always_sends_something(self, uplifts):
        assignment = policy_best_campaign(uplifts)
        assert NO_EMAIL not in set(assignment)
        assert set(assignment) <= {config.MENS_ARM, config.WOMENS_ARM}

    def test_best_campaign_picks_the_higher_predicted_uplift(self, uplifts):
        assignment = policy_best_campaign(uplifts)
        mens_better = (
            uplifts[f"uplift_{config.MENS_ARM}"]
            >= uplifts[f"uplift_{config.WOMENS_ARM}"]
        )
        assert (assignment[mens_better.to_numpy()] == config.MENS_ARM).all()

    def test_threshold_withholds_email_when_uplift_is_low(self, uplifts):
        """A high threshold must send nobody anything."""
        assignment = policy_best_or_none(uplifts, threshold=10.0)
        assert (assignment == NO_EMAIL).all()

    def test_zero_threshold_matches_best_campaign_where_uplift_is_positive(
        self, uplifts
    ):
        best = policy_best_campaign(uplifts)
        gated = policy_best_or_none(uplifts, threshold=0.0)
        positive = uplifts.max(axis=1).to_numpy() > 0

        assert (gated[positive] == best[positive]).all()
        assert (gated[~positive] == NO_EMAIL).all()

    def test_random_policy_uses_all_three_actions(self, processed_df):
        assignment = policy_random(len(processed_df))
        assert set(assignment) == {config.MENS_ARM, config.WOMENS_ARM, NO_EMAIL}

    def test_build_policies_returns_full_length_assignments(
        self, processed_df, uplifts
    ):
        for name, policy in build_policies(processed_df, uplifts).items():
            assert len(policy) == len(processed_df), name


# ==========================================================================
# Cross-fitted uplift
# ==========================================================================
class TestCrossFittedUplifts:
    def test_scores_every_customer_for_every_arm(self, processed_df, uplifts):
        assert len(uplifts) == len(processed_df)
        for arm, _ in config.COMPARISONS:
            assert f"uplift_{arm}" in uplifts.columns
        assert uplifts.notna().all().all()

    def test_scores_customers_outside_the_arm_being_modelled(
        self, processed_df, uplifts
    ):
        """A Womens-arm customer still needs a Mens uplift estimate, because
        the policy may want to assign them Mens."""
        womens_rows = processed_df[config.TREATMENT_COL] == config.WOMENS_ARM
        mens_uplift = uplifts.loc[womens_rows.to_numpy(), f"uplift_{config.MENS_ARM}"]

        assert len(mens_uplift) == 21_387
        assert mens_uplift.notna().all()
        assert mens_uplift.std() > 0


# ==========================================================================
# The policy works when there is something to find
# ==========================================================================
def test_policy_beats_single_arm_when_segments_differ():
    """Validation on data built so that personalisation must pay.

    Segment A responds to Mens, segment B to Womens. A learned policy should
    beat both fixed single-arm policies -- which is what makes the real data's
    null result informative rather than a sign of broken code.
    """
    data = synthetic_three_arm(n=40_000, seed=1)
    uplifts = cross_fitted_arm_uplifts(
        data, outcome="visit", features=["segment_a"]
    )

    learned = policy_value(
        data, policy_best_campaign(uplifts), "visit", "learned"
    ).value
    all_mens = policy_value(
        data, policy_always(config.MENS_ARM, len(data)), "visit", "mens"
    ).value
    all_womens = policy_value(
        data, policy_always(config.WOMENS_ARM, len(data)), "visit", "womens"
    ).value

    assert learned > all_mens
    assert learned > all_womens


def test_policy_routes_each_synthetic_segment_to_its_own_campaign():
    data = synthetic_three_arm(n=40_000, seed=2)
    uplifts = cross_fitted_arm_uplifts(
        data, outcome="visit", features=["segment_a"]
    )
    assignment = policy_best_campaign(uplifts)

    segment_a = data["segment_a"].to_numpy()
    assert (assignment[segment_a] == config.MENS_ARM).mean() > 0.8
    assert (assignment[~segment_a] == config.WOMENS_ARM).mean() > 0.8


# ==========================================================================
# The real dataset
# ==========================================================================
class TestRealData:
    def test_every_policy_is_valued(self, policy_results):
        assert len(policy_results["values"]) == 6

    def test_emailing_anything_beats_emailing_nobody(self, policy_results):
        values = policy_results["values"].set_index("policy")["value"]
        assert values["everyone gets Mens"] > values["email nobody"]
        assert values["everyone gets Womens"] > values["email nobody"]

    def test_mens_beats_womens_as_a_blanket_policy(self, policy_results):
        values = policy_results["values"].set_index("policy")["value"]
        assert values["everyone gets Mens"] > values["everyone gets Womens"]

    def test_personalisation_does_not_significantly_beat_all_mens(
        self, policy_results
    ):
        """The headline finding, and the reason Feature 10 exists.

        The learned policy is directionally right but its aggregate gain is far
        below what this experiment could detect. If a future dataset overturns
        this, that is a real result worth noticing rather than a broken test.
        """
        differences = policy_results["differences"].set_index("policy")
        learned = differences.loc["best campaign per customer"]

        assert learned["difference"] > 0
        assert learned["p_value"] > config.ALPHA
        assert learned["ci_low"] < 0 < learned["ci_high"]

    def test_the_gain_is_below_the_experiments_detection_threshold(
        self, policy_results
    ):
        """Feature 5 put the visit MDE at 0.85pp. The personalisation gain is
        a fraction of that, so the null result is a power limitation rather
        than evidence the policy is worthless."""
        differences = policy_results["differences"].set_index("policy")
        gain = differences.loc["best campaign per customer", "difference"]
        assert 0 < gain < 0.0085

    def test_nobody_is_predicted_to_be_harmed_by_both_campaigns(
        self, policy_results
    ):
        """Consistent with Feature 7 finding no harmed subgroup."""
        assignment = policy_results["policies"][
            "best campaign, or none if uplift <= 0"
        ]
        assert (assignment == NO_EMAIL).mean() < 0.01

    def test_writes_results_files(self, processed_df, tmp_path, monkeypatch):
        monkeypatch.setattr(config, "RESULTS_DIR", tmp_path)
        monkeypatch.setattr(config, "ensure_dirs", lambda: None)
        run_policy_analysis(processed_df, save=True)

        for name in [
            "09_policy_values.csv",
            "09_policy_differences.csv",
            "09_customer_assignments.csv",
        ]:
            assert (tmp_path / name).exists()
