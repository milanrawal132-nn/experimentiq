"""Tests for the heterogeneous treatment effect analysis.

The central claim of this feature is that heterogeneity must be tested by
interaction rather than by comparing per-subgroup p-values. That claim is
asserted directly: a synthetic experiment is built where both subgroups return
significant effects and the interaction between them returns nothing, which is
precisely the case the naive reading gets wrong.
"""

import numpy as np
import pandas as pd
import pytest

from src import config
from src.analysis.heterogeneity import (
    MIN_CELL_SIZE,
    interaction_test,
    run_heterogeneity_analysis,
    subgroup_effects,
)


def synthetic_subgroups(
    effect_group_a: float,
    effect_group_b: float,
    n: int = 40_000,
    base_rate: float = 0.10,
    seed: int = 0,
) -> pd.DataFrame:
    """A randomised experiment whose effect differs between two subgroups.

    Setting both effects equal produces data with no heterogeneity, which is
    what the null cases below rely on.
    """
    rng = np.random.default_rng(seed)
    in_group_a = rng.random(n) < 0.5
    treated = rng.random(n) < 0.5

    effect = np.where(in_group_a, effect_group_a, effect_group_b)
    probability = np.clip(base_rate + treated * effect, 0, 1)

    return pd.DataFrame(
        {
            "visit": rng.binomial(1, probability),
            "newbie": in_group_a.astype(int),
            "treated": treated.astype(int),
            config.TREATMENT_COL: pd.Categorical(
                np.where(treated, config.MENS_ARM, config.CONTROL_ARM),
                categories=[config.CONTROL_ARM, config.MENS_ARM, config.WOMENS_ARM],
            ),
        }
    )


def exact_subgroup_counts(
    visits_treated: tuple[int, int],
    visits_control: tuple[int, int],
    n_per_cell: int,
) -> pd.DataFrame:
    """Build an experiment with exactly the specified cell counts.

    Sampling would leave the demonstration at the mercy of the seed. Fixing the
    counts makes the arithmetic exact and the conclusion reproducible.

    Group index 0 is `newbie = 1` ("New customer"), index 1 is `newbie = 0`.
    """
    rows = []
    for group_index, newbie in enumerate((1, 0)):
        for treated, visits in (
            (1, visits_treated[group_index]),
            (0, visits_control[group_index]),
        ):
            rows.append(
                pd.DataFrame(
                    {
                        "visit": [1] * visits + [0] * (n_per_cell - visits),
                        "newbie": newbie,
                        "treated": treated,
                    }
                )
            )

    frame = pd.concat(rows, ignore_index=True)
    frame[config.TREATMENT_COL] = pd.Categorical(
        np.where(frame["treated"] == 1, config.MENS_ARM, config.CONTROL_ARM),
        categories=[config.CONTROL_ARM, config.MENS_ARM, config.WOMENS_ARM],
    )
    return frame


@pytest.fixture(scope="module")
def heterogeneity(processed_df):
    return run_heterogeneity_analysis(processed_df, save=False)


# ==========================================================================
# The interaction test detects what it should
# ==========================================================================
class TestInteractionTest:
    def test_detects_a_real_difference(self):
        data = synthetic_subgroups(0.15, 0.02, seed=1)
        result = interaction_test(data, "visit", config.MENS_ARM, "newbie")

        assert result.p_value < 1e-6
        assert result.df_num == 1
        assert result.levels == 2

    def test_finds_nothing_when_effects_are_equal(self):
        """Equal effects in both subgroups must not register as heterogeneity."""
        data = synthetic_subgroups(0.08, 0.08, seed=2)
        result = interaction_test(data, "visit", config.MENS_ARM, "newbie")
        assert result.p_value > 0.05

    def test_is_not_the_same_as_comparing_subgroup_p_values(self):
        """The error this whole feature exists to avoid.

        Both subgroups have a strong, significant effect, and the effects are
        identical. Reading two significant p-values as "the effect differs" is
        wrong; the interaction test correctly reports no difference.
        """
        data = synthetic_subgroups(0.10, 0.10, n=60_000, seed=3)

        levels = subgroup_effects(data, "visit", config.MENS_ARM, "newbie")
        assert (levels["p_value"] < 0.001).all(), "both subgroups significant"

        interaction = interaction_test(data, "visit", config.MENS_ARM, "newbie")
        assert interaction.p_value > 0.05, "yet the effects do not differ"

    def test_the_converse_case(self):
        """One subgroup significant, one not, yet no real difference.

        The commonest misreading in practice. A small effect that clears
        significance and a slightly smaller one that does not are easily
        indistinguishable from each other, and reporting "it works for new
        customers but not established ones" treats an accident of where the
        threshold fell as a finding.

        Built from exact cell counts rather than sampled, so the demonstration
        cannot flip on a different random draw.
        """
        data = exact_subgroup_counts(
            visits_treated=(682, 649), visits_control=(600, 600), n_per_cell=6_000
        )

        levels = subgroup_effects(
            data, "visit", config.MENS_ARM, "newbie"
        ).set_index("level")
        assert levels.loc["New customer", "p_value"] < config.ALPHA
        assert levels.loc["Established customer", "p_value"] > config.ALPHA

        interaction = interaction_test(data, "visit", config.MENS_ARM, "newbie")
        assert interaction.p_value > config.ALPHA

    def test_rejects_a_single_level_subgroup(self, processed_df):
        constant = processed_df.assign(newbie=1)
        with pytest.raises(ValueError, match="only one level"):
            interaction_test(constant, "visit", config.MENS_ARM, "newbie")

    def test_degrees_of_freedom_match_the_level_count(self, processed_df):
        result = interaction_test(
            processed_df, "visit", config.MENS_ARM, "recency_bucket"
        )
        assert result.levels == 4
        assert result.df_num == 3

    def test_marks_preregistered_subgroups(self, processed_df):
        registered = interaction_test(processed_df, "visit", config.MENS_ARM, "mens")
        exploratory = interaction_test(
            processed_df, "visit", config.MENS_ARM, "zip_code"
        )
        assert registered.preregistered
        assert registered.rationale
        assert not exploratory.preregistered


# ==========================================================================
# Interactions are expensive
# ==========================================================================
def test_interaction_standard_error_is_about_double_the_main_effect():
    """Why subgroup analysis needs so much more data than a main effect.

    Two things compound. An interaction is a difference of two differences, so
    it carries the variance of both -- a factor of sqrt(2). And each of those
    differences is estimated on half the customers -- another sqrt(2). The
    result is a standard error about twice that of the main effect on the full
    sample, so detecting an interaction of a given size needs roughly four
    times the customers.

    The comparison must be against the main effect fitted on the *whole*
    sample. The `treated` coefficient inside the interaction model is already
    the reference level's effect on half the data, and comparing against that
    recovers only the first sqrt(2).
    """
    import statsmodels.formula.api as smf

    data = synthetic_subgroups(0.10, 0.10, n=80_000, seed=4)

    main_only = smf.ols("visit ~ treated", data=data).fit(cov_type="HC3")
    with_interaction = smf.ols(
        "visit ~ treated * C(newbie)", data=data
    ).fit(cov_type="HC3")

    main = main_only.bse["treated"]
    term = [i for i in with_interaction.params.index if "treated:" in i][0]
    interaction = with_interaction.bse[term]

    assert interaction / main == pytest.approx(2.0, abs=0.15)

    # The within-model comparison recovers only one of the two factors.
    reference_level = with_interaction.bse["treated"]
    assert interaction / reference_level == pytest.approx(np.sqrt(2), abs=0.1)


# ==========================================================================
# Per-level effects
# ==========================================================================
class TestSubgroupEffects:
    def test_returns_one_row_per_level(self, processed_df):
        effects = subgroup_effects(
            processed_df, "visit", config.MENS_ARM, "recency_bucket"
        )
        assert len(effects) == 4

    def test_effects_match_the_interval(self, processed_df):
        effects = subgroup_effects(processed_df, "visit", config.MENS_ARM, "mens")
        assert (effects["ci_low"] < effects["effect"]).all()
        assert (effects["effect"] < effects["ci_high"]).all()

    def test_drops_cells_that_are_too_small(self):
        """A level with a handful of customers per arm is noise, not a finding."""
        rng = np.random.default_rng(5)
        data = synthetic_subgroups(0.1, 0.1, n=20_000, seed=5)
        # Make one level tiny.
        data.loc[data.index[:19_900], "newbie"] = 0
        data.loc[data.index[19_900:], "newbie"] = 1

        effects = subgroup_effects(
            data, "visit", config.MENS_ARM, "newbie", min_cell_size=MIN_CELL_SIZE
        )
        assert len(effects) == 1

    def test_level_sample_sizes_sum_to_the_arm_totals(self, processed_df):
        effects = subgroup_effects(processed_df, "visit", config.MENS_ARM, "mens")
        assert effects["n_treated"].sum() == 21_307
        assert effects["n_control"].sum() == 21_306


# ==========================================================================
# The real dataset
# ==========================================================================
class TestRealData:
    def test_womens_campaign_is_heterogeneous_by_purchase_history(self, heterogeneity):
        """The headline finding: the Womens email works far better on customers
        who previously bought womens merchandise."""
        interactions = heterogeneity["interactions"]
        result = interactions[
            (interactions["treatment_arm"] == config.WOMENS_ARM)
            & (interactions["outcome"] == "visit")
            & (interactions["subgroup"] == "womens")
        ].iloc[0]

        assert result["significant"]
        assert result["p_value"] < 1e-10

    def test_mens_campaign_is_uniform(self, heterogeneity):
        """The counterpart finding, and the one that limits Feature 9: no
        pre-registered subgroup moderates the Mens campaign on visits."""
        interactions = heterogeneity["interactions"]
        mens = interactions[
            (interactions["treatment_arm"] == config.MENS_ARM)
            & (interactions["outcome"] == "visit")
            & interactions["preregistered"]
        ]
        assert not mens["significant"].any()

    def test_the_womens_effect_gap_is_large(self, heterogeneity):
        effects = heterogeneity["effects"]
        womens = effects[
            (effects["treatment_arm"] == config.WOMENS_ARM)
            & (effects["outcome"] == "visit")
            & (effects["subgroup"] == "womens")
        ].set_index("level")

        strong = womens.loc["Bought womens", "effect"]
        weak = womens.loc["No womens purchase", "effect"]
        assert strong > 5 * weak
        assert womens["ci_low"].min() > 0, "both levels still benefit"

    def test_no_subgroup_is_actively_harmed(self, heterogeneity):
        """No level has an interval lying entirely below zero, so there is no
        'do not email' segment identifiable at this resolution."""
        assert not heterogeneity["effects"]["harmful"].any()

    def test_correction_applies_only_to_the_primary_family(self, heterogeneity):
        """Exploratory and underpowered tests must not dilute the correction
        applied to the pre-registered hypotheses."""
        interactions = heterogeneity["interactions"]
        primary = interactions["preregistered"] & (
            interactions["outcome"] == config.HETEROGENEITY_PRIMARY_OUTCOME
        )
        assert interactions.loc[primary, "p_value_adjusted"].notna().all()
        assert interactions.loc[~primary, "p_value_adjusted"].isna().all()

    def test_adjusted_p_values_are_never_smaller(self, heterogeneity):
        interactions = heterogeneity["interactions"]
        primary = interactions[interactions["p_value_adjusted"].notna()]
        assert (
            primary["p_value_adjusted"] >= primary["p_value"] - 1e-12
        ).all()

    def test_covers_every_subgroup_outcome_and_arm(self, heterogeneity):
        expected = (
            len(config.COMPARISONS)
            * len(config.OUTCOMES)
            * (len(config.PREREGISTERED_SUBGROUPS) + len(config.EXPLORATORY_SUBGROUPS))
        )
        assert len(heterogeneity["interactions"]) == expected

    def test_writes_results_files(self, processed_df, tmp_path, monkeypatch):
        monkeypatch.setattr(config, "RESULTS_DIR", tmp_path)
        monkeypatch.setattr(config, "ensure_dirs", lambda: None)
        run_heterogeneity_analysis(
            processed_df, outcomes=["visit"], include_exploratory=False, save=True
        )
        assert (tmp_path / "07_interaction_tests.csv").exists()
        assert (tmp_path / "07_subgroup_effects.csv").exists()
