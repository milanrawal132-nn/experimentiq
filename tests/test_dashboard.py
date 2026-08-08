"""Tests for the dashboard's data layer.

The dashboard makes one promise worth testing: what it shows is what the
pipeline produced. Two of its behaviours could break that promise, and both are
pinned here against the modules they mirror.

**The live profit model.** Moving the margin slider recomputes profitability by
arithmetic rather than by re-running Feature 10. `test_live_economics_matches_
the_committed_pipeline` asserts the two agree to floating point, so the fast
path is provably the same number and not an approximation of it.

**The reconstructed randomisation test.** `03_srm.csv` saves arm counts but not
the statistic they produced, so the dashboard rebuilds the chi-square test from
those counts. The same equivalence is asserted against `diagnostics.srm_test`.

Streamlit itself is not imported. The loaders are deliberately free of it, so
this suite runs without a browser or a server.
"""

import numpy as np
import pandas as pd
import pytest

from src import config
from src.analysis.diagnostics import srm_test
from src.dashboard import loaders
from src.models.budget import campaign_economics


@pytest.fixture(scope="module")
def ab_results():
    return loaders.result("ab_tests")


def fake_ab_results(spend_effect=1.0, ci_low=0.5, ci_high=1.5) -> pd.DataFrame:
    """A minimal results frame, so the transform can be tested on known inputs."""
    return pd.DataFrame({
        "outcome": ["visit", "spend"],
        "treatment_arm": [config.MENS_ARM, config.MENS_ARM],
        "absolute_effect": [0.07, spend_effect],
        "ci_low": [0.06, ci_low],
        "ci_high": [0.08, ci_high],
    })


# ==========================================================================
# Result files
# ==========================================================================
class TestResultAccess:
    def test_every_registered_result_maps_to_a_rebuild_command(self):
        for name in loaders.RESULT_FILES:
            assert loaders.rebuild_command(name).startswith("python -m src.")

    def test_unknown_result_names_are_rejected(self):
        with pytest.raises(KeyError):
            loaders.result_path("not_a_result")

    def test_missing_files_name_the_command_that_makes_them(
        self, tmp_path, monkeypatch
    ):
        """A fresh clone has no generated results. The reader should be told
        which command produces what they asked for, not shown a traceback."""
        monkeypatch.setattr(config, "RESULTS_DIR", tmp_path)

        with pytest.raises(loaders.MissingArtifact, match="src.models.budget"):
            loaders.result("economics")

    def test_rebuild_plan_is_ordered_as_the_pipeline_runs(
        self, tmp_path, monkeypatch
    ):
        monkeypatch.setattr(config, "RESULTS_DIR", tmp_path)
        plan = loaders.rebuild_plan()

        assert plan == list(loaders.REBUILD_COMMANDS.values())

    def test_nothing_is_missing_once_the_pipeline_has_run(self):
        """Guards against a result being renamed in one place only."""
        assert loaders.missing() == []


# ==========================================================================
# The live profit model
# ==========================================================================
class TestLiveEconomics:
    def test_matches_the_committed_pipeline(self, processed_df, ab_results):
        """The load-bearing test of this feature.

        The dashboard recomputes profit from Feature 4's spend effect rather
        than re-running Feature 10. That is only defensible if the two agree
        exactly -- which they do, because profit is a linear transform of spend
        and a constant shift changes no variance.
        """
        live = loaders.live_economics(ab_results, margin=0.30, cost=0.10).set_index(
            "treatment_arm"
        )
        committed = {
            e.treatment_arm: e for e in campaign_economics(processed_df, 0.30, 0.10)
        }

        for arm, expected in committed.items():
            assert live.loc[arm, "profit_per_email"] == pytest.approx(
                expected.profit_per_email
            )
            assert live.loc[arm, "profit_ci_low"] == pytest.approx(
                expected.profit_ci_low
            )
            assert live.loc[arm, "profit_ci_high"] == pytest.approx(
                expected.profit_ci_high
            )
            assert live.loc[arm, "break_even_margin"] == pytest.approx(
                expected.break_even_margin
            )

    @pytest.mark.parametrize(
        "margin,cost", [(0.10, 0.05), (0.45, 0.25), (0.30, 0.0)]
    )
    def test_agrees_with_the_pipeline_at_other_assumptions(
        self, processed_df, ab_results, margin, cost
    ):
        """Not just at the configured values -- the sliders move."""
        live = loaders.live_economics(ab_results, margin, cost).set_index(
            "treatment_arm"
        )
        for expected in campaign_economics(processed_df, margin, cost):
            assert live.loc[
                expected.treatment_arm, "profit_per_email"
            ] == pytest.approx(expected.profit_per_email)

    def test_profit_is_linear_in_the_margin(self):
        economics = loaders.live_economics(
            fake_ab_results(spend_effect=1.0), margin=0.50, cost=0.10
        )
        assert economics["profit_per_email"].iloc[0] == pytest.approx(0.40)

    def test_a_free_email_never_loses_money_on_a_positive_effect(self):
        economics = loaders.live_economics(fake_ab_results(), margin=0.30, cost=0.0)
        assert economics["break_even_spend"].iloc[0] == 0.0
        assert economics["profit_per_email"].iloc[0] > 0

    def test_break_even_margin_ignores_the_assumed_margin(self):
        """It is a property of the spend effect and the send cost alone, which
        is why it is the number to quote when the assumption is in question."""
        first = loaders.live_economics(fake_ab_results(), margin=0.20, cost=0.10)
        second = loaders.live_economics(fake_ab_results(), margin=0.55, cost=0.10)

        assert first["break_even_margin"].iloc[0] == pytest.approx(
            second["break_even_margin"].iloc[0]
        )

    def test_a_zero_effect_can_never_break_even(self):
        economics = loaders.live_economics(
            fake_ab_results(spend_effect=0.0, ci_low=-0.5, ci_high=0.5),
            margin=0.30, cost=0.10,
        )
        assert np.isinf(economics["break_even_margin"].iloc[0])

    def test_non_positive_margin_is_rejected(self):
        with pytest.raises(ValueError):
            loaders.live_economics(fake_ab_results(), margin=0.0, cost=0.10)

    def test_results_without_spend_rows_are_rejected(self):
        visit_only = fake_ab_results().query("outcome == 'visit'")
        with pytest.raises(ValueError, match="no spend rows"):
            loaders.live_economics(visit_only)


class TestVerdict:
    def test_an_interval_above_zero_pays_for_itself(self):
        row = pd.Series({"profit_ci_low": 0.05, "profit_ci_high": 0.20})
        assert loaders.verdict(row) == "pays for itself"

    def test_an_interval_below_zero_loses_money(self):
        row = pd.Series({"profit_ci_low": -0.20, "profit_ci_high": -0.05})
        assert loaders.verdict(row) == "loses money"

    def test_an_interval_straddling_zero_is_undecided(self):
        """Three states, not two. Collapsing this into "not profitable" would
        report a wide interval as a negative finding."""
        row = pd.Series({"profit_ci_low": -0.05, "profit_ci_high": 0.20})
        assert loaders.verdict(row) == "cannot be determined"

    def test_the_real_campaigns_land_in_the_expected_states(self, ab_results):
        economics = loaders.live_economics(ab_results).set_index("treatment_arm")
        assert loaders.verdict(economics.loc[config.MENS_ARM]) == "pays for itself"
        assert (
            loaders.verdict(economics.loc[config.WOMENS_ARM])
            == "cannot be determined"
        )


class TestBudgetProjection:
    def test_starts_at_zero_and_scales_linearly(self, ab_results):
        economics = loaders.live_economics(ab_results)
        projection = loaders.budget_projection(
            economics, config.MENS_ARM, n_customers=64_000, cost=0.10
        )
        per_email = economics.set_index("treatment_arm").loc[
            config.MENS_ARM, "profit_per_email"
        ]

        assert projection["net_gain"].iloc[0] == 0.0
        assert projection["net_gain"].iloc[-1] == pytest.approx(64_000 * per_email)
        assert projection["budget"].iloc[-1] == pytest.approx(6_400.0)

    def test_the_interval_brackets_the_expectation(self, ab_results):
        projection = loaders.budget_projection(
            loaders.live_economics(ab_results), config.MENS_ARM, 64_000
        )
        assert (projection["ci_low"] <= projection["net_gain"]).all()
        assert (projection["net_gain"] <= projection["ci_high"]).all()

    def test_an_unknown_arm_is_rejected(self, ab_results):
        with pytest.raises(ValueError, match="No economics row"):
            loaders.budget_projection(
                loaders.live_economics(ab_results), "Postcard", 64_000
            )


# ==========================================================================
# The reconstructed randomisation test
# ==========================================================================
class TestSRMVerdict:
    def test_matches_the_diagnostics_module(self, processed_df):
        """`03_srm.csv` saves the counts but not the statistic, so the
        dashboard rebuilds the test. It must produce the same numbers."""
        expected = srm_test(processed_df)
        rebuilt = loaders.srm_verdict(expected.to_frame())

        assert rebuilt["chi2"] == pytest.approx(expected.chi2)
        assert rebuilt["dof"] == expected.dof
        assert rebuilt["p_value"] == pytest.approx(expected.p_value)
        assert rebuilt["passed"] == expected.passed

    def test_matches_the_saved_file(self, processed_df):
        expected = srm_test(processed_df)
        rebuilt = loaders.srm_verdict(loaders.result("srm"))
        assert rebuilt["p_value"] == pytest.approx(expected.p_value)

    def test_a_perfect_split_produces_no_evidence_of_mismatch(self):
        counts = pd.DataFrame({"observed": [100.0] * 3, "expected": [100.0] * 3})
        rebuilt = loaders.srm_verdict(counts)

        assert rebuilt["chi2"] == pytest.approx(0.0)
        assert rebuilt["p_value"] == pytest.approx(1.0)
        assert rebuilt["passed"]

    def test_a_gross_imbalance_is_flagged(self):
        counts = pd.DataFrame({
            "observed": [5_000.0, 10_000.0, 15_000.0],
            "expected": [10_000.0] * 3,
        })
        rebuilt = loaders.srm_verdict(counts)

        assert rebuilt["p_value"] < config.SRM_ALPHA
        assert not rebuilt["passed"]


# ==========================================================================
# Warehouse access
# ==========================================================================
class TestWarehouseAccess:
    def test_every_outcome_maps_to_a_real_column(self, warehouse):
        metrics = loaders.arm_metrics()
        for outcome in config.OUTCOMES:
            assert loaders.arm_metric_column(outcome) in metrics.columns

    def test_the_arm_column_exists_under_the_name_the_app_uses(self, warehouse):
        assert loaders.ARM_COLUMN in loaders.arm_metrics().columns

    def test_unknown_outcomes_are_rejected(self):
        with pytest.raises(KeyError):
            loaders.arm_metric_column("churn")

    def test_filtering_to_one_dimension_returns_only_that_dimension(
        self, warehouse
    ):
        dimension = loaders.dimensions()[0]
        rows = loaders.segment_metrics(dimension)

        assert not rows.empty
        assert (rows["dimension"] == dimension).all()

    def test_unfiltered_segment_metrics_covers_every_dimension(self, warehouse):
        assert set(loaders.segment_metrics()["dimension"]) == set(
            loaders.dimensions()
        )

    def test_control_mirror_columns_are_not_offered_as_metrics(self, warehouse):
        """The view repeats the control arm's rates on every treated row so
        that lift can be computed in SQL. Charting those by arm would plot the
        same number once per arm."""
        rows = loaders.segment_metrics()
        offered = loaders.treatment_metrics(rows)

        assert "visit_rate" in offered
        assert "visit_lift" in offered
        assert not any(column.startswith("control_") for column in offered)


# ==========================================================================
# Headline numbers
# ==========================================================================
class TestHeadline:
    def test_reports_the_full_experiment(self):
        assert loaders.headline()["customers"] == 64_000

    def test_mens_lift_exceeds_womens(self):
        numbers = loaders.headline()
        assert numbers["mens_visit_lift"] > numbers["womens_visit_lift"] > 0

    def test_agrees_with_the_generated_tables(self):
        """The overview page and the README must not be able to drift apart."""
        numbers = loaders.headline()
        ab = loaders.result("ab_tests")
        visit = ab[(ab["outcome"] == "visit")].set_index("treatment_arm")

        assert numbers["mens_visit_lift"] == pytest.approx(
            visit.loc[config.MENS_ARM, "absolute_effect"]
        )
        assert numbers["significant_results"] == int(ab["significant"].sum())

    def test_the_best_policy_is_the_highest_valued_one(self):
        numbers = loaders.headline()
        values = loaders.result("policy_values").set_index("policy")["value"]
        assert values[numbers["best_policy"]] == pytest.approx(
            numbers["best_policy_value"]
        )
        assert numbers["best_policy_value"] == values.max()
