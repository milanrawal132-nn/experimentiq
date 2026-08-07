"""Tests for the DuckDB warehouse.

The most valuable tests here recompute the SQL views' numbers in pandas and
assert the two agree. A view that is merely *self-consistent* can still be
quietly wrong; agreement between two independent implementations is what
catches a mis-specified GROUP BY or a denominator taken over the wrong set.
"""

import numpy as np
import pytest

from src import config
from src.db import warehouse as wh

TOLERANCE = 1e-9


@pytest.fixture(scope="module")
def metrics(warehouse):
    return warehouse.table("v_arm_metrics").set_index("arm")


@pytest.fixture(scope="module")
def lift(warehouse):
    return warehouse.table("v_arm_lift")


@pytest.fixture(scope="module")
def funnel(warehouse):
    return warehouse.table("v_funnel")


@pytest.fixture(scope="module")
def dimensions(warehouse):
    return warehouse.table("v_customer_dimensions")


@pytest.fixture(scope="module")
def segments(warehouse):
    return warehouse.table("v_segment_metrics")


# --------------------------------------------------------------------------
# Structure
# --------------------------------------------------------------------------
class TestWarehouseStructure:
    def test_base_table_row_count(self, warehouse):
        rows = warehouse.query("SELECT count(*) AS n FROM customers")["n"].iloc[0]
        assert rows == 64_000

    def test_all_expected_views_exist(self, warehouse):
        present = set(
            warehouse.query(
                "SELECT table_name FROM information_schema.tables"
            )["table_name"]
        )
        assert set(wh.EXPECTED_VIEWS) <= present

    def test_customer_id_is_unique(self, warehouse):
        result = warehouse.query(
            "SELECT count(*) AS n, count(DISTINCT customer_id) AS d FROM customers"
        )
        assert result["n"].iloc[0] == result["d"].iloc[0] == 64_000

    def test_arm_enum_orders_control_first(self, warehouse):
        """Control must sort first so it reads as the baseline everywhere."""
        arms = warehouse.query("SELECT arm FROM v_arm_metrics")["arm"].tolist()
        assert arms[0] == config.CONTROL_ARM

    def test_table_rejects_unknown_relation(self, warehouse):
        with pytest.raises(ValueError, match="Unknown relation"):
            warehouse.table("v_does_not_exist")

    def test_sql_files_are_ordered(self):
        """Views depend on earlier views, so filename order is load order."""
        names = [p.name for p in wh.sql_files()]
        assert names == sorted(names)
        assert len(names) == len(wh.EXPECTED_VIEWS)


# --------------------------------------------------------------------------
# SQL agrees with pandas
# --------------------------------------------------------------------------
class TestMetricsMatchPandas:
    @pytest.mark.parametrize(
        ("sql_column", "source_column"),
        [
            ("visit_rate", "visit"),
            ("conversion_rate", "conversion"),
            ("mean_spend", "spend"),
        ],
    )
    def test_rates_match(self, metrics, processed_df, sql_column, source_column):
        expected = processed_df.groupby(
            config.TREATMENT_COL, observed=True
        )[source_column].mean()

        for arm, value in expected.items():
            assert metrics.loc[arm, sql_column] == pytest.approx(value, abs=TOLERANCE)

    def test_customer_counts_match(self, metrics, processed_df):
        expected = processed_df[config.TREATMENT_COL].value_counts()
        for arm, count in expected.items():
            assert metrics.loc[arm, "customers"] == count

    def test_spend_per_converter_uses_converter_denominator(self, metrics, processed_df):
        """Mean spend per converter, not per customer -- a denominator that is
        easy to get wrong and produces a plausible-looking number when it is."""
        grouped = processed_df.groupby(config.TREATMENT_COL, observed=True)
        expected = grouped["spend"].sum() / grouped["conversion"].sum()

        for arm, value in expected.items():
            assert metrics.loc[arm, "mean_spend_per_converter"] == pytest.approx(value)


# --------------------------------------------------------------------------
# Lift view
# --------------------------------------------------------------------------
class TestArmLift:
    def test_control_arm_is_excluded(self, lift):
        assert config.CONTROL_ARM not in set(lift["arm"])

    def test_shape(self, lift):
        """Two treatment arms x three outcomes."""
        assert len(lift) == 6
        assert set(lift["outcome"]) == {"visit_rate", "conversion_rate", "mean_spend"}

    def test_absolute_lift_is_the_difference(self, lift):
        assert np.allclose(
            lift["absolute_lift"], lift["treatment_value"] - lift["control_value"]
        )

    def test_relative_lift_is_the_ratio(self, lift):
        assert np.allclose(
            lift["relative_lift"], lift["absolute_lift"] / lift["control_value"]
        )

    def test_control_value_is_constant_per_outcome(self, lift):
        """Both arms are compared against the same control, so the control
        value for a given outcome must not vary by arm."""
        assert (lift.groupby("outcome")["control_value"].nunique() == 1).all()


# --------------------------------------------------------------------------
# Funnel view
# --------------------------------------------------------------------------
class TestFunnel:
    def test_funnel_counts_are_monotone(self, funnel):
        """Feature 1's contract: conversion implies a visit."""
        assert (funnel["assigned"] >= funnel["visited"]).all()
        assert (funnel["visited"] >= funnel["converted"]).all()

    def test_unconditional_rates_use_assigned_denominator(self, funnel):
        assert np.allclose(funnel["visit_rate"], funnel["visited"] / funnel["assigned"])
        assert np.allclose(
            funnel["conversion_rate"], funnel["converted"] / funnel["assigned"]
        )

    def test_conditional_rate_uses_visitor_denominator(self, funnel):
        assert np.allclose(
            funnel["conversion_rate_given_visit"],
            funnel["converted"] / funnel["visited"],
        )

    def test_conditional_rate_exceeds_unconditional(self, funnel):
        """A sanity check on the two denominators being genuinely different."""
        assert (
            funnel["conversion_rate_given_visit"] > funnel["conversion_rate"]
        ).all()


# --------------------------------------------------------------------------
# Unpivoted dimensions
# --------------------------------------------------------------------------
class TestCustomerDimensions:
    def test_one_row_per_customer_per_dimension(self, dimensions):
        n_dimensions = dimensions["dimension"].nunique()
        assert n_dimensions == 7
        assert len(dimensions) == 64_000 * n_dimensions

    def test_no_customer_has_two_levels_in_one_dimension(self, warehouse):
        duplicates = warehouse.query(
            """
            SELECT count(*) AS n FROM (
                SELECT customer_id, dimension
                FROM v_customer_dimensions
                GROUP BY customer_id, dimension
                HAVING count(*) > 1
            )
            """
        )["n"].iloc[0]
        assert duplicates == 0

    def test_no_null_levels(self, dimensions):
        assert dimensions["level"].notna().all()

    def test_excludes_outcome_columns(self, dimensions):
        """Slicing by a post-treatment variable would break the comparison."""
        levels = set(dimensions["level"])
        for outcome in config.OUTCOMES:
            assert outcome not in {d.lower() for d in dimensions["dimension"]}
            assert outcome not in levels


# --------------------------------------------------------------------------
# Segment metrics
# --------------------------------------------------------------------------
class TestSegmentMetrics:
    def test_customers_sum_to_arm_total_within_each_dimension(
        self, segments, processed_df
    ):
        """Every customer falls in exactly one level of every dimension, so the
        cells of a dimension must partition each arm exactly."""
        arm_totals = processed_df[config.TREATMENT_COL].value_counts()
        totals = segments.groupby(["dimension", "arm"], observed=True)[
            "customers"
        ].sum()

        for (_, arm), total in totals.items():
            assert total == arm_totals[arm]

    def test_lifts_are_differences(self, segments):
        assert np.allclose(
            segments["visit_lift"],
            segments["visit_rate"] - segments["control_visit_rate"],
        )
        assert np.allclose(
            segments["spend_lift"],
            segments["mean_spend"] - segments["control_mean_spend"],
        )

    def test_control_arm_is_excluded(self, segments):
        assert config.CONTROL_ARM not in set(segments["arm"])

    def test_min_arm_customers_is_the_smaller_arm(self, segments):
        assert (
            segments["min_arm_customers"]
            == segments[["customers", "control_customers"]].min(axis=1)
        ).all()

    def test_a_known_cell_matches_pandas(self, segments, processed_df):
        """Spot-check one cell end to end through the unpivot and the join."""
        cell = segments[
            (segments["dimension"] == "Tenure")
            & (segments["level"] == "New customer")
            & (segments["arm"] == config.MENS_ARM)
        ]
        assert len(cell) == 1

        expected = processed_df[
            (processed_df["newbie"] == 1)
            & (processed_df[config.TREATMENT_COL] == config.MENS_ARM)
        ]
        assert cell["customers"].iloc[0] == len(expected)
        assert cell["visit_rate"].iloc[0] == pytest.approx(expected["visit"].mean())


# --------------------------------------------------------------------------
# Rebuild is deterministic
# --------------------------------------------------------------------------
def test_rebuild_is_idempotent(warehouse, processed_df):
    """The warehouse is a pure function of the processed data: building twice
    must not duplicate rows or leave stale views behind."""
    before = warehouse.table("v_arm_metrics")
    warehouse.build(df=processed_df)
    after = warehouse.table("v_arm_metrics")

    assert len(after) == len(before) == 3
    assert np.allclose(after["visit_rate"], before["visit_rate"])
