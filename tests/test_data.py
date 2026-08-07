"""Tests for the data layer.

Two kinds of test live here. The first kind asserts that the published data
satisfies the contracts. The second, and more useful, kind corrupts a copy of
the data and asserts that `validate` actually rejects it — a contract that
cannot fail is not a contract.
"""

import numpy as np
import pandas as pd
import pytest

from src import config
from src.data import load as data_load
from src.data.load import (
    DataContractError,
    add_derived_columns,
    make_comparison_frame,
    validate,
)


# --------------------------------------------------------------------------
# The published data satisfies its contracts
# --------------------------------------------------------------------------
class TestRawDataContracts:
    def test_shape(self, raw_df):
        assert len(raw_df) == data_load.EXPECTED_ROWS
        assert list(raw_df.columns) == data_load.RAW_COLUMNS

    def test_no_missing_values(self, raw_df):
        assert raw_df.isna().sum().sum() == 0

    def test_arm_sizes(self, raw_df):
        counts = raw_df[config.TREATMENT_COL].value_counts()
        assert set(counts.index) == set(config.ARMS)
        assert counts.to_dict() == {
            "Womens E-Mail": 21_387,
            "Mens E-Mail": 21_307,
            "No E-Mail": 21_306,
        }

    def test_outcome_hierarchy(self, raw_df):
        """Conversion implies a visit, and spend occurs exactly on conversion."""
        assert ((raw_df.conversion == 1) & (raw_df.visit == 0)).sum() == 0
        assert ((raw_df.spend > 0) & (raw_df.conversion == 0)).sum() == 0
        assert ((raw_df.conversion == 1) & (raw_df.spend <= 0)).sum() == 0

    def test_validate_passes(self, raw_df):
        validate(raw_df)  # must not raise

    def test_duplicates_are_present_and_retained(self, raw_df):
        """The 6,562 duplicate rows are a documented decision, not an oversight.

        If this count ever changes, the justification in the README no longer
        describes the data and needs revisiting.
        """
        assert raw_df.duplicated().sum() == 6_562


# --------------------------------------------------------------------------
# The contracts reject corrupted data
# --------------------------------------------------------------------------
class TestValidateRejectsBadData:
    @pytest.fixture
    def sample(self, raw_df):
        """A small mutable copy. Row-count checks are asserted separately."""
        return raw_df.head(1_000).copy()

    def test_rejects_wrong_row_count(self, sample):
        with pytest.raises(DataContractError, match="rows"):
            validate(sample)

    def test_rejects_missing_column(self, raw_df):
        broken = raw_df.drop(columns=["spend"])
        with pytest.raises(DataContractError, match="missing columns"):
            validate(broken)

    def test_rejects_nulls(self, raw_df):
        broken = raw_df.copy()
        broken.loc[0, "history"] = np.nan
        with pytest.raises(DataContractError, match="nulls"):
            validate(broken)

    def test_rejects_renamed_arm(self, raw_df):
        broken = raw_df.copy()
        broken[config.TREATMENT_COL] = broken[config.TREATMENT_COL].replace(
            "Mens E-Mail", "Men E-Mail"
        )
        with pytest.raises(DataContractError, match="treatment arms changed"):
            validate(broken)

    def test_rejects_unexpected_category(self, raw_df):
        broken = raw_df.copy()
        broken.loc[0, "channel"] = "Carrier Pigeon"
        with pytest.raises(DataContractError, match="channel"):
            validate(broken)

    def test_rejects_non_binary_column(self, raw_df):
        broken = raw_df.copy()
        broken.loc[0, "visit"] = 2
        with pytest.raises(DataContractError, match="not binary"):
            validate(broken)

    def test_rejects_out_of_range_recency(self, raw_df):
        broken = raw_df.copy()
        broken.loc[0, "recency"] = 99
        with pytest.raises(DataContractError, match="recency outside"):
            validate(broken)

    def test_rejects_negative_spend(self, raw_df):
        broken = raw_df.copy()
        broken.loc[0, "spend"] = -1.0
        with pytest.raises(DataContractError, match="negative"):
            validate(broken)

    def test_rejects_conversion_without_visit(self, raw_df):
        broken = raw_df.copy()
        broken.loc[0, ["visit", "conversion", "spend"]] = [0, 1, 10.0]
        with pytest.raises(DataContractError, match="converted without a visit"):
            validate(broken)

    def test_rejects_spend_without_conversion(self, raw_df):
        broken = raw_df.copy()
        broken.loc[0, ["visit", "conversion", "spend"]] = [1, 0, 10.0]
        with pytest.raises(DataContractError, match="spend without a conversion"):
            validate(broken)


# --------------------------------------------------------------------------
# Derived features
# --------------------------------------------------------------------------
class TestDerivedColumns:
    def test_row_count_preserved(self, raw_df, processed_df):
        assert len(processed_df) == len(raw_df)

    def test_raw_columns_preserved(self, processed_df):
        for column in data_load.RAW_COLUMNS:
            assert column in processed_df.columns

    def test_no_nulls_introduced(self, processed_df):
        assert processed_df.isna().sum().sum() == 0

    def test_history_segment_is_ordered(self, processed_df):
        dtype = processed_df["history_segment"].dtype
        assert isinstance(dtype, pd.CategoricalDtype)
        assert dtype.ordered
        assert list(dtype.categories) == data_load.HISTORY_SEGMENT_ORDER

    def test_control_is_reference_level(self, processed_df):
        """Control must sort first so regressions treat it as the baseline."""
        categories = list(processed_df[config.TREATMENT_COL].dtype.categories)
        assert categories[0] == config.CONTROL_ARM

    def test_history_segment_rank_matches_label(self, processed_df):
        """The ordinal rank must agree with the leading digit of the label."""
        expected = (
            processed_df["history_segment"].astype(str).str[0].astype(int)
        )
        assert (processed_df["history_segment_rank"] == expected).all()

    def test_recency_bucket_covers_all_rows(self, processed_df):
        assert processed_df["recency_bucket"].notna().all()
        assert processed_df["recency_bucket"].value_counts().sum() == len(processed_df)

    def test_log_history_is_monotone_in_history(self, processed_df):
        assert np.allclose(
            processed_df["log_history"], np.log1p(processed_df["history"])
        )

    def test_no_outcome_leakage_in_derived_features(self, raw_df, processed_df):
        """Derived features must depend only on pre-treatment attributes.

        Shuffling the outcomes must leave every derived column untouched. If a
        feature were computed from an outcome, this would change it, and it
        would silently contaminate the balance checks, CUPED and uplift models.
        """
        shuffled = raw_df.copy()
        rng = np.random.default_rng(config.RANDOM_SEED)
        order = rng.permutation(len(shuffled))
        shuffled[["visit", "conversion", "spend"]] = (
            shuffled[["visit", "conversion", "spend"]].to_numpy()[order]
        )

        derived_from_shuffled = add_derived_columns(shuffled)
        for column in ["history_segment_rank", "recency_bucket", "log_history"]:
            pd.testing.assert_series_equal(
                derived_from_shuffled[column], processed_df[column]
            )


# --------------------------------------------------------------------------
# Two-arm comparison frames
# --------------------------------------------------------------------------
class TestComparisonFrame:
    @pytest.mark.parametrize(
        ("arm", "expected_treated", "expected_control"),
        [
            (config.MENS_ARM, 21_307, 21_306),
            (config.WOMENS_ARM, 21_387, 21_306),
        ],
    )
    def test_sizes(self, processed_df, arm, expected_treated, expected_control):
        frame = make_comparison_frame(processed_df, arm)
        assert (frame.treated == 1).sum() == expected_treated
        assert (frame.treated == 0).sum() == expected_control
        assert len(frame) == expected_treated + expected_control

    def test_excludes_the_third_arm(self, processed_df):
        frame = make_comparison_frame(processed_df, config.MENS_ARM)
        present = set(frame[config.TREATMENT_COL].unique())
        assert config.WOMENS_ARM not in present

    def test_treated_flag_matches_arm(self, processed_df):
        frame = make_comparison_frame(processed_df, config.MENS_ARM)
        assert (
            frame.loc[frame.treated == 1, config.TREATMENT_COL] == config.MENS_ARM
        ).all()
        assert (
            frame.loc[frame.treated == 0, config.TREATMENT_COL] == config.CONTROL_ARM
        ).all()

    def test_rejects_unknown_arm(self, processed_df):
        with pytest.raises(ValueError, match="Unknown arm"):
            make_comparison_frame(processed_df, "Postcard")

    def test_rejects_identical_arms(self, processed_df):
        with pytest.raises(ValueError, match="must differ"):
            make_comparison_frame(processed_df, config.CONTROL_ARM)


# --------------------------------------------------------------------------
# Persistence
# --------------------------------------------------------------------------
def test_parquet_roundtrip_preserves_dtypes(processed_df, tmp_path):
    path = tmp_path / "roundtrip.parquet"
    processed_df.to_parquet(path, index=False)
    restored = pd.read_parquet(path)

    pd.testing.assert_frame_equal(processed_df, restored)
