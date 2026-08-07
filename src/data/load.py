"""Load, validate and persist the Hillstrom experiment data.

The raw CSV is treated as immutable. This module applies explicit data
contracts to it, derives a small set of strictly pre-treatment features, and
writes a typed parquet file that every downstream feature reads.

Run as a script to rebuild the processed dataset:

    python -m src.data.load
"""

from __future__ import annotations

import argparse
import logging

import numpy as np
import pandas as pd

from src import config

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------
# Expected shape of the raw file
# --------------------------------------------------------------------------
EXPECTED_ROWS = 64_000

RAW_COLUMNS = [
    "recency",
    "history_segment",
    "history",
    "mens",
    "womens",
    "zip_code",
    "newbie",
    "channel",
    "segment",
    "visit",
    "conversion",
    "spend",
]

BINARY_COLUMNS = ["mens", "womens", "newbie", "visit", "conversion"]

# Ordered because history_segment is a binned version of a monetary amount;
# preserving the order lets it be used directly in ordered subgroup plots.
HISTORY_SEGMENT_ORDER = [
    "1) $0 - $100",
    "2) $100 - $200",
    "3) $200 - $350",
    "4) $350 - $500",
    "5) $500 - $750",
    "6) $750 - $1,000",
    "7) $1,000 +",
]

# 'Surburban' is misspelled in the source file. It is left verbatim so the
# processed data stays faithful to the raw source; the dashboard relabels it
# at display time rather than mutating the data.
ZIP_CODE_VALUES = ["Rural", "Surburban", "Urban"]
CHANNEL_VALUES = ["Multichannel", "Phone", "Web"]

RECENCY_MIN, RECENCY_MAX = 1, 12

# Recency is in months since last purchase. Four equal quarters of the
# observed range give balanced buckets for the subgroup analysis in Feature 7.
RECENCY_BUCKET_EDGES = [0, 3, 6, 9, 12]
RECENCY_BUCKET_LABELS = ["1-3 months", "4-6 months", "7-9 months", "10-12 months"]


class DataContractError(AssertionError):
    """Raised when the data violates an assumption the analysis depends on."""


# --------------------------------------------------------------------------
# Load
# --------------------------------------------------------------------------
def load_raw() -> pd.DataFrame:
    """Read the raw CSV exactly as published, with no cleaning applied."""
    if not config.RAW_CSV.exists():
        raise FileNotFoundError(
            f"Raw data not found at {config.RAW_CSV}. "
            "It is version-controlled; check out the repository fully."
        )
    df = pd.read_csv(config.RAW_CSV)
    logger.info("Loaded raw data: %d rows, %d columns", len(df), df.shape[1])
    return df


# --------------------------------------------------------------------------
# Validate
# --------------------------------------------------------------------------
def validate(df: pd.DataFrame) -> None:
    """Assert every contract the downstream analysis relies on.

    These are not defensive checks for their own sake. Each one corresponds to
    an assumption some later feature would silently break on: a changed arm
    label would misalign the treatment comparisons, an out-of-range binary
    column would corrupt a proportion test, and a broken outcome hierarchy
    would make the conversion and spend models mutually inconsistent.

    Raises:
        DataContractError: on the first violated contract.
    """
    failures: list[str] = []

    # --- Structure ---
    missing = set(RAW_COLUMNS) - set(df.columns)
    if missing:
        failures.append(f"missing columns: {sorted(missing)}")

    if len(df) != EXPECTED_ROWS:
        failures.append(f"expected {EXPECTED_ROWS:,} rows, found {len(df):,}")

    # Bail out early: the remaining contracts index columns by name.
    if missing:
        raise DataContractError("; ".join(failures))

    # --- Completeness ---
    null_counts = df[RAW_COLUMNS].isna().sum()
    if null_counts.any():
        offending = null_counts[null_counts > 0].to_dict()
        failures.append(f"unexpected nulls: {offending}")

    # --- Categorical domains ---
    actual_arms = set(df[config.TREATMENT_COL].unique())
    if actual_arms != set(config.ARMS):
        failures.append(
            f"treatment arms changed: expected {sorted(config.ARMS)}, "
            f"found {sorted(actual_arms)}"
        )

    for column, allowed in [
        ("history_segment", HISTORY_SEGMENT_ORDER),
        ("zip_code", ZIP_CODE_VALUES),
        ("channel", CHANNEL_VALUES),
    ]:
        unexpected = set(df[column].unique()) - set(allowed)
        if unexpected:
            failures.append(f"{column} has unexpected values: {sorted(unexpected)}")

    # --- Numeric ranges ---
    for column in BINARY_COLUMNS:
        if not df[column].isin([0, 1]).all():
            failures.append(f"{column} is not binary")

    if not df["recency"].between(RECENCY_MIN, RECENCY_MAX).all():
        failures.append(
            f"recency outside [{RECENCY_MIN}, {RECENCY_MAX}]: "
            f"observed [{df['recency'].min()}, {df['recency'].max()}]"
        )

    if (df["history"] <= 0).any():
        failures.append("history contains non-positive prior spend")

    if (df["spend"] < 0).any():
        failures.append("spend contains negative values")

    # --- Outcome hierarchy ---
    # A customer cannot convert without visiting, and spend is recorded if and
    # only if a conversion occurred. Both hold exactly in the published data.
    if (n := int(((df["conversion"] == 1) & (df["visit"] == 0)).sum())):
        failures.append(f"{n} rows converted without a visit")

    if (n := int(((df["spend"] > 0) & (df["conversion"] == 0)).sum())):
        failures.append(f"{n} rows have spend without a conversion")

    if (n := int(((df["conversion"] == 1) & (df["spend"] <= 0)).sum())):
        failures.append(f"{n} rows converted with no spend")

    if failures:
        raise DataContractError(
            "Data contract violated:\n  - " + "\n  - ".join(failures)
        )

    logger.info("All data contracts passed.")


# --------------------------------------------------------------------------
# Transform
# --------------------------------------------------------------------------
def add_derived_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Apply dtypes and derive pre-treatment features.

    Every derived column is a function of pre-treatment attributes only. No
    feature is derived from `visit`, `conversion` or `spend`, so nothing here
    can leak the outcome into the covariates used for balance checks, CUPED
    or uplift modelling.
    """
    out = df.copy()

    out["history_segment"] = pd.Categorical(
        out["history_segment"], categories=HISTORY_SEGMENT_ORDER, ordered=True
    )
    out["zip_code"] = pd.Categorical(out["zip_code"], categories=ZIP_CODE_VALUES)
    out["channel"] = pd.Categorical(out["channel"], categories=CHANNEL_VALUES)
    # Control first, so it is the reference level in any regression.
    out[config.TREATMENT_COL] = pd.Categorical(
        out[config.TREATMENT_COL],
        categories=[config.CONTROL_ARM, config.MENS_ARM, config.WOMENS_ARM],
    )

    # Ordinal rank 1-7 parsed from the leading digit of history_segment, for
    # use as a single numeric covariate instead of six dummies.
    out["history_segment_rank"] = (
        out["history_segment"].cat.codes.astype("int8") + 1
    )

    out["recency_bucket"] = pd.cut(
        out["recency"],
        bins=RECENCY_BUCKET_EDGES,
        labels=RECENCY_BUCKET_LABELS,
        ordered=True,
    )

    # Prior spend is right-skewed (mean $242, max $3,346). The log transform
    # is carried alongside the raw value; CUPED uses raw `history` so its
    # adjustment stays on the outcome's own scale.
    out["log_history"] = np.log1p(out["history"])

    for column in BINARY_COLUMNS:
        out[column] = out[column].astype("int8")
    out["recency"] = out["recency"].astype("int8")

    return out


def make_comparison_frame(
    df: pd.DataFrame,
    treatment_arm: str,
    control_arm: str = config.CONTROL_ARM,
) -> pd.DataFrame:
    """Reduce the three-arm data to one two-arm comparison.

    Returns only the two named arms, with a `treated` indicator (1 for the
    treatment arm, 0 for control). This is the shape every downstream test and
    uplift model expects.
    """
    for arm in (treatment_arm, control_arm):
        if arm not in config.ARMS:
            raise ValueError(f"Unknown arm {arm!r}; expected one of {config.ARMS}")
    if treatment_arm == control_arm:
        raise ValueError("treatment_arm and control_arm must differ")

    subset = df[df[config.TREATMENT_COL].isin([treatment_arm, control_arm])].copy()
    subset["treated"] = (
        subset[config.TREATMENT_COL] == treatment_arm
    ).astype("int8")
    return subset.reset_index(drop=True)


# --------------------------------------------------------------------------
# Build and read
# --------------------------------------------------------------------------
def build(write: bool = True) -> pd.DataFrame:
    """Run the full pipeline: load, validate, derive, and optionally persist."""
    df = load_raw()
    validate(df)
    processed = add_derived_columns(df)

    if write:
        config.ensure_dirs()
        processed.to_parquet(config.PROCESSED_PARQUET, index=False)
        logger.info("Wrote processed data to %s", config.PROCESSED_PARQUET)

    return processed


def load_processed() -> pd.DataFrame:
    """Read the processed parquet, building it first if it is absent."""
    if not config.PROCESSED_PARQUET.exists():
        logger.info("Processed data missing; building it now.")
        return build(write=True)
    return pd.read_parquet(config.PROCESSED_PARQUET)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the processed dataset.")
    parser.add_argument(
        "--no-write",
        action="store_true",
        help="Validate and transform without writing the parquet file.",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    df = build(write=not args.no_write)

    print(f"\nProcessed dataset: {len(df):,} rows x {df.shape[1]} columns")
    print("\nCustomers per arm:")
    print(df[config.TREATMENT_COL].value_counts().sort_index().to_string())


if __name__ == "__main__":
    main()
