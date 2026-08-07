"""Build and query the DuckDB analytical store.

The warehouse holds one table, `customers`, and a set of views defined as SQL
files in `src/db/sql/`. Keeping the SQL in `.sql` files rather than in Python
string literals means it is readable, diffable and syntax-highlighted as SQL,
and it can be run against the database by hand while developing.

The store is a derived artifact: it is rebuilt from the processed parquet and
is not version-controlled.

Run as a script to rebuild it:

    python -m src.db.warehouse
"""

from __future__ import annotations

import logging
from pathlib import Path

import duckdb
import pandas as pd

from src import config
from src.data.load import load_processed

logger = logging.getLogger(__name__)

SQL_DIR = Path(__file__).parent / "sql"

BASE_TABLE = "customers"

# Views are created in filename order; the numeric prefixes encode the
# dependency chain (v_arm_lift reads v_arm_metrics, v_segment_metrics reads
# v_customer_dimensions).
EXPECTED_VIEWS = [
    "v_arm_metrics",
    "v_arm_lift",
    "v_funnel",
    "v_customer_dimensions",
    "v_segment_metrics",
]


def sql_files() -> list[Path]:
    """Return the view definition files in dependency order."""
    files = sorted(SQL_DIR.glob("*.sql"))
    if not files:
        raise FileNotFoundError(f"No .sql files found in {SQL_DIR}")
    return files


def connect(read_only: bool = False) -> duckdb.DuckDBPyConnection:
    """Open a connection to the warehouse, building it first if absent."""
    if not config.DUCKDB_PATH.exists():
        if read_only:
            raise FileNotFoundError(
                f"No warehouse at {config.DUCKDB_PATH}. "
                "Build it with `python -m src.db.warehouse`."
            )
        logger.info("Warehouse missing; building it now.")
        build()
    return duckdb.connect(str(config.DUCKDB_PATH), read_only=read_only)


def build(df: pd.DataFrame | None = None) -> Path:
    """Create the warehouse from the processed dataset.

    The base table is rebuilt from scratch every time rather than appended to,
    so the warehouse is always a pure function of the processed parquet and
    cannot drift from it.
    """
    config.ensure_dirs()
    frame = load_processed() if df is None else df

    con = duckdb.connect(str(config.DUCKDB_PATH))
    try:
        con.register("processed_df", frame)

        # A surrogate row key, needed to join the unpivoted dimension view back
        # to the base table. It identifies a ROW, not a person: the dataset has
        # no customer identifier, and the 6,562 duplicated rows are retained
        # deliberately (see README). Nothing should treat this as a person ID.
        con.execute(
            f"""
            CREATE OR REPLACE TABLE {BASE_TABLE} AS
            SELECT row_number() OVER () AS customer_id, *
            FROM processed_df
            """
        )
        con.unregister("processed_df")

        rows = con.execute(f"SELECT count(*) FROM {BASE_TABLE}").fetchone()[0]
        logger.info("Built table %s: %d rows", BASE_TABLE, rows)

        for path in sql_files():
            con.execute(path.read_text())
            logger.info("Created view from %s", path.name)

        con.execute("CHECKPOINT")
    finally:
        con.close()

    logger.info("Warehouse ready at %s", config.DUCKDB_PATH)
    return config.DUCKDB_PATH


def query(sql: str, params: list | None = None) -> pd.DataFrame:
    """Run a read-only query against the warehouse and return a DataFrame."""
    con = connect()
    try:
        relation = con.execute(sql, params) if params else con.execute(sql)
        return relation.df()
    finally:
        con.close()


def table(name: str) -> pd.DataFrame:
    """Read an entire table or view."""
    if name != BASE_TABLE and name not in EXPECTED_VIEWS:
        raise ValueError(
            f"Unknown relation {name!r}. "
            f"Available: {[BASE_TABLE, *EXPECTED_VIEWS]}"
        )
    return query(f"SELECT * FROM {name}")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    build()

    metrics = table("v_arm_metrics")
    print("\nOutcome metrics by arm\n")
    print(
        metrics[
            ["arm", "customers", "visit_rate", "conversion_rate", "mean_spend"]
        ].to_string(index=False, formatters={
            "customers": "{:,}".format,
            "visit_rate": "{:.2%}".format,
            "conversion_rate": "{:.3%}".format,
            "mean_spend": "${:.3f}".format,
        })
    )

    lift = table("v_arm_lift")
    print("\nLift over control (descriptive, no inference)\n")
    print(
        lift[["arm", "outcome", "absolute_lift", "relative_lift"]].to_string(
            index=False,
            formatters={
                "absolute_lift": "{:+.4f}".format,
                "relative_lift": "{:+.1%}".format,
            },
        )
    )


if __name__ == "__main__":
    main()
