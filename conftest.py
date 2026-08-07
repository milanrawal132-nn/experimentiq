"""Pytest configuration.

Living at the repository root, this file puts the project on `sys.path` so the
tests can `import src...` without the package needing to be installed.
"""

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.load import add_derived_columns, load_raw  # noqa: E402


@pytest.fixture(scope="session")
def raw_df():
    """The raw CSV, read once per test session."""
    return load_raw()


@pytest.fixture(scope="session")
def processed_df(raw_df):
    """The derived dataset, built once per test session."""
    return add_derived_columns(raw_df)


@pytest.fixture(scope="session")
def warehouse(processed_df, tmp_path_factory):
    """A DuckDB warehouse built into a temporary location.

    Redirecting `config.DUCKDB_PATH` keeps the suite from overwriting the
    developer's real database, and guarantees each run starts from an empty
    file rather than inheriting stale views from a previous build.
    """
    from src import config
    from src.db import warehouse as wh

    original = config.DUCKDB_PATH
    config.DUCKDB_PATH = tmp_path_factory.mktemp("db") / "test.duckdb"
    try:
        wh.build(df=processed_df)
        yield wh
    finally:
        config.DUCKDB_PATH = original
