import json
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

RESULTS = ROOT / "results"
DB_PATH = ROOT / "data" / "duckdb" / "scoreguard.duckdb"


@pytest.fixture(scope="session")
def results_dir() -> Path:
    return RESULTS


@pytest.fixture(scope="session")
def audit() -> dict:
    with open(RESULTS / "audit_summary.json", encoding="utf-8") as f:
        return json.load(f)


@pytest.fixture(scope="session")
def read_csv():
    def _read(name: str) -> pd.DataFrame:
        return pd.read_csv(RESULTS / name)
    return _read


@pytest.fixture(scope="session")
def db():
    """Read-only DuckDB connection. Tests that need it are skipped when the database is absent
    (it is excluded from Git and rebuilt by running the src/ pipeline)."""
    if not DB_PATH.exists():
        pytest.skip("DuckDB database not present (excluded from Git); run the src/ pipeline to rebuild it")
    import duckdb
    con = duckdb.connect(str(DB_PATH), read_only=True)
    yield con
    con.close()
