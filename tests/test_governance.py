"""Step 1 controls: pseudonymisation, quarantine routing, protected-attribute exclusion."""
import json

import pandas as pd
import pytest

import data_governance as dg


# ------------------------------------------------------------------ pseudonymisation
def test_pseudonymisation_is_deterministic():
    assert dg.pseudonymise_id(1, salt="s3cret") == dg.pseudonymise_id(1, salt="s3cret")
    assert dg.pseudonymise_id(1, salt="s3cret") == dg.pseudonymise_id("1", salt="s3cret")


def test_pseudonymisation_depends_on_salt_and_id():
    assert dg.pseudonymise_id(1, salt="salt-a") != dg.pseudonymise_id(1, salt="salt-b")
    assert dg.pseudonymise_id(1, salt="salt-a") != dg.pseudonymise_id(2, salt="salt-a")


def test_pseudonymisation_output_is_sha256_hex():
    h = dg.pseudonymise_id(12345, salt="x")
    assert len(h) == 64 and all(c in "0123456789abcdef" for c in h)


def test_pseudonymisation_fails_without_salt(monkeypatch):
    monkeypatch.delenv("SCOREGUARD_SALT", raising=False)
    with pytest.raises(RuntimeError):
        dg.pseudonymise_id(1)


@pytest.mark.parametrize("bad", ["", "   "])
def test_pseudonymisation_fails_with_empty_salt(monkeypatch, bad):
    monkeypatch.setenv("SCOREGUARD_SALT", bad)
    with pytest.raises(RuntimeError):
        dg.pseudonymise_id(1)


def test_no_hard_coded_salt_in_source():
    src = (dg.PROJECT_ROOT / "src" / "data_governance.py").read_text(encoding="utf-8")
    assert "DEFAULT_SALT" not in src


# ------------------------------------------------------------------ quarantine routing
def _row(**overrides) -> dict:
    base = {c: 0 for c in dg.REAL_FEATURE_COLS}
    base.update({"ID": 0, "LIMIT_BAL": 50000, "SEX": 1, "EDUCATION": 2, "MARRIAGE": 1, "AGE": 30,
                 dg.RAW_TARGET_COL: 0})
    base.update(overrides)
    return base


def test_quarantine_routing_on_known_example():
    rows = [
        _row(ID=1),                                   # clean
        _row(ID=2, EDUCATION=0),                      # DQ07 -> quarantine
        _row(ID=3, MARRIAGE=0),                       # DQ08 -> quarantine
        _row(ID=4, LIMIT_BAL=0),                      # DQ03 -> quarantine
        _row(ID=5, AGE=15),                           # DQ04 -> quarantine
        _row(ID=6, PAY_AMT1=-10),                     # DQ05 -> quarantine
        _row(ID=7, LIMIT_BAL=70000),                  # clean
        _row(ID=8, LIMIT_BAL=70000),                  # exact duplicate of ID 7 -> DQ09 flag, retained
    ]
    df = pd.DataFrame(rows)
    clean, quarantine, summary = dg.evaluate_data_quality_rules(df)

    assert sorted(quarantine["ID"]) == [2, 3, 4, 5, 6]
    assert sorted(clean["ID"]) == [1, 7, 8]
    assert int(clean.loc[clean["ID"] == 8, "is_exact_dup_flag"].iloc[0]) == 1
    assert int(clean.loc[clean["ID"] == 7, "is_exact_dup_flag"].iloc[0]) == 0

    s = summary.set_index("rule_id")["flagged_rows"]
    assert s["DQ07_UNDOCUMENTED_EDUCATION_ZERO"] == 1
    assert s["DQ08_UNDOCUMENTED_MARRIAGE_ZERO"] == 1
    assert s["DQ09_EXACT_FEATURE_TARGET_DUPLICATE"] == 1
    assert quarantine.loc[quarantine["ID"] == 2, "failed_rules"].iloc[0] == "DQ07_UNDOCUMENTED_EDUCATION_ZERO"


def test_executed_quarantine_counts(read_csv, audit):
    dq = read_csv("dq_rule_summary.csv").set_index("rule_id")
    hard = dq[dq["governance_action"] == "HARD_QUARANTINE"]["flagged_rows"]
    rc = audit["row_counts"]
    # every quarantined row fails exactly one hard rule in this dataset, so the counts add up
    assert hard.sum() == rc["quarantine.credit_card_quarantine"]
    assert rc["bronze.credit_card_raw"] - rc["quarantine.credit_card_quarantine"] == rc["silver.credit_card_cleansed"]
    assert rc["silver.credit_card_cleansed"] == rc["gold.credit_card_modelling"] == rc["dev_rows"] + rc["holdout_rows"]
    assert dq.loc["DQ09_EXACT_FEATURE_TARGET_DUPLICATE", "governance_action"] == "FLAG_AND_RETAIN"
    assert dq.loc["DQ10_UNDOCUMENTED_CODE_MAPPING_NOTE", "governance_action"] == "DOCUMENTED_MAPPING_NOTE"


def test_quarantine_table_matches_database(db, audit):
    n = db.execute("select count(*) from quarantine.credit_card_quarantine").fetchone()[0]
    g = db.execute("select count(*) from gold.credit_card_modelling").fetchone()[0]
    assert n == audit["row_counts"]["quarantine.credit_card_quarantine"]
    assert g == audit["row_counts"]["gold.credit_card_modelling"]
    cols = [r[0] for r in db.execute(
        "select column_name from information_schema.columns where table_name = 'credit_card_modelling'").fetchall()]
    assert "ID" not in cols, "raw ID must not reach the gold layer"


# ------------------------------------------------------------------ protected attributes
PROTECTED = {"SEX", "MARRIAGE"}


def test_protected_attributes_not_in_champion():
    import scorecard_champion as sc
    feats = set(sc.FINAL_SCORECARD_FEATURES)
    assert not any(f.startswith(p) for f in feats for p in PROTECTED)
    assert PROTECTED <= sc.EXCLUDED_COLS


def test_protected_attributes_not_in_challenger(results_dir):
    with open(results_dir / "step3_challenger_summary.json", encoding="utf-8") as f:
        s3 = json.load(f)
    for key in ("challenger_core_10f", "challenger_expanded_19f"):
        feats = [r["feature"] for r in s3[key]["shap_importance"]]
        assert feats, key
        assert not any(f.startswith(p) for f in feats for p in PROTECTED), key
