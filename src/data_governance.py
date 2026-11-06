"""ScoreGuard-IRB Data Governance Layer.

Implements:
- Salted SHA-256 pseudonymisation of account IDs (GDPR / Data Governance control).
  Reads salt strictly from the SCOREGUARD_SALT environment variable (no hard-coded fallback).
  Note on Reversibility: Because the public UCI ID 350 dataset uses sequential integer IDs (1..30,000),
  pseudonymisation with a known salt would be trivially reversible via a rainbow-table dictionary attack
  over 1..30,000 unless the salt is kept strictly secret outside the repository (e.g., in an unversioned
  secret store / .env excluded via .gitignore).
- Documented treatment of undocumented codes in EDUCATION, MARRIAGE, and PAY_0..PAY_6.
  Specifically, PAY = -1 is documented ("pay duly"); only PAY in {-2, 0} and EDUCATION in {0, 5, 6}
  and MARRIAGE == 0 are undocumented in the original UCI data dictionary.
- Explicit data quality (DQ) rules and quarantine routing:
  - DQ07 (EDUCATION == 0, 14 rows) and DQ08 (MARRIAGE == 0, 54 rows) are routed to HARD_QUARANTINE (68 rows total).
  - DQ09 (35 exact feature+target duplicate rows, of which 30 are all-zero inactive accounts with unique IDs
    representing distinct dormant customers) is set to FLAG_AND_RETAIN (`is_exact_dup_flag = 1`), not quarantined.
  - DQ10 (undocumented codes EDUCATION in {5, 6} or PAY_0..PAY_6 in {-2, 0}; note PAY = -1 is documented "pay duly")
    is classified as DOCUMENTED_MAPPING_NOTE, not a data quality violation.
- DuckDB Bronze / Silver / Gold + Quarantine medallion architecture.
"""
from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple

import duckdb
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_CSV_PATH = PROJECT_ROOT / "data" / "raw" / "uci_credit_card_default_raw.csv"
CSO_CSV_PATH = PROJECT_ROOT / "data" / "raw" / "cso_lrm01_raw.csv"
DB_PATH = PROJECT_ROOT / "data" / "duckdb" / "scoreguard.duckdb"
RESULTS_DIR = PROJECT_ROOT / "results"

RANDOM_SEED = 42

REAL_FEATURE_COLS = [
    "LIMIT_BAL",
    "SEX",
    "EDUCATION",
    "MARRIAGE",
    "AGE",
    "PAY_0",
    "PAY_2",
    "PAY_3",
    "PAY_4",
    "PAY_5",
    "PAY_6",
    "BILL_AMT1",
    "BILL_AMT2",
    "BILL_AMT3",
    "BILL_AMT4",
    "BILL_AMT5",
    "BILL_AMT6",
    "PAY_AMT1",
    "PAY_AMT2",
    "PAY_AMT3",
    "PAY_AMT4",
    "PAY_AMT5",
    "PAY_AMT6",
]
PAY_STATUS_COLS = ["PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"]
BILL_AMT_COLS = ["BILL_AMT1", "BILL_AMT2", "BILL_AMT3", "BILL_AMT4", "BILL_AMT5", "BILL_AMT6"]
PAY_AMT_COLS = ["PAY_AMT1", "PAY_AMT2", "PAY_AMT3", "PAY_AMT4", "PAY_AMT5", "PAY_AMT6"]
RAW_TARGET_COL = "default payment next month"


def get_required_salt(salt: str | None = None) -> str:
    """Retrieve cryptographic salt strictly from explicit argument or SCOREGUARD_SALT env var.

    Raises RuntimeError if SCOREGUARD_SALT is missing or empty.
    """
    active_salt = salt if salt is not None else os.environ.get("SCOREGUARD_SALT")
    if active_salt is None or not str(active_salt).strip():
        raise RuntimeError(
            "SCOREGUARD_SALT environment variable is missing or empty. "
            "Refusing to run pseudonymisation with a hard-coded salt."
        )
    return str(active_salt).strip()


def pseudonymise_id(raw_id: int | str, salt: str | None = None) -> str:
    """Return deterministic salted SHA-256 hex digest for an account identifier."""
    active_salt = get_required_salt(salt)
    payload = f"{active_salt}::{str(raw_id).strip()}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def evaluate_data_quality_rules(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Evaluate 9 governance data quality rules and 1 documented mapping note on a Bronze-schema DataFrame.

    Returns:
        clean_df: Records passing all hard quarantine rules (includes `is_exact_dup_flag` for DQ09).
        quarantine_df: Records failing >= 1 hard quarantine rule, with `failed_rules`.
        rule_summary_df: Summary table of all 10 entries, their governance action, and exact counts.
    """
    n_total = len(df)
    target_col = RAW_TARGET_COL if RAW_TARGET_COL in df.columns else "default_flag"

    # Rule evaluations
    r01_null = df[["ID"] + REAL_FEATURE_COLS + [target_col]].isnull().any(axis=1)
    r02_dup_id = df["ID"].duplicated(keep="first")
    r03_limit = df["LIMIT_BAL"] <= 0
    r04_age = (df["AGE"] < 18) | (df["AGE"] > 100)
    r05_neg_pay = (df[PAY_AMT_COLS] < 0).any(axis=1)
    r06_domain = (~df["SEX"].isin([1, 2])) | (~df[target_col].isin([0, 1]))
    r07_edu_zero = df["EDUCATION"] == 0
    r08_mar_zero = df["MARRIAGE"] == 0

    # DQ09: 35 exact feature+target duplicates (keep='first'); 30 of the 35 are all-zero inactive accounts
    # with unique IDs (likely distinct dormant customers). Action: FLAG_AND_RETAIN, not quarantine.
    r09_exact_dup = df.duplicated(subset=REAL_FEATURE_COLS + [target_col], keep="first")

    # DQ10: Documented Mapping Note (NOT a data quality violation).
    # PAY = -1 is documented ("pay duly"). Only PAY in {-2, 0} and EDUCATION in {5, 6} are undocumented
    # among retained Silver records.
    r10_undoc_note = df["EDUCATION"].isin([5, 6]) | df[PAY_STATUS_COLS].isin([-2, 0]).any(axis=1)

    rules_meta = [
        (
            "DQ01_MISSING_CRITICAL_FIELDS",
            "Null/NaN in ID, any of the 23 features, or default flag",
            "HARD_QUARANTINE",
            r01_null,
        ),
        (
            "DQ02_DUPLICATE_PRIMARY_KEY",
            "Duplicate raw account ID",
            "HARD_QUARANTINE",
            r02_dup_id,
        ),
        (
            "DQ03_NON_POSITIVE_CREDIT_LIMIT",
            "Credit limit LIMIT_BAL <= 0",
            "HARD_QUARANTINE",
            r03_limit,
        ),
        (
            "DQ04_AGE_OUT_OF_BOUNDS",
            "Cardholder age < 18 or > 100",
            "HARD_QUARANTINE",
            r04_age,
        ),
        (
            "DQ05_NEGATIVE_REPAYMENT_AMOUNT",
            "Negative repayment amount in any PAY_AMT1..PAY_AMT6",
            "HARD_QUARANTINE",
            r05_neg_pay,
        ),
        (
            "DQ06_INVALID_SEX_OR_TARGET_DOMAIN",
            "SEX not in {1, 2} or target default flag not in {0, 1}",
            "HARD_QUARANTINE",
            r06_domain,
        ),
        (
            "DQ07_UNDOCUMENTED_EDUCATION_ZERO",
            "EDUCATION == 0 (unmapped/missing zero code outside UCI dictionary 1..4 and 5..6)",
            "HARD_QUARANTINE",
            r07_edu_zero,
        ),
        (
            "DQ08_UNDOCUMENTED_MARRIAGE_ZERO",
            "MARRIAGE == 0 (unmapped/missing zero code outside UCI dictionary 1..3)",
            "HARD_QUARANTINE",
            r08_mar_zero,
        ),
        (
            "DQ09_EXACT_FEATURE_TARGET_DUPLICATE",
            "Exact duplicate across 23 features + target (30/35 are all-zero inactive accounts with unique IDs; flagged & retained)",
            "FLAG_AND_RETAIN",
            r09_exact_dup,
        ),
        (
            "DQ10_UNDOCUMENTED_CODE_MAPPING_NOTE",
            "Documented mapping note (not a DQ violation): EDUCATION in {5, 6} or PAY_0..6 in {-2, 0} (PAY=-1 is documented 'pay duly')",
            "DOCUMENTED_MAPPING_NOTE",
            r10_undoc_note,
        ),
    ]

    hard_rule_masks = {code: mask for code, _, action, mask in rules_meta if action == "HARD_QUARANTINE"}
    quarantine_mask = pd.Series(False, index=df.index)
    for mask in hard_rule_masks.values():
        quarantine_mask |= mask

    failed_rules_list: List[str] = []
    for idx in df.index:
        triggered = [code for code, mask in hard_rule_masks.items() if bool(mask.loc[idx])]
        failed_rules_list.append(";".join(triggered))

    df_out = df.copy()
    df_out["is_exact_dup_flag"] = r09_exact_dup.astype(int)
    df_out["failed_rules"] = failed_rules_list
    df_out["is_quarantined"] = quarantine_mask

    quarantine_df = df_out.loc[quarantine_mask].copy()
    clean_df = df_out.loc[~quarantine_mask].drop(columns=["failed_rules", "is_quarantined"]).copy()

    summary_rows = []
    for code, desc, action, mask in rules_meta:
        cnt = int(mask.sum())
        def_rate = float(df.loc[mask, target_col].mean()) if cnt > 0 else 0.0
        summary_rows.append(
            {
                "rule_id": code,
                "description": desc,
                "governance_action": action,
                "flagged_rows": cnt,
                "pct_of_bronze": round(100.0 * cnt / max(n_total, 1), 4),
                "flagged_default_rate": round(def_rate, 6),
            }
        )
    rule_summary_df = pd.DataFrame(summary_rows)
    return clean_df, quarantine_df, rule_summary_df


def apply_silver_transformations(clean_df: pd.DataFrame) -> pd.DataFrame:
    """Apply documented Silver-layer standardisation to clean records.

    Documented treatment of undocumented codes:
    1. EDUCATION:
       - Raw 0 was quarantined by DQ07 (14 rows).
       - Raw 1 (Graduate school), 2 (University), 3 (High school) preserved as 1, 2, 3.
       - Raw 4 (documented Others, 123 rows), 5 (undocumented Unknown, 280 rows),
         6 (undocumented Unknown, 51 rows) consolidated into EDUCATION_CLEAN = 4 ('Other / Unknown').
       - Original code preserved in EDUCATION_RAW for auditability.
    2. MARRIAGE:
       - Raw 0 was quarantined by DQ08 (54 rows).
       - Raw 1 (Married), 2 (Single), 3 (Others) preserved in MARRIAGE_CLEAN (and MARRIAGE_RAW).
    3. PAY_0..PAY_6:
       - Raw -1 is documented in the UCI dictionary as "pay duly" (paid in full on time).
       - Raw -2 (no consumption / inactive) and 0 (revolving credit / minimum paid on time) are undocumented
         in the original UCI note but well-established in the literature.
       - All three (-2, -1, 0) represent 0 months past due (performing / current) and map to PAY_x_DPD = 0,
         while raw values >= 1 (months in arrears) are preserved as-is.
       - Raw PAY_0..PAY_6 columns and explicit state indicators (_IS_REVOLVING, _IS_INACTIVE, _IS_PAID_DULY)
         are preserved alongside PAY_0_DPD..PAY_6_DPD.
    """
    silver = clean_df.copy()
    if RAW_TARGET_COL in silver.columns:
        silver = silver.rename(columns={RAW_TARGET_COL: "default_flag"})

    silver["EDUCATION_RAW"] = silver["EDUCATION"].astype(int)
    silver["EDUCATION_CLEAN"] = silver["EDUCATION_RAW"].replace({5: 4, 6: 4}).astype(int)
    edu_labels = {1: "Graduate School", 2: "University", 3: "High School", 4: "Other / Unknown (4,5,6)"}
    silver["EDUCATION_LABEL"] = silver["EDUCATION_CLEAN"].map(edu_labels)

    silver["MARRIAGE_RAW"] = silver["MARRIAGE"].astype(int)
    silver["MARRIAGE_CLEAN"] = silver["MARRIAGE_RAW"].astype(int)
    mar_labels = {1: "Married", 2: "Single", 3: "Other"}
    silver["MARRIAGE_LABEL"] = silver["MARRIAGE_CLEAN"].map(mar_labels)

    sex_labels = {1: "Male", 2: "Female"}
    silver["SEX_LABEL"] = silver["SEX"].map(sex_labels)

    for col in PAY_STATUS_COLS:
        silver[f"{col}_DPD"] = silver[col].clip(lower=0).astype(int)
        silver[f"{col}_IS_REVOLVING"] = (silver[col] == 0).astype(int)
        silver[f"{col}_IS_INACTIVE"] = (silver[col] == -2).astype(int)
        silver[f"{col}_IS_PAID_DULY"] = (silver[col] == -1).astype(int)

    return silver


def engineer_gold_features(silver_df: pd.DataFrame) -> pd.DataFrame:
    """Create Gold modelling table with credit-domain features and deterministic 70/30 dev/holdout split."""
    gold = silver_df.copy()

    dpd_cols = [f"{c}_DPD" for c in PAY_STATUS_COLS]
    gold["MAX_DPD_6M"] = gold[dpd_cols].max(axis=1)
    gold["MONTHS_DELINQUENT_6M"] = (gold[dpd_cols] >= 1).sum(axis=1)
    gold["MONTHS_30PLUS_DPD_6M"] = (gold[dpd_cols] >= 1).sum(axis=1)
    gold["MONTHS_60PLUS_DPD_6M"] = (gold[dpd_cols] >= 2).sum(axis=1)
    gold["MONTHS_REVOLVING_6M"] = (gold[PAY_STATUS_COLS] == 0).sum(axis=1)
    gold["MONTHS_INACTIVE_6M"] = (gold[PAY_STATUS_COLS] == -2).sum(axis=1)

    limit_safe = gold["LIMIT_BAL"].clip(lower=1.0)
    gold["UTIL_RATIO_M1"] = (gold["BILL_AMT1"] / limit_safe).clip(lower=-1.0, upper=5.0)
    gold["MEAN_BILL_AMT_6M"] = gold[BILL_AMT_COLS].mean(axis=1)
    gold["MAX_BILL_AMT_6M"] = gold[BILL_AMT_COLS].max(axis=1)
    gold["MEAN_UTIL_RATIO_6M"] = (gold["MEAN_BILL_AMT_6M"] / limit_safe).clip(lower=-1.0, upper=5.0)

    gold["MEAN_PAY_AMT_6M"] = gold[PAY_AMT_COLS].mean(axis=1)
    gold["SUM_PAY_AMT_6M"] = gold[PAY_AMT_COLS].sum(axis=1)
    gold["SUM_BILL_POS_6M"] = gold[BILL_AMT_COLS].clip(lower=0.0).sum(axis=1)

    gold["PAY_TO_BILL_RATIO_6M"] = np.where(
        gold["SUM_BILL_POS_6M"] > 0,
        (gold["SUM_PAY_AMT_6M"] / gold["SUM_BILL_POS_6M"]).clip(lower=0.0, upper=2.0),
        1.0,
    )
    gold["PAY_TO_LIMIT_RATIO_6M"] = (gold["SUM_PAY_AMT_6M"] / (6.0 * limit_safe)).clip(lower=0.0, upper=2.0)
    gold["BILL_TREND_6M_RATIO"] = ((gold["BILL_AMT1"] - gold["BILL_AMT6"]) / limit_safe).clip(lower=-2.0, upper=2.0)

    # Deterministic stratified 70% dev / 30% holdout split
    dev_idx, hold_idx = train_test_split(
        gold.index,
        test_size=0.30,
        random_state=RANDOM_SEED,
        stratify=gold["default_flag"],
    )
    gold["sample_split"] = "DEV"
    gold.loc[hold_idx, "sample_split"] = "HOLDOUT"

    return gold


def build_duckdb_layers(
    raw_csv_path: Path = RAW_CSV_PATH,
    cso_csv_path: Path = CSO_CSV_PATH,
    db_path: Path = DB_PATH,
    salt: str | None = None,
) -> Dict:
    """Execute full Step 1 pipeline and persist Bronze, Quarantine, Silver, Gold, and CSO tables in DuckDB."""
    active_salt = get_required_salt(salt)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    raw_df = pd.read_csv(raw_csv_path)
    ingested_at = datetime.now(timezone.utc).isoformat()

    # 1. Bronze layer: raw columns + pseudonymised account_id_hash + ingestion metadata
    bronze_df = raw_df.copy()
    bronze_df.insert(0, "account_id_hash", [pseudonymise_id(i, salt=active_salt) for i in bronze_df["ID"]])
    bronze_df["ingested_at_utc"] = ingested_at
    bronze_df["source_file"] = raw_csv_path.name

    assert bronze_df["account_id_hash"].nunique() == len(bronze_df), "Hash collision detected!"

    # 2. Evaluate Data Quality rules & route Quarantine vs Clean
    clean_df, quarantine_df, rule_summary_df = evaluate_data_quality_rules(bronze_df)

    # Verify breakdown of DQ09 (35 exact duplicates: 30 all-zero inactive vs 5 active)
    dup_rows = bronze_df.loc[bronze_df.duplicated(subset=REAL_FEATURE_COLS + [RAW_TARGET_COL], keep="first")]
    dup_inactive_cnt = int(
        ((dup_rows[BILL_AMT_COLS] == 0).all(axis=1) & (dup_rows[PAY_AMT_COLS] == 0).all(axis=1)).sum()
    )
    dup_active_cnt = int(len(dup_rows) - dup_inactive_cnt)

    # Drop raw 'ID' from Silver and Gold to enforce pseudonymisation downstream
    clean_no_raw_id = clean_df.drop(columns=["ID"]).copy()

    # 3. Silver layer: documented treatment of undocumented codes
    silver_df = apply_silver_transformations(clean_no_raw_id)

    # 4. Gold layer: domain features + stratified 70/30 dev/holdout split
    gold_df = engineer_gold_features(silver_df)

    # 5. Load CSO LRM01 macro dataset into bronze.cso_lrm01_raw for Step 5
    cso_df = pd.read_csv(cso_csv_path)

    if db_path.exists():
        db_path.unlink()
    con = duckdb.connect(str(db_path))
    try:
        con.execute("CREATE SCHEMA IF NOT EXISTS bronze;")
        con.execute("CREATE SCHEMA IF NOT EXISTS quarantine;")
        con.execute("CREATE SCHEMA IF NOT EXISTS silver;")
        con.execute("CREATE SCHEMA IF NOT EXISTS gold;")

        con.register("v_bronze", bronze_df)
        con.execute("CREATE TABLE bronze.credit_card_raw AS SELECT * FROM v_bronze;")

        con.register("v_cso", cso_df)
        con.execute("CREATE TABLE bronze.cso_lrm01_raw AS SELECT * FROM v_cso;")

        con.register("v_quarantine", quarantine_df)
        con.execute("CREATE TABLE quarantine.credit_card_quarantine AS SELECT * FROM v_quarantine;")

        con.register("v_dq_summary", rule_summary_df)
        con.execute("CREATE TABLE quarantine.dq_rule_summary AS SELECT * FROM v_dq_summary;")

        con.register("v_silver", silver_df)
        con.execute("CREATE TABLE silver.credit_card_cleansed AS SELECT * FROM v_silver;")

        con.register("v_gold", gold_df)
        con.execute("CREATE TABLE gold.credit_card_modelling AS SELECT * FROM v_gold;")

        counts = {
            "bronze.credit_card_raw": con.execute("SELECT COUNT(*) FROM bronze.credit_card_raw").fetchone()[0],
            "bronze.cso_lrm01_raw": con.execute("SELECT COUNT(*) FROM bronze.cso_lrm01_raw").fetchone()[0],
            "quarantine.credit_card_quarantine": con.execute(
                "SELECT COUNT(*) FROM quarantine.credit_card_quarantine"
            ).fetchone()[0],
            "quarantine.dq_rule_summary": con.execute("SELECT COUNT(*) FROM quarantine.dq_rule_summary").fetchone()[0],
            "silver.credit_card_cleansed": con.execute("SELECT COUNT(*) FROM silver.credit_card_cleansed").fetchone()[
                0
            ],
            "gold.credit_card_modelling": con.execute("SELECT COUNT(*) FROM gold.credit_card_modelling").fetchone()[0],
        }
    finally:
        con.close()

    undoc_summary = {
        "EDUCATION_raw_counts_in_bronze": {
            str(k): int(v) for k, v in bronze_df["EDUCATION"].value_counts().sort_index().items()
        },
        "EDUCATION_clean_counts_in_silver": {
            str(k): int(v) for k, v in silver_df["EDUCATION_CLEAN"].value_counts().sort_index().items()
        },
        "MARRIAGE_raw_counts_in_bronze": {
            str(k): int(v) for k, v in bronze_df["MARRIAGE"].value_counts().sort_index().items()
        },
        "MARRIAGE_clean_counts_in_silver": {
            str(k): int(v) for k, v in silver_df["MARRIAGE_CLEAN"].value_counts().sort_index().items()
        },
        "PAY_0_raw_counts_in_bronze": {
            str(k): int(v) for k, v in bronze_df["PAY_0"].value_counts().sort_index().items()
        },
        "PAY_0_DPD_counts_in_silver": {
            str(k): int(v) for k, v in silver_df["PAY_0_DPD"].value_counts().sort_index().items()
        },
        "PAY_documented_vs_undocumented_note": {
            "PAY_minus_1_documented_pay_duly_in_bronze_PAY_0": int((bronze_df["PAY_0"] == -1).sum()),
            "PAY_minus_2_undocumented_inactive_in_bronze_PAY_0": int((bronze_df["PAY_0"] == -2).sum()),
            "PAY_0_undocumented_revolving_in_bronze_PAY_0": int((bronze_df["PAY_0"] == 0).sum()),
            "accounts_with_any_undocumented_PAY_minus2_or_0_or_EDU_5_6_in_bronze": int(
                (bronze_df["EDUCATION"].isin([5, 6]) | bronze_df[PAY_STATUS_COLS].isin([-2, 0]).any(axis=1)).sum()
            ),
            "accounts_with_any_undocumented_PAY_minus2_or_0_or_EDU_5_6_in_silver": int(
                (silver_df["EDUCATION_RAW"].isin([5, 6]) | silver_df[PAY_STATUS_COLS].isin([-2, 0]).any(axis=1)).sum()
            ),
        },
    }

    split_summary = {}
    for split_name, grp in gold_df.groupby("sample_split"):
        split_summary[split_name] = {
            "rows": int(len(grp)),
            "defaults": int(grp["default_flag"].sum()),
            "default_rate": round(float(grp["default_flag"].mean()), 6),
        }

    step1_report = {
        "step": "Step 1 — Data Governance, DuckDB Medallion Layers, Pseudonymisation & Quarantine (Revised)",
        "generated_at_utc": ingested_at,
        "pseudonymisation_policy": {
            "algorithm": "Salted SHA-256 (SHA256(f'{SCOREGUARD_SALT}::{raw_id}'))",
            "salt_source": "Environment variable SCOREGUARD_SALT only (no hard-coded default; fails if missing)",
            "reversibility_governance_note": (
                "Because raw IDs in UCI 350 are sequential integers (1..30,000), a known or committed salt "
                "would make hashes trivially reversible via brute-force over 30,000 integers. The salt must "
                "therefore be stored exclusively in an unversioned environment secret (.env in .gitignore), "
                "and in production must be paired with an HSM-managed secret key or random UUID tokenisation vault."
            ),
        },
        "uci_350_columns": list(raw_df.columns),
        "cso_lrm01_columns": list(cso_df.columns),
        "duckdb_path": Path(db_path).resolve().relative_to(PROJECT_ROOT).as_posix() if Path(db_path).resolve().is_relative_to(PROJECT_ROOT) else Path(db_path).name,
        "table_row_counts": counts,
        "bronze_default_rate": round(float(bronze_df[RAW_TARGET_COL].mean()), 6),
        "quarantine_default_rate": round(float(quarantine_df[RAW_TARGET_COL].mean()), 6),
        "silver_default_rate": round(float(silver_df["default_flag"].mean()), 6),
        "quarantine_breakdown_by_reason": {
            str(k): int(v) for k, v in quarantine_df["failed_rules"].value_counts().items()
        },
        "dq09_flag_and_retain_breakdown": {
            "total_flagged_retained": int(silver_df["is_exact_dup_flag"].sum()),
            "all_zero_inactive_unique_id_accounts": dup_inactive_cnt,
            "active_bill_or_payment_accounts": dup_active_cnt,
            "flagged_default_rate": round(float(dup_rows[RAW_TARGET_COL].mean()), 6),
        },
        "dq_rule_summary": rule_summary_df.to_dict(orient="records"),
        "undocumented_code_treatment": undoc_summary,
        "gold_sample_splits": split_summary,
        "sample_pseudonymised_ids": [
            {"raw_id": int(bronze_df.loc[i, "ID"]), "account_id_hash": bronze_df.loc[i, "account_id_hash"]}
            for i in range(3)
        ],
    }

    with open(RESULTS_DIR / "step1_governance_summary.json", "w", encoding="utf-8") as f:
        json.dump(step1_report, f, indent=2)
    rule_summary_df.to_csv(RESULTS_DIR / "dq_rule_summary.csv", index=False)

    return step1_report


if __name__ == "__main__":
    report = build_duckdb_layers()
    print(json.dumps(report, indent=2))
