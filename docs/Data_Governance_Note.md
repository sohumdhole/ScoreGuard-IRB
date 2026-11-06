# ScoreGuard-IRB: Data Governance and Medallion Architecture Note

**Author:** Sohum Vivek Dhole (`github.com/sohumdhole` | MSc in Business Analytics, First Class Honours, Dublin Business School)  
**Target Role Alignment:** Quantitative Risk Analyst (Risk Analytics, 2nd Line of Defence)  
**Step 1 generated (UTC):** `2026-11-06T18:57:26.271361+00:00`  
**Updated in Step 6:** sections 5 and 6 added.

---

## 1. Data Sources & Licensing Attribution

1. **UCI Default of Credit Card Clients (ID 350)**
   - **Citation:** Yeh, I-Cheng (2016). *Default of Credit Card Clients* [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C55S3H.
   - **License:** Creative Commons Attribution 4.0 International (**CC BY 4.0**).
   - **Verified Raw Columns (`30,000 rows × 25 columns`):**
     ```text
     ['ID', 'LIMIT_BAL', 'SEX', 'EDUCATION', 'MARRIAGE', 'AGE',
      'PAY_0', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6',
      'BILL_AMT1', 'BILL_AMT2', 'BILL_AMT3', 'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6',
      'PAY_AMT1', 'PAY_AMT2', 'PAY_AMT3', 'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6',
      'default payment next month']
     ```
     *(Confirmed: `PAY_0` is followed directly by `PAY_2`; there is no `PAY_1` column.)*
2. **CSO Ireland LRM01 (Persons on Live Register)**
   - **Source API:** `https://ws.cso.ie/public/api.restful/PxStat.Data.Cube_API.ReadDataset/LRM01/CSV/1.0/en` (**CC BY 4.0**).
   - **Verified Raw Columns (`32,310 rows × 12 columns`):**
     ```text
     ['STATISTIC', 'STATISTIC Label', 'TLIST(M1)', 'Month',
      'C02076V02508', 'Age Group', 'C02199V02655', 'Sex',
      'C02454V02972', 'Social Welfare Scheme', 'UNIT', 'VALUE']
     ```

---

## 2. Pseudonymisation Control & Reversibility Governance Note

- **Implementation:** Account identifiers (`ID`) are pseudonymised at Bronze ingestion via [`pseudonymise_id`](../src/data_governance.py) using `SHA256(f"{SCOREGUARD_SALT}::{raw_id}")`, producing `account_id_hash`. Raw `ID` is removed prior to Silver and Gold persistence.
- **No Hard-Coded Salt:** `src/data_governance.py` reads the cryptographic salt exclusively from the `SCOREGUARD_SALT` environment variable (`get_required_salt`) and raises a `RuntimeError` if `SCOREGUARD_SALT` is unset or empty. Both `.env` and `*.duckdb` (`data/duckdb/`) are excluded in `.gitignore`.
- **Second-Line Governance Note on Sequential ID Reversibility:**
  > Because the public UCI ID 350 dataset assigns sequential integers (`1..30,000`) to `ID`, salted SHA-256 hashing with a known or version-controlled salt would be trivially reversible by precomputing 30,000 hashes (a rainbow/dictionary lookup over `1..30,000`). Keeping `SCOREGUARD_SALT` outside version control in `.env` prevents casual exposure, and in a regulated banking environment this control must be backed by an HSM-managed secret key or a non-deterministic tokenisation vault mapping customer IDs to random UUIDs.

---

## 3. DuckDB Medallion Architecture & Updated Table Counts

| DuckDB Table | Row Count | Defaults | Default Rate | Description |
| :--- | ---: | ---: | ---: | :--- |
| `bronze.credit_card_raw` | 30,000 | 6,636 | 22.1200% | Raw UCI 350 records + `account_id_hash` + ingestion metadata |
| `bronze.cso_lrm01_raw` | 32,310 | n/a | n/a | CSO Ireland Live Register macro series (for Step 5 stress sensitivity) |
| `quarantine.credit_card_quarantine` | **68** | **5** | **7.3529%** | Hard-quarantined records (`DQ07`: 14 rows; `DQ08`: 54 rows) |
| `quarantine.dq_rule_summary` | 10 | n/a | n/a | Audit log of 9 DQ rules and 1 documented mapping note |
| `silver.credit_card_cleansed` | **29,932** | **6,631** | **22.1535%** | Cleansed records, raw `ID` stripped, undocumented codes mapped, `DQ09` flagged & retained (`is_exact_dup_flag`) |
| `gold.credit_card_modelling` | **29,932** | **6,631** | **22.1535%** | Engineered credit features + stratified 70/30 `DEV` / `HOLDOUT` split |

### Updated Stratified Development vs. Holdout Split (`random_state=42`)
- **Development (`DEV`, 70%):** `20,952` accounts, `4,642` defaults (`22.1554%` default rate)
- **Holdout (`HOLDOUT`, 30%):** `8,980` accounts, `1,989` defaults (`22.1492%` default rate)

---

## 4. Data Quality Rules, Quarantine Routing & Documented Mapping Notes

| Rule ID | Description | Governance Action | Flagged Rows | % of Bronze | Flagged Default Rate |
| :--- | :--- | :--- | ---: | ---: | ---: |
| `DQ01_MISSING_CRITICAL_FIELDS` | Null/NaN in ID, any of the 23 features, or default flag | `HARD_QUARANTINE` | 0 | 0.0000% | 0.0000% |
| `DQ02_DUPLICATE_PRIMARY_KEY` | Duplicate raw account ID | `HARD_QUARANTINE` | 0 | 0.0000% | 0.0000% |
| `DQ03_NON_POSITIVE_CREDIT_LIMIT` | Credit limit `LIMIT_BAL <= 0` | `HARD_QUARANTINE` | 0 | 0.0000% | 0.0000% |
| `DQ04_AGE_OUT_OF_BOUNDS` | Cardholder `AGE < 18` or `AGE > 100` | `HARD_QUARANTINE` | 0 | 0.0000% | 0.0000% |
| `DQ05_NEGATIVE_REPAYMENT_AMOUNT` | Negative repayment amount in any `PAY_AMT1..PAY_AMT6` | `HARD_QUARANTINE` | 0 | 0.0000% | 0.0000% |
| `DQ06_INVALID_SEX_OR_TARGET_DOMAIN` | `SEX` not in `{1, 2}` or target default flag not in `{0, 1}` | `HARD_QUARANTINE` | 0 | 0.0000% | 0.0000% |
| `DQ07_UNDOCUMENTED_EDUCATION_ZERO` | `EDUCATION == 0` (unmapped zero code outside dictionary) | `HARD_QUARANTINE` | **14** | 0.0467% | 0.0000% |
| `DQ08_UNDOCUMENTED_MARRIAGE_ZERO` | `MARRIAGE == 0` (unmapped zero code outside dictionary) | `HARD_QUARANTINE` | **54** | 0.1800% | 9.2593% |
| `DQ09_EXACT_FEATURE_TARGET_DUPLICATE` | Exact duplicate across 23 features + target (30/35 are all-zero inactive accounts with unique IDs; flagged & retained) | `FLAG_AND_RETAIN` | **35** | 0.1167% | 17.1429% |
| `DQ10_UNDOCUMENTED_CODE_MAPPING_NOTE` | Documented mapping note (not a DQ violation): `EDUCATION in {5, 6}` or `PAY_0..6 in {-2, 0}` (`PAY=-1` is documented 'pay duly') | `DOCUMENTED_MAPPING_NOTE` | **25,956** | 86.5200% | 20.0301% |

### Rationale for `DQ09` (`FLAG_AND_RETAIN`)
- Across the 30,000 Bronze accounts, 35 rows repeat an earlier row across all 23 features and the default flag (`keep='first'`).
- Inspection shows that **30 of these 35 rows** have `0` across all 6 bill amounts (`BILL_AMT1..6`) and `0` across all 6 repayment amounts (`PAY_AMT1..6`), with distinct primary key `ID`s. Two dormant cardholders of the same age, gender, education, marital status, and credit limit naturally share identical all-zero billing vectors. Quarantining them would introduce selection bias against dormant accounts. They are therefore **retained** in Silver and Gold with `is_exact_dup_flag = 1`.

### Rationale for `DQ10` (`DOCUMENTED_MAPPING_NOTE`)
- In the original UCI documentation, `PAY_x = -1` is explicitly documented as **"pay duly"** (paid in full on time; 5,686 rows in `PAY_0`), whereas `PAY_x = -2` (no consumption / inactive; 2,759 rows in `PAY_0`) and `PAY_x = 0` (revolving credit / minimum paid on time; 14,737 rows in `PAY_0`) are undocumented in the initial UCI note, alongside `EDUCATION in {5, 6}` (331 rows).
- Because these codes represent standard account states rather than corrupt records, `DQ10` is recorded as a **Documented Mapping Note** (`25,956` accounts in Bronze; `25,902` in Silver):
  - `EDUCATION in {5, 6}` (280 + 51 = 331 rows) are consolidated with `EDUCATION = 4` (123 rows) into `EDUCATION_CLEAN = 4` (`Other / Unknown (4,5,6)` = **454 rows** in Silver; `1`: 10,581; `2`: 14,024; `3`: 4,873).
  - `MARRIAGE == 0` (54 rows) is quarantined (`DQ08`), leaving clean Silver counts `1` (Married): **13,655**, `2` (Single): **15,954**, `3` (Other): **323**.
  - `PAY_0..PAY_6` values `-2` (inactive), `-1` (documented "pay duly"), and `0` (revolving) all represent 0 months in arrears and map to `0` in `PAY_0_DPD..PAY_6_DPD` (`PAY_0_DPD` in Silver: `0`: **23,129**, `1`: **3,678**, `2`: **2,663**, `3`: **322**, `4`: **76**, `5`: **25**, `6`: **11**, `7`: **9**, `8`: **19**), while preserving the raw `PAY_0..PAY_6` columns and state indicators (`_IS_REVOLVING`, `_IS_INACTIVE`, `_IS_PAID_DULY`).

---

## 5. Version control and data lineage (Step 6)

| Artefact | Committed to Git? | Reason |
| :--- | :--- | :--- |
| `data/raw/` (UCI 350 and CSO LRM01 files) | Yes | Public data under CC BY 4.0, attributed in the README; SHA-256 hashes recorded in `results/audit_summary.json` |
| `data/duckdb/scoreguard.duckdb` | No (`.gitignore`) | Rebuildable from `src/`; contains the pseudonymised account hashes |
| `.env` / `SCOREGUARD_SALT` | No (`.gitignore`) | Secret; a leaked salt makes the sequential IDs reversible (Challenge Log CL-15) |
| `results/` | Yes | Executed outputs that the app, docs and tests read |
| `docs/`, `README.md` | Yes | Rendered from `results/` by `src/render_docs.py`, never edited by hand |

**Provenance record.** `src/build_audit_summary.py` writes `results/audit_summary.json` with raw file hashes, row counts, step timestamps, key metrics and every stated assumption. `tests/test_docs_consistency.py` rebuilds the record from `results/` and re-renders the README and docs, and fails if any copy on disk has drifted.

## 6. Access to protected attributes

`SEX` and `MARRIAGE` are kept in silver and gold only as governance metadata for the fair-lending check. They are excluded from both models, and `tests/test_governance.py` checks that neither appears in the champion or challenger feature lists.

