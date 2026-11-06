"""
ScoreGuard-IRB, Step 6: render README.md and the docs/ pack from executed outputs.

Every number in the rendered files comes from results/audit_summary.json or from the
results/*.csv / *.json files written by Steps 1-5. Nothing is typed in by hand.
tests/test_docs_consistency.py re-renders these files and fails if the copies on disk differ.

Run order:  python src/build_audit_summary.py  ->  python src/render_docs.py
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_ROOT / "results"
DOCS_DIR = PROJECT_ROOT / "docs"

OUTPUTS = {
    "README.md": PROJECT_ROOT / "README.md",
    "Model_Development_Document.md": DOCS_DIR / "Model_Development_Document.md",
    "Validation_and_Challenge_Log.md": DOCS_DIR / "Validation_and_Challenge_Log.md",
    "cv_bullets.md": PROJECT_ROOT / "private" / "cv_bullets.md",  # git-ignored: personal, not for the public repo
}


def _load() -> dict:
    with open(RESULTS_DIR / "audit_summary.json", encoding="utf-8") as f:
        audit = json.load(f)
    with open(RESULTS_DIR / "step2_scorecard_summary.json", encoding="utf-8") as f:
        s2 = json.load(f)
    with open(RESULTS_DIR / "step3_challenger_summary.json", encoding="utf-8") as f:
        s3 = json.load(f)
    return {
        "audit": audit,
        "m": audit["readme_metrics"],
        "s2": s2,
        "s3": s3,
        "iv": pd.read_csv(RESULTS_DIR / "step2_iv_summary.csv"),
        "grades": pd.read_csv(RESULTS_DIR / "step2_rating_grades.csv"),
        "cvc": pd.read_csv(RESULTS_DIR / "step3_champion_vs_challenger.csv"),
        "bt": pd.read_csv(RESULTS_DIR / "step4_grade_binomial_backtest.csv"),
        "flcal": pd.read_csv(RESULTS_DIR / "step4_fair_lending_calibration.csv"),
        "paycode": pd.read_csv(RESULTS_DIR / "step5_uci_pay_coding_audit.csv"),
        "stage": pd.read_csv(RESULTS_DIR / "step5_ifrs9_stage_summary.csv"),
    }


def _pct(x: float, dp: int = 2) -> str:
    return f"{x * 100:.{dp}f}%"


# --------------------------------------------------------------------------------------
# README
# --------------------------------------------------------------------------------------
def render_readme(d: dict) -> str:
    m = d["m"]
    return f"""# ScoreGuard-IRB

**Retail credit PD scorecard with second-line validation, an IFRS 9 staging illustration and illustrative Basel capital**

Independent portfolio project by Sohum Vivek Dhole ([github.com/sohumdhole](https://github.com/sohumdhole)).

> **Read this first.** This project uses real public data: 30,000 Taiwanese credit card accounts from 2005 (UCI, CC BY 4.0) and the CSO Ireland Live Register series.
> The model target is **default in the next month**, not a 12-month default. The capital and IFRS 9 pages use the model output as a 12-month PD proxy, which is a stated assumption, so those figures are **illustrative**. The Irish stress test is a **scenario sensitivity, not a calibrated model**. Every limitation is listed in [docs/Validation_and_Challenge_Log.md](docs/Validation_and_Challenge_Log.md).

---

## What it does

| Step | What was built | Main output |
| :--- | :--- | :--- |
| 1. Data governance | DuckDB bronze / silver / gold layers, salted SHA-256 pseudonymisation, 9 data quality rules plus 1 mapping note, quarantine table | {m['raw_rows']} raw rows, {m['quarantine_rows']} quarantined, {m['gold_rows']} modelling rows |
| 2. Champion scorecard | Monotonic WoE/IV binning, VIF checks, logistic regression, points scaling, 6-grade rating scale | {m['champion_features']} features, max VIF {m['champion_max_vif']} |
| 3. Challenger | Monotone-constrained LightGBM ({m['challenger_features']} features), SHAP, Platt calibration, written production defence | Champion stays in production |
| 4. Validation pack | AUROC, Gini, KS, Brier, Hosmer-Lemeshow, per-grade binomial backtest with Jeffreys intervals, PSI/CSI, fair-lending check | Findings logged openly, including failures |
| 5. Capital, IFRS 9, stress | Behavioural segmentation, Basel QRRE ASRF capital, 3-state Markov IFRS 9 staging, CSO LRM01 stress scenarios | Labelled illustrative throughout |
| 6. Documentation | Model Development Document, Validation & Challenge Log, governance note, Streamlit app, pytest suite, provenance record | This repository |

---

## Key results (holdout sample, {m['holdout_rows']} accounts, {m['holdout_default_rate']} default rate)

| Metric | Champion scorecard | Challenger (LightGBM, Platt) |
| :--- | ---: | ---: |
| AUROC | {m['champion_holdout_auroc']} | {m['challenger_holdout_auroc']} |
| Gini | {m['champion_holdout_gini']} | {m['challenger_holdout_gini']} |
| KS | {m['champion_holdout_ks']} | not used for decision |
| Brier score | {m['champion_holdout_brier']} | {m['challenger_holdout_brier']} |
| Hosmer-Lemeshow p-value (10 groups, df = 8) | {m['champion_hl_p']} | {m['challenger_hl_p']} |

**Production decision.** The challenger is only marginally better: its holdout Gini is {m['gini_gap_points']} points higher, below the {m['gini_trigger_points']}-point review trigger. The champion scorecard stays in production because every score is an additive sum of published bin points, it is monotonic by construction, and its development-to-holdout drop is smaller. The challenger runs in shadow mode as a benchmark.

### Validation findings (reported as they are)

- **Calibration test fails for the champion.** Hosmer-Lemeshow statistic {m['champion_hl_chi2']} (p = {m['champion_hl_p']}) on the holdout. The test is very sensitive at this sample size, but the miss is real and is logged.
- **G1 holdout uplift.** The best grade defaulted at {m['g1_observed_rate']} on the holdout against a {m['g1_dev_pd']} development rate (one-sided p = {m['g1_p']}, {m['g1_light']}). The breakdown by payment status does not explain it, so it is recorded as unexplained and flagged for monitoring.
- **G5 over-prediction.** Mean predicted PD {m['g5_predicted_pd']} against {m['g5_observed_rate']} observed (two-sided p = {m['g5_p']}). This is conservative but still a calibration miss.
- **Stability.** Score PSI is {m['score_psi']}, but the holdout is a random split, not an out-of-time sample, so this is expected and is **not** evidence of stability over time.
- **Fair lending.** Sex and marital status are excluded from both models. The lowest approval disparate-impact ratio is {m['min_dir']} (the four-fifths rule is a US heuristic, not an EU legal test). Calibration by group shows the model under-predicts for married borrowers by {m['married_gap_bps']} bps (p = {m['married_p']}); the male gap of {m['male_gap_bps']} bps is not significant (p = {m['male_p']}). {m['n_group_tests']} group-level tests were run, so some significance is expected by chance.

### Illustrative capital, IFRS 9 and stress (holdout)

| Item | Value |
| :--- | ---: |
| EAD (NT$ m, CCF {m['ccf']}) | {m['ead_m']} |
| RWA (NT$ m, QRRE, R = 0.04, LGD {m['lgd']}, PD floor {m['pd_floor']}) | {m['rwa_m']} |
| RWA density | {m['rwa_density']} |
| Capital at 8% (NT$ m) | {m['capital_m']} |
| Basel expected loss (NT$ m) | {m['el_m']} |
| IFRS 9 stage mix (Stage 1 / 2 / 3) | {m['stage1_share']} / {m['stage2_share']} / {m['stage3_share']} |
| IFRS 9 allocated ECL (NT$ m) and coverage | {m['ecl_m']} ({m['ecl_coverage']}) |
| Stage 2 lifetime uplift (NT$ m) | {m['stage2_uplift_m']} |
| Mild stress (Live Register +{m['mild_shock']} YoY): RWA / ECL uplift | +{m['mild_rwa_uplift']} / +{m['mild_ecl_uplift']} |
| Severe stress (+{m['severe_shock']} YoY, May 2009 peak): RWA / ECL uplift | +{m['severe_rwa_uplift']} / +{m['severe_ecl_uplift']} |

**Two baseline mean PDs, defined.** The capital and stress tables use the mean grade-level regulatory PD ({m['mean_reg_pd']}), where every account carries its grade PD. The IFRS 9 table uses the mean 12-month PD proxy ({m['mean_ifrs9_pd']}), where Stage 3 accounts are set to 100%. The difference comes entirely from that Stage 3 override.

Stage 1 ECL coverage is far higher than at a real bank because a one-month default rate from a high-default 2005 portfolio is being used as a 12-month PD. See the Challenge Log.

---

## Repository layout

```text
src/
  download_data.py            UCI 350 and CSO LRM01 downloads
  data_governance.py          Step 1: bronze/silver/gold, pseudonymisation, DQ rules
  scorecard_champion.py       Step 2: WoE/IV binning, logistic scorecard, grades
  challenger_lgbm.py          Step 3: monotone LightGBM, SHAP, calibration
  validation_pack.py          Step 4: validation pack and fair-lending check
  capital_ifrs9_stress.py     Step 5: segmentation, capital, IFRS 9, stress
  build_audit_summary.py      Step 6: provenance record (results/audit_summary.json)
  render_docs.py              Step 6: renders this README and docs/ from results
app.py                        3-page Streamlit app (reads results/ only)
LICENSE                       MIT licence for the code
.env.example                  template for the SCOREGUARD_SALT secret
tests/                        pytest suite
docs/                         Model Development Document, Challenge Log, governance note
results/                      All step outputs (CSV, JSON, PNG, step reports)
```

## How to run

```bash
pip install -r requirements.txt

# A secret salt is required for pseudonymisation. Never commit it.
export SCOREGUARD_SALT="choose-a-long-random-secret"      # Windows PowerShell: $env:SCOREGUARD_SALT="..."

python src/download_data.py
python src/data_governance.py
python src/scorecard_champion.py
python src/challenger_lgbm.py
python src/validation_pack.py
python src/capital_ifrs9_stress.py
python src/build_audit_summary.py
python src/render_docs.py

streamlit run app.py
pytest -q
```

The app and the test suite read the committed `results/` folder, so they run without rebuilding the DuckDB database (which is excluded from Git).

## Data and attribution

- **UCI Default of Credit Card Clients (ID 350).** Yeh, I-Cheng (2016). UCI Machine Learning Repository. https://doi.org/10.24432/C55S3H. Licence: CC BY 4.0.
- **CSO Ireland, LRM01 Persons on the Live Register.** Central Statistics Office, PxStat API. Licence: CC BY 4.0. Used only for the labelled scenario sensitivity.

Code licence: MIT (see [LICENSE](LICENSE)). The datasets keep their own CC BY 4.0 licences.

This is an independent learning project. It is not a production model and is not affiliated with any bank or regulator.
"""


# --------------------------------------------------------------------------------------
# Model Development Document
# --------------------------------------------------------------------------------------
def render_mdd(d: dict) -> str:
    m, s2, a = d["m"], d["s2"], d["audit"]
    ps = s2["points_scaling_parameters"]

    feat_rows = "\n".join(
        f"| `{r['feature']}` | `{r['formula']}` | {r['divide_by_zero_handling']} |"
        for r in s2["engineered_feature_dictionary"]
    )
    iv = d["iv"]
    iv_rows = "\n".join(
        f"| `{r.feature}` | {r.iv_dev:.4f} | {r.iv_strength_band.split(' (')[0]} | {r.selection_status} |"
        for r in iv.itertuples()
    )
    coef_rows = "\n".join(
        f"| `{r['feature']}` | {r['iv_dev']:.4f} | {r['vif_dev']:.2f} | {r['coefficient_beta']:.6f} | {r['std_error']:.6f} | {r['p_value']:.2e} |"
        for r in s2["final_model_coefficients_and_vif"]
    )
    g = d["grades"]
    grade_rows = "\n".join(
        f"| {r.grade} | `{r.score_band}` | {r.dev_count:,} | {_pct(r.dev_observed_default_rate)} | {_pct(r.dev_mean_predicted_pd)} | {r.holdout_count:,} | {_pct(r.holdout_observed_default_rate)} |"
        for r in g.itertuples()
    )
    cvc = d["cvc"]
    cvc_rows = "\n".join(
        f"| {r.model_name} | {r.feature_count} | {r.dev_auroc:.4f} ({r.dev_eval_type}) | {r.holdout_auroc:.4f} | {r.holdout_gini:.4f} | {r.holdout_brier_platt:.4f} |"
        for r in cvc.itertuples()
    )
    asm = a["assumptions"]
    rc = a["row_counts"]

    return f"""# ScoreGuard-IRB: Model Development Document

Rendered from executed outputs by `src/render_docs.py`. Do not edit by hand.

## 1. Purpose and scope

A retail credit card probability-of-default (PD) scorecard built and validated in the style of a bank's second line of defence, with illustrative Basel capital, IFRS 9 staging and Irish stress sensitivities. It is an independent portfolio project, not a production model.

**Target definition.** `default payment next month`: whether the account defaulted in October 2005, after the April to September 2005 observation window. This is a one-month-ahead default event, not a 12-month PD. See CL-01 in the Challenge Log.

## 2. Data

| Item | Value |
| :--- | :--- |
| Source | {a['data_sources']['uci_350']['citation']} |
| Licence | {a['data_sources']['uci_350']['licence']} |
| Raw rows | {m['raw_rows']} |
| Quarantined rows | {m['quarantine_rows']} (EDUCATION = 0 or MARRIAGE = 0) |
| Modelling rows (gold) | {m['gold_rows']} |
| Development / holdout split | {rc['dev_rows']:,} / {rc['holdout_rows']:,}, stratified 70/30, random_state = 42 |
| Macro series | CSO LRM01, {m['lrm01_rows']} rows, used only for stress sensitivity |

Governance controls (pseudonymisation, data quality rules, quarantine) are described in [Data_Governance_Note.md](Data_Governance_Note.md).

## 3. Feature engineering

| Feature | Formula | Divide-by-zero handling |
| :--- | :--- | :--- |
{feat_rows}

Excluded from all candidate features: `SEX`, `MARRIAGE` and derived columns, `account_id_hash`, `is_exact_dup_flag`.

## 4. Binning and feature selection (DEV only)

Monotonic WoE bins, minimum 5% of DEV per bin. WoE = ln(%Good / %Bad), so higher WoE means lower risk and every coefficient should be negative.

| Feature | DEV IV | Strength | Status |
| :--- | ---: | :--- | :--- |
{iv_rows}

## 5. Champion logistic scorecard

VIF threshold {s2['vif_threshold_stated']}. All retained coefficients are negative, consistent with WoE coding.

| Feature | DEV IV | VIF | Coefficient | Std. error | p-value |
| :--- | ---: | ---: | ---: | ---: | ---: |
{coef_rows}

Intercept {ps['logistic_intercept_b0']}.

**Points scaling.** Base points {ps['base_points']:.0f} at odds {ps['base_odds_good_to_bad']} (PD {_pct(ps['base_pd'])}), PDO {ps['pdo']:.0f}, factor {ps['factor']:.6f}, offset {ps['offset']:.6f}. Score = offset - factor x ln(PD / (1 - PD)).

## 6. Rating scale

| Grade | Score band | DEV count | DEV default rate | DEV mean PD | Holdout count | Holdout default rate |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: |
{grade_rows}

The original 7-grade scale merged its top two grades because they did not separate (see CL-17).

## 7. Champion versus challenger

| Model | Features | DEV AUROC | Holdout AUROC | Holdout Gini | Holdout Brier (Platt) |
| :--- | ---: | ---: | ---: | ---: | ---: |
{cvc_rows}

**Decision.** Champion in production, Challenger-Expanded in shadow mode. Holdout Gini gap {m['gini_gap_points']} points against a {m['gini_trigger_points']}-point review trigger. The challenger constrains `PAY_x_DPD = max(PAY_x, 0)` with +1 and leaves the inactive, paid-duly and revolving indicators unconstrained, because raw PAY codes are not monotonic in risk below zero (see CL-02).

## 8. Downstream illustrations (Step 5)

| Component | Method | Key assumptions |
| :--- | :--- | :--- |
| Basel capital | QRRE ASRF, K = LGD x (CDR(PD) - PD), RWA = 12.5 x K x EAD | R = {asm['asset_correlation_r']}, PD floor {_pct(asm['pd_floor_applied'])}, LGD {_pct(asm['lgd'], 0)}, CCF {_pct(asm['ccf'], 0)}, grade PD = DEV rate + Jeffreys MoC |
| IFRS 9 | 3-state monthly Markov matrix (current, 30-89 DPD, 90+), stage from PAY_0, account curves scaled to the PD proxy | EIR {_pct(asm['eir_annual'], 0)} a year, {asm['ifrs9_lifetime_horizon_months']}-month lifetime, no origination-based SICR |
| Stress | LRM01 year-on-year changes (P90 mild, historical max severe) mapped to PD multipliers | Elasticity {asm['stress_elasticity_epsilon']}; SICR overlay rule: {asm['stress_sicr_overlay_rule']} |

## 9. Use and limitations

Suitable for demonstrating scorecard development and validation technique. Not suitable for any credit decision. The full list of limitations, null findings and assumptions is in [Validation_and_Challenge_Log.md](Validation_and_Challenge_Log.md).
"""


# --------------------------------------------------------------------------------------
# Validation & Challenge Log
# --------------------------------------------------------------------------------------
def render_challenge_log(d: dict) -> str:
    m, s3, a = d["m"], d["s3"], d["audit"]
    nd = s3["non_delinquent_pay_code_non_monotonicity_check_dev"]["PAY_0"]
    pc = d["paycode"].set_index("column")
    pay1 = ", ".join(f"{int(pc.loc[c, 'gold_code_1']):,}" for c in ["PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"])
    pay2 = ", ".join(f"{int(pc.loc[c, 'gold_code_2']):,}" for c in ["PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"])
    st = d["stage"]
    s1_cov = float(st[(st.sample_split == "HOLDOUT") & (st.ifrs9_stage == 1)]["coverage_ratio_allocated_ecl_to_ead_pct"].iloc[0])
    asm = a["assumptions"]
    km = a["key_metrics"]

    return f"""# ScoreGuard-IRB: Validation and Challenge Log

Rendered from executed outputs by `src/render_docs.py`. Each entry gives the finding, the evidence, the impact and the treatment. Status values: **Open** (limitation accepted and disclosed), **Monitored** (flagged for ongoing review), **Closed** (fixed during development).

| ID | Finding | Evidence | Impact | Treatment | Status |
| :--- | :--- | :--- | :--- | :--- | :--- |
| CL-01 | Target is a one-month default, used as a 12-month PD proxy | Target = `default payment next month` (October 2005) | Basel and IFRS 9 figures are not true 12-month quantities | Labelled "1M-as-12M PD proxy" in all reports and the app; capital and IFRS 9 marked illustrative | Open |
| CL-02 | PAY_0 bin 2 merges code -1 with code 0 | DEV default rate {_pct(nd['-1']['dev_bad_rate'])} for -1 (n = {nd['-1']['dev_count']:,}) vs {_pct(nd['0']['dev_bad_rate'])} for 0 (n = {nd['0']['dev_count']:,}) | Champion slightly under-ranks paid-duly accounts | Monotonic binning on raw PAY_0 cannot split them; challenger uses separate state indicators | Open |
| CL-03 | No out-of-time sample | Single 6-month cohort; holdout is a random 70/30 split | Stability over time cannot be tested | Score PSI {m['score_psi']} reported with the caveat that a low value is expected by randomisation | Open |
| CL-04 | No recovery data | Dataset has no post-default cash flows | LGD cannot be estimated | LGD {m['lgd']} stated as an assumption; sensitivity grid from the CRR3 QRRE input floor of {_pct(asm['lgd_crr3_qrre_input_floor'], 0)} to 95% | Open |
| CL-05 | No origination data | No origination date or origination PD | True IFRS 9 SICR (change since origination) is impossible | Staging uses 30+ and 90+ days-past-due backstops only | Open |
| CL-06 | UCI PAY code 1 coding inconsistency | Code 1 appears {int(pc.loc['PAY_0', 'gold_code_1']):,} times in PAY_0 but {pay1} times in PAY_2 to PAY_6; code 2 appears {pay2} times | A 4-state transition matrix had an empty, absorbing "1 month late" state | Codes 1 and 2 merged into one 30-89 DPD state; 3-state matrix with no current-to-90+ transitions | Closed |
| CL-07 | G1 holdout uplift | Holdout default rate {m['g1_observed_rate']} vs DEV {m['g1_dev_pd']}; one-sided p = {m['g1_p']} ({m['g1_light']}) | Best grade may be under-predicted | Breakdown by PAY_0 code shows the uplift across all codes, so it is recorded as unexplained, most likely sampling variation | Monitored |
| CL-08 | G5 over-prediction | Mean predicted PD {m['g5_predicted_pd']} vs observed {m['g5_observed_rate']}; two-sided p = {m['g5_p']} | Conservative, but a calibration miss | Grade-level capital PDs use DEV observed rates plus margin of conservatism rather than the raw logistic PD | Open |
| CL-09 | Hosmer-Lemeshow rejects the champion | Holdout chi-square {m['champion_hl_chi2']}, p = {m['champion_hl_p']}; challenger p = {m['challenger_hl_p']} | Logistic link miscalibrated in mid-score deciles | Test is very sensitive at this sample size; disclosed rather than tuned away | Open |
| CL-10 | Fair-lending calibration gaps | Married: {m['married_gap_bps']} bps, p = {m['married_p']}; male: {m['male_gap_bps']} bps, p = {m['male_p']} (not significant); lowest DIR {m['min_dir']}; {m['n_group_tests']} tests run | Excluding protected attributes pools risk across groups | Protected attributes stay excluded; gaps disclosed; multiple testing noted | Open |
| CL-11 | Portfolio and macro data mismatch | Taiwan 2005 card portfolio vs Irish Live Register 1967-2026 | Stress cannot be estimated, only assumed | Labelled "scenario sensitivity, not a calibrated model" | Open |
| CL-12 | Stated assumptions | LGD {m['lgd']}, CCF {_pct(asm['ccf'], 0)}, EIR {_pct(asm['eir_annual'], 0)}, {asm['ifrs9_lifetime_horizon_months']}-month lifetime, elasticity {asm['stress_elasticity_epsilon']}, PD floor {m['pd_floor']}, SICR overlay: {asm['stress_sicr_overlay_rule']} | Results depend on these choices | Stated in code, docs and app; LGD/CCF sensitivity table provided | Open |
| CL-13 | Challenger only marginally better | Holdout Gini gap {m['gini_gap_points']} points | Little gain from the black-box model | Champion kept; review trigger at {m['gini_trigger_points']} Gini points | Monitored |
| CL-14 | Stage 1 ECL coverage unrealistically high | Holdout Stage 1 coverage {s1_cov:.2f}% | Not comparable with bank provisioning levels | Consequence of CL-01 on a high-default portfolio; disclosed | Open |
| CL-15 | Pseudonymisation is reversible if the salt leaks | IDs are sequential integers 1 to 30,000 | A known salt allows a 30,000-hash lookup | Salt read only from the `SCOREGUARD_SALT` environment variable; production would need an HSM key or token vault | Open |
| CL-16 | Two baseline mean PDs | Holdout mean regulatory grade PD {m['mean_reg_pd']} vs mean IFRS 9 12-month PD proxy {m['mean_ifrs9_pd']} | Looked inconsistent between tables | Both defined explicitly; difference is the Stage 3 = 100% override | Closed |
| CL-17 | Corrections made during development review | See list below | Shows the review trail | All fixed and rerun | Closed |

### CL-17: development review trail

1. A hard-coded pseudonymisation salt was removed; the salt now comes only from the environment.
2. Exact-duplicate rows (mostly all-zero dormant accounts with unique IDs) changed from quarantine to flag-and-retain.
3. PAY = -1 reclassified as a documented code; only -2 and 0 are undocumented.
4. The top two grades of a 7-grade scale did not separate, so they were merged into a 6-grade scale.
5. The challenger first applied +1 monotone constraints to raw PAY codes; replaced by constraints on days past due plus unconstrained state indicators.
6. An unsupported claim that PAY_0 code -1 caused the G1 uplift was withdrawn after the breakdown contradicted it.
7. A grade profile table that disagreed with the database was regenerated from the database.
8. A 4-state transition matrix with impossible transitions was rebuilt as a 3-state matrix (CL-06).
9. Hard-coded narrative figures in the Step 4 summary were replaced with values generated from the computed tables.

### Null and weak findings

- Score PSI {m['score_psi']} and all CSI values are near zero, which says nothing about stability over time (CL-03).
- The male calibration gap is not statistically significant (p = {m['male_p']}).
- The challenger's AUROC gain over the champion is {km['challenger_expanded_platt']['holdout_auroc'] - km['champion']['holdout_auroc']:.4f} on the holdout.
"""


# --------------------------------------------------------------------------------------
# CV bullets
# --------------------------------------------------------------------------------------
def render_cv_bullets(d: dict) -> str:
    m = d["m"]
    return f"""# ScoreGuard-IRB: CV bullets

Private file (git-ignored). Rendered from `results/audit_summary.json` by `src/render_docs.py`. Each bullet lists the audit keys it relies on. Label the project as independent work on any CV. Do not add these to a CV until the repository is public.

1. Built a retail credit PD scorecard (monotonic WoE logistic regression, {m['champion_features']} features, maximum VIF {m['champion_max_vif']}) on {m['gold_rows']} real credit card accounts, reaching a holdout AUROC of {m['champion_holdout_auroc']} (Gini {m['champion_holdout_gini']}).
   - Sources: `readme_metrics.champion_features`, `champion_max_vif`, `gold_rows`, `champion_holdout_auroc`, `champion_holdout_gini`

2. Benchmarked a monotone-constrained LightGBM challenger with SHAP (holdout Gini {m['challenger_holdout_gini']}, {m['gini_gap_points']} points higher) and wrote the production defence for keeping the explainable scorecard, with a {m['gini_trigger_points']}-point Gini review trigger.
   - Sources: `challenger_holdout_gini`, `gini_gap_points`, `gini_trigger_points`

3. Ran a second-line validation pack (KS, Hosmer-Lemeshow, Jeffreys per-grade backtests, PSI, fair-lending calibration across {m['n_group_tests']} group tests) and reported the failures openly, including a top-grade holdout uplift ({m['g1_observed_rate']} observed against a {m['g1_dev_pd']} development rate).
   - Sources: `n_group_tests`, `g1_observed_rate`, `g1_dev_pd`

4. Illustrated Basel QRRE capital (holdout RWA density {m['rwa_density']}) and IFRS 9 staging with a 3-state Markov matrix (ECL coverage {m['ecl_coverage']}) under stated assumptions, plus Irish Live Register stress scenarios raising ECL by up to {m['severe_ecl_uplift']}.
   - Sources: `rwa_density`, `ecl_coverage`, `severe_ecl_uplift`
"""


def render_all() -> dict[str, str]:
    d = _load()
    return {
        "README.md": render_readme(d),
        "Model_Development_Document.md": render_mdd(d),
        "Validation_and_Challenge_Log.md": render_challenge_log(d),
        "cv_bullets.md": render_cv_bullets(d),
    }


def main() -> None:
    DOCS_DIR.mkdir(exist_ok=True)
    (PROJECT_ROOT / "private").mkdir(exist_ok=True)
    for name, text in render_all().items():
        OUTPUTS[name].write_text(text, encoding="utf-8", newline="\n")
        print(f"Wrote {OUTPUTS[name].relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
