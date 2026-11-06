# ScoreGuard-IRB

**Retail credit PD scorecard with second-line validation, an IFRS 9 staging illustration and illustrative Basel capital**

Independent portfolio project by Sohum Vivek Dhole ([github.com/sohumdhole](https://github.com/sohumdhole)).

> **Read this first.** This project uses real public data: 30,000 Taiwanese credit card accounts from 2005 (UCI, CC BY 4.0) and the CSO Ireland Live Register series.
> The model target is **default in the next month**, not a 12-month default. The capital and IFRS 9 pages use the model output as a 12-month PD proxy, which is a stated assumption, so those figures are **illustrative**. The Irish stress test is a **scenario sensitivity, not a calibrated model**. Every limitation is listed in [docs/Validation_and_Challenge_Log.md](docs/Validation_and_Challenge_Log.md).

---

## What it does

| Step | What was built | Main output |
| :--- | :--- | :--- |
| 1. Data governance | DuckDB bronze / silver / gold layers, salted SHA-256 pseudonymisation, 9 data quality rules plus 1 mapping note, quarantine table | 30,000 raw rows, 68 quarantined, 29,932 modelling rows |
| 2. Champion scorecard | Monotonic WoE/IV binning, VIF checks, logistic regression, points scaling, 6-grade rating scale | 7 features, max VIF 3.19 |
| 3. Challenger | Monotone-constrained LightGBM (19 features), SHAP, Platt calibration, written production defence | Champion stays in production |
| 4. Validation pack | AUROC, Gini, KS, Brier, Hosmer-Lemeshow, per-grade binomial backtest with Jeffreys intervals, PSI/CSI, fair-lending check | Findings logged openly, including failures |
| 5. Capital, IFRS 9, stress | Behavioural segmentation, Basel QRRE ASRF capital, 3-state Markov IFRS 9 staging, CSO LRM01 stress scenarios | Labelled illustrative throughout |
| 6. Documentation | Model Development Document, Validation & Challenge Log, governance note, Streamlit app, pytest suite, provenance record | This repository |

---

## Key results (holdout sample, 8,980 accounts, 22.15% default rate)

| Metric | Champion scorecard | Challenger (LightGBM, Platt) |
| :--- | ---: | ---: |
| AUROC | 0.7598 | 0.7730 |
| Gini | 0.5197 | 0.5460 |
| KS | 0.3930 | not used for decision |
| Brier score | 0.1403 | 0.1368 |
| Hosmer-Lemeshow p-value (10 groups, df = 8) | 0.0013 | 0.3447 |

**Production decision.** The challenger is only marginally better: its holdout Gini is 2.63 points higher, below the 5.0-point review trigger. The champion scorecard stays in production because every score is an additive sum of published bin points, it is monotonic by construction, and its development-to-holdout drop is smaller. The challenger runs in shadow mode as a benchmark.

### Validation findings (reported as they are)

- **Calibration test fails for the champion.** Hosmer-Lemeshow statistic 25.49 (p = 0.0013) on the holdout. The test is very sensitive at this sample size, but the miss is real and is logged.
- **G1 holdout uplift.** The best grade defaulted at 7.56% on the holdout against a 5.67% development rate (one-sided p = 0.0056, AMBER). The breakdown by payment status does not explain it, so it is recorded as unexplained and flagged for monitoring.
- **G5 over-prediction.** Mean predicted PD 36.01% against 31.55% observed (two-sided p = 0.0032). This is conservative but still a calibration miss.
- **Stability.** Score PSI is 0.000607, but the holdout is a random split, not an out-of-time sample, so this is expected and is **not** evidence of stability over time.
- **Fair lending.** Sex and marital status are excluded from both models. The lowest approval disparate-impact ratio is 0.9382 (the four-fifths rule is a US heuristic, not an EU legal test). Calibration by group shows the model under-predicts for married borrowers by +190.50 bps (p = 0.0032); the male gap of +113.01 bps is not significant (p = 0.1129). 30 group-level tests were run, so some significance is expected by chance.

### Illustrative capital, IFRS 9 and stress (holdout)

| Item | Value |
| :--- | ---: |
| EAD (NT$ m, CCF 50%) | 987.36 |
| RWA (NT$ m, QRRE, R = 0.04, LGD 85%, PD floor 0.10%) | 1,761.22 |
| RWA density | 178.38% |
| Capital at 8% (NT$ m) | 140.90 |
| Basel expected loss (NT$ m) | 155.98 |
| IFRS 9 stage mix (Stage 1 / 2 / 3) | 77.33% / 21.17% / 1.50% |
| IFRS 9 allocated ECL (NT$ m) and coverage | 168.85 (17.10%) |
| Stage 2 lifetime uplift (NT$ m) | 21.94 |
| Mild stress (Live Register +22.82% YoY): RWA / ECL uplift | +2.82% / +12.26% |
| Severe stress (+95.35% YoY, May 2009 peak): RWA / ECL uplift | +5.91% / +53.20% |

**Two baseline mean PDs, defined.** The capital and stress tables use the mean grade-level regulatory PD (23.08%), where every account carries its grade PD. The IFRS 9 table uses the mean 12-month PD proxy (23.59%), where Stage 3 accounts are set to 100%. The difference comes entirely from that Stage 3 override.

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
