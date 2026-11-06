# ScoreGuard-IRB: Step 6 Report (Docs, App, Tests, README, Provenance)

Built from the approved Step 5 project. Python 3.13, pytest 9.1, Streamlit 1.67, DuckDB 2.0.

## 1. What was added

| File | Purpose |
| :--- | :--- |
| `src/build_audit_summary.py` | Writes `results/audit_summary.json`: raw file SHA-256 hashes, row counts, step timestamps, key metrics, every stated assumption, and a flat `readme_metrics` block. Reads only executed outputs. |
| `src/render_docs.py` | Renders `README.md`, `docs/Model_Development_Document.md`, `docs/Validation_and_Challenge_Log.md` and `private/cv_bullets.md` (git-ignored) from the audit record and results files. No number is typed by hand. |
| `app.py` | 3-page Streamlit app reading `results/` only, with the one-month target caveat, "illustrative" and "scenario sensitivity, not a calibrated model" labels. |
| `tests/` (5 files) | 66 tests, listed in section 3. |
| `docs/Data_Governance_Note.md` | Updated: version control and lineage table, protected-attribute access note, em dashes removed. |
| `.env.example` | Salt template with no secret; `.gitignore` keeps `.env` excluded. |

## 2. Fixes made during Step 6

1. **Baseline mean PD (23.59% vs 23.08%).** Both are correct but different quantities. 23.08% is the mean grade-level regulatory PD (every account at its grade PD, used for capital and the stress baseline). 23.59% is the mean IFRS 9 12-month PD proxy, where Stage 3 accounts are set to 100%. Both definitions are now in the audit record, README, app and Challenge Log (CL-16), and a test recomputes both from the database.
2. **Stale hard-coded text in Step 4 output.** `step4_validation_summary.json` still contained an old G1 breakdown (n = 211 / 471 / 1,877) that contradicted its own computed table (n = 243 / 434 / 1,882). The narrative and a hard-coded "+162.04 bps" are now generated from the computed values. Step 4 was rerun; every Step 4 CSV is numerically identical to before (maximum difference 0).
3. **Absolute Windows paths** to the calibration PNGs in the Step 4 JSON replaced with relative paths.
4. **Wrong comment** for the approval PD cutoff (said 28.43%; the code computes 26.82%) corrected.
5. **Streamlit API**: `use_container_width` is deprecated; replaced with `width="stretch"` and `requirements.txt` raised to `streamlit>=1.50.0`.

## 3. Test results

```text
66 passed
  tests/test_governance.py        12  pseudonymisation (deterministic, salt-dependent, fails without salt),
                                      no hard-coded salt, quarantine routing on a known example, executed
                                      counts, raw ID absent from gold, SEX/MARRIAGE excluded from both models
  tests/test_scorecard.py         18  WoE/IV on a hand-computed example, monotonic merging, executed bins
                                      monotonic, points fall as bin risk rises, coefficients negative,
                                      points-scaling identity, grade boundaries at every edge
  tests/test_ifrs9_capital.py     24  Markov rows sum to 1, counts match probabilities, no current-to-90+,
                                      absorbing curves, staging rules, ECL mechanics, K rises below the
                                      ~39% peak and falls above it, EL+UL rises everywhere, K/RWA/EL+UL
                                      rise with LGD, PD floor, capital table reproduces the formula,
                                      LGD x CCF grid monotonic
  tests/test_docs_consistency.py   9  audit record equals a fresh rebuild, README and all docs equal a
                                      fresh render, every README results number traces to the audit,
                                      both baseline PD definitions, key figures recomputed from DuckDB,
                                      raw file hashes
  tests/test_app.py                3  all 3 pages render without exceptions; required caveats present
```

No test was edited to make it pass. Two first-run failures were in my own test code: a DuckDB API call (`.sql().fetchone()` replaced by `.execute().fetchone()`), and the README trace check, which flagged `G5`, `IFRS 9`, `8%`, `2005` and `CCF 50%`. The first four are labels or regulatory constants and are whitelisted with a comment. CCF was a genuine gap and is now sourced from the code constant rather than whitelisted. A mutation check (changing one README number) makes two tests fail, as intended.

## 4. Key numbers (holdout, all from `results/audit_summary.json`)

| Metric | Value |
| :--- | ---: |
| Champion AUROC / Gini / KS | 0.7598 / 0.5197 / 0.3930 |
| Challenger AUROC / Gini (Platt) | 0.7730 / 0.5460 |
| Gini gap / review trigger | 2.63 / 5.0 points |
| Hosmer-Lemeshow p (champion / challenger) | 0.0013 / 0.3447 |
| G1 observed vs DEV rate | 7.56% vs 5.67% (AMBER) |
| G5 predicted vs observed | 36.01% vs 31.55% |
| RWA density / capital at 8% | 178.38% / NT$ 140.90m |
| IFRS 9 allocated ECL / coverage | NT$ 168.85m / 17.10% |
| Severe stress RWA / ECL uplift | +5.91% / +53.20% |

## 5. What was not rerun

Steps 1, 2, 3 and 5 were not rerun here; their outputs are used exactly as approved. The challenger (Step 3) needs LightGBM and SHAP, which are listed in `requirements.txt`. If you rerun the full pipeline locally with your own salt, the account hashes change (expected) and all metrics should be unchanged; then run `python src/build_audit_summary.py`, `python src/render_docs.py` and `pytest -q`.
