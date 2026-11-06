# ScoreGuard-IRB: Validation and Challenge Log

Rendered from executed outputs by `src/render_docs.py`. Each entry gives the finding, the evidence, the impact and the treatment. Status values: **Open** (limitation accepted and disclosed), **Monitored** (flagged for ongoing review), **Closed** (fixed during development).

| ID | Finding | Evidence | Impact | Treatment | Status |
| :--- | :--- | :--- | :--- | :--- | :--- |
| CL-01 | Target is a one-month default, used as a 12-month PD proxy | Target = `default payment next month` (October 2005) | Basel and IFRS 9 figures are not true 12-month quantities | Labelled "1M-as-12M PD proxy" in all reports and the app; capital and IFRS 9 marked illustrative | Open |
| CL-02 | PAY_0 bin 2 merges code -1 with code 0 | DEV default rate 16.49% for -1 (n = 3,965) vs 12.67% for 0 (n = 10,294) | Champion slightly under-ranks paid-duly accounts | Monotonic binning on raw PAY_0 cannot split them; challenger uses separate state indicators | Open |
| CL-03 | No out-of-time sample | Single 6-month cohort; holdout is a random 70/30 split | Stability over time cannot be tested | Score PSI 0.000607 reported with the caveat that a low value is expected by randomisation | Open |
| CL-04 | No recovery data | Dataset has no post-default cash flows | LGD cannot be estimated | LGD 85% stated as an assumption; sensitivity grid from the CRR3 QRRE input floor of 50% to 95% | Open |
| CL-05 | No origination data | No origination date or origination PD | True IFRS 9 SICR (change since origination) is impossible | Staging uses 30+ and 90+ days-past-due backstops only | Open |
| CL-06 | UCI PAY code 1 coding inconsistency | Code 1 appears 3,678 times in PAY_0 but 28, 4, 2, 0, 0 times in PAY_2 to PAY_6; code 2 appears 3,919, 3,809, 3,149, 2,621, 2,760 times | A 4-state transition matrix had an empty, absorbing "1 month late" state | Codes 1 and 2 merged into one 30-89 DPD state; 3-state matrix with no current-to-90+ transitions | Closed |
| CL-07 | G1 holdout uplift | Holdout default rate 7.56% vs DEV 5.67%; one-sided p = 0.0056 (AMBER) | Best grade may be under-predicted | Breakdown by PAY_0 code shows the uplift across all codes, so it is recorded as unexplained, most likely sampling variation | Monitored |
| CL-08 | G5 over-prediction | Mean predicted PD 36.01% vs observed 31.55%; two-sided p = 0.0032 | Conservative, but a calibration miss | Grade-level capital PDs use DEV observed rates plus margin of conservatism rather than the raw logistic PD | Open |
| CL-09 | Hosmer-Lemeshow rejects the champion | Holdout chi-square 25.49, p = 0.0013; challenger p = 0.3447 | Logistic link miscalibrated in mid-score deciles | Test is very sensitive at this sample size; disclosed rather than tuned away | Open |
| CL-10 | Fair-lending calibration gaps | Married: +190.50 bps, p = 0.0032; male: +113.01 bps, p = 0.1129 (not significant); lowest DIR 0.9382; 30 tests run | Excluding protected attributes pools risk across groups | Protected attributes stay excluded; gaps disclosed; multiple testing noted | Open |
| CL-11 | Portfolio and macro data mismatch | Taiwan 2005 card portfolio vs Irish Live Register 1967-2026 | Stress cannot be estimated, only assumed | Labelled "scenario sensitivity, not a calibrated model" | Open |
| CL-12 | Stated assumptions | LGD 85%, CCF 50%, EIR 18%, 36-month lifetime, elasticity 0.5, PD floor 0.10%, SICR overlay: A Stage 1 account moves to Stage 2 under stress if it is downgraded by at least one grade AND its post-stress grade is G4, G5 or G6. | Results depend on these choices | Stated in code, docs and app; LGD/CCF sensitivity table provided | Open |
| CL-13 | Challenger only marginally better | Holdout Gini gap 2.63 points | Little gain from the black-box model | Champion kept; review trigger at 5.0 Gini points | Monitored |
| CL-14 | Stage 1 ECL coverage unrealistically high | Holdout Stage 1 coverage 9.11% | Not comparable with bank provisioning levels | Consequence of CL-01 on a high-default portfolio; disclosed | Open |
| CL-15 | Pseudonymisation is reversible if the salt leaks | IDs are sequential integers 1 to 30,000 | A known salt allows a 30,000-hash lookup | Salt read only from the `SCOREGUARD_SALT` environment variable; production would need an HSM key or token vault | Open |
| CL-16 | Two baseline mean PDs | Holdout mean regulatory grade PD 23.08% vs mean IFRS 9 12-month PD proxy 23.59% | Looked inconsistent between tables | Both defined explicitly; difference is the Stage 3 = 100% override | Closed |
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

- Score PSI 0.000607 and all CSI values are near zero, which says nothing about stability over time (CL-03).
- The male calibration gap is not statistically significant (p = 0.1129).
- The challenger's AUROC gain over the champion is 0.0132 on the holdout.
