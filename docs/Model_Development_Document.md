# ScoreGuard-IRB: Model Development Document

Rendered from executed outputs by `src/render_docs.py`. Do not edit by hand.

## 1. Purpose and scope

A retail credit card probability-of-default (PD) scorecard built and validated in the style of a bank's second line of defence, with illustrative Basel capital, IFRS 9 staging and Irish stress sensitivities. It is an independent portfolio project, not a production model.

**Target definition.** `default payment next month`: whether the account defaulted in October 2005, after the April to September 2005 observation window. This is a one-month-ahead default event, not a 12-month PD. See CL-01 in the Challenge Log.

## 2. Data

| Item | Value |
| :--- | :--- |
| Source | Yeh, I-Cheng (2016). Default of Credit Card Clients. UCI Machine Learning Repository. https://doi.org/10.24432/C55S3H |
| Licence | CC BY 4.0 |
| Raw rows | 30,000 |
| Quarantined rows | 68 (EDUCATION = 0 or MARRIAGE = 0) |
| Modelling rows (gold) | 29,932 |
| Development / holdout split | 20,952 / 8,980, stratified 70/30, random_state = 42 |
| Macro series | CSO LRM01, 32,310 rows, used only for stress sensitivity |

Governance controls (pseudonymisation, data quality rules, quarantine) are described in [Data_Governance_Note.md](Data_Governance_Note.md).

## 3. Feature engineering

| Feature | Formula | Divide-by-zero handling |
| :--- | :--- | :--- |
| `MONTHS_DELINQUENT_6M` | `sum_{t in {0,2,3,4,5,6}} I(PAY_t_DPD >= 1), where PAY_t_DPD = max(PAY_t, 0)` | N/A (integer count in {0..6}; no division). |
| `MONTHS_60PLUS_DPD_6M` | `sum_{t in {0,2,3,4,5,6}} I(PAY_t_DPD >= 2), where PAY_t_DPD = max(PAY_t, 0)` | N/A (integer count in {0..6}; no division). |
| `MAX_DPD_6M` | `max_{t in {0,2,3,4,5,6}} (PAY_t_DPD), where PAY_t_DPD = max(PAY_t, 0)` | N/A (integer maximum in {0..8}; no division). |
| `MONTHS_REVOLVING_6M` | `sum_{t in {0,2,3,4,5,6}} I(PAY_t == 0)` | N/A (integer count in {0..6}; no division). |
| `MONTHS_INACTIVE_6M` | `sum_{t in {0,2,3,4,5,6}} I(PAY_t == -2)` | N/A (integer count in {0..6}; no division). |
| `MEAN_PAY_AMT_6M` | `(1 / 6) * sum_{k=1}^{6} PAY_AMT_k` | Fixed denominator = 6 (no variable denominator). |
| `MEAN_UTIL_RATIO_6M` | `clip( ((1/6) * sum_{k=1}^{6} BILL_AMT_k) / max(LIMIT_BAL, 1.0), -1.0, 5.0 )` | Denominator protected via limit_safe = max(LIMIT_BAL, 1.0) plus DQ03 hard quarantine (LIMIT_BAL <= 0); ratio clipped to [-1.0, 5.0]. |
| `UTIL_RATIO_M1` | `clip( BILL_AMT1 / max(LIMIT_BAL, 1.0), -1.0, 5.0 )` | Denominator protected via limit_safe = max(LIMIT_BAL, 1.0) plus DQ03 hard quarantine (LIMIT_BAL <= 0); ratio clipped to [-1.0, 5.0]. |
| `BILL_TREND_6M_RATIO` | `clip( (BILL_AMT1 - BILL_AMT6) / max(LIMIT_BAL, 1.0), -2.0, 2.0 )` | Denominator protected via limit_safe = max(LIMIT_BAL, 1.0) plus DQ03 hard quarantine (LIMIT_BAL <= 0); ratio clipped to [-2.0, 2.0]. |
| `PAY_TO_BILL_RATIO_6M` | `clip( sum_{k=1}^{6} PAY_AMT_k / sum_{k=1}^{6} max(BILL_AMT_k, 0), 0.0, 2.0 ) if sum_{k=1}^{6} max(BILL_AMT_k, 0) > 0 else 1.0` | Explicitly guarded by np.where(SUM_BILL_POS_6M > 0, ..., 1.0): when 6-month positive billed balances sum to zero, the ratio defaults to 1.0 (100% paid / no unpaid positive bill); otherwise clipped to [0.0, 2.0]. |
| `PAY_TO_LIMIT_RATIO_6M` | `clip( sum_{k=1}^{6} PAY_AMT_k / (6.0 * max(LIMIT_BAL, 1.0)), 0.0, 2.0 )` | Denominator protected via 6.0 * max(LIMIT_BAL, 1.0) plus DQ03 hard quarantine (LIMIT_BAL <= 0); ratio clipped to [0.0, 2.0]. |

Excluded from all candidate features: `SEX`, `MARRIAGE` and derived columns, `account_id_hash`, `is_exact_dup_flag`.

## 4. Binning and feature selection (DEV only)

Monotonic WoE bins, minimum 5% of DEV per bin. WoE = ln(%Good / %Bad), so higher WoE means lower risk and every coefficient should be negative.

| Feature | DEV IV | Strength | Status |
| :--- | ---: | :--- | :--- |
| `PAY_0` | 0.8964 | Very Strong | KEPT |
| `MONTHS_DELINQUENT_6M` | 0.8955 | Very Strong | KEPT |
| `MONTHS_60PLUS_DPD_6M` | 0.8744 | Very Strong | DROPPED_COLLINEAR_OR_REDUNDANT |
| `MAX_DPD_6M` | 0.7770 | Very Strong | DROPPED_COLLINEAR_OR_REDUNDANT |
| `PAY_2` | 0.5701 | Very Strong | DROPPED_COLLINEAR_OR_REDUNDANT |
| `PAY_3` | 0.4221 | Strong | DROPPED_COLLINEAR_OR_REDUNDANT |
| `PAY_4` | 0.3729 | Strong | DROPPED_COLLINEAR_OR_REDUNDANT |
| `PAY_5` | 0.3410 | Strong | DROPPED_COLLINEAR_OR_REDUNDANT |
| `PAY_6` | 0.3038 | Strong | DROPPED_COLLINEAR_OR_REDUNDANT |
| `MONTHS_REVOLVING_6M` | 0.2738 | Medium | KEPT |
| `MEAN_PAY_AMT_6M` | 0.1816 | Medium | KEPT |
| `SUM_PAY_AMT_6M` | 0.1816 | Medium | DROPPED_COLLINEAR_OR_REDUNDANT |
| `LIMIT_BAL` | 0.1678 | Medium | KEPT |
| `PAY_AMT1` | 0.1675 | Medium | DROPPED_COLLINEAR_OR_REDUNDANT |
| `PAY_AMT2` | 0.1383 | Medium | DROPPED_COLLINEAR_OR_REDUNDANT |
| `PAY_AMT3` | 0.1123 | Medium | DROPPED_COLLINEAR_OR_REDUNDANT |
| `PAY_AMT4` | 0.0985 | Weak | DROPPED_COLLINEAR_OR_REDUNDANT |
| `PAY_AMT6` | 0.0965 | Weak | DROPPED_COLLINEAR_OR_REDUNDANT |
| `MEAN_UTIL_RATIO_6M` | 0.0947 | Weak | KEPT |
| `PAY_TO_BILL_RATIO_6M` | 0.0852 | Weak | DROPPED_COLLINEAR_OR_REDUNDANT |
| `PAY_AMT5` | 0.0803 | Weak | DROPPED_COLLINEAR_OR_REDUNDANT |
| `BILL_TREND_6M_RATIO` | 0.0687 | Weak | KEPT |
| `UTIL_RATIO_M1` | 0.0528 | Weak | DROPPED_COLLINEAR_OR_REDUNDANT |
| `MONTHS_INACTIVE_6M` | 0.0276 | Weak | DROPPED_COLLINEAR_OR_REDUNDANT |
| `MAX_BILL_AMT_6M` | 0.0259 | Weak | DROPPED_COLLINEAR_OR_REDUNDANT |
| `PAY_TO_LIMIT_RATIO_6M` | 0.0186 | Unpredictive | DROPPED_LOW_IV |
| `EDUCATION_CLEAN` | 0.0175 | Unpredictive | DROPPED_LOW_IV |
| `BILL_AMT1` | 0.0073 | Unpredictive | DROPPED_LOW_IV |
| `MEAN_BILL_AMT_6M` | 0.0072 | Unpredictive | DROPPED_LOW_IV |
| `AGE` | 0.0051 | Unpredictive | DROPPED_LOW_IV |
| `BILL_AMT2` | 0.0041 | Unpredictive | DROPPED_LOW_IV |
| `BILL_AMT3` | 0.0033 | Unpredictive | DROPPED_LOW_IV |
| `BILL_AMT6` | 0.0031 | Unpredictive | DROPPED_LOW_IV |
| `BILL_AMT4` | 0.0022 | Unpredictive | DROPPED_LOW_IV |
| `BILL_AMT5` | 0.0020 | Unpredictive | DROPPED_LOW_IV |

## 5. Champion logistic scorecard

VIF threshold 5.0. All retained coefficients are negative, consistent with WoE coding.

| Feature | DEV IV | VIF | Coefficient | Std. error | p-value |
| :--- | ---: | ---: | ---: | ---: | ---: |
| `PAY_0` | 0.8964 | 2.44 | -0.580556 | 0.027073 | 5.17e-102 |
| `MONTHS_DELINQUENT_6M` | 0.8955 | 3.19 | -0.338698 | 0.032715 | 4.06e-25 |
| `MONTHS_REVOLVING_6M` | 0.2738 | 1.76 | -0.330738 | 0.049517 | 2.40e-11 |
| `MEAN_PAY_AMT_6M` | 0.1816 | 1.42 | -0.621535 | 0.052903 | 7.17e-32 |
| `LIMIT_BAL` | 0.1678 | 1.59 | -0.166338 | 0.056922 | 3.48e-03 |
| `MEAN_UTIL_RATIO_6M` | 0.0947 | 1.77 | -0.609451 | 0.079328 | 1.56e-14 |
| `BILL_TREND_6M_RATIO` | 0.0687 | 1.09 | -0.393267 | 0.074557 | 1.33e-07 |

Intercept -1.249184.

**Points scaling.** Base points 600 at odds 19:1 (PD 5.00%), PDO 50, factor 72.134752, offset 387.603624. Score = offset - factor x ln(PD / (1 - PD)).

## 6. Rating scale

| Grade | Score band | DEV count | DEV default rate | DEV mean PD | Holdout count | Holdout default rate |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: |
| G1 (Prime / Strong) | `[565, +inf)` | 2,559 | 5.67% | 6.81% | 1,084 | 7.56% |
| G2 (Satisfactory) | `[535, 565)` | 5,198 | 8.98% | 9.57% | 2,267 | 9.09% |
| G3 (Moderate Risk) | `[500, 535)` | 6,105 | 15.69% | 14.22% | 2,606 | 16.50% |
| G4 (High Risk) | `[460, 500)` | 1,807 | 23.46% | 21.28% | 814 | 23.96% |
| G5 (Very High Risk) | `[400, 460)` | 2,356 | 32.00% | 36.10% | 1,011 | 31.55% |
| G6 (Substandard) | `(-inf, 400)` | 2,927 | 64.71% | 63.79% | 1,198 | 63.19% |

The original 7-grade scale merged its top two grades because they did not separate (see CL-17).

## 7. Champion versus challenger

| Model | Features | DEV AUROC | Holdout AUROC | Holdout Gini | Holdout Brier (Platt) |
| :--- | ---: | ---: | ---: | ---: | ---: |
| Champion: Monotonic WoE Logistic Scorecard (7 Features) | 7 | 0.7761 (DEV In-Sample (MLE)) | 0.7598 | 0.5197 | 0.1403 |
| Challenger-Core: Monotone LightGBM (10 Features: Champion 7 w/ PAY_0_DPD + State Flags) | 10 | 0.7861 (DEV 5-Fold CV Out-of-Fold) | 0.7706 | 0.5413 | 0.1370 |
| Challenger-Expanded: Monotone LightGBM (19 Features: Clean DPD Lags + State Flags) | 19 | 0.7867 (DEV 5-Fold CV Out-of-Fold) | 0.7730 | 0.5460 | 0.1368 |

**Decision.** Champion in production, Challenger-Expanded in shadow mode. Holdout Gini gap 2.63 points against a 5.0-point review trigger. The challenger constrains `PAY_x_DPD = max(PAY_x, 0)` with +1 and leaves the inactive, paid-duly and revolving indicators unconstrained, because raw PAY codes are not monotonic in risk below zero (see CL-02).

## 8. Downstream illustrations (Step 5)

| Component | Method | Key assumptions |
| :--- | :--- | :--- |
| Basel capital | QRRE ASRF, K = LGD x (CDR(PD) - PD), RWA = 12.5 x K x EAD | R = 0.04, PD floor 0.10%, LGD 85%, CCF 50%, grade PD = DEV rate + Jeffreys MoC |
| IFRS 9 | 3-state monthly Markov matrix (current, 30-89 DPD, 90+), stage from PAY_0, account curves scaled to the PD proxy | EIR 18% a year, 36-month lifetime, no origination-based SICR |
| Stress | LRM01 year-on-year changes (P90 mild, historical max severe) mapped to PD multipliers | Elasticity 0.5; SICR overlay rule: A Stage 1 account moves to Stage 2 under stress if it is downgraded by at least one grade AND its post-stress grade is G4, G5 or G6. |

## 9. Use and limitations

Suitable for demonstrating scorecard development and validation technique. Not suitable for any credit decision. The full list of limitations, null findings and assumptions is in [Validation_and_Challenge_Log.md](Validation_and_Challenge_Log.md).
