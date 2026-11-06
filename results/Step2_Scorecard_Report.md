# ScoreGuard-IRB — Step 2 Champion Logistic Scorecard Report

**Author:** Sohum Vivek Dhole (`github.com/sohumdhole` | MSc in Business Analytics, First Class Honours, Dublin Business School)  
**Target Role Alignment:** Quantitative Risk Analyst (Risk Analytics, 2nd Line of Defence)  
**Generated UTC:** `2026-11-06T19:10:42.756860+00:00`

---

## 1. Development & Holdout Governance Controls

- **Strict DEV-Only Fitting:** All quantile/discrete cut-points, monotonic bin merges (minimum `5%` of `DEV` per bin), Weight of Evidence ($\text{WoE}$) values, Information Value ($\text{IV}$) calculations, Variance Inflation Factor ($\text{VIF}$) checks, and Maximum Likelihood Logistic Regression parameters were fitted strictly on the **Development (`DEV`) sample (`n = 20,952`, `4,642` defaults, `22.1554%` default rate)** and applied unchanged to the **Holdout (`HOLDOUT`) sample (`n = 8,980`, `1,989` defaults, `22.1492%` default rate)**.
- **Strict Feature Exclusions:** `SEX`, `MARRIAGE` (and `SEX_LABEL`, `MARRIAGE_RAW`, `MARRIAGE_CLEAN`, `MARRIAGE_LABEL`), `account_id_hash`, and `is_exact_dup_flag` were excluded from all candidate features.
- **WoE Definition & Sign Convention:**
  $$\text{WoE}_i = \ln\left(\frac{\%\text{Good}_i}{\%\text{Bad}_i}\right), \qquad \text{IV} = \sum_{i=1}^{B} \left(\%\text{Good}_i - \%\text{Bad}_i\right) \cdot \text{WoE}_i$$
  where $\text{Good}$ = non-default (`default_flag = 0`) and $\text{Bad}$ = default (`default_flag = 1`). Under this coding, higher $\text{WoE}$ indicates lower default risk, so every well-behaved feature in a logistic regression predicting $\Pr(\text{default} = 1)$ must have a **strictly negative coefficient ($\beta_j < 0$)**.

---

## 2. Candidate Feature Information Value (IV) Table & Selection Decisions (`DEV`, `n = 20,952`)

Standard Siddiqi (2006) IV strength bands:
- `< 0.02`: Unpredictive
- `0.02 – 0.10`: Weak
- `0.10 – 0.30`: Medium
- `0.30 – 0.50`: Strong
- `>= 0.50`: Very Strong (behavioural delinquency indicators; checked for collinearity)

| Rank | Feature | Domain Group | Bins | DEV IV | IV Strength Band | Monotonic Direction | Min Bin Bad Rate | Max Bin Bad Rate | Status | Selection / Drop Rationale |
| ---: | :--- | :--- | ---: | ---: | :--- | :--- | ---: | ---: | :--- | :--- |
| 1 | `PAY_0` | Delinquency Recency | 4 | **0.8964** | Very Strong (`>= 0.50`) | Increasing Bad Rate | 12.88% | 69.97% | **KEPT** | Strongest delinquency recency indicator (Month 1 repayment status; `IV=0.8964`, `VIF=2.44`). |
| 2 | `MONTHS_DELINQUENT_6M` | Delinquency Frequency | 5 | **0.8955** | Very Strong (`>= 0.50`) | Increasing Bad Rate | 11.50% | 68.75% | **KEPT** | Captures 6-month delinquency persistence/frequency across Apr–Sep 2005 (`IV=0.8955`, `VIF=3.19`). |
| 3 | `MONTHS_60PLUS_DPD_6M` | Delinquency Severity | 5 | 0.8744 | Very Strong (`>= 0.50`) | Increasing Bad Rate | 12.52% | 70.50% | DROPPED | Sub-component of `MONTHS_DELINQUENT_6M` & `PAY_0`; causes severe multicollinearity if combined. |
| 4 | `MAX_DPD_6M` | Delinquency Severity | 3 | 0.7770 | Very Strong (`>= 0.50`) | Increasing Bad Rate | 11.50% | 46.85% | DROPPED | High VIF (`VIF=12.98 > 5.0`) when combined with `PAY_0` and `MONTHS_DELINQUENT_6M`. |
| 5 | `PAY_2` | Delinquency History | 2 | 0.5701 | Very Strong (`>= 0.50`) | Increasing Bad Rate | 16.13% | 56.64% | DROPPED | Month-2 arrears lag captured by `MONTHS_DELINQUENT_6M` + `PAY_0` (`VIF=3.45` in unpruned set). |
| 6 | `PAY_3` | Delinquency History | 3 | 0.4221 | Strong (`0.30 - 0.50`) | Increasing Bad Rate | 16.45% | 52.51% | DROPPED | Month-3 arrears lag captured by `MONTHS_DELINQUENT_6M` (insignificant `p=0.143` when added). |
| 7 | `PAY_4` | Delinquency History | 3 | 0.3729 | Strong (`0.30 - 0.50`) | Increasing Bad Rate | 16.79% | 53.92% | DROPPED | Month-4 arrears lag captured by `MONTHS_DELINQUENT_6M`. |
| 8 | `PAY_5` | Delinquency History | 3 | 0.3410 | Strong (`0.30 - 0.50`) | Increasing Bad Rate | 17.24% | 56.03% | DROPPED | Month-5 arrears lag captured by `MONTHS_DELINQUENT_6M`. |
| 9 | `PAY_6` | Delinquency History | 3 | 0.3038 | Strong (`0.30 - 0.50`) | Increasing Bad Rate | 17.97% | 53.28% | DROPPED | Month-6 arrears lag captured by `MONTHS_DELINQUENT_6M`. |
| 10 | `MONTHS_REVOLVING_6M` | Revolving Behaviour | 3 | **0.2738** | Medium (`0.10 - 0.30`) | Decreasing Bad Rate | 10.38% | 28.58% | **KEPT** | Captures chronic revolving behaviour (`PAY_x == 0` count over 6m; `IV=0.2738`, `VIF=1.76`). |
| 11 | `MEAN_PAY_AMT_6M` | Repayment Capacity | 6 | **0.1816** | Medium (`0.10 - 0.30`) | Decreasing Bad Rate | 11.23% | 31.29% | **KEPT** | Strongest repayment capacity metric over 6 months (`IV=0.1816`, `VIF=1.42`). |
| 12 | `SUM_PAY_AMT_6M` | Repayment Capacity | 6 | 0.1816 | Medium (`0.10 - 0.30`) | Decreasing Bad Rate | 11.23% | 31.29% | DROPPED | Exact `6x` linear multiple of `MEAN_PAY_AMT_6M` (identical `IV=0.1816`). |
| 13 | `LIMIT_BAL` | Credit Limit / Line | 6 | **0.1678** | Medium (`0.10 - 0.30`) | Decreasing Bad Rate | 12.76% | 32.08% | **KEPT** | Core underwriting credit line / customer tier indicator (`IV=0.1678`, `VIF=1.59`). |
| 14 | `PAY_AMT1` | Repayment Capacity | 6 | 0.1675 | Medium (`0.10 - 0.30`) | Decreasing Bad Rate | 13.29% | 35.68% | DROPPED | Single-month component of `MEAN_PAY_AMT_6M`; smoother 6m mean preferred. |
| 15 | `PAY_AMT2` | Repayment Capacity | 6 | 0.1383 | Medium (`0.10 - 0.30`) | Decreasing Bad Rate | 13.09% | 33.50% | DROPPED | Single-month component of `MEAN_PAY_AMT_6M`. |
| 16 | `PAY_AMT3` | Repayment Capacity | 6 | 0.1123 | Medium (`0.10 - 0.30`) | Decreasing Bad Rate | 14.26% | 31.87% | DROPPED | Single-month component of `MEAN_PAY_AMT_6M`. |
| 17 | `PAY_AMT4` | Repayment Capacity | 5 | 0.0985 | Weak (`0.02 - 0.10`) | Decreasing Bad Rate | 14.92% | 30.88% | DROPPED | Single-month component of `MEAN_PAY_AMT_6M`. |
| 18 | `PAY_AMT6` | Repayment Capacity | 5 | 0.0965 | Weak (`0.02 - 0.10`) | Decreasing Bad Rate | 14.03% | 29.33% | DROPPED | Single-month component of `MEAN_PAY_AMT_6M`. |
| 19 | `MEAN_UTIL_RATIO_6M` | Credit Utilisation | 4 | **0.0947** | Weak (`0.02 - 0.10`) | Increasing Bad Rate | 17.75% | 32.62% | **KEPT** | Captures 6-month credit line utilisation pressure (`IV=0.0947`, `VIF=1.77`). |
| 20 | `PAY_TO_BILL_RATIO_6M` | Repayment Burden | 4 | 0.0852 | Weak (`0.02 - 0.10`) | Decreasing Bad Rate | 16.85% | 29.81% | DROPPED | Exhibits positive coefficient (`+0.2007`, sign flip) when combined with `MEAN_PAY_AMT_6M` and `MEAN_UTIL_RATIO_6M`. |
| 21 | `PAY_AMT5` | Repayment Capacity | 6 | 0.0803 | Weak (`0.02 - 0.10`) | Decreasing Bad Rate | 14.72% | 29.30% | DROPPED | Single-month component of `MEAN_PAY_AMT_6M`. |
| 22 | `BILL_TREND_6M_RATIO` | Balance Trajectory | 4 | **0.0687** | Weak (`0.02 - 0.10`) | Decreasing Bad Rate | 18.03% | 30.21% | **KEPT** | Captures 6-month balance trajectory / draw-down velocity (`IV=0.0687`, `VIF=1.09`). |
| 23 | `UTIL_RATIO_M1` | Credit Utilisation | 4 | 0.0528 | Weak (`0.02 - 0.10`) | Increasing Bad Rate | 18.32% | 27.29% | DROPPED | High VIF (`VIF=5.13 > 5.0`) when paired with `MEAN_UTIL_RATIO_6M`. |
| 24 | `MONTHS_INACTIVE_6M` | Account Activity | 3 | 0.0276 | Weak (`0.02 - 0.10`) | Decreasing Bad Rate | 12.96% | 23.17% | DROPPED | Weak IV (`0.0276`); activity state already captured by `MONTHS_REVOLVING_6M` and `MEAN_UTIL_RATIO_6M`. |
| 25 | `MAX_BILL_AMT_6M` | Balance Magnitude | 5 | 0.0259 | Weak (`0.02 - 0.10`) | Decreasing Bad Rate | 18.67% | 26.95% | DROPPED | Weak IV (`0.0259`); unnormalised bill size less informative than `MEAN_UTIL_RATIO_6M`. |
| 26 | `PAY_TO_LIMIT_RATIO_6M` | Repayment Capacity | 3 | 0.0186 | Unpredictive (`< 0.02`) | Decreasing Bad Rate | 17.47% | 25.14% | DROPPED | Below `0.02` minimum IV threshold. |
| 27 | `EDUCATION_CLEAN` | Demographics (Permitted) | 3 | 0.0175 | Unpredictive (`< 0.02`) | Increasing Bad Rate | 19.11% | 24.20% | DROPPED | Below `0.02` minimum IV threshold. |
| 28 | `BILL_AMT1` | Balance Magnitude | 4 | 0.0073 | Unpredictive (`< 0.02`) | Decreasing Bad Rate | 19.62% | 24.16% | DROPPED | Below `0.02` minimum IV threshold. |
| 29 | `MEAN_BILL_AMT_6M` | Balance Magnitude | 4 | 0.0072 | Unpredictive (`< 0.02`) | Decreasing Bad Rate | 20.39% | 25.22% | DROPPED | Below `0.02` minimum IV threshold. |
| 30 | `AGE` | Demographics (Permitted) | 2 | 0.0051 | Unpredictive (`< 0.02`) | Increasing Bad Rate | 21.62% | 25.09% | DROPPED | Below `0.02` minimum IV threshold. |
| 31 | `BILL_AMT2` | Balance Magnitude | 3 | 0.0041 | Unpredictive (`< 0.02`) | Decreasing Bad Rate | 19.79% | 22.78% | DROPPED | Below `0.02` minimum IV threshold. |
| 32 | `BILL_AMT3` | Balance Magnitude | 3 | 0.0033 | Unpredictive (`< 0.02`) | Decreasing Bad Rate | 20.10% | 22.76% | DROPPED | Below `0.02` minimum IV threshold. |
| 33 | `BILL_AMT6` | Balance Magnitude | 3 | 0.0031 | Unpredictive (`< 0.02`) | Increasing Bad Rate | 21.19% | 23.11% | DROPPED | Below `0.02` minimum IV threshold. |
| 34 | `BILL_AMT4` | Balance Magnitude | 4 | 0.0022 | Unpredictive (`< 0.02`) | Decreasing Bad Rate | 20.42% | 22.71% | DROPPED | Below `0.02` minimum IV threshold. |
| 35 | `BILL_AMT5` | Balance Magnitude | 4 | 0.0020 | Unpredictive (`< 0.02`) | Decreasing Bad Rate | 20.76% | 23.22% | DROPPED | Below `0.02` minimum IV threshold. |

---

## 3. VIF Check & Coefficient Sign-Consistency Audit (`Stated Threshold: VIF < 5.0`)

### 3.1 Pre-Pruning Collinearity & Sign-Flip Diagnostic (16 High-IV Features on `DEV`)
When all 16 non-duplicate features with $\text{IV} \ge 0.05$ are entered simultaneously into a logistic regression on `DEV`, three features breach $\text{VIF} < 5.0$ and two exhibit **positive coefficient sign flips** ($\beta_j > 0$, contradicting WoE coding):
- `MONTHS_DELINQUENT_6M`: **$\text{VIF} = 40.91$** (exceeds `5.0`) and **$\beta = +0.6816$** (flagged sign flip due to collinearity with `MAX_DPD_6M` and `PAY_0..PAY_6`).
- `MAX_DPD_6M`: **$\text{VIF} = 12.98$** (exceeds `5.0`).
- `UTIL_RATIO_M1`: **$\text{VIF} = 5.13$** (exceeds `5.0`, collinear with `MEAN_UTIL_RATIO_6M`).
- `PAY_TO_BILL_RATIO_6M`: **$\beta = +0.3285$** (flagged sign flip due to collinearity with `MEAN_PAY_AMT_6M` and `MEAN_UTIL_RATIO_6M`).

### 3.2 Final 7-Feature Champion Logistic Regression (`DEV`, `n = 20,952`)
After pruning collinear lags and redundant components, all 7 retained domain features have **$\text{VIF} \le 3.19$** (well below the stated `5.0` threshold), are highly statistically significant ($p \le 0.0035$), and have **strictly negative coefficients ($\beta_j < 0$)** consistent with WoE coding:

- **Intercept ($\beta_0$):** `-1.249184` ($\text{SE} = 0.018502$, $z = -67.52$, $p < 10^{-300}$)

| Feature | DEV IV | DEV VIF | Threshold (`< 5.0`) | Coefficient ($\beta_j$) | Std. Error | $z$-Stat | $p$-Value | Expected Sign | Sign Consistent? |
| :--- | ---: | ---: | :---: | ---: | ---: | ---: | ---: | :---: | :---: |
| `PAY_0` | 0.8964 | **2.4413** | PASS | **-0.580556** | 0.027073 | -21.44 | $5.17 \times 10^{-102}$ | `< 0` | **YES** |
| `MONTHS_DELINQUENT_6M` | 0.8955 | **3.1940** | PASS | **-0.338698** | 0.032715 | -10.35 | $4.06 \times 10^{-25}$ | `< 0` | **YES** |
| `MONTHS_REVOLVING_6M` | 0.2738 | **1.7576** | PASS | **-0.330738** | 0.049517 | -6.68 | $2.40 \times 10^{-11}$ | `< 0` | **YES** |
| `MEAN_PAY_AMT_6M` | 0.1816 | **1.4199** | PASS | **-0.621535** | 0.052903 | -11.75 | $7.17 \times 10^{-32}$ | `< 0` | **YES** |
| `LIMIT_BAL` | 0.1678 | **1.5911** | PASS | **-0.166338** | 0.056922 | -2.92 | $3.48 \times 10^{-3}$ | `< 0` | **YES** |
| `MEAN_UTIL_RATIO_6M` | 0.0947 | **1.7677** | PASS | **-0.609451** | 0.079328 | -7.68 | $1.56 \times 10^{-14}$ | `< 0` | **YES** |
| `BILL_TREND_6M_RATIO` | 0.0687 | **1.0862** | PASS | **-0.393267** | 0.074557 | -5.27 | $1.33 \times 10^{-7}$ | `< 0` | **YES** |

---

## 4. Monotonic WoE Bin Tables & Scorecard Points (`DEV` Fitted, Applied to `HOLDOUT`)

### Stated Points Scaling Parameters
- **Base Points ($P_0$):** `600` points
- **Base Odds ($\theta_0$, Good : Bad):** `19 : 1` (corresponding to a base probability of default $\text{PD}_0 = \frac{1}{1+19} = 5.00\%$)
- **Points to Double the Odds ($\text{PDO}$):** `50` points
- **Scaling Factor ($B = \frac{\text{PDO}}{\ln 2}$):** `72.134752`
- **Scaling Offset ($A = P_0 - B \ln(\theta_0)$):** `387.603624`
- **Score Formula:** $\text{Score} = A - B \cdot \ln\left(\frac{\text{PD}}{1 - \text{PD}}\right) = \sum_{j=1}^{7} \left[\frac{A - B \beta_0}{7} - B \beta_j \text{WoE}_j\right]$
  *(where base points per feature $\frac{A - B\beta_0}{7} = 68.244741$)*

| Feature | Bin | Interval / Values | DEV Count | DEV Share | DEV Bads | DEV Bad Rate | Fitted WoE | Bin IV | Points (Exact) | Points (Int) | HOLDOUT Count | HOLDOUT Bad Rate |
| :--- | ---: | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `PAY_0` | 1 | `(-inf, -1.5]` (`-2` Inactive) | 1,926 | 9.19% | 248 | **12.88%** | `+0.655296` | 0.0324 | `95.69` | **96** | 823 | 14.09% |
| `PAY_0` | 2 | `(-1.5, 0.5]` (`-1` Duly, `0` Rev) | 14,259 | 68.06% | 1,958 | **13.73%** | `+0.581124` | 0.1932 | `92.58` | **93** | 6,121 | 14.41% |
| `PAY_0` | 3 | `(0.5, 1.5]` (`1` = 1m DPD) | 2,559 | 12.21% | 891 | **34.82%** | `-0.629597` | 0.0565 | `41.88` | **42** | 1,119 | 32.17% |
| `PAY_0` | 4 | `(1.5, inf]` (`2+` = 60+ DPD) | 2,208 | 10.54% | 1,545 | **69.97%** | `-2.102637` | 0.6144 | `-19.81` | **-20** | 917 | 68.81% |
| `MONTHS_DELINQUENT_6M` | 1 | `(-inf, 0.5]` (`0` months) | 13,917 | 66.42% | 1,601 | **11.50%** | `+0.783638` | 0.3215 | `87.39` | **87** | 5,969 | 12.25% |
| `MONTHS_DELINQUENT_6M` | 2 | `(0.5, 1.5]` (`1` month) | 3,045 | 14.53% | 909 | **29.85%** | `-0.402288` | 0.0261 | `58.42` | **58** | 1,369 | 29.95% |
| `MONTHS_DELINQUENT_6M` | 3 | `(1.5, 2.5]` (`2` months) | 1,310 | 6.25% | 508 | **38.78%** | `-0.800006` | 0.0482 | `48.70` | **49** | 586 | 38.91% |
| `MONTHS_DELINQUENT_6M` | 4 | `(2.5, 4.5]` (`3–4` months) | 1,528 | 7.29% | 832 | **54.45%** | `-1.435116` | 0.1960 | `33.18` | **33** | 574 | 52.09% |
| `MONTHS_DELINQUENT_6M` | 5 | `(4.5, inf]` (`5–6` months) | 1,152 | 5.50% | 792 | **68.75%** | `-2.045090` | 0.3038 | `18.28` | **18** | 482 | 66.60% |
| `MONTHS_REVOLVING_6M` | 1 | `(-inf, 4.5]` (`0–4` months) | 12,003 | 57.29% | 3,431 | **28.58%** | `-0.340984` | 0.0728 | `60.11` | **60** | 5,079 | 28.25% |
| `MONTHS_REVOLVING_6M` | 2 | `(4.5, 5.5]` (`5` months) | 2,111 | 10.08% | 501 | **23.73%** | `-0.089250` | 0.0008 | `66.12` | **66** | 930 | 25.59% |
| `MONTHS_REVOLVING_6M` | 3 | `(5.5, inf]` (`6` months) | 6,838 | 32.64% | 710 | **10.38%** | `+0.898726` | 0.2002 | `89.69` | **90** | 2,971 | 10.64% |
| `MEAN_PAY_AMT_6M` | 1 | `(-inf, 775.3]` | 3,493 | 16.67% | 1,093 | **31.29%** | `-0.470091` | 0.0415 | `47.17` | **47** | 1,503 | 33.33% |
| `MEAN_PAY_AMT_6M` | 2 | `(775.3, 1440.7]` | 3,491 | 16.66% | 1,015 | **29.07%** | `-0.364877` | 0.0244 | `51.89` | **52** | 1,472 | 28.26% |
| `MEAN_PAY_AMT_6M` | 3 | `(1440.7, 2382.8]` | 3,492 | 16.67% | 893 | **25.57%** | `-0.188338` | 0.0062 | `59.80` | **60** | 1,451 | 24.33% |
| `MEAN_PAY_AMT_6M` | 4 | `(2382.8, 4172.2]` | 3,493 | 16.67% | 663 | **18.98%** | `+0.194624` | 0.0060 | `76.97` | **77** | 1,530 | 19.54% |
| `MEAN_PAY_AMT_6M` | 5 | `(4172.2, 7980.9]` | 3,491 | 16.66% | 586 | **16.79%** | `+0.344236` | 0.0179 | `83.68` | **84** | 1,553 | 15.71% |
| `MEAN_PAY_AMT_6M` | 6 | `(7980.9, inf]` | 3,492 | 16.67% | 392 | **11.23%** | `+0.811262` | 0.0857 | `104.62` | **105** | 1,471 | 11.96% |
| `LIMIT_BAL` | 1 | `(-inf, 50,000]` | 5,387 | 25.71% | 1,728 | **32.08%** | `-0.506408` | 0.0749 | `62.17` | **62** | 2,275 | 31.30% |
| `LIMIT_BAL` | 2 | `(50,000, 80,000]` | 2,188 | 10.44% | 552 | **25.23%** | `-0.170172` | 0.0032 | `66.20` | **66** | 927 | 27.29% |
| `LIMIT_BAL` | 3 | `(80,000, 140,000]` | 3,166 | 15.11% | 759 | **23.97%** | `-0.102498` | 0.0016 | `67.01` | **67** | 1,314 | 23.97% |
| `LIMIT_BAL` | 4 | `(140,000, 200,000]` | 3,529 | 16.84% | 616 | **17.46%** | `+0.297059` | 0.0136 | `71.81` | **72** | 1,541 | 18.11% |
| `LIMIT_BAL` | 5 | `(200,000, 300,000]` | 3,453 | 16.48% | 575 | **16.65%** | `+0.353848` | 0.0186 | `72.49` | **72** | 1,592 | 14.89% |
| `LIMIT_BAL` | 6 | `(300,000, inf]` | 3,229 | 15.41% | 412 | **12.76%** | `+0.665771` | 0.0559 | `76.23` | **76** | 1,331 | 14.50% |
| `MEAN_UTIL_RATIO_6M` | 1 | `(-inf, 0.2887]` | 10,476 | 50.00% | 1,860 | **17.75%** | `+0.276411` | 0.0353 | `80.40` | **80** | 4,546 | 18.39% |
| `MEAN_UTIL_RATIO_6M` | 2 | `(0.2887, 0.5677]` | 3,492 | 16.67% | 756 | **21.65%** | `+0.029578` | 0.0001 | `69.55` | **70** | 1,553 | 21.57% |
| `MEAN_UTIL_RATIO_6M` | 3 | `(0.5677, 0.8056]` | 3,492 | 16.67% | 887 | **25.40%** | `-0.179290` | 0.0056 | `60.36` | **60** | 1,447 | 25.78% |
| `MEAN_UTIL_RATIO_6M` | 4 | `(0.8056, inf]` | 3,492 | 16.67% | 1,139 | **32.62%** | `-0.531093` | 0.0537 | `44.90` | **45** | 1,434 | 31.03% |
| `BILL_TREND_6M_RATIO` | 1 | `(-inf, -0.0652]` | 3,492 | 16.67% | 1,055 | **30.21%** | `-0.419406` | 0.0327 | `56.35` | **56** | 1,466 | 28.51% |
| `BILL_TREND_6M_RATIO` | 2 | `(-0.0652, -0.0056]` | 3,492 | 16.67% | 865 | **24.77%** | `-0.145765` | 0.0037 | `64.11` | **64** | 1,502 | 22.50% |
| `BILL_TREND_6M_RATIO` | 3 | `(-0.0056, 0.0068]` | 3,492 | 16.67% | 833 | **23.85%** | `-0.095961` | 0.0016 | `65.52` | **66** | 1,510 | 23.91% |
| `BILL_TREND_6M_RATIO` | 4 | `(0.0068, inf]` | 10,476 | 50.00% | 1,889 | **18.03%** | `+0.257569` | 0.0308 | `75.55` | **76** | 4,502 | 19.37% |

---

## 5. Scorecard Distribution & 7-Grade Master Rating Scale

### 5.1 Score Distribution (`DEV` vs `HOLDOUT`)

| Sample | Mean | Std | Min | P01 | P05 | P10 | P25 | Median (P50) | P75 | P90 | P95 | P99 | Max |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **DEV (`n = 20,952`)** | 495.81 | 74.95 | 269.16 | 296.54 | 331.42 | 367.98 | 459.27 | **519.63** | 549.23 | 567.37 | 576.88 | 592.68 | 606.46 |
| **HOLDOUT (`n = 8,980`)** | 496.95 | 74.03 | 269.16 | 295.97 | 333.95 | 371.53 | 461.85 | **519.63** | 549.56 | 567.37 | 576.88 | 595.01 | 606.46 |

### 5.2 Rating Grade Master Scale (`G1 Prime` to `G7 Substandard`)

| Grade | Score Band | Implied PD Band | DEV Count | DEV Share | DEV Defaults | DEV Obs. Bad Rate | DEV Mean Pred. PD | HOLDOUT Count | HOLDOUT Share | HOLDOUT Defaults | HOLDOUT Obs. Bad Rate | HOLDOUT Mean Pred. PD |
| :--- | :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **G1 (Prime)** | `[585, +inf)` | `0.00% – 6.09%` | 518 | 2.47% | 28 | **5.41%** | 5.56% | 220 | 2.45% | 16 | **7.27%** | 5.50% |
| **G2 (Strong)** | `[565, 585)` | `6.09% – 7.88%` | 2,041 | 9.74% | 117 | **5.73%** | 7.12% | 864 | 9.62% | 66 | **7.64%** | 7.11% |
| **G3 (Satisfactory)** | `[535, 565)` | `7.88% – 11.47%` | 5,198 | 24.81% | 467 | **8.98%** | 9.57% | 2,267 | 25.24% | 206 | **9.09%** | 9.53% |
| **G4 (Moderate Risk)** | `[500, 535)` | `11.47% – 17.39%` | 6,105 | 29.14% | 958 | **15.69%** | 14.22% | 2,606 | 29.02% | 430 | **16.50%** | 14.22% |
| **G5 (High Risk)** | `[460, 500)` | `17.39% – 26.82%` | 1,807 | 8.62% | 424 | **23.46%** | 21.28% | 814 | 9.06% | 195 | **23.96%** | 21.41% |
| **G6 (Very High Risk)** | `[400, 460)` | `26.82% – 45.71%` | 2,356 | 11.24% | 754 | **32.00%** | 36.10% | 1,011 | 11.26% | 319 | **31.55%** | 36.01% |
| **G7 (Substandard)** | `(-inf, 400)` | `45.71% – 100.0%` | 2,927 | 13.97% | 1,894 | **64.71%** | 63.79% | 1,198 | 13.34% | 757 | **63.19%** | 63.67% |
| **Total / Portfolio** | **All Scores** | **0.00% – 100.0%** | **20,952** | **100.00%** | **4,642** | **22.16%** | **22.16%** | **8,980** | **100.00%** | **1,989** | **22.15%** | **21.95%** |

*(Observed default rates increase strictly monotonically from `G1` through `G7` on both `DEV` (`5.41% -> 64.71%`) and `HOLDOUT` (`7.27% -> 63.19%`).)*

---

## 6. Champion Discrimination Performance (`DEV` and `HOLDOUT`)

| Sample | Accounts ($N$) | Defaults | Observed Default Rate | **AUROC** | **Gini ($2 \cdot \text{AUROC} - 1$)** |
| :--- | ---: | ---: | ---: | ---: | ---: |
| **Development (`DEV`)** | 20,952 | 4,642 | 22.1554% | **0.776123** | **0.552245** |
| **Holdout (`HOLDOUT`)** | 8,980 | 1,989 | 22.1492% | **0.759844** | **0.519688** |

