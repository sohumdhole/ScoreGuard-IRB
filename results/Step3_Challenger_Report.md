# ScoreGuard-IRB — Step 2 Fixes & Step 3 Challenger Report (Revised)

**Author:** Sohum Vivek Dhole (`github.com/sohumdhole` | MSc in Business Analytics, First Class Honours, Dublin Business School)  
**Target Role Alignment:** Quantitative Risk Analyst (Risk Analytics, 2nd Line of Defence)  
**Generated UTC:** `2026-11-06T19:35:33.987023+00:00`

---

## Part I — Step 2 Governance Fixes (Verified Before Step 3)

### 1. Re-Cut / Merged 6-Grade Master Rating Scale (`G1 Prime/Strong` to `G6 Substandard`)
In the initial 7-grade scale, splitting `[565, 585)` and `[585, +inf)` produced insufficient separation between `G1` and `G2` (`DEV` observed default rate `5.41%` vs `5.73%`; `HOLDOUT` `7.27%` vs `7.64%`). Merging `[565, +inf)` into **`G1 (Prime / Strong)`** produces a **6-grade master scale** where every adjacent grade has a clearly separated, strictly increasing observed default rate on both `DEV` and `HOLDOUT` ([step2_rating_grades.csv](step2_rating_grades.csv)):

| Grade | Score Band | Implied PD Band | DEV Count | DEV Share | DEV Defaults | DEV Obs. Bad Rate | DEV Mean Pred. PD | HOLDOUT Count | HOLDOUT Share | HOLDOUT Defaults | HOLDOUT Obs. Bad Rate | HOLDOUT Mean Pred. PD |
| :--- | :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **G1 (Prime / Strong)** | `[565, +inf)` | `0.00% – 7.88%` | 2,559 | 12.21% | 145 | **5.67%** | 6.81% | 1,084 | 12.07% | 82 | **7.56%** | 6.78% |
| **G2 (Satisfactory)** | `[535, 565)` | `7.88% – 11.47%` | 5,198 | 24.81% | 467 | **8.98%** | 9.57% | 2,267 | 25.24% | 206 | **9.09%** | 9.53% |
| **G3 (Moderate Risk)** | `[500, 535)` | `11.47% – 17.39%` | 6,105 | 29.14% | 958 | **15.69%** | 14.22% | 2,606 | 29.02% | 430 | **16.50%** | 14.22% |
| **G4 (High Risk)** | `[460, 500)` | `17.39% – 26.82%` | 1,807 | 8.62% | 424 | **23.46%** | 21.28% | 814 | 9.06% | 195 | **23.96%** | 21.41% |
| **G5 (Very High Risk)** | `[400, 460)` | `26.82% – 45.71%` | 2,356 | 11.24% | 754 | **32.00%** | 36.10% | 1,011 | 11.26% | 319 | **31.55%** | 36.01% |
| **G6 (Substandard)** | `(-inf, 400)` | `45.71% – 100.0%` | 2,927 | 13.97% | 1,894 | **64.71%** | 63.79% | 1,198 | 13.34% | 757 | **63.19%** | 63.67% |
| **Total / Portfolio** | **All Scores** | **0.00% – 100.0%** | **20,952** | **100.00%** | **4,642** | **22.16%** | **22.16%** | **8,980** | **100.00%** | **1,989** | **22.15%** | **21.95%** |

---

### 2. Engineered Feature Dictionary & Divide-by-Zero Governance

All 11 engineered features in `gold.credit_card_modelling` ([src/data_governance.py](../src/data_governance.py) and [src/scorecard_champion.py](../src/scorecard_champion.py)):

| Engineered Feature | Exact Mathematical Formula ($t \in \{0,2,3,4,5,6\}$, $k \in \{1..6\}$) | Divide-by-Zero & Boundary Handling | Domain Meaning |
| :--- | :--- | :--- | :--- |
| `MONTHS_DELINQUENT_6M` | $\sum_{t} \mathbb{I}(\text{PAY\_}t\text{\_DPD} \ge 1)$, where $\text{PAY\_}t\text{\_DPD} = \max(\text{PAY\_}t, 0)$ | N/A (integer count in $\{0..6\}$; no division). | Count of months in Apr–Sep 2005 with $1+$ months past due ($30+$ DPD). |
| `MONTHS_60PLUS_DPD_6M` | $\sum_{t} \mathbb{I}(\text{PAY\_}t\text{\_DPD} \ge 2)$ | N/A (integer count in $\{0..6\}$; no division). | Count of months in Apr–Sep 2005 with $2+$ months past due ($60+$ DPD). |
| `MAX_DPD_6M` | $\max_{t} (\text{PAY\_}t\text{\_DPD})$ | N/A (integer maximum in $\{0..8\}$; no division). | Worst delinquency severity (in months past due) over the 6-month window. |
| `MONTHS_REVOLVING_6M` | $\sum_{t} \mathbb{I}(\text{PAY\_}t = 0)$ | N/A (integer count in $\{0..6\}$; no division). | Count of months where the cardholder used revolving credit ($\text{PAY\_}t = 0$). |
| `MONTHS_INACTIVE_6M` | $\sum_{t} \mathbb{I}(\text{PAY\_}t = -2)$ | N/A (integer count in $\{0..6\}$; no division). | Count of months with no consumption / inactive status ($\text{PAY\_}t = -2$). |
| `MEAN_PAY_AMT_6M` | $\frac{1}{6}\sum_{k=1}^{6} \text{PAY\_AMT}k$ | Fixed constant denominator $= 6$. | Average monthly repayment amount (NT\$) over the 6-month window. |
| `MEAN_UTIL_RATIO_6M` | $\text{clip}\!\left(\frac{\frac{1}{6}\sum_{k=1}^{6} \text{BILL\_AMT}k}{\max(\text{LIMIT\_BAL}, 1.0)}, -1.0, 5.0\right)$ | Denominator guarded via `limit_safe = clip(LIMIT_BAL, lower=1.0)` plus `DQ03` hard quarantine (`LIMIT_BAL <= 0`); clipped to $[-1.0, 5.0]$. | 6-month mean statement balance as a fraction of credit limit. |
| `UTIL_RATIO_M1` | $\text{clip}\!\left(\frac{\text{BILL\_AMT1}}{\max(\text{LIMIT\_BAL}, 1.0)}, -1.0, 5.0\right)$ | Denominator guarded via `limit_safe = clip(LIMIT_BAL, lower=1.0)` plus `DQ03`; clipped to $[-1.0, 5.0]$. | Month-1 (September 2005) statement balance as a fraction of credit limit. |
| `BILL_TREND_6M_RATIO` | $\text{clip}\!\left(\frac{\text{BILL\_AMT1} - \text{BILL\_AMT6}}{\max(\text{LIMIT\_BAL}, 1.0)}, -2.0, 2.0\right)$ | Denominator guarded via `limit_safe = clip(LIMIT_BAL, lower=1.0)` plus `DQ03`; clipped to $[-2.0, 2.0]$. | 6-month statement balance change (Sep minus Apr 2005) scaled by credit limit. |
| `PAY_TO_BILL_RATIO_6M` | $\begin{cases} \text{clip}\!\left(\frac{\sum_{k=1}^6 \text{PAY\_AMT}k}{\sum_{k=1}^6 \max(\text{BILL\_AMT}k, 0)}, 0.0, 2.0\right) & \text{if } \sum_{k=1}^6 \max(\text{BILL\_AMT}k, 0) > 0 \\ 1.0 & \text{if } \sum_{k=1}^6 \max(\text{BILL\_AMT}k, 0) = 0 \end{cases}$ | Explicitly guarded via `np.where(SUM_BILL_POS_6M > 0, ..., 1.0)`: when 6-month positive billed balances sum to zero, the ratio defaults to `1.0` (full coverage / zero unpaid positive bill); otherwise clipped to $[0.0, 2.0]$. | Total 6-month repayments divided by total 6-month positive billed balances. |
| `PAY_TO_LIMIT_RATIO_6M` | $\text{clip}\!\left(\frac{\sum_{k=1}^{6} \text{PAY\_AMT}k}{6.0 \times \max(\text{LIMIT\_BAL}, 1.0)}, 0.0, 2.0\right)$ | Denominator guarded via `6.0 * clip(LIMIT_BAL, lower=1.0)` plus `DQ03`; clipped to $[0.0, 2.0]$. | Total 6-month repayments divided by 6 months of credit limit exposure. |

---

### 3. Why `PAY_0` Bin 1 (`-2` Inactive) and Bin 2 (`-1` Paid Duly / `0` Revolving) Stay Separate
In `PAY_0`, Bin 1 (`(-inf, -1.5]`, i.e., `PAY_0 == -2` "no consumption / inactive", `DEV n = 1,926`, `9.19%` share) has an observed default rate of **12.88%** ($\text{WoE} = +0.6553$, **96 points**), while Bin 2 (`(-1.5, 0.5]`, i.e., `PAY_0 == -1` "pay duly" and `PAY_0 == 0` "revolving minimum paid", `DEV n = 14,259`, `68.06%` share) has an observed default rate of **13.73%** ($\text{WoE} = +0.5811$, **93 points**). They stay separate for three governance reasons:
1. **Population Credibility:** Bin 1 contains `1,926` accounts (`9.19%` of `DEV`, `248` defaults), nearly double the `5.0%` (`1,048` rows) minimum bin-size rule.
2. **Out-of-Sample Monotonic Persistence:** The risk ordering (`Bin 1 < Bin 2 < Bin 3 < Bin 4`) holds strictly on both `DEV` (`12.88% < 13.73% < 34.82% < 69.97%`) and `HOLDOUT` (`14.09% < 14.41% < 32.17% < 68.81%`), awarding a `+3` point premium (`96` vs `93` points) to dormant/zero-consumption accounts over active transacting/revolving accounts.
3. **Behavioural Distinctness:** `PAY_0 == -2` represents inactive accounts with no credit utilisation, whereas `PAY_0 in {-1, 0}` represents active accounts carrying or settling statement balances. Retaining separate bins preserves behavioural interpretability in scorecard monitoring.

---

## Part II — Step 3 (Revised): Monotone-Constrained LightGBM Challenger, SHAP, Calibration & Production Defence

### 1. Monotone Constraint Governance & Decomposition of Non-Delinquent `PAY_x` Codes
- **Why `+1` Monotone Constraints Are NOT Applied to Raw `PAY_0, PAY_2..PAY_6`:**
  Raw `PAY_x` codes mix non-delinquent behavioural states (`-2` inactive, `-1` paid duly, `0` revolving) with delinquent months-past-due (`1..8`), and **the non-delinquent codes are not monotonic in default rate**:
  - In Bronze (`N = 30,000`), `PAY_0` default rates are **13.23%** at `-2`, **16.78%** at `-1`, and **12.81%** at `0` (on `DEV`: `12.88%` at `-2`, `16.49%` at `-1`, and `12.67%` at `0`).
  - In Bronze, `PAY_2` default rates are **18.27%** at `-2`, **15.97%** at `-1`, and **15.91%** at `0` (on `DEV`: `17.95%` at `-2`, `15.88%` at `-1`, and `15.79%` at `0`).
- **Corrected Feature & Constraint Architecture ([src/challenger_lgbm.py](../src/challenger_lgbm.py)):**
  - Clean delinquency counters **`PAY_0_DPD, PAY_2_DPD..PAY_6_DPD = max(PAY_x, 0)`** receive strict **`+1`** monotone constraints.
  - Non-delinquent behavioural state indicators (**`PAY_0_IS_INACTIVE`**, **`PAY_0_IS_PAID_DULY`**, **`PAY_0_IS_REVOLVING`**, and **`MONTHS_INACTIVE_6M`**) are entered as separate **unconstrained (`0`)** features so LightGBM can model their true non-monotonic risk differences without artificial distortion.
  - **`MONTHS_REVOLVING_6M`** receives **`-1`**, matching its strictly decreasing bad-rate / increasing WoE relationship in Step 2 (`0–4m: 28.58%`, `5m: 23.73%`, `6m: 10.38%`).
- **Two Challenger Specifications Evaluated (Tuned via 5-Fold Stratified CV on `DEV` Only):**
  1. **`Challenger-Core (10 Features)`:** The Champion's 7 risk dimensions with `PAY_0` decomposed into `PAY_0_DPD` (`+1`) + `PAY_0_IS_INACTIVE` (`0`) + `PAY_0_IS_PAID_DULY` (`0`) + `PAY_0_IS_REVOLVING` (`0`) + the other 6 Champion features (`MONTHS_DELINQUENT_6M` `[+1]`, `MONTHS_REVOLVING_6M` `[-1]`, `MEAN_PAY_AMT_6M` `[-1]`, `LIMIT_BAL` `[-1]`, `MEAN_UTIL_RATIO_6M` `[+1]`, `BILL_TREND_6M_RATIO` `[-1]`).
  2. **`Challenger-Expanded (19 Features)`:** Adds `MAX_DPD_6M` (`+1`), clean lagged arrears counters `PAY_2_DPD..PAY_6_DPD` (`+1`), `PAY_AMT1..2` (`-1`), and `MONTHS_INACTIVE_6M` (`0`).

#### 5-Fold Stratified CV Tuning Grid on `DEV` (`Challenger-Expanded`, [step3_cv_tuning_grid.csv](step3_cv_tuning_grid.csv))

| Grid Configuration | `n_estimators` | `learning_rate` | `num_leaves` | `max_depth` | `min_child_samples` | `reg_alpha` / `reg_lambda` | DEV 5-Fold OOF AUROC | DEV 5-Fold OOF Gini | Fold AUROC SD | DEV OOF Brier |
| :--- | ---: | ---: | ---: | ---: | ---: | :---: | ---: | ---: | ---: | ---: |
| `Grid_1_Shallow_Conservative` | 150 | 0.03 | 7 | 3 | 150 | `2.0 / 10.0` | 0.784573 | 0.569146 | 0.004632 | 0.133346 |
| `Grid_2_Balanced_Monotone` | 250 | 0.03 | 15 | 4 | 100 | `1.0 / 5.0` | 0.786446 | 0.572892 | 0.004925 | 0.133033 |
| **`Grid_3_Moderate_Depth` (Selected)** | **300** | **0.03** | **31** | **5** | **80** | **`1.0 / 5.0`** | **0.786740** | **0.573481** | **0.005070** | **0.133033** |
| `Grid_4_High_Shrinkage` | 400 | 0.02 | 15 | 4 | 120 | `1.5 / 8.0` | 0.786691 | 0.573381 | 0.004716 | 0.133004 |

---

### 2. Stated Monotone Constraints & TreeSHAP Global Importance / Dependence

#### Table 2A — `Challenger-Core (10 Features)` Monotone Constraints & Global TreeSHAP Importance ([step3_shap_importance_core.csv](step3_shap_importance_core.csv))

| Rank | Feature | Monotone Constraint | DEV Mean $|\text{SHAP}|$ | **Share of Total $|\text{SHAP}|$** | HOLDOUT Mean $|\text{SHAP}|$ | Spearman $\rho(X, \text{SHAP})$ | Low Group Mean SHAP | High Group Mean SHAP | Matches Constraint / Champion? |
| ---: | :--- | :---: | ---: | ---: | ---: | ---: | ---: | ---: | :---: |
| 1 | `MONTHS_DELINQUENT_6M` | **`+1`** | **0.469639** | **26.64%** | 0.464238 | `+0.8379` | `-0.3373` | `+0.9827` | **YES** |
| 2 | `MEAN_PAY_AMT_6M` | **`-1`** | **0.282125** | **16.00%** | 0.285212 | `-0.9764` | `+0.4140` | `-0.3359` | **YES** |
| 3 | `PAY_0_DPD` | **`+1`** | **0.279137** | **15.83%** | 0.276346 | `+0.7315` | `-0.1963` | `+1.1498` | **YES** |
| 4 | `MEAN_UTIL_RATIO_6M` | **`+1`** | **0.250959** | **14.23%** | 0.250476 | `+0.9120` | `-0.2633` | `+0.3480` | **YES** |
| 5 | `BILL_TREND_6M_RATIO` | **`-1`** | **0.140108** | **7.95%** | 0.139429 | `-0.8495` | `+0.1383` | `-0.1616` | **YES** |
| 6 | `MONTHS_REVOLVING_6M` | **`-1`** | **0.123764** | **7.02%** | 0.125299 | `-0.9384` | `+0.1536` | `-0.1622` | **YES** |
| 7 | `LIMIT_BAL` | **`-1`** | **0.117232** | **6.65%** | 0.116850 | `-0.9565` | `+0.1660` | `-0.1460` | **YES** |
| 8 | `PAY_0_IS_PAID_DULY` | **`0` (Unconstrained)** | **0.065619** | **3.72%** | 0.066186 | `+0.6702` | `-0.0375` | `+0.1823` | **YES (Unconstrained)** |
| 9 | `PAY_0_IS_REVOLVING` | **`0` (Unconstrained)** | **0.029801** | **1.69%** | 0.029835 | `-0.8471` | `+0.0146` | `-0.0411` | **YES (Unconstrained)** |
| 10 | `PAY_0_IS_INACTIVE` | **`0` (Unconstrained)** | **0.004691** | **0.27%** | 0.004681 | `+0.1299` | `-0.0010` | `+0.0005` | **YES (Unconstrained)** |

*(Exact SHAP shares in `Challenger-Core`: `MONTHS_DELINQUENT_6M` (`26.64%`) and `PAY_0_DPD` (`15.83%`) together account for **42.47%** of total $|\text{SHAP}|$ — or **48.15%** when adding the three Month-1 non-delinquent state indicators (`5.68%`) — while repayment capacity (`MEAN_PAY_AMT_6M`, `16.00%`), utilisation (`MEAN_UTIL_RATIO_6M`, `14.23%`), balance trend (`7.95%`), revolving frequency (`7.02%`), and credit limit (`6.65%`) account for the remaining **51.85%**.)*

#### Table 2B — `Challenger-Expanded (19 Features)` Monotone Constraints & Global TreeSHAP Importance ([step3_shap_importance_expanded.csv](step3_shap_importance_expanded.csv))

| Rank | Feature | Monotone Constraint | DEV Mean $|\text{SHAP}|$ | **Share of Total $|\text{SHAP}|$** | HOLDOUT Mean $|\text{SHAP}|$ | Spearman $\rho(X, \text{SHAP})$ | Low Group Mean SHAP | High Group Mean SHAP | Matches Constraint / Champion? |
| ---: | :--- | :---: | ---: | ---: | ---: | ---: | ---: | ---: | :---: |
| 1 | `PAY_0_DPD` | **`+1`** | **0.316667** | **17.43%** | 0.313291 | `+0.7315` | `-0.2284` | `+1.2629` | **YES** |
| 2 | `MAX_DPD_6M` | **`+1`** | **0.278664** | **15.34%** | 0.276419 | `+0.8255` | `-0.1821` | `+0.5410` | **YES** |
| 3 | `MEAN_UTIL_RATIO_6M` | **`+1`** | **0.247968** | **13.65%** | 0.247796 | `+0.8822` | `-0.2506` | `+0.3374` | **YES** |
| 4 | `MEAN_PAY_AMT_6M` | **`-1`** | **0.205703** | **11.32%** | 0.208285 | `-0.9660` | `+0.3089` | `-0.2325` | **YES** |
| 5 | `MONTHS_DELINQUENT_6M` | **`+1`** | **0.131919** | **7.26%** | 0.128736 | `+0.8360` | `-0.0969` | `+0.3179` | **YES** |
| 6 | `BILL_TREND_6M_RATIO` | **`-1`** | **0.131355** | **7.23%** | 0.130925 | `-0.8324` | `+0.1311` | `-0.1456` | **YES** |
| 7 | `PAY_AMT1` | **`-1`** | **0.104974** | **5.78%** | 0.106595 | `-0.9490` | `+0.1422` | `-0.1347` | **YES** |
| 8 | `LIMIT_BAL` | **`-1`** | **0.101635** | **5.60%** | 0.101292 | `-0.9538` | `+0.1452` | `-0.1287` | **YES** |
| 9 | `MONTHS_REVOLVING_6M` | **`-1`** | **0.095028** | **5.23%** | 0.096049 | `-0.8780` | `+0.0966` | `-0.1307` | **YES** |
| 10 | `PAY_AMT2` | **`-1`** | **0.057395** | **3.16%** | 0.057664 | `-0.9336` | `+0.0703` | `-0.0831` | **YES** |
| 11 | `PAY_0_IS_PAID_DULY` | **`0` (Unconstrained)** | **0.051332** | **2.83%** | 0.051785 | `+0.6757` | `-0.0299` | `+0.1416` | **YES (Unconstrained)** |
| 12 | `PAY_0_IS_REVOLVING` | **`0` (Unconstrained)** | **0.037702** | **2.08%** | 0.037692 | `-0.8659` | `+0.0330` | `-0.0425` | **YES (Unconstrained)** |
| 13 | `MONTHS_INACTIVE_6M` | **`0` (Unconstrained)** | **0.022282** | **1.23%** | 0.022533 | `+0.2069` | `-0.0067` | `-0.0056` | **YES (Unconstrained)** |
| 14 | `PAY_2_DPD` | **`+1`** | **0.015602** | **0.86%** | 0.015502 | `+0.6112` | `-0.0094` | `+0.0509` | **YES** |
| 15 | `PAY_5_DPD` | **`+1`** | **0.005669** | **0.31%** | 0.005586 | `+0.5183` | `-0.0042` | `+0.0187` | **YES** |
| 16 | `PAY_0_IS_INACTIVE` | **`0` (Unconstrained)** | **0.005643** | **0.31%** | 0.005579 | `-0.5004` | `+0.0038` | `-0.0242` | **YES (Unconstrained)** |
| 17 | `PAY_6_DPD` | **`+1`** | **0.004581** | **0.25%** | 0.004507 | `+0.5290` | `-0.0031` | `+0.0171` | **YES** |
| 18 | `PAY_4_DPD` | **`+1`** | **0.002073** | **0.11%** | 0.002013 | `+0.5613` | `-0.0011` | `+0.0091` | **YES** |
| 19 | `PAY_3_DPD` | **`+1`** | **0.000295** | **0.02%** | 0.000283 | `+0.6049` | `-0.0001` | `+0.0013` | **YES** |

*(Exact SHAP shares in `Challenger-Expanded`: all 8 DPD arrears variables combined (`PAY_0_DPD`, `MAX_DPD_6M`, `MONTHS_DELINQUENT_6M`, `PAY_2_DPD..PAY_6_DPD`) account for **41.59%** of total $|\text{SHAP}|$, while individual lagged arrears `PAY_2_DPD..PAY_6_DPD` contribute just **1.55%** combined. Note also how unconstrained `PAY_0_IS_PAID_DULY` (`PAY_0 == -1`) has a positive SHAP contribution (`+0.1416` when active vs `-0.0299` when `0`), capturing the higher empirical default rate of `-1` (`16.49%` on `DEV`) relative to revolving `0` (`12.67%`) once low utilisation is controlled for.)*

---

### 3. Discrimination & Probability Calibration Comparison (`DEV` and `HOLDOUT`)

#### Table 3A — Discrimination (`AUROC` and `Gini`) & Gap vs. Champion ([step3_champion_vs_challenger.csv](step3_champion_vs_challenger.csv))

| Model | Features | DEV Evaluation Type | DEV AUROC | DEV Gini | HOLDOUT AUROC | HOLDOUT Gini | HOLDOUT AUROC Gap vs. Champion | HOLDOUT Gini Gap vs. Champion |
| :--- | ---: | :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| **Champion: Monotonic WoE Logistic Scorecard** | **7** | **DEV In-Sample (MLE)** | **0.776123** | **0.552245** | **0.759844** | **0.519688** | `0.000000` (Baseline) | `0.000000` (Baseline) |
| **Challenger-Core: Monotone LightGBM (Champion 7 w/ `PAY_0_DPD` + State Flags)** | **10** | **DEV 5-Fold CV (OOF)** | **0.786086** | **0.572171** | **0.770632** | **0.541264** | **`+0.010788` (`+1.08%p`)** | **`+0.021576` (`+2.16%p`)** |
| **Challenger-Expanded: Monotone LightGBM (Clean DPD Lags + State Flags)** | **19** | **DEV 5-Fold CV (OOF)** | **0.786740** | **0.573481** | **0.772998** | **0.545996** | **`+0.013154` (`+1.32%p`)** | **`+0.026308` (`+2.63%p`)** |

#### Table 3B — Calibration Comparison: Raw vs. Platt Scaling vs. Isotonic Regression (`HOLDOUT`, `n = 8,980`)

| Model & Calibration Method | DEV Evaluation Basis | DEV Brier Score | HOLDOUT Brier Score | HOLDOUT Log-Loss | HOLDOUT AUROC | HOLDOUT Mean Pred. PD | HOLDOUT Obs. Default Rate |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| **Champion Logistic Scorecard (MLE)** | **DEV In-Sample (MLE)** | `0.135787` | **`0.140250`** | `0.445512` | `0.759844` | `21.9535%` | `22.1492%` |
| **Challenger-Core (10F) — Raw Probabilities** | DEV 5-Fold CV (OOF) | `0.133000` | `0.137076` | `0.436442` | `0.770632` | `21.8432%` | `22.1492%` |
| **Challenger-Core (10F) — Platt Scaling (`slope=0.9763, int=-0.0240`)** | DEV 5-Fold CV (OOF) | `0.133018` | **`0.137015`** | `0.436173` | `0.770632` | `21.8673%` | `22.1492%` |
| **Challenger-Core (10F) — Isotonic Regression** | DEV 5-Fold CV (OOF) | `0.133299` | `0.137282` | `0.437934` | `0.770183` | `21.8660%` | `22.1492%` |
| **Challenger-Expanded (19F) — Raw Probabilities** | DEV 5-Fold CV (OOF) | `0.133033` | `0.136865` | `0.435388` | `0.772998` | `21.8854%` | `22.1492%` |
| **Challenger-Expanded (19F) — Platt Scaling (`slope=0.9727, int=-0.0317`)** | DEV 5-Fold CV (OOF) | `0.133041` | **`0.136793`** | `0.435092` | `0.772998` | `21.8546%` | `22.1492%` |
| **Challenger-Expanded (19F) — Isotonic Regression** | DEV 5-Fold CV (OOF) | `0.133268` | `0.136964` | `0.435562` | `0.772651` | `21.8630%` | `22.1492%` |

---

### 4. Second-Line (2LoD) Champion vs. Challenger Defence: Which Model Goes to Production and Why

> **Production Decision:** **Deploy the Champion (7-Feature Monotonic WoE Logistic Scorecard) to Production.** Retain `Challenger-Expanded` (Platt-calibrated Monotone LightGBM) as the **Second-Line Benchmark / Shadow Challenger**.

1. **Plain Assessment of the Performance Gap (Marginal Uplift):**
   - **The Challenger is only marginally better than the Champion.**
   - On `HOLDOUT` (`n = 8,980`), `Challenger-Core` improves AUROC by **`+0.0108`** (`0.7598` $\rightarrow$ `0.7706`, Gini `+0.0216`), and `Challenger-Expanded` (19 features) improves AUROC by **`+0.0132`** (`0.7598` $\rightarrow$ `0.7730`, Gini `+0.0263`).
   - On `HOLDOUT`, Brier score improves by **`0.0035`** (`0.14025` on the Champion vs `0.13679` Platt-calibrated on `Challenger-Expanded`).
   - In `Challenger-Core`, `MONTHS_DELINQUENT_6M` (`26.64%`) and `PAY_0_DPD` (`15.83%`) account for **42.47%** of total $|\text{SHAP}|$ (or **48.15%** including the Month-1 state flags), while repayment capacity (`16.00%`), utilisation (`14.23%`), balance trend (`7.95%`), revolving frequency (`7.02%`), and credit limit (`6.65%`) account for the remaining **51.85%**. In `Challenger-Expanded`, all 8 DPD arrears variables combined account for **41.59%** of total $|\text{SHAP}|$, with individual lagged DPD variables (`PAY_2_DPD..PAY_6_DPD`) contributing just **1.55%** combined.
2. **Explainability & GDPR Article 22:**
   - Under EU GDPR Article 22 (automated individual decision-making, including profiling), data subjects have a right to obtain **meaningful information about the logic involved** in automated credit decisions.
   - The Champion Scorecard expresses its decision logic as an exact, deterministic additive points schedule (32 bins across 7 features), so the exact points awarded or deducted for every input variable can be stated directly to a customer, credit officer, or model risk committee.
   - By contrast, LightGBM requires post-hoc TreeSHAP decompositions across 300 depth-5 trees (`num_leaves=31`), where a feature's marginal log-odds contribution varies across borrowers depending on higher-order interaction paths.
3. **Monotonicity, Non-Delinquent Code Handling & Known Champion Limitation for the Challenge Log:**
   - Raw `PAY_x` codes (`-2`, `-1`, `0`) are non-monotonic in empirical default rates. While the revised Challenger handles this properly by constraining `PAY_x_DPD = max(PAY_x, 0)` (`+1`) and leaving the binary state indicators unconstrained (`0`), the Champion Scorecard achieves global univariate and multivariate monotonicity directly within a 1-page coarse-binned WoE lookup table.
   - **Known Champion Limitation (Recorded for the Challenge Log):** In the Champion Scorecard, **`PAY_0` Bin 2 merges `-1` (`16.49%` `DEV` default rate, `n = 3,965`) with `0` (`12.67%` `DEV` default rate, `n = 10,294`)**, which are materially different risks (`+3.82` percentage points higher default rate on `-1` than `0`, blended to `13.73%`). Because numeric `PAY_0` orders `-2`, `-1`, `0` sequentially, monotonic interval binning on raw `PAY_0` cannot separate `-1` from `0` without violating monotonicity against Bin 1 (`-2`, `12.88%` `DEV` default rate). By contrast, the Challenger decomposes `PAY_0` into `PAY_0_DPD = max(PAY_0, 0)` plus separate unconstrained state indicators (`PAY_0_IS_PAID_DULY`, `PAY_0_IS_REVOLVING`, `PAY_0_IS_INACTIVE`).
4. **Stability & Generalisation:**
   - The Champion estimates only **8 parameters** (`intercept + 7 slopes`) over 32 coarse bins, exhibiting a modest `DEV`-to-`HOLDOUT` AUROC drop of **`0.0163`** (`0.7761` $\rightarrow$ `0.7598`).
   - `Challenger-Expanded` fits 300 trees (`num_leaves=31`) on continuous inputs and shows a larger full-`DEV`-to-`HOLDOUT` AUROC drop of **`0.0259`** (`0.7989` $\rightarrow$ `0.7730`).
5. **Regulatory Auditability (CRR Art. 174–180 & EBA/GL/2017/16) & Quantitative Governance Trigger:**
   - Under CRR Articles 174–180 and EBA Guidelines on PD/LGD estimation (EBA/GL/2017/16), retail IRB rating systems require transparent parameterisation, clear risk differentiation across grades (`G1`–`G6`), and straightforward Second-Line (2LoD) replicability and challenge.
   - **Quantitative Governance Trigger:** **A `HOLDOUT` Gini gap of 5 points (`0.0500`) or more (`Challenger Gini - Champion Gini >= 0.0500`) triggers a formal Second-Line Champion review and scorecard redevelopment.** On `HOLDOUT`, the observed Gini gap is **`+2.16` points (`+0.021576`)** against `Challenger-Core (10F)` and **`+2.63` points (`+0.026308`)** against `Challenger-Expanded (19F)` — well below the `5.0`-point governance trigger.
