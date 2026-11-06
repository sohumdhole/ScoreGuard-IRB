# ScoreGuard-IRB — Step 5: Segmentation, Illustrative Basel Capital, IFRS 9 Staging & CSO LRM01 Stress Sensitivity

**Author:** Sohum Vivek Dhole (`github.com/sohumdhole` | MSc in Business Analytics, First Class Honours, Dublin Business School)  
**Target Role:** AIB Quantitative Risk Analyst (Risk Analytics, 2nd Line of Defence)  
**Primary Portfolio Data:** UCI Default of Credit Card Clients (ID 350, CC BY 4.0) — $N = 29,932$ Gold accounts (`DEV` $n = 20,952$; `HOLDOUT` $n = 8,980$)  
**Macro Stress Data:** CSO Ireland `LRM01` (Persons on Live Register, 1967M01–2026M10, $n = 718$ months; $n = 706$ valid YoY observations)

> [!IMPORTANT]
> **Fundamental Target Horizon Limitation (`1M-as-12M PD Proxy` Assumption):**  
> The UCI ID 350 target variable (`default payment next month`) is a **one-month-ahead default event** (measuring whether the borrower defaults in October 2005 following the April–September 2005 billing/repayment window), **not an empirical 12-month default probability**. Because the dataset provides only a single outcome month and no 12-month performance horizon, using the scorecard's predicted probability and grade default rates (plus Jeffreys Margin of Conservatism) as a **12-month PD proxy (`1M-as-12M PD proxy`)** in the Basel IRB capital formula and IFRS 9 ECL calculations is an **explicit methodological assumption and limitation** adopted solely to illustrate regulatory capital and provisioning mechanics.

---

## 1. Risk Segmentation (`gold.segmentation_grade_profiles` & `gold.segmentation_behavioural_profiles`)

### 1.1 Rating Grade Risk Driver Profiles (`G1`–`G6`)

Every count, default count, bad rate, and risk driver mean below is generated directly from `gold.credit_card_modelling` and verified by automated assertions against DuckDB:

| Split | Grade | Score Band | Accounts ($n$) | Share (%) | Defaults ($d$) | Observed Bad Rate (1M Target, %) | Mean Champion PD (1M Target, %) | Mean Score | `PAY_0 = -2` Inactive (%) | `PAY_0 = -1` Paid Duly (%) | `PAY_0 = 0` Revolving (%) | `PAY_0 = 1` 1M DPD (%) | `PAY_0 >= 2` 2M+ DPD (%) | Mean Months Delinq (6M) | Mean Months Revolv (6M) | Mean Util Ratio 6M (%) | Mean Monthly Pay (`NT$`) | Mean Credit Limit (`NT$`) |
| :--- | :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **DEV** | **G1** | `[565, +inf)` | 2,559 | 12.21% | 145 | **5.67%** | 6.81% | 576.90 | 9.50% | 16.96% | 73.54% | 0.00% | 0.00% | 0.00 | 4.39 | 32.47% | 14,965 | 321,578 |
| **DEV** | **G2** | `[535, 565)` | 5,198 | 24.81% | 467 | **8.98%** | 9.57% | 549.98 | 11.91% | 22.85% | 65.24% | 0.00% | 0.00% | 0.01 | 3.99 | 33.75% | 7,680 | 201,089 |
| **DEV** | **G3** | `[500, 535)` | 6,105 | 29.14% | 958 | **15.69%** | 14.22% | 517.76 | 16.25% | 28.58% | 55.17% | 0.00% | 0.00% | 0.07 | 3.12 | 32.21% | 2,215 | 136,280 |
| **DEV** | **G4** | `[460, 500)` | 1,807 | 8.62% | 424 | **23.46%** | 21.28% | 482.48 | 3.82% | 28.22% | 57.94% | 10.02% | 0.00% | 0.99 | 3.18 | 39.64% | 3,493 | 115,794 |
| **DEV** | **G5** | `[400, 460)` | 2,356 | 11.24% | 754 | **32.00%** | 36.10% | 429.29 | 0.13% | 3.52% | 25.64% | 67.40% | 3.31% | 1.79 | 1.92 | 32.59% | 2,403 | 142,220 |
| **DEV** | **G6** | `(-inf, 400)` | 2,927 | 13.97% | 1,894 | **64.71%** | 63.79% | 344.72 | 0.00% | 0.17% | 0.07% | 26.99% | 72.77% | 3.81 | 1.93 | 62.62% | 2,387 | 88,946 |
| **HOLDOUT** | **G1** | `[565, +inf)` | 1,084 | 12.07% | 82 | **7.56%** | 6.78% | 577.22 | 8.86% | 15.41% | 75.74% | 0.00% | 0.00% | 0.00 | 4.54 | 32.75% | 14,178 | 316,946 |
| **HOLDOUT** | **G2** | `[535, 565)` | 2,267 | 25.25% | 206 | **9.09%** | 9.53% | 550.37 | 12.17% | 21.48% | 66.34% | 0.00% | 0.00% | 0.01 | 4.04 | 33.62% | 7,588 | 200,865 |
| **HOLDOUT** | **G3** | `[500, 535)` | 2,606 | 29.02% | 430 | **16.50%** | 14.22% | 517.77 | 16.08% | 29.55% | 54.37% | 0.00% | 0.00% | 0.09 | 3.09 | 31.98% | 2,401 | 135,668 |
| **HOLDOUT** | **G4** | `[460, 500)` | 814 | 9.06% | 195 | **23.96%** | 21.40% | 481.93 | 3.69% | 29.36% | 56.02% | 10.93% | 0.00% | 1.01 | 3.14 | 38.96% | 3,662 | 121,069 |
| **HOLDOUT** | **G5** | `[400, 460)` | 1,011 | 11.26% | 319 | **31.55%** | 36.01% | 429.54 | 0.20% | 3.86% | 21.76% | 71.41% | 2.77% | 1.77 | 1.81 | 30.67% | 2,562 | 150,857 |
| **HOLDOUT** | **G6** | `(-inf, 400)` | 1,198 | 13.34% | 757 | **63.19%** | 63.67% | 345.01 | 0.00% | 0.00% | 0.08% | 25.71% | 74.21% | 3.74 | 1.93 | 61.00% | 2,401 | 88,431 |
| **ALL_SILVER_GOLD** | **G1** | `[565, +inf)` | 3,643 | 12.17% | 227 | **6.23%** | 6.80% | 576.99 | 9.31% | 16.50% | 74.20% | 0.00% | 0.00% | 0.00 | 4.44 | 32.55% | 14,731 | 320,200 |
| **ALL_SILVER_GOLD** | **G2** | `[535, 565)` | 7,465 | 24.94% | 673 | **9.02%** | 9.56% | 550.10 | 11.99% | 22.44% | 65.57% | 0.00% | 0.00% | 0.01 | 4.01 | 33.71% | 7,652 | 201,021 |
| **ALL_SILVER_GOLD** | **G3** | `[500, 535)` | 8,711 | 29.10% | 1,388 | **15.93%** | 14.22% | 517.76 | 16.20% | 28.87% | 54.93% | 0.00% | 0.00% | 0.08 | 3.11 | 32.14% | 2,271 | 136,097 |
| **ALL_SILVER_GOLD** | **G4** | `[460, 500)` | 2,621 | 8.76% | 619 | **23.62%** | 21.32% | 482.31 | 3.78% | 28.58% | 57.34% | 10.30% | 0.00% | 0.99 | 3.17 | 39.43% | 3,545 | 117,432 |
| **ALL_SILVER_GOLD** | **G5** | `[400, 460)` | 3,367 | 11.25% | 1,073 | **31.87%** | 36.07% | 429.36 | 0.15% | 3.62% | 24.47% | 68.61% | 3.15% | 1.78 | 1.89 | 32.01% | 2,451 | 144,813 |
| **ALL_SILVER_GOLD** | **G6** | `(-inf, 400)` | 4,125 | 13.78% | 2,651 | **64.27%** | 63.76% | 344.81 | 0.00% | 0.12% | 0.07% | 26.62% | 73.19% | 3.79 | 1.93 | 62.15% | 2,392 | 88,797 |

### 1.2 Behavioural Customer Segmentation (`S1`–`S5`)

- **`S1: Inactive / Dormant Transactors`**: `MONTHS_DELINQUENT_6M == 0` and (`MONTHS_INACTIVE_6M >= 3` or (`PAY_0 == -2` and `MEAN_UTIL_RATIO_6M < 5%`)), excluding active revolvers (`MONTHS_REVOLVING_6M >= 3`).
- **`S2: Active Full-Payer Transactors`**: `MONTHS_DELINQUENT_6M == 0`, not `S1`, and `MONTHS_REVOLVING_6M <= 2`.
- **`S3: Seasoned Performing Revolvers`**: `MONTHS_DELINQUENT_6M == 0` and `MONTHS_REVOLVING_6M >= 3`.
- **`S4: Episodic / Recently Delinquent (1–2M)`**: `1 <= MONTHS_DELINQUENT_6M <= 2` and `MAX_DPD_6M <= 2`.
- **`S5: Chronic / Severe Delinquent (3+M or 90+ DPD)`**: `MONTHS_DELINQUENT_6M >= 3` or `MAX_DPD_6M >= 3`.

| Split | Behavioural Segment | Accounts ($n$) | Share (%) | Defaults ($d$) | Observed Bad Rate (1M Target, %) | Mean Champion PD (1M Target, %) | Mean Challenger Platt PD (%) | Mean Score | Mean Months Delinq (6M) | Mean Months Revolv (6M) | Mean Months Inact (6M) | Mean Util Ratio 6M (%) | Mean Monthly Pay (`NT$`) | Mean Credit Limit (`NT$`) |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **DEV** | **S1: Inactive / Dormant Transactors** | 2,140 | 10.21% | 284 | **13.27%** | 12.75% | 13.44% | 528.96 | 0.00 | 0.19 | 5.25 | 3.66% | 5,050 | 239,864 |
| **DEV** | **S2: Active Full-Payer Transactors** | 2,854 | 13.62% | 382 | **13.38%** | 11.93% | 13.18% | 534.69 | 0.00 | 0.59 | 0.26 | 5.16% | 7,602 | 217,999 |
| **DEV** | **S3: Seasoned Performing Revolvers** | 8,923 | 42.59% | 935 | **10.48%** | 10.73% | 10.57% | 544.02 | 0.00 | 5.56 | 0.11 | 50.46% | 6,276 | 165,771 |
| **DEV** | **S4: Episodic / Recently Delinquent (1-2M)** | 4,261 | 20.34% | 1,366 | **32.06%** | 33.92% | 32.40% | 440.32 | 1.29 | 2.69 | 0.89 | 35.25% | 3,571 | 146,831 |
| **DEV** | **S5: Chronic / Severe Delinquent (3+M or 90+ DPD)** | 2,774 | 13.24% | 1,675 | **60.38%** | 58.61% | 59.59% | 360.41 | 4.38 | 1.36 | 0.07 | 59.05% | 2,515 | 95,357 |
| **HOLDOUT** | **S1: Inactive / Dormant Transactors** | 912 | 10.16% | 138 | **15.13%** | 12.71% | 13.38% | 529.30 | 0.00 | 0.20 | 5.19 | 3.96% | 5,109 | 236,447 |
| **HOLDOUT** | **S2: Active Full-Payer Transactors** | 1,191 | 13.26% | 173 | **14.53%** | 12.00% | 13.47% | 534.25 | 0.00 | 0.57 | 0.27 | 4.85% | 7,174 | 215,911 |
| **HOLDOUT** | **S3: Seasoned Performing Revolvers** | 3,866 | 43.05% | 420 | **10.86%** | 10.69% | 10.51% | 544.33 | 0.00 | 5.58 | 0.10 | 50.15% | 6,113 | 164,850 |
| **HOLDOUT** | **S4: Episodic / Recently Delinquent (1-2M)** | 1,915 | 21.33% | 621 | **32.43%** | 33.83% | 32.47% | 440.66 | 1.29 | 2.59 | 0.96 | 33.58% | 3,966 | 153,632 |
| **HOLDOUT** | **S5: Chronic / Severe Delinquent (3+M or 90+ DPD)** | 1,096 | 12.20% | 637 | **58.12%** | 58.49% | 59.50% | 360.70 | 4.44 | 1.31 | 0.07 | 57.73% | 2,598 | 95,967 |
| **ALL_SILVER_GOLD** | **S1: Inactive / Dormant Transactors** | 3,052 | 10.20% | 422 | **13.83%** | 12.74% | 13.43% | 529.06 | 0.00 | 0.19 | 5.23 | 3.75% | 5,068 | 238,843 |
| **ALL_SILVER_GOLD** | **S2: Active Full-Payer Transactors** | 4,045 | 13.51% | 555 | **13.72%** | 11.95% | 13.26% | 534.56 | 0.00 | 0.58 | 0.26 | 5.07% | 7,476 | 217,384 |
| **ALL_SILVER_GOLD** | **S3: Seasoned Performing Revolvers** | 12,789 | 42.73% | 1,355 | **10.60%** | 10.72% | 10.55% | 544.11 | 0.00 | 5.57 | 0.11 | 50.37% | 6,227 | 165,493 |
| **ALL_SILVER_GOLD** | **S4: Episodic / Recently Delinquent (1-2M)** | 6,176 | 20.63% | 1,987 | **32.17%** | 33.90% | 32.42% | 440.43 | 1.29 | 2.66 | 0.91 | 34.73% | 3,693 | 148,940 |
| **ALL_SILVER_GOLD** | **S5: Chronic / Severe Delinquent (3+M or 90+ DPD)** | 3,870 | 12.93% | 2,312 | **59.74%** | 58.58% | 59.56% | 360.49 | 4.40 | 1.35 | 0.07 | 58.68% | 2,539 | 95,530 |

---

## 2. Illustrative Basel Retail Capital (`gold.basel_capital_by_grade` & `gold.basel_capital_sensitivity`)

### 2.1 Regulatory Citations, Stated Assumptions & Limitations

- **Target Horizon Assumption (`1M-as-12M PD Proxy`):** The Basel II/III/CRR3 IRB formula requires a 12-month Through-the-Cycle (TTC) or hybrid PD. Because the UCI ID 350 target (`default payment next month`) is a **1-month default indicator**, we use each grade's `DEV` 1-month observed default rate plus its 95% Jeffreys Margin of Conservatism ($\text{PD}_{\text{IRB},g}^{\text{proxy}} = \text{ODR}_{\text{DEV},g} + \text{MoC}_g = \text{Jeffreys}_{95\%,\text{upper}}(\text{DEV}_g)$) as an **illustrative 12-month PD proxy**.
- **Asset Class & Correlation:** Qualifying Revolving Retail Exposures (QRRE) under CRR Art. 154(4) / Basel CRE30.23 & CRE31.9 with fixed asset correlation **$R = 0.04$**.
- **Regulatory PD Floors (CRR3 vs CRR2):**
  - **CRR3 QRRE Revolver Floor (Applied):** **$0.10\%$ (`0.0010`)** under CRR3 Art. 163(1) / Basel III CRE30.55 for QRRE revolvers.
  - **CRR3 Standard Retail / QRRE Transactor Floor:** **$0.05\%$ (`0.0005`)** under CRR3 Art. 163(1).
  - **Legacy CRR2 Retail Floor:** **$0.03\%$ (`0.0003`)** under CRR2 Art. 163.
  - *(Note: All three floors are non-binding on this portfolio because the lowest grade PD proxy is `G1 = 6.6126%`.)*
- **LGD Assumption & CRR3 Input Floor:**
  - **Baseline LGD Assumption:** **$85.0\%$ (`0.85`)**, stated as an explicit assumption for unsecured retail credit card downturn LGD because UCI ID 350 contains no post-default recovery cash flows.
  - **CRR3 QRRE LGD Input Floor:** **$50.0\%$ (`0.50`)** under CRR3 Art. 164(4) / Basel III CRE30.56 for unsecured QRRE exposures (included as the lower bound of the LGD sensitivity grid `[50%, 65%, 75%, 85%, 95%]`).
- **EAD Construction (`CCF = 50%` Baseline):** $\text{Drawn}_i = \max(\text{BILL\_AMT1}_i, 0)$, $\text{Undrawn}_i = \max(\text{LIMIT\_BAL}_i - \text{Drawn}_i, 0)$, and $\text{EAD}_i = \text{Drawn}_i + \text{CCF} \times \text{Undrawn}_i$.

### 2.2 Grade-Level Basel QRRE Capital Schedule (`LGD = 85%`, `CCF = 50%`, `R = 0.04`, `PD Floor = 0.10%`)

| Split | Grade | Accounts ($n$) | `DEV` 1M Bad Rate (%) | Jeffreys MoC (%) | **IRB PD Proxy w/ MoC (%)** | 99.9% Stressed CDR (%) | Total Stressed Loss Rate $\text{EL}+\text{UL}$ (% EAD) | UL Capital Factor $K_{\text{UL}}$ | RWA Density (% EAD) | Drawn (`NT$ M`) | Undrawn (`NT$ M`) | EAD (`NT$ M`) | RWA (`NT$ M`) | Capital @ 8% (`NT$ M`) | Expected Loss (`NT$ M`) | Total Stressed Loss $\text{EL}+\text{UL}$ (`NT$ M`) |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **DEV** | **G1** | 2,559 | 5.6663% | +0.9463% | **6.6126%** | 18.2592% | 15.5203% | 0.098996 | **123.75%** | 300.28 | 526.08 | 563.32 | **697.08** | 55.77 | 31.66 | 87.43 |
| **DEV** | **G2** | 5,198 | 8.9842% | +0.8002% | **9.7844%** | 24.5153% | 20.8380% | 0.125212 | **156.52%** | 318.10 | 731.15 | 683.68 | **1,070.06** | 85.61 | 56.86 | 142.47 |
| **DEV** | **G3** | 6,105 | 15.6921% | +0.9286% | **16.6206%** | 35.9999% | 30.5999% | 0.164724 | **205.90%** | 181.33 | 654.53 | 508.60 | **1,047.23** | 83.78 | 71.85 | 155.63 |
| **DEV** | **G4** | 1,807 | 23.4643% | +1.9953% | **25.4596%** | 48.2884% | 41.0452% | 0.194045 | **242.56%** | 57.38 | 152.56 | 133.66 | **324.20** | 25.94 | 28.92 | 54.86 |
| **DEV** | **G5** | 2,356 | 32.0034% | +1.9049% | **33.9083%** | 58.2099% | 49.4784% | 0.206564 | **258.20%** | 68.42 | 268.00 | 202.42 | **522.66** | 41.81 | 58.34 | 100.15 |
| **DEV** | **G6** | 2,927 | 64.7079% | +1.7160% | **66.4239%** | 85.6244% | 72.7807% | 0.163204 | **204.01%** | 154.20 | 109.66 | 209.03 | **426.43** | 34.11 | 118.02 | 152.13 |
| **DEV** | **Total** | **20,952** | — | — | — | — | — | — | **177.67%** | **1,079.72** | **2,441.98** | **2,300.71** | **4,087.66** | **327.01** | **365.66** | **692.67** |
| **HOLDOUT** | **G1** | 1,084 | 5.6663% | +0.9463% | **6.6126%** | 18.2592% | 15.5203% | 0.098996 | **123.75%** | 126.97 | 218.69 | 236.31 | **292.42** | 23.39 | 13.28 | 36.68 |
| **HOLDOUT** | **G2** | 2,267 | 8.9842% | +0.8002% | **9.7844%** | 24.5153% | 20.8380% | 0.125212 | **156.52%** | 137.82 | 319.32 | 297.48 | **465.60** | 37.25 | 24.74 | 61.99 |
| **HOLDOUT** | **G3** | 2,606 | 15.6921% | +0.9286% | **16.6206%** | 35.9999% | 30.5999% | 0.164724 | **205.90%** | 74.90 | 280.87 | 215.33 | **443.38** | 35.47 | 30.42 | 65.89 |
| **HOLDOUT** | **G4** | 814 | 23.4643% | +1.9953% | **25.4596%** | 48.2884% | 41.0452% | 0.194045 | **242.56%** | 24.26 | 74.70 | 61.61 | **149.44** | 11.96 | 13.33 | 25.29 |
| **HOLDOUT** | **G5** | 1,011 | 32.0034% | +1.9049% | **33.9083%** | 58.2099% | 49.4784% | 0.206564 | **258.20%** | 31.13 | 122.41 | 92.33 | **238.41** | 19.07 | 26.61 | 45.69 |
| **HOLDOUT** | **G6** | 1,198 | 64.7079% | +1.7160% | **66.4239%** | 85.6244% | 72.7807% | 0.163204 | **204.01%** | 61.33 | 45.94 | 84.30 | **171.97** | 13.76 | 47.59 | 61.35 |
| **HOLDOUT** | **Total** | **8,980** | — | — | — | — | — | — | **178.38%** | **456.40** | **1,061.93** | **987.36** | **1,761.22** | **140.90** | **155.98** | **296.88** |
| **ALL_SILVER_GOLD** | **G1** | 3,643 | 5.6663% | +0.9463% | **6.6126%** | 18.2592% | 15.5203% | 0.098996 | **123.75%** | 427.24 | 744.77 | 799.63 | **989.50** | 79.16 | 44.94 | 124.10 |
| **ALL_SILVER_GOLD** | **G2** | 7,465 | 8.9842% | +0.8002% | **9.7844%** | 24.5153% | 20.8380% | 0.125212 | **156.52%** | 455.92 | 1,050.48 | 981.16 | **1,535.66** | 122.85 | 81.60 | 204.45 |
| **ALL_SILVER_GOLD** | **G3** | 8,711 | 15.6921% | +0.9286% | **16.6206%** | 35.9999% | 30.5999% | 0.164724 | **205.90%** | 256.23 | 935.40 | 723.93 | **1,490.61** | 119.25 | 102.27 | 221.52 |
| **ALL_SILVER_GOLD** | **G4** | 2,621 | 23.4643% | +1.9953% | **25.4596%** | 48.2884% | 41.0452% | 0.194045 | **242.56%** | 81.64 | 227.26 | 195.27 | **473.64** | 37.89 | 42.26 | 80.15 |
| **ALL_SILVER_GOLD** | **G5** | 3,367 | 32.0034% | +1.9049% | **33.9083%** | 58.2099% | 49.4784% | 0.206564 | **258.20%** | 99.55 | 390.41 | 294.76 | **761.07** | 60.89 | 84.95 | 145.84 |
| **ALL_SILVER_GOLD** | **G6** | 4,125 | 64.7079% | +1.7160% | **66.4239%** | 85.6244% | 72.7807% | 0.163204 | **204.01%** | 215.53 | 155.60 | 293.33 | **598.40** | 47.87 | 165.61 | 213.49 |
| **ALL_SILVER_GOLD** | **Total** | **29,932** | — | — | — | — | — | — | **177.88%** | **1,536.11** | **3,503.91** | **3,288.07** | **5,848.88** | **467.91** | **521.64** | **989.55** |

### 2.3 Mathematical Monotonicity Verification (PD Below/Above Peak & LGD)

1. **$K_{\text{UL}}$ and RWA Density Monotonic in PD Below the Vasicek Peak ($\text{PD}^* = 38.9795\%$ for $R = 0.04$):**
   - Across `G1` $\rightarrow$ `G5` ($\text{PD} \in [6.6126\%, 33.9083\%] < 38.9795\%$), $K_{\text{UL}}$ increases strictly (`0.098996` $\rightarrow$ `0.125212` $\rightarrow$ `0.164724` $\rightarrow$ `0.194045` $\rightarrow$ `0.206564`) and RWA density increases strictly (`123.75%` $\rightarrow$ `156.52%` $\rightarrow$ `205.90%` $\rightarrow$ `242.56%` $\rightarrow$ `258.20%`).
   - Above the Vasicek inflection point $\text{PD}^* = 38.9795\%$ (where $\partial \text{CDR}/\partial \text{PD} = 1$), incremental PD shifts loss out of Unexpected Loss ($K_{\text{UL}}$) and into Expected Loss ($\text{EL} = \text{LGD} \times \text{PD}$), so `G6` (`PD = 66.4239%`) has $K_{\text{UL}} = 0.163204$ (`204.01%` RWA density) and $\text{EL} = 56.4603\%$ of EAD.
2. **Total Stressed Loss Rate ($\text{EL} + \text{UL} = \text{LGD} \times \text{CDR}(\text{PD})$) Strictly Monotonic in PD Everywhere (`G1` $\rightarrow$ `G6`):**
   - Across all six grades `G1` $\rightarrow$ `G6`, 99.9% Stressed CDR (`18.26%` $\rightarrow$ `24.52%` $\rightarrow$ `36.00%` $\rightarrow$ `48.29%` $\rightarrow$ `58.21%` $\rightarrow$ `85.62%`) and Total Stressed Loss $\text{EL} + \text{UL}$ (`15.52%` $\rightarrow$ `20.84%` $\rightarrow$ `30.60%` $\rightarrow$ `41.05%` $\rightarrow$ `49.48%` $\rightarrow$ `72.78%` of EAD) increase **strictly monotonically**.
3. **$K_{\text{UL}}$, RWA, and $\text{EL} + \text{UL}$ Strictly Monotonic in LGD Everywhere (`50% CRR3 Floor` $\rightarrow$ `95%`):**

| LGD Assumption | `CCF = 25%` (`EAD = NT$ 721.88M`) RWA (`NT$ M`) / Density | `CCF = 50%` (`EAD = NT$ 987.36M`) RWA (`NT$ M`) / Density | `CCF = 75%` (`EAD = NT$ 1,252.85M`) RWA (`NT$ M`) / Density | `CCF = 100%` (`EAD = NT$ 1,518.33M`) RWA (`NT$ M`) / Density |
| :--- | ---: | ---: | ---: | ---: |
| **`LGD = 50%`** *(CRR3 QRRE Floor)* | `750.76` (`104.00%`) | `1,036.01` (`104.93%`) | `1,321.27` (`105.46%`) | `1,606.52` (`105.81%`) |
| **`LGD = 65%`** | `975.99` (`135.20%`) | `1,346.82` (`136.41%`) | `1,717.65` (`137.10%`) | `2,088.47` (`137.55%`) |
| **`LGD = 75%`** | `1,126.15` (`156.00%`) | `1,554.02` (`157.39%`) | `1,981.90` (`158.19%`) | `2,409.78` (`158.71%`) |
| **`LGD = 85%`** *(Baseline Assumption)* | `1,276.30` (`176.80%`) | `1,761.22` (`178.38%`) | `2,246.15` (`179.28%`) | `2,731.08` (`179.87%`) |
| **`LGD = 95%`** | `1,426.45` (`197.60%`) | `1,968.43` (`199.36%`) | `2,510.40` (`200.38%`) | `3,052.38` (`201.04%`) |

---

## 3. IFRS 9 Staging Illustration (`gold.ifrs9_markov_matrix` & `gold.ifrs9_stage_summary`)

> [!WARNING]
> **Illustrative Status & Target Horizon Caveat:**  
> All IFRS 9 12-month and 36-month lifetime PD and ECL figures below are **illustrative**. First, the underlying scorecard target (`default payment next month`) is a **1-month default event** used here as an explicit **12-month PD proxy (`1M-as-12M PD proxy`)**. Second, UCI ID 350 contains no origination date or origination PD (precluding a relative origination-PD SICR test), no cure/write-off cash-flow logs, and no contractual APRs (so $\text{LGD} = 85\%$, $\text{EIR} = 18.0\%$ p.a., and a $T = 36\text{-month}$ revolving behavioural horizon are stated assumptions).

### 3.1 Documentation of the UCI `PAY` Code `1` Coding Inconsistency & 3-State Markov Redesign

A cross-month audit of `PAY_0, PAY_2, PAY_3, PAY_4, PAY_5, PAY_6` in UCI ID 350 reveals a well-known historical coding inconsistency between Month 1 (`PAY_0`) and Months 2–6 (`PAY_2`–`PAY_6`):

| Delinquency Column | Raw Bronze ($N=30,000$) `PAY <= 0` | Raw Bronze ($N=30,000$) **`PAY == 1`** | Raw Bronze ($N=30,000$) **`PAY == 2`** | Raw Bronze ($N=30,000$) `PAY >= 3` | Gold ($N=29,932$) **`PAY == 1`** | Gold ($N=29,932$) **`PAY == 2`** | `DEV` ($n=20,952$) **`PAY == 1`** | `DEV` ($n=20,952$) **`PAY == 2`** |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **`PAY_0`** | 23,182 | **3,688** | **2,667** | 463 | **3,678** | **2,663** | 2,559 | 1,881 |
| **`PAY_2`** | 25,562 | **28** | **3,927** | 483 | **28** | **3,919** | 14 | 2,770 |
| **`PAY_3`** | 25,787 | **4** | **3,819** | 390 | **4** | **3,809** | 3 | 2,693 |
| **`PAY_4`** | 26,490 | **2** | **3,159** | 349 | **2** | **3,149** | 2 | 2,249 |
| **`PAY_5`** | 27,032 | **0** | **2,626** | 342 | **0** | **2,621** | 0 | 1,842 |
| **`PAY_6`** | 26,921 | **0** | **2,766** | 313 | **0** | **2,760** | 0 | 1,960 |

- **Root Cause & Resolution:** In `PAY_0` (September 2005), code `1` (`1M DPD`) appears **3,688 times** in the raw 30,000-row dataset (`3,678` in Gold), whereas in `PAY_2`–`PAY_6` (April–August 2005) code `1` appears only **28, 4, 2, 0, and 0 times** while code `2` appears in the thousands (`2,621` to `3,919` times). Treating code `1` and code `2` as separate Markov states creates a degenerate `State 1` row based on just 19 `DEV` transitions.
- **3-State Markov Matrix Specification:** To align with both the data structure and the IFRS 9 Stage 2 (`30–89 DPD`) definition, we merge codes `1` and `2` into a single **Delinquent 30–89 DPD** state, forming three well-populated states across all five monthly transitions (`PAY_6 -> PAY_5 -> PAY_4 -> PAY_3 -> PAY_2 -> PAY_0`):
  1. **`State 0: Current / 0 DPD (`PAY_x <= 0`)`** $\rightarrow$ **Stage 1**
  2. **`State 1: Delinquent 30–89 DPD (`PAY_x in {1, 2}` merged)`** $\rightarrow$ **Stage 2**
  3. **`State 2: Default / 90+ DPD (`PAY_x >= 3`)`** $\rightarrow$ **Stage 3**

### 3.2 Empirical 3-State Monthly Markov Transition Matrix (`gold.ifrs9_markov_matrix`)

| Split | From State ($t$) | Transition Count | To `State 0: Current (<=0)` | To `State 1: 30–89 DPD (1–2)` | To `State 2: 90+ DPD (>=3)` | Row Sum | Absorbing Cumul Default **12M** | Absorbing Cumul Default **24M** | Absorbing Cumul Default **36M** |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **DEV** | **State 0: Current / 0 DPD (PAY_x <= 0 -> Stage 1)** | 91,888 | `0.938697` (86,255) | `0.061303` (5,633) | `0.000000` (0) | **`1.000000`** | 9.37% | 20.21% | **29.77%** |
| **DEV** | **State 1: Delinquent 30-89 DPD (PAY_x in {1, 2} merged -> Stage 2)** | 11,533 | `0.252146` (2,908) | `0.684731` (7,897) | `0.063123` (728) | **`1.000000`** | 24.93% | 33.97% | **41.88%** |
| **DEV** | **State 2: Default / 90+ DPD (PAY_x >= 3 -> Stage 3)** | 1,339 | `0.105302` (141) | `0.360717` (483) | `0.533981` (715) | **`1.000000`** | 100.00% | 100.00% | **100.00%** |
| **HOLDOUT** | **State 0: Current / 0 DPD (PAY_x <= 0 -> Stage 1)** | 39,608 | `0.938926` (37,189) | `0.061074` (2,419) | `0.000000` (0) | **`1.000000`** | 9.31% | 20.06% | **29.55%** |
| **HOLDOUT** | **State 1: Delinquent 30-89 DPD (PAY_x in {1, 2} merged -> Stage 2)** | 4,759 | `0.254885` (1,213) | `0.681866` (3,245) | `0.063249` (301) | **`1.000000`** | 24.79% | 33.77% | **41.64%** |
| **HOLDOUT** | **State 2: Default / 90+ DPD (PAY_x >= 3 -> Stage 3)** | 533 | `0.110694` (59) | `0.367730` (196) | `0.521576` (278) | **`1.000000`** | 100.00% | 100.00% | **100.00%** |
| **ALL_SILVER_GOLD** | **State 0: Current / 0 DPD (PAY_x <= 0 -> Stage 1)** | 131,496 | `0.938766` (123,444) | `0.061234` (8,052) | `0.000000` (0) | **`1.000000`** | 9.35% | 20.17% | **29.70%** |
| **ALL_SILVER_GOLD** | **State 1: Delinquent 30-89 DPD (PAY_x in {1, 2} merged -> Stage 2)** | 16,292 | `0.252946` (4,121) | `0.683894` (11,142) | `0.063160` (1,029) | **`1.000000`** | 24.89% | 33.91% | **41.81%** |
| **ALL_SILVER_GOLD** | **State 2: Default / 90+ DPD (PAY_x >= 3 -> Stage 3)** | 1,872 | `0.106838` (200) | `0.362714` (679) | `0.530449` (993) | **`1.000000`** | 100.00% | 100.00% | **100.00%** |

### 3.3 Illustrative IFRS 9 Stage Allocation, Lifetime PD & ECL (`LGD = 85%`, `EIR = 18.0%`, `Horizon = 36M`)

For each account $i$ in state $s_i \in \{0, 1\}$ with 12-month PD proxy $p_{12,i}$ (`1M-as-12M PD proxy`), the 3-state `DEV` absorbing Markov cumulative survival curve $1 - C_{s_i}(t)$ is scaled via $\kappa_i = \ln(1 - p_{12,i}) / \ln(1 - C_{s_i}(12))$ so that 12-step cumulative default equals $p_{12,i}$ and 36-step illustrative lifetime default is $\text{PD}_{\text{LT},36\text{M},i} = 1 - (1 - C_{s_i}(36))^{\kappa_i}$. Monthly marginal defaults $d_i(t) = S_i(t-1) - S_i(t)$ are discounted at monthly $\text{EIR} = 1.3888\%$.

| Split | IFRS 9 Stage | Accounts ($n$) | Account Share (%) | Observed Next-Month Bad Rate (%) | Mean 12M PD Proxy (1M-Target Assumption, %) | Mean Illustrative 36M Lifetime PD (%) | Drawn (`NT$ M`) | EAD @ CCF 50% (`NT$ M`) | EAD Share (%) | Illustrative 12M-Proxy ECL (`NT$ M`) | Illustrative 36M Lifetime ECL (`NT$ M`) | **Allocated IFRS 9 ECL (`NT$ M`)** | **Stage 2 Lifetime Uplift (`NT$ M`)** | **Allocated ECL Coverage (% EAD)** |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **DEV** | **Stage 1 (Performing: Current <= 0 / 0 DPD)** | 16,185 | 77.25% | 13.63% | 14.49% | **41.35%** | 870.53 | 1,888.80 | 82.10% | 172.60 | 445.40 | **172.60** | +0.00 | **9.14%** |
| **DEV** | **Stage 2 (SICR Backstop: Delinquent 30-89 DPD / PAY_0 in {1,2})** | 4,440 | 21.19% | 49.59% | 52.56% | **73.13%** | 191.34 | 387.88 | 16.86% | 153.04 | 202.16 | **202.16** | +49.13 | **52.12%** |
| **DEV** | **Stage 3 (Credit-Impaired: 90+ DPD / PAY_0 >= 3)** | 327 | 1.56% | 71.56% | 100.00% | **100.00%** | 17.85 | 24.03 | 1.04% | 20.43 | 20.43 | **20.43** | +0.00 | **85.00%** |
| **DEV** | **TOTAL PORTFOLIO (Stages 1 + 2 + 3)** | 20,952 | 100.00% | 22.16% | 23.89% | **49.00%** | 1,079.72 | 2,300.71 | 100.00% | 346.06 | 667.99 | **395.19** | +49.13 | **17.18%** |
| **HOLDOUT** | **Stage 1 (Performing: Current <= 0 / 0 DPD)** | 6,944 | 77.33% | 14.37% | 14.41% | **41.22%** | 368.15 | 805.65 | 81.60% | 73.40 | 189.70 | **73.40** | +0.00 | **9.11%** |
| **HOLDOUT** | **Stage 2 (SICR Backstop: Delinquent 30-89 DPD / PAY_0 in {1,2})** | 1,901 | 21.17% | 46.92% | 51.69% | **72.23%** | 82.11 | 173.68 | 17.59% | 66.69 | 88.63 | **88.63** | +21.94 | **51.03%** |
| **HOLDOUT** | **Stage 3 (Credit-Impaired: 90+ DPD / PAY_0 >= 3)** | 135 | 1.50% | 73.33% | 100.00% | **100.00%** | 6.13 | 8.04 | 0.81% | 6.83 | 6.83 | **6.83** | +0.00 | **85.00%** |
| **HOLDOUT** | **TOTAL PORTFOLIO (Stages 1 + 2 + 3)** | 8,980 | 100.00% | 22.15% | 23.59% | **48.67%** | 456.40 | 987.36 | 100.00% | 146.92 | 285.15 | **168.85** | +21.94 | **17.10%** |
| **ALL_SILVER_GOLD** | **Stage 1 (Performing: Current <= 0 / 0 DPD)** | 23,129 | 77.27% | 13.85% | 14.46% | **41.31%** | 1,238.69 | 2,694.45 | 81.95% | 246.00 | 635.09 | **246.00** | +0.00 | **9.13%** |
| **ALL_SILVER_GOLD** | **Stage 2 (SICR Backstop: Delinquent 30-89 DPD / PAY_0 in {1,2})** | 6,341 | 21.18% | 48.79% | 52.30% | **72.86%** | 273.45 | 561.55 | 17.08% | 219.73 | 290.79 | **290.79** | +71.06 | **51.78%** |
| **ALL_SILVER_GOLD** | **Stage 3 (Credit-Impaired: 90+ DPD / PAY_0 >= 3)** | 462 | 1.54% | 72.08% | 100.00% | **100.00%** | 23.98 | 32.07 | 0.98% | 27.26 | 27.26 | **27.26** | +0.00 | **85.00%** |
| **ALL_SILVER_GOLD** | **TOTAL PORTFOLIO (Stages 1 + 2 + 3)** | 29,932 | 100.00% | 22.15% | 23.80% | **48.90%** | 1,536.11 | 3,288.07 | 100.00% | 492.98 | 953.14 | **564.04** | +71.06 | **17.15%** |

---

## 4. Irish Stress Sensitivity Illustration via CSO `LRM01` (`gold.stress_scenario_summary` & `gold.stress_grade_migration`)

> [!NOTE]
> **Scenario Sensitivity Caveat (Not an Empirical Macro-Credit Model):**  
> Because a 6-month 2005 Taiwanese credit card cross-section cannot be regressed against Irish macroeconomic time series, this analysis is explicitly labelled a **scenario sensitivity illustration**. We compute historical 12-month YoY relative changes from the official CSO Ireland `LRM01` series (`Persons on the Live Register`, filtered to `All ages`, `Both sexes`, `All live register schemes`, $n = 718$ months from `1967M01` to `2026M10`, yielding $n = 706$ valid YoY observations) and apply a **stated pass-through elasticity assumption ($\varepsilon = 0.50$, so $\text{PD Multiplier} = 1 + 0.50 \times \Delta_{\text{YoY}}$)** to the 1M-as-12M PD proxies.

### 4.1 CSO `LRM01` YoY Distribution & Stress Scenarios

- **CSO `LRM01` Historical YoY Percentiles ($n = 706$ months):** Min $= -24.68\%$, Median (`P50`) $= -0.40\%$, `P75` $= +9.38\%$, **`P90` (Mild Stress) $= +22.82\%$**, `P95` $= +33.89\%$, `P99` $= +76.45\%$, **Historical Max (`2009 May`, `394,134` persons, Severe Stress) $= +95.35\%$**.

### 4.2 Portfolio Stress Impact Summary (`HOLDOUT` $n = 8,980$ & Full Portfolio $n = 29,932$)

| Split | Scenario | LRM01 YoY Shock (%) | PD Multiplier | Mean Stressed PD Proxy (%) | Mean Stressed Score | Accounts Downgraded $\ge 1$ Grade ($n$ / %) | `G5 + G6` Share (%) | Direct Stressed RWA (`NT$ M` / $\Delta\%$) | Grade-Migration RWA (`NT$ M` / $\Delta\%$) | Stressed IRB EL (`NT$ M`) | Total Stressed IRB Loss $\text{EL}+\text{UL}$ (`NT$ M`) | Fixed-Stage IFRS 9 ECL (`NT$ M` / $\Delta\%$) | SICR-Overlay Stage 2 Share (%) | SICR-Overlay IFRS 9 ECL (`NT$ M` / $\Delta\%$) |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **HOLDOUT** | **`S0_BASELINE`** | `+0.00%` | `1.0000x` | 23.08% | 496.95 | `0` (`0.00%`) | 24.60% | **`1,761.22` (`+0.00%`)** | **`1,761.22` (`+0.00%`)** | 155.98 | 296.88 | **`168.85` (`+0.00%`)** | 21.17% | **`168.85` (`+0.00%`)** |
| **HOLDOUT** | **`S1_MILD_P90`** | `+22.82%` | `1.1141x` | 25.71% | 485.09 | `2,028` (`22.58%`) | 26.46% | **`1,810.84` (`+2.82%`)** | **`1,834.60` (`+4.17%`)** | 173.78 | 318.65 | **`184.42` (`+9.22%`)** | 30.14% | **`189.55` (`+12.26%`)** |
| **HOLDOUT** | **`S2_SEVERE_MAX`** | `+95.35%` | `1.4768x` | 34.08% | 426.32 | `7,206` (`80.24%`) | 32.73% | **`1,865.23` (`+5.91%`)** | **`2,023.15` (`+14.87%`)** | 230.35 | 379.57 | **`229.95` (`+36.18%`)** | 56.71% | **`258.68` (`+53.20%`)** |
| **ALL_SILVER_GOLD** | **`S0_BASELINE`** | `+0.00%` | `1.0000x` | 23.28% | 496.15 | `0` (`0.00%`) | 25.03% | **`5,848.88` (`+0.00%`)** | **`5,848.88` (`+0.00%`)** | 521.64 | 989.55 | **`564.04` (`+0.00%`)** | 21.18% | **`564.04` (`+0.00%`)** |
| **ALL_SILVER_GOLD** | **`S1_MILD_P90`** | `+22.82%` | `1.1141x` | 25.94% | 484.22 | `6,860` (`22.92%`) | 26.84% | **`6,009.82` (`+2.75%`)** | **`6,098.47` (`+4.27%`)** | 581.17 | 1,061.95 | **`615.52` (`+9.13%`)** | 30.10% | **`632.79` (`+12.19%`)** |
| **ALL_SILVER_GOLD** | **`S2_SEVERE_MAX`** | `+95.35%` | `1.4768x` | 34.38% | 424.19 | `23,878` (`79.77%`) | 32.78% | **`6,173.91` (`+5.56%`)** | **`6,733.19` (`+15.12%`)** | 770.34 | 1,264.26 | **`765.91` (`+35.79%`)** | 56.65% | **`861.88` (`+52.80%`)** |

### 4.3 Rating Grade Migration Schedule Under Stress (`gold.stress_grade_migration`)

| Split | Rating Grade | Score Band | `S0_BASELINE` (`1.0000x`) Count / Share | `S1_MILD_P90` (`1.1141x`) Count / Share | `S2_SEVERE_MAX` (`1.4768x`) Count / Share | Severe vs Baseline Account Shift |
| :--- | :--- | :--- | ---: | ---: | ---: | ---: |
| **HOLDOUT** | **G1 (Prime / Strong)** | `[565, +inf)` | `1,084` (`12.07%`) | `609` (`6.78%`) | `89` (`0.99%`) | `-995` (`-11.08 pp`) |
| **HOLDOUT** | **G2 (Satisfactory)** | `[535, 565)` | `2,267` (`25.25%`) | `2,160` (`24.05%`) | `937` (`10.43%`) | `-1,330` (`-14.81 pp`) |
| **HOLDOUT** | **G3 (Moderate Risk)** | `[500, 535)` | `2,606` (`29.02%`) | `2,527` (`28.14%`) | `2,530` (`28.17%`) | `-76` (`-0.85 pp`) |
| **HOLDOUT** | **G4 (High Risk)** | `[460, 500)` | `814` (`9.06%`) | `1,308` (`14.57%`) | `2,485` (`27.67%`) | `+1,671` (`+18.61 pp`) |
| **HOLDOUT** | **G5 (Very High Risk)** | `[400, 460)` | `1,011` (`11.26%`) | `1,035` (`11.53%`) | `928` (`10.33%`) | `-83` (`-0.92 pp`) |
| **HOLDOUT** | **G6 (Substandard)** | `(-inf, 400)` | `1,198` (`13.34%`) | `1,341` (`14.93%`) | `2,011` (`22.39%`) | `+813` (`+9.05 pp`) |
| **ALL_SILVER_GOLD** | **G1 (Prime / Strong)** | `[565, +inf)` | `3,643` (`12.17%`) | `2,027` (`6.77%`) | `272` (`0.91%`) | `-3,371` (`-11.26 pp`) |
| **ALL_SILVER_GOLD** | **G2 (Satisfactory)** | `[535, 565)` | `7,465` (`24.94%`) | `7,067` (`23.61%`) | `3,123` (`10.43%`) | `-4,342` (`-14.51 pp`) |
| **ALL_SILVER_GOLD** | **G3 (Moderate Risk)** | `[500, 535)` | `8,711` (`29.10%`) | `8,548` (`28.56%`) | `8,390` (`28.03%`) | `-321` (`-1.07 pp`) |
| **ALL_SILVER_GOLD** | **G4 (High Risk)** | `[460, 500)` | `2,621` (`8.76%`) | `4,256` (`14.22%`) | `8,334` (`27.84%`) | `+5,713` (`+19.09 pp`) |
| **ALL_SILVER_GOLD** | **G5 (Very High Risk)** | `[400, 460)` | `3,367` (`11.25%`) | `3,398` (`11.35%`) | `3,001` (`10.03%`) | `-366` (`-1.22 pp`) |
| **ALL_SILVER_GOLD** | **G6 (Substandard)** | `(-inf, 400)` | `4,125` (`13.78%`) | `4,636` (`15.49%`) | `6,812` (`22.76%`) | `+2,687` (`+8.98 pp`) |
