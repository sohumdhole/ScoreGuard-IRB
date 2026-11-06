# ScoreGuard-IRB — Step 4: Second-Line (2LoD) Validation Pack & Fair-Lending Report

**Author:** Sohum Vivek Dhole ([github.com/sohumdhole](https://github.com/sohumdhole) | MSc in Business Analytics, First Class Honours, Dublin Business School)  
**Target Role Alignment:** AIB Quantitative Risk Analyst (Risk Analytics, 2nd Line of Defence)  
**Execution Script:** `src/validation_pack.py`  
**Summary Artifact:** `results/step4_validation_summary.json`  
**Calibration Plots:** `results/step4_calibration_by_decile.png`, `results/step4_calibration_by_grade.png`

---

## 1. Discrimination & Probability Accuracy (`DEV` vs. `HOLDOUT`)

Both the **Champion (7-Feature Monotonic WoE Logistic Scorecard)** and the **Challenger-Expanded (19-Feature Platt-Calibrated Monotone LightGBM)** were evaluated on `DEV` (`n = 20,952`, `4,642` defaults, `22.1554%` default rate) and `HOLDOUT` (`n = 8,980`, `1,989` defaults, `22.1492%` default rate).

- For comparison on the same points scale (`Base Points = 600` at `Base Odds = 19:1`, `PDO = 50`, `Factor = 72.1348`, `Offset = 387.6036`), the Challenger's Platt-calibrated probability $\hat{p}_{\text{Platt}}$ is mapped to its equivalent score $\text{Score}_{\text{equiv}} = 387.6036 - 72.1348 \ln(\hat{p}_{\text{Platt}} / (1 - \hat{p}_{\text{Platt}}))$.
- As recorded in Step 3, the **Quantitative Governance Trigger** is a **`HOLDOUT` Gini gap of `5.0` points (`0.0500`) or more** (`Challenger Gini - Champion Gini >= 0.0500`). On `HOLDOUT`, the observed Gini gap is **`+2.63` points (`+0.026308`)**, which remains well below the `5.0`-point trigger.

### Table 1A — Discrimination, KS Peak & Brier Accuracy Summary

| Model | Sample Split | Evaluation Basis | Accounts ($N$) | Defaults ($D$) | Obs. Default Rate | Mean Pred. PD | AUROC | Gini | KS Statistic | Score at KS Peak | PD at KS Peak | Cum. Bad Capture at KS Peak | Cum. Good Capture at KS Peak | Brier Score | Log-Loss |
| :--- | :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **Champion (7F WoE Logistic Scorecard)** | `DEV` | `DEV In-Sample (MLE)` | 20,952 | 4,642 | `22.1554%` | `22.1554%` | **`0.776123`** | **`0.552245`** | **`0.419458`** | **`481.11`** | `21.4795%` | `61.6760%` | `19.7302%` | **`0.135787`** | `0.433063` |
| **Champion (7F WoE Logistic Scorecard)** | `HOLDOUT` | `HOLDOUT Out-of-Sample` | 8,980 | 1,989 | `22.1492%` | `21.8374%` | **`0.759844`** | **`0.519688`** | **`0.392953`** | **`482.40`** | `21.1797%` | `59.7788%` | `20.4835%` | **`0.140250`** | `0.445020` |
| **Challenger-Expanded (19F Platt LGBM)** | `DEV` | `DEV 5-Fold CV OOF` | 20,952 | 4,642 | `22.1554%` | `22.1553%` | `0.786740` | `0.573481` | `0.432110` | `475.00` | `22.9715%` | `63.1624%` | `19.9514%` | `0.135036` | `0.430118` |
| **Challenger-Expanded (19F Platt LGBM)** | `DEV` | `DEV Full-Fit (Platt)` | 20,952 | 4,642 | `22.1554%` | `22.1507%` | `0.798857` | `0.597714` | `0.458461` | `474.00` | `23.2163%` | `65.0797%` | `19.2336%` | `0.130406` | `0.416268` |
| **Challenger-Expanded (19F Platt LGBM)** | `HOLDOUT` | `HOLDOUT Out-of-Sample` | 8,980 | 1,989 | `22.1492%` | `21.8546%` | **`0.772998`** | **`0.545996`** | **`0.419081`** | **`468.00`** | `24.6741%` | `60.3318%` | `18.4237%` | **`0.136793`** | `0.435092` |

**Key Findings on KS & Discrimination:**
- On `DEV`, the Champion's Kolmogorov-Smirnov (KS) statistic peaks at **`0.4195` (`41.95%`)** at **Score `481.11` (`PD = 21.48%`, inside Grade `G4 [460, 500)`)**, where the lowest-scoring `29.02%` of accounts capture `61.68%` of all defaults against only `19.73%` of non-defaults.
- On `HOLDOUT`, the Champion's KS statistic peaks at **`0.3930` (`39.30%`)** at **Score `482.40` (`PD = 21.18%`)**, capturing `59.78%` of defaults against `20.48%` of non-defaults.
- The Platt-calibrated `Challenger-Expanded` reaches a `HOLDOUT` KS of **`0.4191` (`41.91%`)** at score-equivalent **`468.00` (`PD = 24.67%`)**, improving `HOLDOUT` KS by `+2.61` percentage points and `HOLDOUT` Brier score by `0.003457` (`0.140250` $\rightarrow$ `0.136793`).

---

## 2. Hosmer-Lemeshow Calibration Test & Decile Calibration Tables

 accounts are ordered deterministically by `(predicted_pd, account_id_hash)` and partitioned into $G = 10$ equal-sized risk deciles (`D1` lowest risk to `D10` highest risk). The Hosmer-Lemeshow statistic is computed as:

$$\chi^2_{\text{HL}} = \sum_{g=1}^{10} \frac{(O_{1g} - E_{1g})^2}{N_g \, \bar{\pi}_g (1 - \bar{\pi}_g)}$$

Both the standard internal degrees of freedom ($df = G - 2 = 8$) and the external validation degrees of freedom ($df = G = 10$) are reported below.

> **Methodological Caveat on Large Sample Sizes:** The Hosmer-Lemeshow $\chi^2$ statistic scales linearly with sample size $N$. At retail portfolio scale ($N = 20,952$ on `DEV` and $N = 8,980$ on `HOLDOUT`), a `1.5%` to `3.0%` local deviation between a parametric 7-feature logistic link and empirical decile bad rates produces statistically small $p$-values ($p < 0.005$) even when portfolio-level calibration (`21.84%` predicted vs `22.15%` observed on `HOLDOUT`) and Brier score (`0.140250`) are well-behaved.

### Table 2A — Hosmer-Lemeshow Summary (`G = 10` Risk Deciles)

| Model | Sample Split | Sample Size ($N$) | HL Statistic ($\chi^2_{\text{HL}}$) | Standard $df$ ($G-2$) | $p$-Value ($df=8$) | External $df$ ($G$) | $p$-Value ($df=10$) | Primary Source of $\chi^2_{\text{HL}}$ |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | :--- |
| **Champion (7F WoE Logit)** | `DEV` | 20,952 | `42.7019` | `8` | `0.000001` | `10` | `0.000006` | Deciles `D6` (`15.71`) and `D7` (`8.68`) = `57.1%` of total $\chi^2$ |
| **Champion (7F WoE Logit)** | `HOLDOUT` | 8,980 | `25.4922` | `8` | `0.001282` | `10` | `0.004487` | Deciles `D6` (`8.61`) and `D7` (`6.19`) = `58.1%` of total $\chi^2$ |
| **Challenger-Expanded (19F Platt)** | `DEV` | 20,952 | `24.2477` | `8` | `0.002083` | `10` | `0.006969` | Decile `D1` (`14.62`, in-sample tree tail compression) |
| **Challenger-Expanded (19F Platt)** | `HOLDOUT` | 8,980 | **`8.9716`** | `8` | **`0.344699`** | `10` | **`0.534803`** | Passes ($p = 0.345$); all 10 deciles within 95% Jeffreys CI except minor `D1` drift |

### Table 2B — Champion Risk Decile Calibration on `HOLDOUT` (`n = 8,980`)

| Decile | Accounts ($N_g$) | Min PD | Max PD | Mean Pred. PD | Expected Defaults ($E_g$) | Observed Defaults ($O_g$) | Observed Bad Rate | 95% Jeffreys CI (Lower, Upper) | HL $\chi^2$ Component |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :---: | ---: |
| `D1` | 898 | `4.59%` | `7.64%` | `6.58%` | `59.13` | 67 | `7.46%` | `[5.88%, 9.32%]` | `1.1206` |
| `D2` | 898 | `7.64%` | `8.84%` | `8.20%` | `73.66` | 74 | `8.24%` | `[6.58%, 10.17%]` | `0.0017` |
| `D3` | 898 | `8.84%` | `10.16%` | `9.58%` | `86.05` | 68 | `7.57%` | `[5.98%, 9.44%]` | `4.1865` |
| `D4` | 898 | `10.17%` | `11.84%` | `10.99%` | `98.73` | 115 | `12.81%` | `[10.74%, 15.11%]` | `3.0118` |
| `D5` | 898 | `11.85%` | `13.82%` | `12.78%` | `114.73` | 114 | `12.69%` | `[10.64%, 14.99%]` | `0.0054` |
| `D6` | 898 | `13.82%` | `15.79%` | `14.89%` | `133.69` | 165 | `18.37%` | `[15.95%, 21.01%]` | `8.6128` |
| `D7` | 898 | `15.80%` | `20.51%` | `17.35%` | `155.78` | 184 | `20.49%` | `[17.95%, 23.22%]` | `6.1864` |
| `D8` | 898 | `20.51%` | `35.62%` | `26.90%` | `241.58` | 230 | `25.61%` | `[22.84%, 28.54%]` | `0.7594` |
| `D9` | 898 | `35.62%` | `55.53%` | `43.00%` | `386.17` | 371 | `41.31%` | `[38.13%, 44.56%]` | `1.0451` |
| `D10` | 898 | `55.72%` | `83.78%` | `68.09%` | `611.48` | 601 | `66.93%` | `[63.80%, 69.95%]` | `0.5626` |
| **Total / HL** | **8,980** | `4.59%` | `83.78%` | **`21.84%`** | **`1,961.00`** | **1,989** | **`22.15%`** | — | **`25.4922`** |

### Table 2C — Platt-Calibrated Challenger-Expanded Risk Decile Calibration on `HOLDOUT` (`n = 8,980`)

| Decile | Accounts ($N_g$) | Min PD | Max PD | Mean Pred. PD | Expected Defaults ($E_g$) | Observed Defaults ($O_g$) | Observed Bad Rate | 95% Jeffreys CI (Lower, Upper) | HL $\chi^2$ Component |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :---: | ---: |
| `D1` | 898 | `1.68%` | `5.30%` | `4.06%` | `36.50` | 48 | `5.35%` | `[4.02%, 6.96%]` | `3.7769` |
| `D2` | 898 | `5.31%` | `7.81%` | `6.52%` | `58.53` | 65 | `7.24%` | `[5.68%, 9.07%]` | `0.7641` |
| `D3` | 898 | `7.81%` | `10.15%` | `9.03%` | `81.07` | 95 | `10.58%` | `[8.69%, 12.72%]` | `2.6324` |
| `D4` | 898 | `10.15%` | `11.88%` | `11.03%` | `99.05` | 106 | `11.80%` | `[9.82%, 14.04%]` | `0.5475` |
| `D5` | 898 | `11.88%` | `14.45%` | `12.95%` | `116.29` | 110 | `12.25%` | `[10.23%, 14.51%]` | `0.3914` |
| `D6` | 898 | `14.47%` | `17.35%` | `15.75%` | `141.40` | 142 | `15.81%` | `[13.54%, 18.31%]` | `0.0030` |
| `D7` | 898 | `17.36%` | `22.46%` | `19.72%` | `177.08` | 181 | `20.16%` | `[17.63%, 22.87%]` | `0.1083` |
| `D8` | 898 | `22.47%` | `32.77%` | `27.46%` | `246.63` | 251 | `27.95%` | `[25.09%, 30.95%]` | `0.1070` |
| `D9` | 898 | `32.80%` | `56.90%` | `42.53%` | `381.96` | 377 | `41.98%` | `[38.78%, 45.23%]` | `0.1120` |
| `D10` | 898 | `56.95%` | `87.38%` | `69.49%` | `624.03` | 614 | `68.37%` | `[65.28%, 71.35%]` | `0.5289` |
| **Total / HL** | **8,980** | `1.68%` | `87.38%` | **`21.85%`** | **`1,962.55`** | **1,989** | **`22.15%`** | — | **`8.9716`** |

---

## 3. Per-Grade Binomial Backtest (`HOLDOUT`, `n = 8,980`) & Honest Analysis of `G1` Under-Prediction

### Backtest Methodology & Stated Traffic-Light Rule
For each master rating grade $g \in \{\text{G1}, \dots, \text{G6}\}$ with $N_g$ accounts and $D_g$ observed defaults on `HOLDOUT`:
1. **Jeffreys Non-Informative Bayesian Credible Intervals:** Computed from $\text{Beta}(D_g + 0.5, \, N_g - D_g + 0.5)$ at **95%** (`[2.5%, 97.5%]`) and **99.9%** (`[0.05%, 99.95%]`).
2. **One-Sided Exact Binomial Test (Regulatory Conservatism / Under-Prediction Test):** Tests $H_0: p_{\text{true},g} \le \text{PD}_g$ against $H_1: p_{\text{true},g} > \text{PD}_g$, where $p\text{-value}_{\text{1-sided}} = \mathbb{P}(\text{Binom}(N_g, \text{PD}_g) \ge D_g)$ and $\text{Confidence} = 1 - p\text{-value}_{\text{1-sided}}$.
3. **Traffic-Light Rule:**
   - **GREEN:** Confidence $< 95.0\%$ ($p\text{-value}_{\text{1-sided}} > 0.050$)
   - **AMBER:** Confidence $95.0\%$ to $99.9\%$ ($0.001 < p\text{-value}_{\text{1-sided}} \le 0.050$)
   - **RED:** Confidence $> 99.9\%$ ($p\text{-value}_{\text{1-sided}} \le 0.001$)

Because an IRB master scale can be backtested either against **(A) the Grade `DEV` Empirical Default Rate** (pooled master-scale calibration, where each grade is assigned its `DEV` observed default rate $\text{ODR}_{g,\text{DEV}}$) or **(B) the Grade Mean Continuous Model PD** ($\overline{\text{PD}}_g$), both tests are reported side by side below, along with **(C) the Platt-Calibrated Challenger-Expanded Mean PD** in each grade.

### Table 3A — Per-Grade Binomial Backtest on `HOLDOUT` (`n = 8,980`)

| Grade | Score Band | `HOLDOUT` Count ($N_g$) | `HOLDOUT` Defaults ($D_g$) | `HOLDOUT` Obs. Default Rate | 95% Jeffreys CI for Obs. Rate | **Ref A: `DEV` Empirical Grade PD** | One-Sided $p$ (Conf. %) vs Ref A | Two-Sided $p$ vs Ref A | Traffic Light (Ref A) | **Ref B: Champion Mean Logistic PD** | One-Sided $p$ (Conf. %) vs Ref B | Two-Sided $p$ (Two-Sided Check) vs Ref B | Traffic Light (Ref B) | **Ref C: Challenger Platt Mean PD** | One-Sided $p$ (Conf. %) vs Ref C | Traffic Light (Ref C) |
| :--- | :--- | ---: | ---: | ---: | :---: | ---: | ---: | ---: | :---: | ---: | ---: | :--- | :---: | ---: | ---: | :---: |
| **`G1`** | `[565, +inf)` | 1,084 | 82 | **`7.56%`** | **`[6.10%, 9.25%]`** | **`5.67%`** (`61.42` exp) | **`0.0056`** (`99.44%`) | `0.0102` | **AMBER** | **`6.78%`** (`73.50` exp) | `0.1665` (`83.35%`) | `0.3041` (Calibrated) | **GREEN** | **`5.94%`** (`64.43` exp) | **`0.0167`** (`98.33%`) | **AMBER** |
| **`G2`** | `[535, 565)` | 2,267 | 206 | **`9.09%`** | `[7.96%, 10.32%]` | **`8.98%`** (`203.67` exp) | `0.4427` (`55.73%`) | `0.8543` | **GREEN** | **`9.53%`** (`215.98` exp) | `0.7720` (`22.80%`) | `0.4968` (Calibrated) | **GREEN** | **`8.74%`** (`198.12` exp) | `0.2890` (`71.10%`) | **GREEN** |
| **`G3`** | `[500, 535)` | 2,606 | 430 | **`16.50%`** | `[15.11%, 17.96%]` | **`15.69%`** (`408.93` exp) | `0.1343` (`86.57%`) | `0.2580` | **GREEN** | **`14.22%`** (`370.48` exp) | **`0.0006`** (`99.94%`) | **`0.0010` (Under-pred)** | **RED** | **`15.88%`** (`413.85` exp) | `0.2003` (`79.97%`) | **GREEN** |
| **`G4`** | `[460, 500)` | 814 | 195 | **`23.96%`** | `[21.12%, 26.98%]` | **`23.46%`** (`191.00` exp) | `0.3835` (`61.65%`) | `0.7405` | **GREEN** | **`21.41%`** (`174.24` exp) | **`0.0431`** (`95.69%`) | `0.0797` (Calibrated) | **AMBER** | **`24.08%`** (`196.01` exp) | `0.5466` (`45.34%`) | **GREEN** |
| **`G5`** | `[400, 460)` | 1,011 | 319 | **`31.55%`** | `[28.74%, 34.47%]` | **`32.00%`** (`323.55` exp) | `0.6309` (`36.91%`) | `0.7870` | **GREEN** | **`36.01%`** (`364.04` exp) | `0.9987` (`0.13%`) | **`0.0032` (Conservative Over-pred)** | **GREEN** | **`31.82%`** (`321.65` exp) | `0.5827` (`41.73%`) | **GREEN** |
| **`G6`** | `(-inf, 400)` | 1,198 | 757 | **`63.19%`** | `[60.43%, 65.89%]` | **`64.71%`** (`775.20` exp) | `0.8707` (`12.93%`) | `0.2765` | **GREEN** | **`63.67%`** (`762.76` exp) | `0.6474` (`35.26%`) | `0.7411` (Calibrated) | **GREEN** | **`64.15%`** (`768.47` exp) | `0.7651` (`23.49%`) | **GREEN** |

### Honest Second-Line Assessment of `G1` Uplift, `G5` Conservative Over-Prediction & Margin of Conservatism (MoC)
1. **`G1 (Prime / Strong, Score >= 565)` Unexplained `HOLDOUT` Uplift:**
   - On `HOLDOUT` (`n = 1,084`), `G1` experienced **82 defaults (`7.5646%`)**, with a 95% Jeffreys interval of **`[6.10%, 9.25%]`** and a 99.9% Jeffreys interval of **`[5.21%, 10.50%]`**.
   - **Against the `DEV` Empirical Grade PD (`5.6663%`, `61.42` expected defaults):** `G1` under-predicts by **`+189.83 bps`** (`+20.58` excess defaults). The one-sided binomial $p$-value is **`0.0056` (`99.44%` confidence; two-sided $p = 0.0102$)**, triggering an **AMBER** traffic-light flag, and `5.67%` lies below the 95% Jeffreys interval `[6.10%, 9.25%]`.
   - **Against the Platt-Calibrated Challenger-Expanded Mean PD (`5.9442%`, `64.43` expected defaults):** `G1` also under-predicts by **`+162.04 bps`** (`+17.57` excess defaults), with a one-sided $p$-value of **`0.0167` (`98.33%` confidence -> AMBER)**, also below the 95% Jeffreys interval `[6.10%, 9.25%]`.
   - **Against the Champion Continuous Logistic Mean PD (`6.7805%`, `73.50` expected defaults):** Because the unadjusted logistic tail over-predicted `G1` on `DEV` (`6.81%` predicted vs `5.67%` observed, `-113.98 bps`), that parametric buffer absorbs part of the `HOLDOUT` uplift, leaving a **`+78.41 bps` under-prediction** (`7.56%` observed vs `6.78%` predicted, one-sided $p = 0.1665$, two-sided $p = 0.3041$ -> `GREEN`).
   - **Honest Root-Cause Finding (`G1` Breakdown by `PAY_0` Code):** Within `G1`, observed default rates by `PAY_0` code on `DEV` are **`3.29%` (`-2`, `8/243`)**, **`5.30%` (`-1`, `23/434`)**, and **`6.06%` (`0`, `114/1,882`)**; on `HOLDOUT` they are **`6.25%` (`-2`, `6/96`)**, **`7.19%` (`-1`, `12/167`)**, and **`7.80%` (`0`, `64/821`)**. Code `-1` is not riskier than code `0` within `G1`, the `HOLDOUT` default-rate uplift occurs across all three non-delinquent codes (`-2`, `-1`, and `0`), and the Challenger (which separates `-1`) is also `AMBER` on `G1`. Therefore, the `G1` uplift on `HOLDOUT` is an **unexplained out-of-sample uplift, most likely sampling variation on the random 70/30 split**, and is flagged for ongoing monitoring. (The merge of `PAY_0 = -1` and `0` in Champion Bin 2 remains a separate portfolio-level limitation).
2. **Two-Sided Calibration Check & `G5` Conservative Miscalibration:**
   - In **`G5 (Very High Risk, [400, 460))`**, the Champion continuous logistic mean PD is **`36.01%`** (`364.04` expected defaults) versus an observed `HOLDOUT` default rate of **`31.55%`** (`319` observed defaults, **`-445.53 bps` over-prediction**). While `G5` is `GREEN` under the one-sided under-prediction test ($p = 0.9987$) because over-predicting risk is capital-conservative, the **two-sided calibration test rejects at $p = 0.003176$ on `HOLDOUT`** (and $p = 0.000031$ on `DEV`: `36.10%` predicted vs `32.00%` observed, `-409.76 bps`). We explicitly record `G5` as a **conservative over-prediction miscalibration** of the linear logistic curve, alongside the mid-grade under-prediction in `G3` (`16.50%` observed vs `14.22%` predicted, two-sided $p = 0.001028$).
   - By contrast, when grades are evaluated against their **`DEV` empirical pooled default rates (`Ref A`)**, **`G2`–`G6` pass both the one-sided ($p \ge 0.1343$) and two-sided ($p \ge 0.2580$) calibration checks on `HOLDOUT`**.
3. **Stated Derivation of the Margin of Conservatism (MoC) for Step 5 Capital:**
   - Under `EBA/GL/2017/16` (Section 4.4, Category C estimation error), we derive each grade's **Margin of Conservatism ($\text{MoC}_g$)** directly from the **`DEV` 95% Jeffreys credible interval upper bound minus the `DEV` empirical grade default rate**:
     $$\text{MoC}_g = \text{Jeffreys}_{95\%,\text{upper}}(\text{DEV}_g) - \text{ODR}_{\text{DEV},g}, \quad \text{PD}_{\text{IRB},g} = \text{ODR}_{\text{DEV},g} + \text{MoC}_g = \text{Jeffreys}_{95\%,\text{upper}}(\text{DEV}_g)$$
   - For **`G1`**, `DEV` empirical PD is `5.6663%` (`145 / 2,559`) and the `DEV` 95% Jeffreys upper bound is `6.6126%`, yielding a derived **`G1` MoC of `+94.63 bps` (`+0.9463%`, regulatory PD = `6.6126%`)**. Across all six grades (`G1`–`G6`), the derived MoCs are `+94.63 bps` (`G1`), `+80.02 bps` (`G2`), `+92.85 bps` (`G3`), `+199.53 bps` (`G4`), `+190.49 bps` (`G5`), and `+171.60 bps` (`G6`).

---

## 4. Population Stability Index (PSI) & Characteristic Stability Index (CSI)

Stated governance thresholds:
- **GREEN (`< 0.10`):** Stable distribution — no action required.
- **AMBER (`0.10` to `< 0.25`):** Moderate shift — investigate population drift.
- **RED (`>= 0.25`):** Significant shift — recalibration or redevelopment required.

> **Critical Methodological Caveat (Random Split vs. Out-of-Time Sample):** `DEV` (`n = 20,952`) and `HOLDOUT` (`n = 8,980`) were partitioned via a **stratified random 70/30 split (`random_state=42`)** from the same 6-month cross-sectional cohort (Taiwan, April–September 2005). Because UCI ID 350 contains only a single 6-month window per account with no longitudinal cohort dates, **no Out-of-Time (OOT) validation sample exists**. Consequently, near-zero PSI and CSI values (`0.00009` to `0.00182`) are mathematically expected by randomisation and **must not be interpreted as evidence of stability over time**.

### Table 4A — Score PSI & Feature CSI Summary (`DEV` vs. `HOLDOUT`)

| Score / Feature | Index Type | Number of Bins | PSI / CSI Value | Amber Threshold | Red Threshold | Governance Status |
| :--- | :--- | ---: | ---: | ---: | ---: | :--- |
| **`CHAMPION_SCORE` (6 Master Grades `G1`–`G6`)** | Score PSI (Rating Grades) | 6 | **`0.000607`** | `0.10` | `0.25` | **GREEN (`< 0.10`)** |
| **`CHAMPION_SCORE` (10 `DEV` Score Deciles)** | Score PSI (Score Deciles) | 10 | **`0.001653`** | `0.10` | `0.25` | **GREEN (`< 0.10`)** |
| **`CHALLENGER_EXPANDED_SCORE` (10 `DEV` Deciles)** | Score PSI (Challenger Deciles) | 10 | **`0.001815`** | `0.10` | `0.25` | **GREEN (`< 0.10`)** |
| `MONTHS_DELINQUENT_6M` | Feature CSI (Champion WoE Bins) | 5 | `0.001677` | `0.10` | `0.25` | **GREEN (`< 0.10`)** |
| `LIMIT_BAL` | Feature CSI (Champion WoE Bins) | 6 | `0.001423` | `0.10` | `0.25` | **GREEN (`< 0.10`)** |
| `MEAN_UTIL_RATIO_6M` | Feature CSI (Champion WoE Bins) | 4 | `0.000794` | `0.10` | `0.25` | **GREEN (`< 0.10`)** |
| `MEAN_PAY_AMT_6M` | Feature CSI (Champion WoE Bins) | 6 | `0.000569` | `0.10` | `0.25` | **GREEN (`< 0.10`)** |
| `MONTHS_REVOLVING_6M` | Feature CSI (Champion WoE Bins) | 3 | `0.000232` | `0.10` | `0.25` | **GREEN (`< 0.10`)** |
| `PAY_0` | Feature CSI (Champion WoE Bins) | 4 | `0.000155` | `0.10` | `0.25` | **GREEN (`< 0.10`)** |
| `BILL_TREND_6M_RATIO` | Feature CSI (Champion WoE Bins) | 4 | `0.000090` | `0.10` | `0.25` | **GREEN (`< 0.10`)** |

---

## 5. Fair-Lending & Demographic Calibration Check (`SEX` and `MARRIAGE`)

### 5.1 Policy, Exclusions & Legal Framing
1. **Verified Feature Exclusion:** Both `SEX` (`1 = Male`, `2 = Female`) and `MARRIAGE` (`1 = Married`, `2 = Single`, `3 = Other`) are programmatically verified to be excluded from the Champion (7 features) and Challenger (10F Core and 19F Expanded).
2. **Stated Credit Approval Cutoff:**
   - **Champion Rule:** **Approve Grades `G1`–`G4` (`Champion Score >= 460`, equivalent to `Champion PD <= 26.8229%`)**; **Decline Grades `G5`–`G6` (`Score < 460`)**.
   - **Challenger Rule:** **Approve accounts with `Challenger Platt PD <= 26.8229%` (`Score-Equivalent >= 460`)**.
3. **Regulatory Framing of the Four-Fifths (`0.80`) Rule:**
   - The **four-fifths (`80%` / `0.80`) disparate-impact ratio rule** is a **US regulatory heuristic** originating from the US EEOC Uniform Guidelines on Employee Selection Procedures (`29 CFR § 1607.4D`) and US ECOA supervisory practice. **It is NOT an EU legal test.**
   - Under **EU law** (EU Charter of Fundamental Rights Article 21, Council Directive `2004/113/EC` on equal treatment between men and women in the access to and supply of goods and services, and EBA Guidelines on Loan Origination and Monitoring `EBA/GL/2020/06`), direct discrimination on protected grounds is prohibited, while indirect differential outcomes in risk-based credit scoring are assessed on **objective justification, proportionality, and group calibration accuracy** (whether predicted PDs accurately reflect empirical default risk without systematic penalty against a protected group).

### Table 5A — Approval Rates (`G1`–`G4`, `Score >= 460`) & Disparate-Impact Ratios (`DEV` and `HOLDOUT`)

| Sample | Attribute | Group | Count ($N$) | Pop. Share | Observed Default Rate | Champion Mean Score | Champion Approval Rate | Champion DIR (vs Highest Group) | Challenger Approval Rate | Challenger DIR (vs Highest Group) | US Four-Fifths (`>= 0.80`) Heuristic |
| :--- | :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :---: |
| **`DEV`** | `SEX` | `Male (1)` | 8,350 | `39.85%` | `24.26%` | `491.95` | `73.04%` | **`0.9618`** | `71.71%` | **`0.9438`** | Pass (`>= 0.80`) |
| **`DEV`** | `SEX` | `Female (2)` (Ref) | 12,602 | `60.15%` | `20.76%` | `498.37` | `75.94%` | `1.0000` | `75.98%` | `1.0000` | Reference |
| **`DEV`** | `MARRIAGE` | `Married (1)` | 9,520 | `45.44%` | `23.43%` | `495.64` | `74.46%` | **`0.9917`** | `74.00%` | **`0.9924`** | Pass (`>= 0.80`) |
| **`DEV`** | `MARRIAGE` | `Single (2)` (Ref) | 11,226 | `53.58%` | `21.00%` | `496.14` | `75.08%` | `1.0000` | `74.57%` | `1.0000` | Reference |
| **`DEV`** | `MARRIAGE` | `Other (3)` | 206 | `0.98%` | `26.21%` | `485.87` | `73.30%` | **`0.9762`** | `71.36%` | **`0.9570`** | Pass (`>= 0.80`) |
| **`HOLDOUT`** | `SEX` | `Male (1)` | 3,516 | `39.15%` | `24.03%` | `492.71` | **`73.24%`** | **`0.9537`** | **`71.33%`** | **`0.9351`** | Pass (`>= 0.80`) |
| **`HOLDOUT`** | `SEX` | `Female (2)` (Ref) | 5,464 | `60.85%` | `20.94%` | `499.68` | **`76.79%`** | `1.0000` | **`76.28%`** | `1.0000` | Reference |
| **`HOLDOUT`** | `MARRIAGE` | `Married (1)` | 4,135 | `46.05%` | `23.58%` | `497.78` | **`75.28%`** | **`0.9956`** | **`74.22%`** | **`0.9955`** | Pass (`>= 0.80`) |
| **`HOLDOUT`** | `MARRIAGE` | `Single (2)` (Ref) | 4,728 | `52.65%` | `20.81%` | `496.58` | **`75.61%`** | `1.0000` | **`74.56%`** | `1.0000` | Reference |
| **`HOLDOUT`** | `MARRIAGE` | `Other (3)` | 117 | `1.30%` | `25.64%` | `482.20` | **`70.94%`** | **`0.9382`** | **`70.09%`** | **`0.9400`** | Pass (`>= 0.80`) |

### Table 5B — Calibration by Protected Group (`Observed Default Rate` vs. `Mean Predicted PD`)

| Sample | Attribute | Group | Count ($N$) | Observed Defaults ($D$) | Observed Default Rate | 95% Jeffreys CI for Obs. Rate | Champion Mean Pred. PD | Champion Diff (`Obs - Pred` bps) | Champion `Obs / Pred` Ratio | Champion Two-Sided $p$-Value | Challenger Platt Mean PD | Challenger Diff (`Obs - Pred` bps) | Challenger `Obs / Pred` Ratio | Challenger Two-Sided $p$-Value |
| :--- | :--- | :--- | ---: | ---: | ---: | :---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **`DEV`** | `SEX` | `Male (1)` | 8,350 | 2,026 | **`24.26%`** | `[23.35%, 25.19%]` | **`23.15%`** | **`+110.98 bps`** (Under-pred) | `1.0479` | `0.0170` | `23.41%` | `+85.46 bps` | `1.0365` | `0.0665` |
| **`DEV`** | `SEX` | `Female (2)` | 12,602 | 2,616 | **`20.76%`** | `[20.06%, 21.47%]` | **`21.49%`** | **`-73.53 bps`** (Over-pred) | `0.9658` | `0.0449` | `21.32%` | `-55.85 bps` | `0.9738` | `0.1279` |
| **`DEV`** | `MARRIAGE` | `Married (1)` | 9,520 | 2,231 | **`23.43%`** | `[22.59%, 24.29%]` | **`22.27%`** | **`+116.77 bps`** (Under-pred) | `1.0524` | `0.0065` | `22.26%` | `+117.83 bps` | `1.0529` | `0.0060` |
| **`DEV`** | `MARRIAGE` | `Single (2)` | 11,226 | 2,357 | **`21.00%`** | `[20.25%, 21.76%]` | **`22.03%`** | **`-102.91 bps`** (Over-pred) | `0.9533` | `0.0085` | `22.02%` | `-102.63 bps` | `0.9534` | `0.0085` |
| **`DEV`** | `MARRIAGE` | `Other (3)` | 206 | 54 | **`26.21%`** | `[20.57%, 32.52%]` | **`24.10%`** | `+211.71 bps` (Under-pred) | `1.0879` | `0.4644` | `24.26%` | `+194.87 bps` | `1.0803` | `0.5158` |
| **`HOLDOUT`** | `SEX` | `Male (1)` | 3,516 | 845 | **`24.03%`** | `[22.64%, 25.47%]` | **`22.90%`** | **`+113.01 bps`** (Under-pred) | `1.0493` | `0.1129` | **`23.11%`** | `+91.86 bps` | `1.0397` | `0.2005` |
| **`HOLDOUT`** | `SEX` | `Female (2)` | 5,464 | 1,144 | **`20.94%`** | `[19.87%, 22.03%]` | **`21.15%`** | **`-21.48 bps`** (Over-pred) | `0.9898` | `0.7156` | **`21.04%`** | `-10.69 bps` | `0.9949` | `0.8552` |
| **`HOLDOUT`** | `MARRIAGE` | `Married (1)` | 4,135 | 975 | **`23.58%`** | `[22.30%, 24.89%]` | **`21.67%`** | **`+190.50 bps`** (Under-pred) | `1.0879` | `0.0032` | **`21.68%`** | `+189.63 bps` | `1.0875` | `0.0034` |
| **`HOLDOUT`** | `MARRIAGE` | `Single (2)` | 4,728 | 984 | **`20.81%`** | `[19.67%, 21.99%]` | **`21.90%`** | **`-108.80 bps`** (Over-pred) | `0.9503` | `0.0729` | **`21.92%`** | `-110.76 bps` | `0.9495` | `0.0675` |
| **`HOLDOUT`** | `MARRIAGE` | `Other (3)` | 117 | 30 | **`25.64%`** | `[18.39%, 34.08%]` | **`25.07%`** | `+56.67 bps` (Under-pred) | `1.0226` | `0.9151` | **`25.28%`** | `+35.68 bps` | `1.0141` | `0.9155` |

### Key Second-Line Findings on Fair-Lending & Group Calibration
1. **Approval Parity & Disparate-Impact Ratios:**
   - Across both `DEV` and `HOLDOUT`, all Disparate-Impact Ratios (DIR) lie between **`0.9351` and `0.9956`** (`0.9537` for Males vs Females and `0.9382` for Other vs Single on `HOLDOUT` for the Champion).
   - Lower approval rates for `Male` (`73.24%` vs `76.79%` for `Female` on `HOLDOUT`) and `Other` (`70.94%` vs `75.61%` for `Single` on `HOLDOUT`) align directionally with higher observed default rates in those groups (`Male` `HOLDOUT` bad rate `24.03%` vs `Female` `20.94%`; `Other` `25.64%` vs `Single` `20.81%`).
2. **Calibration by Group (`Observed Default Rate` vs. `Mean Predicted PD`):**
   - On `HOLDOUT`, for `SEX`, the `Male` calibration difference is `+113.01 bps` (`24.03%` observed vs `22.90%` Champion mean PD; `+91.86 bps` for Challenger), with a two-sided binomial **$p = 0.1129$ (`p = 0.2005` for Challenger), which is not statistically significant**. For `Female`, the difference is `-21.48 bps` (`20.94%` observed vs `21.15%` Champion mean PD, **$p = 0.7156$, not statistically significant**).
   - On `HOLDOUT`, for `MARRIAGE`, the `Married` group shows an under-prediction gap of `+190.50 bps` (`23.58%` observed vs `21.67%` Champion mean PD; `+189.63 bps` for Challenger) with a two-sided binomial **$p = 0.0032$ (`p = 0.0034` for Challenger)**. The `Single` group shows `-108.80 bps` (`20.81%` observed vs `21.90%` Champion mean PD, **$p = 0.0729$, not significant at $\alpha = 0.05$**), and the `Other` group shows `+56.67 bps` (`25.64%` observed vs `25.07%` Champion mean PD, **$p = 0.9151$, not significant**).
   - **Multiple-Testing Note:** Several hypothesis tests were run across 5 demographic groups, 2 sample splits (`DEV` and `HOLDOUT`), and 2 models (20 group-level binomial tests in total), which increases the family-wise false-positive rate when evaluating unadjusted $p$-values.


