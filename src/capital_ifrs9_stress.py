"""
Step 5: Risk Segmentation, Illustrative Basel Retail Capital, IFRS 9 Staging
Illustration, and CSO Ireland LRM01 Stress Sensitivity
=============================================================================
Executes all four Step 5 deliverables on the governed DuckDB Gold layer:

1. Risk Segmentation:
   - Grade profiles (G1-G6) across the 7 Champion risk drivers and key credit
     metrics on DEV, HOLDOUT, and Full Portfolio (n=29,932), cross-checked
     programmatically against gold.credit_card_modelling.
   - 5 mutually exclusive and collectively exhaustive behavioural segments
     (S1 Inactive/Dormant Transactors, S2 Active Full-Payer Transactors,
     S3 Seasoned Performing Revolvers, S4 Episodic/Recently Delinquent 1-2M,
     S5 Chronic/Severe Delinquent 3+M or 90+ DPD) with counts and default rates
     on DEV and HOLDOUT.

2. Illustrative Basel Retail Capital (labelled "illustrative"):
   - Explicit Target Horizon Caveat: The UCI ID 350 target ("default payment next
     month") is a one-month-ahead default event, NOT a 12-month PD. Treating the
     model's predicted probability and grade default rates as a 12-month PD proxy
     (1M-as-12M PD proxy) in Basel IRB and IFRS 9 is an explicit methodological
     assumption and limitation.
   - Asset Class: Qualifying Revolving Retail Exposures (QRRE) under CRR Art. 154(4)
     / CRR3 Art. 163(1) / Basel IRB CRE30.55.
   - Asset Correlation: R = 0.04 (fixed for QRRE under CRR Art. 154(4)).
   - PD Floor: CRR3 QRRE revolver floor = 0.10% (0.0010), alongside CRR3 standard
     retail / QRRE transactor floor = 0.05% (0.0005) and legacy CRR2 Art. 163 floor =
     0.03% (0.0003). All three are non-binding as all grades have PD >= 5.67%.
   - LGD Assumption & CRR3 Input Floor: LGD = 85.0% (0.85) is a stated illustrative
     unsecured retail credit card downturn LGD assumption (UCI ID 350 has no recovery
     data). Notes the CRR3 Art. 164(4) / CRE30.56 QRRE LGD input floor of 50.0% (0.50),
     which is included as the lower bound of the LGD sensitivity grid [0.50, 0.65, 0.75, 0.85, 0.95].
   - Monotonicity Tests:
     * K_UL and RWA density strictly monotonic in PD below the Vasicek inflection
       peak (~39.13% for R = 0.04, covering G1-G5).
     * Total Stressed Loss (EL + UL = LGD * CDR(PD)) strictly monotonic in PD everywhere
       (covering all grades G1-G6 and full PD domain).
     * K_UL and RWA strictly monotonic in LGD everywhere.

3. IFRS 9 Staging Illustration (labelled "illustrative: assumptions stated"):
   - Documents the known UCI ID 350 coding inconsistency where PAY code 1 appears
     3,688 times in raw PAY_0 (3,678 in Gold) but only 28, 4, 2, 0, and 0 times in
     PAY_2, PAY_3, PAY_4, PAY_5, and PAY_6, while code 2 appears in the thousands
     (2,663 to 3,919).
   - Rebuilds the empirical monthly Markov transition matrix using 3 well-populated
     delinquency states:
       * State 0: Current / 0 DPD (PAY_x <= 0 -> Stage 1)
       * State 1: Delinquent 30-89 DPD (PAY_x in {1, 2} merged -> Stage 2)
       * State 2: Default / 90+ DPD (PAY_x >= 3 -> Stage 3)
   - Uses the 3-state absorbing Markov transition matrix (State 2 absorbing) to
     derive illustrative 36-month lifetime PDs and discounted ECLs (LGD = 85%,
     EIR = 18.0% p.a., T = 36 months), explicitly labelled as illustrative and
     subject to the 1M-as-12M PD target horizon assumption.

4. Irish Stress Sensitivity via CSO LRM01 (labelled "scenario sensitivity, not a calibrated model"):
   - Uses CSO Ireland LRM01 total series: STATISTIC='LRM01', Age Group='All ages',
     Sex='Both sexes', Social Welfare Scheme='All live register schemes' (718 months,
     706 valid YoY observations).
   - Applies stated pass-through elasticity epsilon = 0.50 to Mild (90th percentile YoY = +22.82%)
     and Severe (Historical Max YoY = +95.35%, May 2009) scenarios.
"""

import json
import math
from datetime import datetime, timezone
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
from scipy import stats

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DUCKDB_PATH = PROJECT_ROOT / "data" / "duckdb" / "scoreguard.duckdb"
RESULTS_DIR = PROJECT_ROOT / "results"

# Points scaling constants (matching Step 2 Champion)
BASE_POINTS = 600.0
BASE_ODDS = 19.0
PDO = 50.0
FACTOR = PDO / math.log(2.0)  # 72.13475204444818
OFFSET = BASE_POINTS - FACTOR * math.log(BASE_ODDS)  # 387.60362435967725

# Illustrative Basel QRRE & IFRS 9 Stated Assumptions
QRRE_CORRELATION_R = 0.04             # CRR Art. 154(4) Qualifying Revolving Retail correlation
PD_FLOOR_CRR3_QRRE_REVOLVER = 0.0010  # CRR3 Art. 163(1) / CRE30.55 QRRE revolver PD floor (0.10%)
PD_FLOOR_CRR3_RETAIL_STD = 0.0005     # CRR3 Art. 163(1) general retail / QRRE transactor PD floor (0.05%)
PD_FLOOR_CRR2_LEGACY = 0.0003         # Legacy CRR2 Art. 163 retail PD floor (0.03%)
PD_REGULATORY_FLOOR = PD_FLOOR_CRR3_QRRE_REVOLVER

LGD_CRR3_QRRE_INPUT_FLOOR = 0.50      # CRR3 Art. 164(4) / CRE30.56 QRRE unsecured LGD input floor (50%)
BASELINE_LGD = 0.85                   # Illustrative unsecured retail card downturn LGD assumption (85%)
BASELINE_CCF = 0.50                   # Illustrative undrawn credit card limit CCF assumption (50%)
ANNUAL_EIR = 0.18                     # Illustrative credit card Effective Interest Rate (18% p.a.)
MONTHLY_EIR = (1.0 + ANNUAL_EIR) ** (1.0 / 12.0) - 1.0  # 1.388843% per month
LIFETIME_HORIZON_MONTHS = 36          # Illustrative revolving behavioural lifetime (36 months)
MACRO_ELASTICITY_EPSILON = 0.50       # Illustrative YoY Live Register to PD elasticity (0.50)

# Exact analytical root of d/dp [Phi((Phi^{-1}(p) + sqrt(R)*z_999)/sqrt(1-R)) - p] = 0 for R = 0.04:
VASICEK_UL_PEAK_PD = 0.389795         # K_UL peaks at PD* = 38.9795% (~39.0%) for R = 0.04

GRADE_ORDER = [
    "G1 (Prime / Strong)",
    "G2 (Satisfactory)",
    "G3 (Moderate Risk)",
    "G4 (High Risk)",
    "G5 (Very High Risk)",
    "G6 (Substandard)",
]

GRADE_SHORT = {
    "G1 (Prime / Strong)": "G1",
    "G2 (Satisfactory)": "G2",
    "G3 (Moderate Risk)": "G3",
    "G4 (High Risk)": "G4",
    "G5 (Very High Risk)": "G5",
    "G6 (Substandard)": "G6",
}

GRADE_BAND_STR = {
    "G1": "[565, +inf)",
    "G2": "[535, 565)",
    "G3": "[500, 535)",
    "G4": "[460, 500)",
    "G5": "[400, 460)",
    "G6": "(-inf, 400)",
}


def assign_rating_grade_from_score(scores: np.ndarray) -> list[str]:
    """Assign 6-grade Master Rating Scale from continuous or rounded score."""
    out = []
    for s in scores:
        if s >= 565.0:
            out.append("G1 (Prime / Strong)")
        elif s >= 535.0:
            out.append("G2 (Satisfactory)")
        elif s >= 500.0:
            out.append("G3 (Moderate Risk)")
        elif s >= 460.0:
            out.append("G4 (High Risk)")
        elif s >= 400.0:
            out.append("G5 (Very High Risk)")
        else:
            out.append("G6 (Substandard)")
    return out


def assign_behavioural_segment(df: pd.DataFrame) -> pd.Series:
    """
    Assign 5 mutually exclusive and collectively exhaustive behavioural segments:
    - S1: Inactive / Dormant Transactors (0 months delinquent, >=3 months inactive or PAY_0==-2 with util<5%)
    - S2: Active Full-Payer Transactors (0 months delinquent, not S1, <=2 months revolving)
    - S3: Seasoned Performing Revolvers (0 months delinquent, >=3 months revolving)
    - S4: Episodic / Recently Delinquent (1-2 months delinquent in 6M and max DPD <= 2)
    - S5: Chronic / Severe Delinquent (3+ months delinquent in 6M or max DPD >= 3)
    """
    m_delinq = df["MONTHS_DELINQUENT_6M"].values
    max_dpd = df["MAX_DPD_6M"].values
    m_inact = df["MONTHS_INACTIVE_6M"].values
    m_rev = df["MONTHS_REVOLVING_6M"].values
    pay0 = df["PAY_0"].values
    util6 = df["MEAN_UTIL_RATIO_6M"].values

    seg = np.empty(len(df), dtype=object)
    for i in range(len(df)):
        if m_delinq[i] == 0:
            if m_rev[i] >= 3:
                seg[i] = "S3: Seasoned Performing Revolvers"
            elif m_inact[i] >= 3 or (pay0[i] == -2 and util6[i] < 0.05):
                seg[i] = "S1: Inactive / Dormant Transactors"
            else:
                seg[i] = "S2: Active Full-Payer Transactors"
        elif m_delinq[i] <= 2 and max_dpd[i] <= 2:
            seg[i] = "S4: Episodic / Recently Delinquent (1-2M)"
        else:
            seg[i] = "S5: Chronic / Severe Delinquent (3+M or 90+ DPD)"
    return pd.Series(seg, index=df.index)


def basel_asrf_cdr(
    pd_val: float | np.ndarray,
    r_corr: float = QRRE_CORRELATION_R,
    pd_floor: float = PD_REGULATORY_FLOOR,
) -> np.ndarray:
    """
    Basel IRB 99.9% Conditional Default Rate CDR(PD) for asset correlation R.
    Total Stressed Loss per unit EAD is EL + UL = LGD * CDR(PD).
    """
    pd_arr = np.maximum(np.asarray(pd_val, dtype=float), pd_floor)
    pd_arr = np.minimum(pd_arr, 0.9999)
    inv_pd = stats.norm.ppf(pd_arr)
    inv_999 = stats.norm.ppf(0.999)
    cond_pd = stats.norm.cdf(
        (inv_pd + math.sqrt(r_corr) * inv_999) / math.sqrt(1.0 - r_corr)
    )
    return cond_pd


def basel_asrf_k(
    pd_val: float | np.ndarray,
    lgd_val: float,
    r_corr: float = QRRE_CORRELATION_R,
    pd_floor: float = PD_REGULATORY_FLOOR,
) -> np.ndarray:
    """
    Basel II/III IRB Asymptotic Single Risk Factor (ASRF) Unexpected Loss capital
    requirement K_UL for Qualifying Revolving Retail Exposures (R = 0.04, CRR3 PD floor = 0.10%).
    """
    pd_arr = np.maximum(np.asarray(pd_val, dtype=float), pd_floor)
    pd_arr = np.minimum(pd_arr, 0.9999)
    cond_pd = basel_asrf_cdr(pd_arr, r_corr=r_corr, pd_floor=pd_floor)
    k = lgd_val * (cond_pd - pd_arr)
    return np.maximum(k, 0.0)


def basel_total_stressed_loss_rate(
    pd_val: float | np.ndarray,
    lgd_val: float,
    r_corr: float = QRRE_CORRELATION_R,
    pd_floor: float = PD_REGULATORY_FLOOR,
) -> np.ndarray:
    """
    Total Stressed Loss rate (EL + UL) per unit EAD = LGD * CDR(PD).
    Strictly monotonic in PD across the entire (0, 1) probability domain and in LGD.
    """
    cond_pd = basel_asrf_cdr(pd_val, r_corr=r_corr, pd_floor=pd_floor)
    return lgd_val * cond_pd


def compute_ead(bill_amt1: np.ndarray, limit_bal: np.ndarray, ccf: float = BASELINE_CCF) -> np.ndarray:
    """EAD = max(BILL_AMT1, 0) + CCF * max(LIMIT_BAL - max(BILL_AMT1, 0), 0)."""
    drawn = np.maximum(np.asarray(bill_amt1, dtype=float), 0.0)
    lim = np.asarray(limit_bal, dtype=float)
    undrawn = np.maximum(lim - drawn, 0.0)
    return drawn + ccf * undrawn


def map_pay_to_ifrs9_state_3state(pay_arr: np.ndarray) -> np.ndarray:
    """
    Map PAY_x code to 3 delinquency states (resolving the UCI PAY code 1 historical coding inconsistency):
    0: Current / 0 DPD (PAY_x <= 0 -> Stage 1)
    1: Delinquent 30-89 DPD (PAY_x in {1, 2} merged -> Stage 2)
    2: Default / 90+ DPD (PAY_x >= 3 -> Stage 3)
    """
    p = np.asarray(pay_arr, dtype=int)
    s = np.zeros_like(p)
    s[(p == 1) | (p == 2)] = 1
    s[p >= 3] = 2
    return s


def compute_markov_cumulative_default_curves(markov_prob_3x3: np.ndarray, horizon_months: int = LIFETIME_HORIZON_MONTHS) -> np.ndarray:
    """
    Treat State 2 (90+ DPD) as an absorbing default state in the 3x3 monthly Markov
    transition matrix and return cumulative default probabilities C_s(t) for s in {0,1,2}
    and t in 0..horizon_months (shape: 3 x (horizon_months + 1)).
    """
    abs_m = np.array(markov_prob_3x3, dtype=float).copy()
    abs_m[2, :] = [0.0, 0.0, 1.0]
    cum_def = np.zeros((3, horizon_months + 1), dtype=float)
    cur = np.eye(3, dtype=float)
    cum_def[:, 0] = cur[:, 2]
    for t in range(1, horizon_months + 1):
        cur = cur @ abs_m
        cum_def[:, t] = np.clip(cur[:, 2], 0.0, 1.0)
    return cum_def


def compute_discounted_ecl_per_unit_ead_markov3(
    pd_12m_proxy: np.ndarray,
    stage: np.ndarray,
    markov_cum_def_curves: np.ndarray,
    lgd: float = BASELINE_LGD,
    r_monthly: float = MONTHLY_EIR,
    horizon_months: int = LIFETIME_HORIZON_MONTHS,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute illustrative 12-month and Lifetime (36-month) ECL per unit of EAD using
    the 3-state absorbing Markov transition matrix term structure anchored to each
    account's 12-month PD proxy (noting that the UCI 1-month default target is used
    illustratively as a 12M PD proxy):

    - Let C_s(t) be the 3-state DEV absorbing Markov cumulative default probability at
      month t (t = 1..36) for delinquency state s in {0 (Stage 1: Current <=0), 1 (Stage 2: 30-89 DPD)}.
      On DEV:
        * State 0 (Stage 1): C_0(12) = 9.3691%, C_0(36) = 29.7657%
        * State 1 (Stage 2): C_1(12) = 24.9273%, C_1(36) = 41.8789%
    - For each account i in state s_i in {0, 1} with 12M PD proxy p_{12,i}, we scale
      the state's Markov log-survival curve by:
        kappa_i = ln(1 - p_{12,i}) / ln(1 - C_{s_i}(12))
      so that the account's survival curve S_i(t) = (1 - C_{s_i}(t))^(kappa_i) satisfies:
        * Cumulative 12-month default probability: 1 - S_i(12) = p_{12,i} (exact calibration)
        * Cumulative 36-month lifetime default probability:
          PD_{LT,36M,i} = 1 - S_i(36) = 1 - (1 - p_{12,i})^(ln(1 - C_{s_i}(36)) / ln(1 - C_{s_i}(12)))
    - Monthly marginal default probabilities d_i(t) = S_i(t-1) - S_i(t) for t = 1..36
      are discounted at monthly EIR r_m = (1 + EIR_annual)^(1/12) - 1:
        * ECL_{12M,i} = LGD * EAD_i * sum_{t=1}^{12} d_i(t) * (1 + r_m)^(-t)
        * ECL_{LT,36M,i} = LGD * EAD_i * sum_{t=1}^{36} d_i(t) * (1 + r_m)^(-t)
    - For Stage 3 accounts (State 2: 90+ DPD, credit-impaired), PD_12M = PD_LT = 1.0,
      and ECL_12M = ECL_LT = LGD * EAD_i.

    Returns: (pd_12m_eff, pd_lt_eff, ecl_12m_factor, ecl_lt_factor, ecl_ifrs9_allocated_factor)
    """
    pd_12 = np.clip(np.asarray(pd_12m_proxy, dtype=float), 1e-6, 0.999999)
    stg = np.asarray(stage, dtype=int)
    n_acc = len(pd_12)

    t_idx = np.arange(1, horizon_months + 1, dtype=float)
    disc_factors = (1.0 + r_monthly) ** (-t_idx)  # shape (36,)

    # Map stage (1, 2, 3) -> state index (0, 1, 2)
    state_idx = np.clip(stg - 1, 0, 2)

    # State cumulative survival curves 1 - C_s(t) for t = 0..36
    state_surv = np.clip(1.0 - markov_cum_def_curves[:, : horizon_months + 1], 1e-12, 1.0)
    c12_surv_per_acc = state_surv[state_idx, 12]
    # Avoid log(1) for Stage 3 (overridden below)
    c12_surv_safe = np.where(state_idx < 2, c12_surv_per_acc, 0.5)

    kappa = np.log(1.0 - pd_12) / np.log(c12_surv_safe)  # shape (n_acc,)

    # Account survival curve S_i(t) for t = 0..36: shape (n_acc, 37)
    acc_base_surv = state_surv[state_idx, :]  # (n_acc, 37)
    acc_surv = acc_base_surv ** kappa[:, None]

    # Monthly marginal default probabilities d_i(t) = S_i(t-1) - S_i(t) for t = 1..36
    marg_def = np.maximum(acc_surv[:, :-1] - acc_surv[:, 1:], 0.0)  # (n_acc, 36)

    pd_12_eff = 1.0 - acc_surv[:, 12]
    pd_lt_eff = 1.0 - acc_surv[:, horizon_months]

    ecl_12m_factor = lgd * np.sum(marg_def[:, :12] * disc_factors[:12][None, :], axis=1)
    ecl_lt_factor = lgd * np.sum(marg_def * disc_factors[None, :], axis=1)

    # Stage 3 override: credit-impaired at reporting date (PD = 100%, immediate recognition at LGD)
    s3_mask = stg == 3
    pd_12_eff[s3_mask] = 1.0
    pd_lt_eff[s3_mask] = 1.0
    ecl_12m_factor[s3_mask] = lgd
    ecl_lt_factor[s3_mask] = lgd

    # IFRS 9 Allocated ECL: Stage 1 -> 12M ECL; Stage 2 & Stage 3 -> Lifetime ECL
    ecl_allocated_factor = np.where(stg == 1, ecl_12m_factor, ecl_lt_factor)

    return pd_12_eff, pd_lt_eff, ecl_12m_factor, ecl_lt_factor, ecl_allocated_factor


def run_step5_pipeline(db_path: Path = DUCKDB_PATH) -> dict:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    con = duckdb.connect(str(db_path))
    try:
        gold_df = con.execute("SELECT * FROM gold.credit_card_modelling").fetchdf()
        raw_df = con.execute("SELECT PAY_0, PAY_2, PAY_3, PAY_4, PAY_5, PAY_6 FROM bronze.credit_card_raw").fetchdf()
        cso_df = con.execute("SELECT * FROM bronze.cso_lrm01_raw").fetchdf()
    finally:
        con.close()

    # Document exact PAY code 1 and code 2 counts across PAY_0..PAY_6 in Bronze (Raw N=30,000) and Gold (N=29,932)
    pay_cols_chronological = ["PAY_6", "PAY_5", "PAY_4", "PAY_3", "PAY_2", "PAY_0"]
    uci_pay_coding_audit = []
    for col in ["PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"]:
        dev_sub = gold_df[gold_df["sample_split"] == "DEV"][col]
        hold_sub = gold_df[gold_df["sample_split"] == "HOLDOUT"][col]
        uci_pay_coding_audit.append({
            "column": col,
            "raw_30k_code_le_0": int((raw_df[col] <= 0).sum()),
            "raw_30k_code_1": int((raw_df[col] == 1).sum()),
            "raw_30k_code_2": int((raw_df[col] == 2).sum()),
            "raw_30k_code_ge_3": int((raw_df[col] >= 3).sum()),
            "gold_code_le_0": int((gold_df[col] <= 0).sum()),
            "gold_code_1": int((gold_df[col] == 1).sum()),
            "gold_code_2": int((gold_df[col] == 2).sum()),
            "gold_code_ge_3": int((gold_df[col] >= 3).sum()),
            "dev_code_1": int((dev_sub == 1).sum()),
            "dev_code_2": int((dev_sub == 2).sum()),
            "holdout_code_1": int((hold_sub == 1).sum()),
            "holdout_code_2": int((hold_sub == 2).sum()),
        })
    uci_pay_coding_audit_df = pd.DataFrame(uci_pay_coding_audit)

    # Add behavioural segment, EAD (baseline CCF=50%), and 3-state IFRS 9 delinquency state/stage
    gold_df["behavioural_segment"] = assign_behavioural_segment(gold_df)
    gold_df["drawn_balance"] = np.maximum(gold_df["BILL_AMT1"].values.astype(float), 0.0)
    gold_df["undrawn_limit"] = np.maximum(gold_df["LIMIT_BAL"].values.astype(float) - gold_df["drawn_balance"].values, 0.0)
    gold_df["ead_ccf50"] = compute_ead(gold_df["BILL_AMT1"].values, gold_df["LIMIT_BAL"].values, ccf=BASELINE_CCF)
    gold_df["dpd_state_m1"] = map_pay_to_ifrs9_state_3state(gold_df["PAY_0"].values)
    gold_df["ifrs9_stage"] = gold_df["dpd_state_m1"] + 1  # State 0 -> Stage 1, State 1 -> Stage 2, State 2 -> Stage 3
    gold_df["ifrs9_stage_label"] = np.where(
        gold_df["ifrs9_stage"] == 1,
        "Stage 1 (Performing: Current <= 0 / 0 DPD)",
        np.where(
            gold_df["ifrs9_stage"] == 2,
            "Stage 2 (SICR Backstop: Delinquent 30-89 DPD / PAY_0 in {1,2})",
            "Stage 3 (Credit-Impaired: 90+ DPD / PAY_0 >= 3)",
        ),
    )

    dev_df = gold_df[gold_df["sample_split"] == "DEV"].copy().reset_index(drop=True)
    hold_df = gold_df[gold_df["sample_split"] == "HOLDOUT"].copy().reset_index(drop=True)

    # =========================================================================
    # PART 1: RISK SEGMENTATION (Grade Profiles G1-G6 & 5 Behavioural Segments)
    # =========================================================================
    grade_profile_rows = []
    for sp_label, sdf in [("DEV", dev_df), ("HOLDOUT", hold_df), ("ALL_SILVER_GOLD", gold_df)]:
        for g_lbl in GRADE_ORDER:
            g = sdf[sdf["champion_grade"] == g_lbl]
            n_g = len(g)
            d_g = int(g["default_flag"].sum())
            grade_profile_rows.append({
                "sample_split": sp_label,
                "rating_grade": GRADE_SHORT[g_lbl],
                "grade_label": g_lbl,
                "score_band": GRADE_BAND_STR[GRADE_SHORT[g_lbl]],
                "count": int(n_g),
                "portfolio_share_pct": round(100.0 * n_g / len(sdf), 4),
                "defaults": int(d_g),
                "observed_default_rate": round(d_g / n_g, 6),
                "observed_default_rate_pct": round(100.0 * d_g / n_g, 2),
                "mean_champion_pd": round(float(g["champion_pd"].mean()), 6),
                "mean_champion_score": round(float(g["champion_score"].mean()), 2),
                "pay0_inactive_m2_pct": round(100.0 * float((g["PAY_0"] == -2).mean()), 2),
                "pay0_paid_duly_m1_pct": round(100.0 * float((g["PAY_0"] == -1).mean()), 2),
                "pay0_revolving_0_pct": round(100.0 * float((g["PAY_0"] == 0).mean()), 2),
                "pay0_dpd1_pct": round(100.0 * float((g["PAY_0"] == 1).mean()), 2),
                "pay0_dpd2plus_pct": round(100.0 * float((g["PAY_0"] >= 2).mean()), 2),
                "mean_pay0_dpd": round(float(np.maximum(g["PAY_0"], 0).mean()), 4),
                "mean_months_delinquent_6m": round(float(g["MONTHS_DELINQUENT_6M"].mean()), 4),
                "mean_months_revolving_6m": round(float(g["MONTHS_REVOLVING_6M"].mean()), 4),
                "mean_util_ratio_6m": round(float(g["MEAN_UTIL_RATIO_6M"].mean()), 4),
                "mean_pay_amt_6m_ntd": round(float(g["MEAN_PAY_AMT_6M"].mean()), 2),
                "mean_limit_bal_ntd": round(float(g["LIMIT_BAL"].mean()), 2),
                "mean_bill_trend_6m_ratio": round(float(g["BILL_TREND_6M_RATIO"].mean()), 4),
                "mean_ead_ccf50_ntd": round(float(g["ead_ccf50"].mean()), 2),
            })
    grade_profiles_df = pd.DataFrame(grade_profile_rows)

    segment_order = [
        "S1: Inactive / Dormant Transactors",
        "S2: Active Full-Payer Transactors",
        "S3: Seasoned Performing Revolvers",
        "S4: Episodic / Recently Delinquent (1-2M)",
        "S5: Chronic / Severe Delinquent (3+M or 90+ DPD)",
    ]
    segment_profile_rows = []
    for sp_label, sdf in [("DEV", dev_df), ("HOLDOUT", hold_df), ("ALL_SILVER_GOLD", gold_df)]:
        for seg_lbl in segment_order:
            g = sdf[sdf["behavioural_segment"] == seg_lbl]
            n_g = len(g)
            d_g = int(g["default_flag"].sum())
            segment_profile_rows.append({
                "sample_split": sp_label,
                "segment": seg_lbl,
                "count": int(n_g),
                "portfolio_share_pct": round(100.0 * n_g / len(sdf), 4),
                "defaults": int(d_g),
                "observed_default_rate": round(d_g / n_g, 6),
                "observed_default_rate_pct": round(100.0 * d_g / n_g, 2),
                "mean_champion_pd": round(float(g["champion_pd"].mean()), 6),
                "mean_challenger_platt_pd": round(float(g["challenger_pd_platt"].mean()), 6),
                "mean_champion_score": round(float(g["champion_score"].mean()), 2),
                "mean_months_delinquent_6m": round(float(g["MONTHS_DELINQUENT_6M"].mean()), 4),
                "mean_months_revolving_6m": round(float(g["MONTHS_REVOLVING_6M"].mean()), 4),
                "mean_months_inactive_6m": round(float(g["MONTHS_INACTIVE_6M"].mean()), 4),
                "mean_util_ratio_6m": round(float(g["MEAN_UTIL_RATIO_6M"].mean()), 4),
                "mean_pay_amt_6m_ntd": round(float(g["MEAN_PAY_AMT_6M"].mean()), 2),
                "mean_limit_bal_ntd": round(float(g["LIMIT_BAL"].mean()), 2),
                "mean_ead_ccf50_ntd": round(float(g["ead_ccf50"].mean()), 2),
            })
    segment_profiles_df = pd.DataFrame(segment_profile_rows)

    # =========================================================================
    # PART 2: ILLUSTRATIVE BASEL RETAIL CAPITAL (QRRE ASRF, R = 0.04)
    # =========================================================================
    # Derive DEV empirical grade default rate (1M-target used as illustrative 12M PD proxy)
    # and Margin of Conservatism (MoC):
    # MoC_g = Jeffreys_95_Upper(DEV_g) - ODR_DEV_g, so Regulatory PD_IRB,g = ODR_DEV_g + MoC_g
    grade_pd_schedule = {}
    for g_lbl in GRADE_ORDER:
        g_dev = dev_df[dev_df["champion_grade"] == g_lbl]
        n_d = len(g_dev)
        d_d = int(g_dev["default_flag"].sum())
        odr_dev = d_d / n_d
        _, j95_up = stats.beta.ppf([0.025, 0.975], d_d + 0.5, n_d - d_d + 0.5)
        moc_g = float(j95_up - odr_dev)
        reg_pd_g = max(float(j95_up), PD_REGULATORY_FLOOR)
        grade_pd_schedule[g_lbl] = {
            "grade_short": GRADE_SHORT[g_lbl],
            "dev_count": n_d,
            "dev_defaults": d_d,
            "dev_empirical_pd": odr_dev,
            "moc_add_on": moc_g,
            "regulatory_pd_with_moc": reg_pd_g,
        }

    # Map Regulatory PD proxy (with MoC) and DEV Empirical PD onto gold_df
    gold_df["irb_grade_pd_emp"] = gold_df["champion_grade"].map(lambda x: grade_pd_schedule[x]["dev_empirical_pd"])
    gold_df["irb_grade_moc"] = gold_df["champion_grade"].map(lambda x: grade_pd_schedule[x]["moc_add_on"])
    gold_df["irb_grade_pd_reg"] = gold_df["champion_grade"].map(lambda x: grade_pd_schedule[x]["regulatory_pd_with_moc"])

    gold_df["irb_k_reg"] = basel_asrf_k(gold_df["irb_grade_pd_reg"].values, BASELINE_LGD, QRRE_CORRELATION_R)
    gold_df["irb_rwa_reg_ntd"] = 12.5 * gold_df["irb_k_reg"] * gold_df["ead_ccf50"]
    gold_df["irb_cap8_reg_ntd"] = 0.08 * gold_df["irb_rwa_reg_ntd"]
    gold_df["irb_el_reg_ntd"] = gold_df["irb_grade_pd_reg"] * BASELINE_LGD * gold_df["ead_ccf50"]

    dev_df = gold_df[gold_df["sample_split"] == "DEV"].copy().reset_index(drop=True)
    hold_df = gold_df[gold_df["sample_split"] == "HOLDOUT"].copy().reset_index(drop=True)

    capital_by_grade_rows = []
    for sp_label, sdf in [("DEV", dev_df), ("HOLDOUT", hold_df), ("ALL_SILVER_GOLD", gold_df)]:
        for g_lbl in GRADE_ORDER:
            g = sdf[sdf["champion_grade"] == g_lbl]
            info = grade_pd_schedule[g_lbl]
            pd_reg = info["regulatory_pd_with_moc"]
            pd_emp = info["dev_empirical_pd"]
            cdr_reg = float(basel_asrf_cdr(pd_reg, r_corr=QRRE_CORRELATION_R))
            k_reg = float(basel_asrf_k(pd_reg, BASELINE_LGD, QRRE_CORRELATION_R))
            k_emp = float(basel_asrf_k(pd_emp, BASELINE_LGD, QRRE_CORRELATION_R))
            total_loss_rate_reg = float(basel_total_stressed_loss_rate(pd_reg, BASELINE_LGD, QRRE_CORRELATION_R))
            drawn_sum = float(g["drawn_balance"].sum())
            undrawn_sum = float(g["undrawn_limit"].sum())
            ead_sum = float(g["ead_ccf50"].sum())
            rwa_reg_sum = 12.5 * k_reg * ead_sum
            cap8_reg_sum = 0.08 * rwa_reg_sum
            el_reg_sum = pd_reg * BASELINE_LGD * ead_sum
            total_stressed_loss_sum = total_loss_rate_reg * ead_sum
            rwa_emp_sum = 12.5 * k_emp * ead_sum

            capital_by_grade_rows.append({
                "sample_split": sp_label,
                "rating_grade": info["grade_short"],
                "grade_label": g_lbl,
                "account_count": int(len(g)),
                "dev_empirical_pd_1m_as_12m_proxy": round(pd_emp, 6),
                "moc_add_on": round(info["moc_add_on"], 6),
                "regulatory_pd_with_moc_proxy": round(pd_reg, 6),
                "pd_floor_crr3_qrre_revolver": PD_REGULATORY_FLOOR,
                "lgd_assumption": BASELINE_LGD,
                "lgd_crr3_input_floor": LGD_CRR3_QRRE_INPUT_FLOOR,
                "correlation_r": QRRE_CORRELATION_R,
                "ccf_assumption": BASELINE_CCF,
                "drawn_balance_ntd_m": round(drawn_sum / 1e6, 4),
                "undrawn_limit_ntd_m": round(undrawn_sum / 1e6, 4),
                "ead_ntd_m": round(ead_sum / 1e6, 4),
                "conditional_default_rate_999": round(cdr_reg, 6),
                "total_stressed_loss_rate_el_plus_ul": round(total_loss_rate_reg, 6),
                "capital_factor_k_reg": round(k_reg, 6),
                "rwa_density_reg_pct": round(12.5 * k_reg * 100.0, 4),
                "rwa_reg_ntd_m": round(rwa_reg_sum / 1e6, 4),
                "capital_8pct_reg_ntd_m": round(cap8_reg_sum / 1e6, 4),
                "expected_loss_reg_ntd_m": round(el_reg_sum / 1e6, 4),
                "total_stressed_loss_el_plus_ul_ntd_m": round(total_stressed_loss_sum / 1e6, 4),
                "capital_factor_k_unadjusted_emp": round(k_emp, 6),
                "rwa_unadjusted_emp_ntd_m": round(rwa_emp_sum / 1e6, 4),
            })
    capital_by_grade_df = pd.DataFrame(capital_by_grade_rows)

    # Monotonicity Verification (matching Step 6 test requirements):
    # 1. K_UL monotonic in PD below the ~38.98% Vasicek peak (G1..G5 and fine grid on [0.0010, 0.389795])
    # 2. EL + UL = LGD * CDR(PD) monotonic in PD everywhere (G1..G6 and fine grid on [0.0010, 0.9990])
    # 3. K_UL and RWA monotonic in LGD everywhere ([0.50, 0.65, 0.75, 0.85, 0.95])
    hold_cap_sub = capital_by_grade_df[capital_by_grade_df["sample_split"] == "HOLDOUT"].reset_index(drop=True)
    el_plus_ul_diffs_grades = np.diff(hold_cap_sub["total_stressed_loss_rate_el_plus_ul"].values)
    k_diffs_below_peak_grades = np.diff(hold_cap_sub["capital_factor_k_reg"].values[:5])  # G1..G5 all < 38.98%

    pd_grid_below_peak = np.linspace(PD_REGULATORY_FLOOR, VASICEK_UL_PEAK_PD, 500)
    k_grid_below_peak = basel_asrf_k(pd_grid_below_peak, BASELINE_LGD, QRRE_CORRELATION_R)
    pd_grid_full = np.linspace(PD_REGULATORY_FLOOR, 0.9990, 1000)
    el_plus_ul_grid_full = basel_total_stressed_loss_rate(pd_grid_full, BASELINE_LGD, QRRE_CORRELATION_R)

    assert np.all(el_plus_ul_diffs_grades > 0) and np.all(np.diff(el_plus_ul_grid_full) > 0), (
        "Total Stressed Loss (EL + UL) must be strictly monotonic in PD everywhere!"
    )
    assert np.all(k_diffs_below_peak_grades > 0) and np.all(np.diff(k_grid_below_peak) > 0), (
        "Basel ASRF K_UL must be strictly monotonic in PD below the ~38.98% Vasicek peak!"
    )

    # Sensitivity Grid for LGD (starting at CRR3 50% input floor) x CCF on HOLDOUT and ALL_SILVER_GOLD
    lgd_grid_values = [0.50, 0.65, 0.75, 0.85, 0.95]
    ccf_grid_values = [0.25, 0.50, 0.75, 1.00]
    cap_sensitivity_rows = []
    for sp_label, sdf in [("HOLDOUT", hold_df), ("ALL_SILVER_GOLD", gold_df)]:
        for lgd_test in lgd_grid_values:
            for ccf_test in ccf_grid_values:
                ead_vec = compute_ead(sdf["BILL_AMT1"].values, sdf["LIMIT_BAL"].values, ccf=ccf_test)
                k_vec = basel_asrf_k(sdf["irb_grade_pd_reg"].values, lgd_test, QRRE_CORRELATION_R)
                rwa_vec = 12.5 * k_vec * ead_vec
                cap8_vec = 0.08 * rwa_vec
                el_vec = sdf["irb_grade_pd_reg"].values * lgd_test * ead_vec
                cap_sensitivity_rows.append({
                    "sample_split": sp_label,
                    "lgd_assumption": lgd_test,
                    "ccf_assumption": ccf_test,
                    "total_ead_ntd_m": round(float(ead_vec.sum()) / 1e6, 4),
                    "total_rwa_ntd_m": round(float(rwa_vec.sum()) / 1e6, 4),
                    "portfolio_rwa_density_pct": round(100.0 * float(rwa_vec.sum() / ead_vec.sum()), 4),
                    "total_capital_8pct_ntd_m": round(float(cap8_vec.sum()) / 1e6, 4),
                    "total_expected_loss_ntd_m": round(float(el_vec.sum()) / 1e6, 4),
                    "total_stressed_loss_el_plus_ul_ntd_m": round(float((cap8_vec + el_vec).sum()) / 1e6, 4),
                })
    cap_sensitivity_df = pd.DataFrame(cap_sensitivity_rows)

    for ccf_test in ccf_grid_values:
        sub_s = cap_sensitivity_df[
            (cap_sensitivity_df["sample_split"] == "HOLDOUT") & (cap_sensitivity_df["ccf_assumption"] == ccf_test)
        ].sort_values("lgd_assumption")
        assert np.all(np.diff(sub_s["total_rwa_ntd_m"].values) > 0), "RWA must increase strictly with LGD!"
        assert np.all(np.diff(sub_s["total_stressed_loss_el_plus_ul_ntd_m"].values) > 0), "EL+UL must increase strictly with LGD!"

    # =========================================================================
    # PART 3: IFRS 9 STAGING ILLUSTRATION (3-State Markov Matrix & Illustrative 12M/LT ECL)
    # =========================================================================
    state_labels_3state = {
        0: "State 0: Current / 0 DPD (PAY_x <= 0 -> Stage 1)",
        1: "State 1: Delinquent 30-89 DPD (PAY_x in {1, 2} merged -> Stage 2)",
        2: "State 2: Default / 90+ DPD (PAY_x >= 3 -> Stage 3)",
    }

    def build_markov_matrix_3state(sdf: pd.DataFrame, sample_label: str) -> tuple[np.ndarray, pd.DataFrame]:
        counts_mat = np.zeros((3, 3), dtype=int)
        for idx_m in range(len(pay_cols_chronological) - 1):
            c_from = pay_cols_chronological[idx_m]
            c_to = pay_cols_chronological[idx_m + 1]
            s_from = map_pay_to_ifrs9_state_3state(sdf[c_from].values)
            s_to = map_pay_to_ifrs9_state_3state(sdf[c_to].values)
            for i in range(3):
                mask_i = s_from == i
                for j in range(3):
                    counts_mat[i, j] += int(np.sum(mask_i & (s_to == j)))

        row_sums = counts_mat.sum(axis=1, keepdims=True)
        prob_mat = counts_mat / np.maximum(row_sums, 1)

        # Verify each row sums to 1.0
        assert np.allclose(prob_mat.sum(axis=1), 1.0, atol=1e-12), "Markov rows do not sum to 1!"

        cum_curves = compute_markov_cumulative_default_curves(prob_mat, horizon_months=LIFETIME_HORIZON_MONTHS)

        rows = []
        for i in range(3):
            r_sum = float(prob_mat[i, :].sum())
            rows.append({
                "sample_split": sample_label,
                "from_state_code": i,
                "from_state_label": state_labels_3state[i],
                "transition_count_total": int(counts_mat[i, :].sum()),
                "count_to_state0_current": int(counts_mat[i, 0]),
                "count_to_state1_30_89dpd": int(counts_mat[i, 1]),
                "count_to_state2_90plus_dpd": int(counts_mat[i, 2]),
                "prob_to_state0_current": round(float(prob_mat[i, 0]), 6),
                "prob_to_state1_30_89dpd": round(float(prob_mat[i, 1]), 6),
                "prob_to_state2_90plus_dpd": round(float(prob_mat[i, 2]), 6),
                "row_sum": round(r_sum, 6),
                "absorbing_cum_default_12m": round(float(cum_curves[i, 12]), 6),
                "absorbing_cum_default_24m": round(float(cum_curves[i, 24]), 6),
                "absorbing_cum_default_36m": round(float(cum_curves[i, 36]), 6),
            })
        return prob_mat, cum_curves, pd.DataFrame(rows)

    markov_prob_dev, markov_cum_dev, markov_df_dev = build_markov_matrix_3state(dev_df, "DEV")
    markov_prob_hold, markov_cum_hold, markov_df_hold = build_markov_matrix_3state(hold_df, "HOLDOUT")
    markov_prob_all, markov_cum_all, markov_df_all = build_markov_matrix_3state(gold_df, "ALL_SILVER_GOLD")
    markov_all_df = pd.concat([markov_df_dev, markov_df_hold, markov_df_all], ignore_index=True)

    # Compute Illustrative IFRS 9 12M-Proxy ECL, 36M-Lifetime ECL, and Stage-Allocated ECL
    # using the 3-state DEV absorbing Markov cumulative default curves
    pd12_eff, pdlt_eff, ecl12_fac, ecllt_fac, ecl_alloc_fac = compute_discounted_ecl_per_unit_ead_markov3(
        gold_df["irb_grade_pd_reg"].values,
        gold_df["ifrs9_stage"].values,
        markov_cum_def_curves=markov_cum_dev,
        lgd=BASELINE_LGD,
        r_monthly=MONTHLY_EIR,
        horizon_months=LIFETIME_HORIZON_MONTHS,
    )
    gold_df["ifrs9_pd_12m"] = pd12_eff
    gold_df["ifrs9_pd_lifetime_36m"] = pdlt_eff
    gold_df["ifrs9_ecl_12m_ntd"] = ecl12_fac * gold_df["ead_ccf50"]
    gold_df["ifrs9_ecl_lifetime_ntd"] = ecllt_fac * gold_df["ead_ccf50"]
    gold_df["ifrs9_ecl_allocated_ntd"] = ecl_alloc_fac * gold_df["ead_ccf50"]

    dev_df = gold_df[gold_df["sample_split"] == "DEV"].copy().reset_index(drop=True)
    hold_df = gold_df[gold_df["sample_split"] == "HOLDOUT"].copy().reset_index(drop=True)

    ifrs9_stage_rows = []
    for sp_label, sdf in [("DEV", dev_df), ("HOLDOUT", hold_df), ("ALL_SILVER_GOLD", gold_df)]:
        for stg_num in [1, 2, 3]:
            g = sdf[sdf["ifrs9_stage"] == stg_num]
            ead_s = float(g["ead_ccf50"].sum())
            ecl12_s = float(g["ifrs9_ecl_12m_ntd"].sum())
            ecllt_s = float(g["ifrs9_ecl_lifetime_ntd"].sum())
            ecl_alloc_s = float(g["ifrs9_ecl_allocated_ntd"].sum())
            ifrs9_stage_rows.append({
                "sample_split": sp_label,
                "ifrs9_stage": stg_num,
                "stage_label": str(g["ifrs9_stage_label"].iloc[0]),
                "account_count": int(len(g)),
                "account_share_pct": round(100.0 * len(g) / len(sdf), 4),
                "observed_default_rate_next_month": round(float(g["default_flag"].mean()), 6),
                "mean_12m_pd_proxy": round(float(g["ifrs9_pd_12m"].mean()), 6),
                "mean_lifetime_36m_pd_illustrative": round(float(g["ifrs9_pd_lifetime_36m"].mean()), 6),
                "drawn_balance_ntd_m": round(float(g["drawn_balance"].sum()) / 1e6, 4),
                "ead_ccf50_ntd_m": round(ead_s / 1e6, 4),
                "ead_share_pct": round(100.0 * ead_s / float(sdf["ead_ccf50"].sum()), 4),
                "ecl_if_12m_ntd_m": round(ecl12_s / 1e6, 4),
                "ecl_if_lifetime_36m_ntd_m": round(ecllt_s / 1e6, 4),
                "ifrs9_allocated_ecl_ntd_m": round(ecl_alloc_s / 1e6, 4),
                "stage2_lifetime_uplift_ntd_m": round((ecl_alloc_s - ecl12_s) / 1e6, 4),
                "coverage_ratio_allocated_ecl_to_ead_pct": round(100.0 * ecl_alloc_s / max(ead_s, 1.0), 4),
            })
        ead_tot = float(sdf["ead_ccf50"].sum())
        ecl12_tot = float(sdf["ifrs9_ecl_12m_ntd"].sum())
        ecllt_tot = float(sdf["ifrs9_ecl_lifetime_ntd"].sum())
        ecl_alloc_tot = float(sdf["ifrs9_ecl_allocated_ntd"].sum())
        ifrs9_stage_rows.append({
            "sample_split": sp_label,
            "ifrs9_stage": 0,
            "stage_label": "TOTAL PORTFOLIO (Stages 1 + 2 + 3)",
            "account_count": int(len(sdf)),
            "account_share_pct": 100.0,
            "observed_default_rate_next_month": round(float(sdf["default_flag"].mean()), 6),
            "mean_12m_pd_proxy": round(float(sdf["ifrs9_pd_12m"].mean()), 6),
            "mean_lifetime_36m_pd_illustrative": round(float(sdf["ifrs9_pd_lifetime_36m"].mean()), 6),
            "drawn_balance_ntd_m": round(float(sdf["drawn_balance"].sum()) / 1e6, 4),
            "ead_ccf50_ntd_m": round(ead_tot / 1e6, 4),
            "ead_share_pct": 100.0,
            "ecl_if_12m_ntd_m": round(ecl12_tot / 1e6, 4),
            "ecl_if_lifetime_36m_ntd_m": round(ecllt_tot / 1e6, 4),
            "ifrs9_allocated_ecl_ntd_m": round(ecl_alloc_tot / 1e6, 4),
            "stage2_lifetime_uplift_ntd_m": round((ecl_alloc_tot - ecl12_tot) / 1e6, 4),
            "coverage_ratio_allocated_ecl_to_ead_pct": round(100.0 * ecl_alloc_tot / max(ead_tot, 1.0), 4),
        })
    ifrs9_stage_df = pd.DataFrame(ifrs9_stage_rows)

    # =========================================================================
    # PART 4: IRISH MACRO STRESS SENSITIVITY VIA CSO LRM01
    # =========================================================================
    cso_total = (
        cso_df[
            (cso_df["STATISTIC"] == "LRM01")
            & (cso_df["Age Group"] == "All ages")
            & (cso_df["Sex"] == "Both sexes")
            & (cso_df["Social Welfare Scheme"] == "All live register schemes")
        ]
        .sort_values("TLIST(M1)")
        .reset_index(drop=True)
    )
    cso_total["value_num"] = pd.to_numeric(cso_total["VALUE"], errors="coerce")
    cso_total["yoy_change"] = (cso_total["value_num"] - cso_total["value_num"].shift(12)) / cso_total["value_num"].shift(12)
    yoy_valid = cso_total.dropna(subset=["yoy_change"]).reset_index(drop=True)

    yoy_p50 = float(yoy_valid["yoy_change"].quantile(0.50))
    yoy_p75 = float(yoy_valid["yoy_change"].quantile(0.75))
    yoy_p90 = float(yoy_valid["yoy_change"].quantile(0.90))
    yoy_p95 = float(yoy_valid["yoy_change"].quantile(0.95))
    yoy_p99 = float(yoy_valid["yoy_change"].quantile(0.99))
    max_idx = int(yoy_valid["yoy_change"].idxmax())
    yoy_max_row = yoy_valid.loc[max_idx]
    yoy_max = float(yoy_max_row["yoy_change"])
    yoy_max_month = str(yoy_max_row["Month"])
    yoy_max_level = float(yoy_max_row["value_num"])

    scenarios = [
        {
            "scenario_id": "S0_BASELINE",
            "scenario_name": "Baseline (No Macro Shock)",
            "cso_benchmark": "Neutral YoY Change (0.00%)",
            "lrm01_yoy_shock_pct": 0.0,
            "elasticity_epsilon": MACRO_ELASTICITY_EPSILON,
            "pd_multiplier": 1.0000,
        },
        {
            "scenario_id": "S1_MILD_P90",
            "scenario_name": "Mild Adverse Stress (CSO LRM01 90th Percentile YoY)",
            "cso_benchmark": f"90th Percentile YoY (+{yoy_p90*100:.2f}%)",
            "lrm01_yoy_shock_pct": round(yoy_p90 * 100.0, 4),
            "elasticity_epsilon": MACRO_ELASTICITY_EPSILON,
            "pd_multiplier": round(1.0 + MACRO_ELASTICITY_EPSILON * yoy_p90, 6),
        },
        {
            "scenario_id": "S2_SEVERE_MAX",
            "scenario_name": f"Severe Crisis Stress (CSO LRM01 Historical Max YoY: {yoy_max_month})",
            "cso_benchmark": f"Historical Maximum YoY ({yoy_max_month}: +{yoy_max*100:.2f}%)",
            "lrm01_yoy_shock_pct": round(yoy_max * 100.0, 4),
            "elasticity_epsilon": MACRO_ELASTICITY_EPSILON,
            "pd_multiplier": round(1.0 + MACRO_ELASTICITY_EPSILON * yoy_max, 6),
        },
    ]

    stress_summary_rows = []
    stress_grade_dist_rows = []

    for sp_label, sdf in [("HOLDOUT", hold_df), ("ALL_SILVER_GOLD", gold_df)]:
        base_rwa_tot = 0.0
        base_ecl_tot = 0.0
        for sc in scenarios:
            mult = float(sc["pd_multiplier"])
            p_champ_stressed = np.minimum(sdf["champion_pd"].values * mult, 0.9999)
            logit_stressed = np.log(p_champ_stressed / (1.0 - p_champ_stressed))
            score_stressed = OFFSET - FACTOR * logit_stressed
            grades_stressed = assign_rating_grade_from_score(score_stressed)

            migrated_reg_pd = np.array([grade_pd_schedule[g]["regulatory_pd_with_moc"] for g in grades_stressed])
            stressed_reg_pd = np.minimum(sdf["irb_grade_pd_reg"].values * mult, 0.9999)

            ead_vec = sdf["ead_ccf50"].values
            k_mig = basel_asrf_k(migrated_reg_pd, BASELINE_LGD, QRRE_CORRELATION_R)
            rwa_mig = float(np.sum(12.5 * k_mig * ead_vec))

            k_str = basel_asrf_k(stressed_reg_pd, BASELINE_LGD, QRRE_CORRELATION_R)
            rwa_str = float(np.sum(12.5 * k_str * ead_vec))
            cap8_str = 0.08 * rwa_str
            el_irb_str = float(np.sum(stressed_reg_pd * BASELINE_LGD * ead_vec))
            total_stressed_loss_irb = cap8_str + el_irb_str

            # IFRS 9 ECL under Stressed PDs using the 3-state DEV Markov cumulative default curves
            _, _, _, _, ecl_fac_fixed_stage = compute_discounted_ecl_per_unit_ead_markov3(
                stressed_reg_pd,
                sdf["ifrs9_stage"].values,
                markov_cum_def_curves=markov_cum_dev,
                lgd=BASELINE_LGD,
                r_monthly=MONTHLY_EIR,
                horizon_months=LIFETIME_HORIZON_MONTHS,
            )
            ecl_fixed_stage = float(np.sum(ecl_fac_fixed_stage * ead_vec))

            # Stress SICR overlay: Stage 1 accounts that downgrade by >= 1 grade into G4/G5/G6 move to Stage 2
            downgraded = np.array([
                GRADE_ORDER.index(g_new) > GRADE_ORDER.index(g_old)
                for g_new, g_old in zip(grades_stressed, sdf["champion_grade"].values)
            ])
            high_risk_after_stress = np.array([GRADE_SHORT[g_new] in ("G4", "G5", "G6") for g_new in grades_stressed])
            stage_stressed = sdf["ifrs9_stage"].values.copy()
            stage_stressed[(stage_stressed == 1) & downgraded & high_risk_after_stress] = 2

            _, _, _, _, ecl_fac_with_sicr = compute_discounted_ecl_per_unit_ead_markov3(
                stressed_reg_pd,
                stage_stressed,
                markov_cum_def_curves=markov_cum_dev,
                lgd=BASELINE_LGD,
                r_monthly=MONTHLY_EIR,
                horizon_months=LIFETIME_HORIZON_MONTHS,
            )
            ecl_with_sicr = float(np.sum(ecl_fac_with_sicr * ead_vec))

            if sc["scenario_id"] == "S0_BASELINE":
                base_rwa_tot = rwa_str
                base_ecl_tot = ecl_fixed_stage

            n_downgraded = int(np.sum(downgraded))
            pct_downgraded = 100.0 * n_downgraded / len(sdf)
            g5_g6_share = 100.0 * float(np.mean([GRADE_SHORT[g] in ("G5", "G6") for g in grades_stressed]))
            stage2_share_sicr = 100.0 * float(np.mean(stage_stressed == 2))

            stress_summary_rows.append({
                "sample_split": sp_label,
                "scenario_id": sc["scenario_id"],
                "scenario_name": sc["scenario_name"],
                "lrm01_yoy_shock_pct": sc["lrm01_yoy_shock_pct"],
                "elasticity_epsilon": sc["elasticity_epsilon"],
                "pd_multiplier": mult,
                "mean_stressed_regulatory_pd_proxy": round(float(np.mean(stressed_reg_pd)), 6),
                "mean_stressed_score": round(float(np.mean(score_stressed)), 2),
                "accounts_downgraded_count": n_downgraded,
                "accounts_downgraded_pct": round(pct_downgraded, 2),
                "subinvestment_g5_g6_share_pct": round(g5_g6_share, 2),
                "total_ead_ntd_m": round(float(ead_vec.sum()) / 1e6, 4),
                "rwa_grade_migration_only_ntd_m": round(rwa_mig / 1e6, 4),
                "rwa_grade_migration_uplift_pct": round(100.0 * (rwa_mig - base_rwa_tot) / max(base_rwa_tot, 1.0), 2),
                "rwa_full_stressed_pd_ntd_m": round(rwa_str / 1e6, 4),
                "rwa_uplift_vs_baseline_pct": round(100.0 * (rwa_str - base_rwa_tot) / max(base_rwa_tot, 1.0), 2),
                "capital_8pct_stressed_ntd_m": round(cap8_str / 1e6, 4),
                "irb_expected_loss_stressed_ntd_m": round(el_irb_str / 1e6, 4),
                "irb_total_stressed_loss_el_plus_ul_ntd_m": round(total_stressed_loss_irb / 1e6, 4),
                "ifrs9_ecl_fixed_stage_ntd_m": round(ecl_fixed_stage / 1e6, 4),
                "ifrs9_ecl_fixed_stage_uplift_pct": round(100.0 * (ecl_fixed_stage - base_ecl_tot) / max(base_ecl_tot, 1.0), 2),
                "ifrs9_sicr_stage2_share_pct": round(stage2_share_sicr, 2),
                "ifrs9_ecl_with_grade_downgrade_sicr_ntd_m": round(ecl_with_sicr / 1e6, 4),
                "ifrs9_ecl_with_sicr_uplift_pct": round(100.0 * (ecl_with_sicr - base_ecl_tot) / max(base_ecl_tot, 1.0), 2),
            })

            for g_lbl in GRADE_ORDER:
                cnt_g = int(sum(1 for x in grades_stressed if x == g_lbl))
                stress_grade_dist_rows.append({
                    "sample_split": sp_label,
                    "scenario_id": sc["scenario_id"],
                    "scenario_name": sc["scenario_name"],
                    "pd_multiplier": mult,
                    "rating_grade": GRADE_SHORT[g_lbl],
                    "grade_label": g_lbl,
                    "account_count": cnt_g,
                    "share_pct": round(100.0 * cnt_g / len(sdf), 4),
                })

    stress_summary_df = pd.DataFrame(stress_summary_rows)
    stress_grade_dist_df = pd.DataFrame(stress_grade_dist_rows)

    # Persist all Step 5 tables to DuckDB Gold schema & CSVs
    con = duckdb.connect(str(db_path))
    try:
        con.register("v_gold_step5", gold_df)
        con.execute("CREATE OR REPLACE TABLE gold.credit_card_modelling AS SELECT * FROM v_gold_step5;")

        con.register("v_gprof", grade_profiles_df)
        con.execute("CREATE OR REPLACE TABLE gold.segmentation_grade_profiles AS SELECT * FROM v_gprof;")

        con.register("v_sprof", segment_profiles_df)
        con.execute("CREATE OR REPLACE TABLE gold.segmentation_behavioural_profiles AS SELECT * FROM v_sprof;")

        con.register("v_cap_grade", capital_by_grade_df)
        con.execute("CREATE OR REPLACE TABLE gold.basel_capital_by_grade AS SELECT * FROM v_cap_grade;")

        con.register("v_cap_sens", cap_sensitivity_df)
        con.execute("CREATE OR REPLACE TABLE gold.basel_capital_sensitivity AS SELECT * FROM v_cap_sens;")

        con.register("v_markov", markov_all_df)
        con.execute("CREATE OR REPLACE TABLE gold.ifrs9_markov_matrix AS SELECT * FROM v_markov;")

        con.register("v_ifrs9", ifrs9_stage_df)
        con.execute("CREATE OR REPLACE TABLE gold.ifrs9_stage_summary AS SELECT * FROM v_ifrs9;")

        con.register("v_stress", stress_summary_df)
        con.execute("CREATE OR REPLACE TABLE gold.stress_scenario_summary AS SELECT * FROM v_stress;")

        con.register("v_stress_g", stress_grade_dist_df)
        con.execute("CREATE OR REPLACE TABLE gold.stress_grade_migration AS SELECT * FROM v_stress_g;")

        # Direct database verification query against gold.credit_card_modelling
        db_check = con.execute("""
            SELECT
                sample_split,
                champion_grade,
                COUNT(*) AS cnt,
                SUM(default_flag) AS defs,
                AVG(default_flag) AS odr
            FROM gold.credit_card_modelling
            GROUP BY sample_split, champion_grade
        """).fetchdf()
        for _, r_chk in db_check.iterrows():
            sp_c = r_chk["sample_split"]
            gr_c = r_chk["champion_grade"]
            prof_row = grade_profiles_df[
                (grade_profiles_df["sample_split"] == sp_c) & (grade_profiles_df["grade_label"] == gr_c)
            ].iloc[0]
            assert int(prof_row["count"]) == int(r_chk["cnt"]), f"Count mismatch on {sp_c} {gr_c}"
            assert int(prof_row["defaults"]) == int(r_chk["defs"]), f"Defaults mismatch on {sp_c} {gr_c}"
            assert abs(float(prof_row["observed_default_rate"]) - float(r_chk["odr"])) < 1e-6, f"ODR mismatch on {sp_c} {gr_c}"
    finally:
        con.close()

    # Save CSV artifacts (save both step5_grade_profiles.csv and step5_grade_driver_profiles.csv so any consumer finds the exact same CSV)
    grade_profiles_df.to_csv(RESULTS_DIR / "step5_grade_profiles.csv", index=False)
    grade_profiles_df.to_csv(RESULTS_DIR / "step5_grade_driver_profiles.csv", index=False)
    segment_profiles_df.to_csv(RESULTS_DIR / "step5_behavioural_segments.csv", index=False)
    capital_by_grade_df.to_csv(RESULTS_DIR / "step5_basel_capital_by_grade.csv", index=False)
    cap_sensitivity_df.to_csv(RESULTS_DIR / "step5_basel_capital_sensitivity.csv", index=False)
    markov_all_df.to_csv(RESULTS_DIR / "step5_ifrs9_markov_matrix.csv", index=False)
    uci_pay_coding_audit_df.to_csv(RESULTS_DIR / "step5_uci_pay_coding_audit.csv", index=False)
    ifrs9_stage_df.to_csv(RESULTS_DIR / "step5_ifrs9_stage_summary.csv", index=False)
    stress_summary_df.to_csv(RESULTS_DIR / "step5_stress_scenario_summary.csv", index=False)
    stress_grade_dist_df.to_csv(RESULTS_DIR / "step5_stress_grade_migration.csv", index=False)

    step5_report = {
        "step": "Step 5 — Risk Segmentation, Illustrative Basel Capital, IFRS 9 Staging Illustration & LRM01 Stress Sensitivity",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "fundamental_target_horizon_caveat": (
            "The UCI ID 350 target ('default payment next month') is a one-month-ahead default event "
            "(October 2005 outcome following the April–September 2005 observation window), NOT an empirical "
            "12-month default horizon. Treating the scorecard's predicted probability and grade default rates "
            "as a 12-month PD proxy (1M-as-12M PD proxy) in the illustrative Basel IRB capital formula and "
            "IFRS 9 ECL calculations is an explicit methodological assumption and limitation."
        ),
        "segmentation": {
            "grade_profiles": grade_profiles_df.to_dict(orient="records"),
            "behavioural_segments": segment_profiles_df.to_dict(orient="records"),
        },
        "illustrative_basel_capital": {
            "label": "illustrative",
            "target_horizon_limitation": "Uses 1-month default target rate + Jeffreys MoC as an illustrative 12-month PD proxy (1M-as-12M PD proxy).",
            "asset_class_citation": "Qualifying Revolving Retail Exposures (QRRE) — CRR Article 154(4) / CRR3 Article 163(1) / Basel IRB CRE30.55",
            "asset_correlation_r": QRRE_CORRELATION_R,
            "pd_floor_crr3_qrre_revolver": PD_FLOOR_CRR3_QRRE_REVOLVER,
            "pd_floor_crr3_retail_standard": PD_FLOOR_CRR3_RETAIL_STD,
            "pd_floor_crr2_legacy": PD_FLOOR_CRR2_LEGACY,
            "lgd_assumption": BASELINE_LGD,
            "lgd_crr3_qrre_input_floor": LGD_CRR3_QRRE_INPUT_FLOOR,
            "lgd_source_note": (
                "85.0% (0.85) is an illustrative unsecured retail credit card downturn LGD assumption "
                "(UCI ID 350 contains no post-default recovery cash flows). Under CRR3 Art. 164(4) / Basel CRE30.56, "
                "the regulatory LGD input floor for unsecured QRRE is 50.0% (0.50), included in the sensitivity table."
            ),
            "ccf_assumption": BASELINE_CCF,
            "ead_formula": "EAD = max(BILL_AMT1, 0) + CCF * max(LIMIT_BAL - max(BILL_AMT1, 0), 0)",
            "vasicek_ul_peak_pd_for_r004": VASICEK_UL_PEAK_PD,
            "monotonicity_verified": {
                "k_ul_monotonic_in_pd_below_peak_38_98pct": True,
                "el_plus_ul_monotonic_in_pd_everywhere": True,
                "k_ul_and_rwa_monotonic_in_lgd_everywhere": True,
            },
            "capital_by_grade": capital_by_grade_df.to_dict(orient="records"),
            "lgd_ccf_sensitivity": cap_sensitivity_df.to_dict(orient="records"),
        },
        "ifrs9_staging_illustration": {
            "label": "illustrative: assumptions stated",
            "target_horizon_limitation": (
                "Year-1 PD is anchored to the grade 1M-as-12M PD proxy; 36-month lifetime PDs and ECLs "
                "are projected via the 3-state absorbing Markov matrix and are strictly illustrative."
            ),
            "uci_pay_code_1_inconsistency_note": (
                "In UCI ID 350 (N=30,000 raw / N=29,932 Gold), PAY code 1 ('payment delay for one month') "
                "appears 3,688 times in raw PAY_0 (3,678 in Gold) but only 28, 4, 2, 0, and 0 times in "
                "PAY_2, PAY_3, PAY_4, PAY_5, and PAY_6, whereas code 2 ('two months delay') appears "
                "3,919, 3,809, 3,149, 2,621, and 2,760 times in Gold PAY_2..PAY_6. To prevent a degenerate "
                "State 1 row in historical transitions, codes 1 and 2 are merged into a single 'Delinquent 30-89 DPD' "
                "state (Stage 2), forming a well-identified 3-state Markov transition matrix."
            ),
            "uci_pay_coding_audit": uci_pay_coding_audit,
            "dpd_state_mapping_3state": state_labels_3state,
            "staging_rules": {
                "stage_1": "State 0: Current / 0 DPD (PAY_0 <= 0) -> Illustrative 12-Month Proxy ECL",
                "stage_2": "State 1: Delinquent 30-89 DPD (PAY_0 in {1, 2} merged, IFRS 9 B5.5.37 30+ DPD backstop) -> Illustrative 36-Month Lifetime ECL",
                "stage_3": "State 2: Default / 90+ DPD (PAY_0 >= 3, IFRS 9 B5.5.37 90+ DPD backstop) -> Credit-Impaired ECL at PD = 100%",
                "sicr_limitation": "No origination date or origination PD exists in UCI ID 350; therefore Stage 2 allocation at reporting date relies on the 30-89 DPD backstop.",
            },
            "stated_ecl_assumptions": {
                "lgd": BASELINE_LGD,
                "annual_eir": ANNUAL_EIR,
                "monthly_eir": round(MONTHLY_EIR, 6),
                "lifetime_horizon_months": LIFETIME_HORIZON_MONTHS,
                "lifetime_pd_method": (
                    "3-state DEV absorbing Markov chain (State 2 = 90+ DPD absorbing) cumulative survival curve "
                    "scaled via log-survival exponent kappa_i = ln(1 - PD_12M_proxy_i) / ln(1 - C_s(12)) so "
                    "12-step default equals PD_12M_proxy_i and 36-step default follows the empirical Markov hazard."
                ),
            },
            "markov_transition_matrix_3state": markov_all_df.to_dict(orient="records"),
            "stage_summary": ifrs9_stage_df.to_dict(orient="records"),
        },
        "irish_lrm01_stress_sensitivity": {
            "label": "scenario sensitivity, not a calibrated model",
            "cso_filter": {
                "STATISTIC": "LRM01",
                "Age Group": "All ages",
                "Sex": "Both sexes",
                "Social Welfare Scheme": "All live register schemes",
                "total_months": int(len(cso_total)),
                "valid_yoy_months": int(len(yoy_valid)),
            },
            "historical_yoy_distribution": {
                "min": round(float(yoy_valid["yoy_change"].min()), 6),
                "p50_median": round(yoy_p50, 6),
                "p75": round(yoy_p75, 6),
                "p90": round(yoy_p90, 6),
                "p95": round(yoy_p95, 6),
                "p99": round(yoy_p99, 6),
                "max": round(yoy_max, 6),
                "max_month": yoy_max_month,
                "max_level_persons": yoy_max_level,
            },
            "stated_pass_through_assumption": f"PD_Multiplier = 1 + epsilon * YoY_Change, with illustrative elasticity epsilon = {MACRO_ELASTICITY_EPSILON:.2f}",
            "scenarios": scenarios,
            "stress_impact_summary": stress_summary_df.to_dict(orient="records"),
            "stress_grade_migration": stress_grade_dist_df.to_dict(orient="records"),
        },
    }

    with open(RESULTS_DIR / "step5_capital_ifrs9_stress_summary.json", "w", encoding="utf-8") as f:
        json.dump(step5_report, f, indent=2)

    # Programmatically generate results/Step5_Capital_IFRS9_Stress_Report.md directly from the verified DataFrames
    write_step5_markdown_report(
        grade_profiles_df=grade_profiles_df,
        segment_profiles_df=segment_profiles_df,
        capital_by_grade_df=capital_by_grade_df,
        cap_sensitivity_df=cap_sensitivity_df,
        uci_pay_coding_audit_df=uci_pay_coding_audit_df,
        markov_all_df=markov_all_df,
        ifrs9_stage_df=ifrs9_stage_df,
        stress_summary_df=stress_summary_df,
        stress_grade_dist_df=stress_grade_dist_df,
        step5_report=step5_report,
    )

    return step5_report


def write_step5_markdown_report(
    grade_profiles_df: pd.DataFrame,
    segment_profiles_df: pd.DataFrame,
    capital_by_grade_df: pd.DataFrame,
    cap_sensitivity_df: pd.DataFrame,
    uci_pay_coding_audit_df: pd.DataFrame,
    markov_all_df: pd.DataFrame,
    ifrs9_stage_df: pd.DataFrame,
    stress_summary_df: pd.DataFrame,
    stress_grade_dist_df: pd.DataFrame,
    step5_report: dict,
) -> None:
    """Generate Step5_Capital_IFRS9_Stress_Report.md directly from database-verified DataFrames."""
    lines = []
    lines.append("# ScoreGuard-IRB — Step 5: Segmentation, Illustrative Basel Capital, IFRS 9 Staging & CSO LRM01 Stress Sensitivity\n")
    lines.append("**Author:** Sohum Vivek Dhole (`github.com/sohumdhole` | MSc in Business Analytics, First Class Honours, Dublin Business School)  ")
    lines.append("**Target Role:** AIB Quantitative Risk Analyst (Risk Analytics, 2nd Line of Defence)  ")
    lines.append("**Primary Portfolio Data:** UCI Default of Credit Card Clients (ID 350, CC BY 4.0) — $N = 29,932$ Gold accounts (`DEV` $n = 20,952$; `HOLDOUT` $n = 8,980$)  ")
    lines.append("**Macro Stress Data:** CSO Ireland `LRM01` (Persons on Live Register, 1967M01–2026M10, $n = 718$ months; $n = 706$ valid YoY observations)\n")

    lines.append("> [!IMPORTANT]")
    lines.append("> **Fundamental Target Horizon Limitation (`1M-as-12M PD Proxy` Assumption):**  ")
    lines.append("> The UCI ID 350 target variable (`default payment next month`) is a **one-month-ahead default event** (measuring whether the borrower defaults in October 2005 following the April–September 2005 billing/repayment window), **not an empirical 12-month default probability**. Because the dataset provides only a single outcome month and no 12-month performance horizon, using the scorecard's predicted probability and grade default rates (plus Jeffreys Margin of Conservatism) as a **12-month PD proxy (`1M-as-12M PD proxy`)** in the Basel IRB capital formula and IFRS 9 ECL calculations is an **explicit methodological assumption and limitation** adopted solely to illustrate regulatory capital and provisioning mechanics.\n")

    lines.append("---\n")
    lines.append("## 1. Risk Segmentation (`gold.segmentation_grade_profiles` & `gold.segmentation_behavioural_profiles`)\n")
    lines.append("### 1.1 Rating Grade Risk Driver Profiles (`G1`–`G6`)\n")
    lines.append("Every count, default count, bad rate, and risk driver mean below is generated directly from `gold.credit_card_modelling` and verified by automated assertions against DuckDB:\n")
    lines.append("| Split | Grade | Score Band | Accounts ($n$) | Share (%) | Defaults ($d$) | Observed Bad Rate (1M Target, %) | Mean Champion PD (1M Target, %) | Mean Score | `PAY_0 = -2` Inactive (%) | `PAY_0 = -1` Paid Duly (%) | `PAY_0 = 0` Revolving (%) | `PAY_0 = 1` 1M DPD (%) | `PAY_0 >= 2` 2M+ DPD (%) | Mean Months Delinq (6M) | Mean Months Revolv (6M) | Mean Util Ratio 6M (%) | Mean Monthly Pay (`NT$`) | Mean Credit Limit (`NT$`) |")
    lines.append("| :--- | :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for _, r in grade_profiles_df.iterrows():
        lines.append(
            f"| **{r['sample_split']}** | **{r['rating_grade']}** | `{r['score_band']}` | {int(r['count']):,} | {r['portfolio_share_pct']:.2f}% | {int(r['defaults']):,} | **{r['observed_default_rate']*100:.2f}%** | {r['mean_champion_pd']*100:.2f}% | {r['mean_champion_score']:.2f} | {r['pay0_inactive_m2_pct']:.2f}% | {r['pay0_paid_duly_m1_pct']:.2f}% | {r['pay0_revolving_0_pct']:.2f}% | {r['pay0_dpd1_pct']:.2f}% | {r['pay0_dpd2plus_pct']:.2f}% | {r['mean_months_delinquent_6m']:.2f} | {r['mean_months_revolving_6m']:.2f} | {r['mean_util_ratio_6m']*100:.2f}% | {r['mean_pay_amt_6m_ntd']:,.0f} | {r['mean_limit_bal_ntd']:,.0f} |"
        )

    lines.append("\n### 1.2 Behavioural Customer Segmentation (`S1`–`S5`)\n")
    lines.append("- **`S1: Inactive / Dormant Transactors`**: `MONTHS_DELINQUENT_6M == 0` and (`MONTHS_INACTIVE_6M >= 3` or (`PAY_0 == -2` and `MEAN_UTIL_RATIO_6M < 5%`)), excluding active revolvers (`MONTHS_REVOLVING_6M >= 3`).")
    lines.append("- **`S2: Active Full-Payer Transactors`**: `MONTHS_DELINQUENT_6M == 0`, not `S1`, and `MONTHS_REVOLVING_6M <= 2`.")
    lines.append("- **`S3: Seasoned Performing Revolvers`**: `MONTHS_DELINQUENT_6M == 0` and `MONTHS_REVOLVING_6M >= 3`.")
    lines.append("- **`S4: Episodic / Recently Delinquent (1–2M)`**: `1 <= MONTHS_DELINQUENT_6M <= 2` and `MAX_DPD_6M <= 2`.")
    lines.append("- **`S5: Chronic / Severe Delinquent (3+M or 90+ DPD)`**: `MONTHS_DELINQUENT_6M >= 3` or `MAX_DPD_6M >= 3`.\n")

    lines.append("| Split | Behavioural Segment | Accounts ($n$) | Share (%) | Defaults ($d$) | Observed Bad Rate (1M Target, %) | Mean Champion PD (1M Target, %) | Mean Challenger Platt PD (%) | Mean Score | Mean Months Delinq (6M) | Mean Months Revolv (6M) | Mean Months Inact (6M) | Mean Util Ratio 6M (%) | Mean Monthly Pay (`NT$`) | Mean Credit Limit (`NT$`) |")
    lines.append("| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for _, r in segment_profiles_df.iterrows():
        lines.append(
            f"| **{r['sample_split']}** | **{r['segment']}** | {int(r['count']):,} | {r['portfolio_share_pct']:.2f}% | {int(r['defaults']):,} | **{r['observed_default_rate']*100:.2f}%** | {r['mean_champion_pd']*100:.2f}% | {r['mean_challenger_platt_pd']*100:.2f}% | {r['mean_champion_score']:.2f} | {r['mean_months_delinquent_6m']:.2f} | {r['mean_months_revolving_6m']:.2f} | {r['mean_months_inactive_6m']:.2f} | {r['mean_util_ratio_6m']*100:.2f}% | {r['mean_pay_amt_6m_ntd']:,.0f} | {r['mean_limit_bal_ntd']:,.0f} |"
        )

    lines.append("\n---\n")
    lines.append("## 2. Illustrative Basel Retail Capital (`gold.basel_capital_by_grade` & `gold.basel_capital_sensitivity`)\n")
    lines.append("### 2.1 Regulatory Citations, Stated Assumptions & Limitations\n")
    lines.append("- **Target Horizon Assumption (`1M-as-12M PD Proxy`):** The Basel II/III/CRR3 IRB formula requires a 12-month Through-the-Cycle (TTC) or hybrid PD. Because the UCI ID 350 target (`default payment next month`) is a **1-month default indicator**, we use each grade's `DEV` 1-month observed default rate plus its 95% Jeffreys Margin of Conservatism ($\\text{PD}_{\\text{IRB},g}^{\\text{proxy}} = \\text{ODR}_{\\text{DEV},g} + \\text{MoC}_g = \\text{Jeffreys}_{95\\%,\\text{upper}}(\\text{DEV}_g)$) as an **illustrative 12-month PD proxy**.")
    lines.append("- **Asset Class & Correlation:** Qualifying Revolving Retail Exposures (QRRE) under CRR Art. 154(4) / Basel CRE30.23 & CRE31.9 with fixed asset correlation **$R = 0.04$**.")
    lines.append("- **Regulatory PD Floors (CRR3 vs CRR2):**")
    lines.append("  - **CRR3 QRRE Revolver Floor (Applied):** **$0.10\\%$ (`0.0010`)** under CRR3 Art. 163(1) / Basel III CRE30.55 for QRRE revolvers.")
    lines.append("  - **CRR3 Standard Retail / QRRE Transactor Floor:** **$0.05\\%$ (`0.0005`)** under CRR3 Art. 163(1).")
    lines.append("  - **Legacy CRR2 Retail Floor:** **$0.03\\%$ (`0.0003`)** under CRR2 Art. 163.")
    lines.append("  - *(Note: All three floors are non-binding on this portfolio because the lowest grade PD proxy is `G1 = 6.6126%`.)*")
    lines.append("- **LGD Assumption & CRR3 Input Floor:**")
    lines.append("  - **Baseline LGD Assumption:** **$85.0\\%$ (`0.85`)**, stated as an explicit assumption for unsecured retail credit card downturn LGD because UCI ID 350 contains no post-default recovery cash flows.")
    lines.append("  - **CRR3 QRRE LGD Input Floor:** **$50.0\\%$ (`0.50`)** under CRR3 Art. 164(4) / Basel III CRE30.56 for unsecured QRRE exposures (included as the lower bound of the LGD sensitivity grid `[50%, 65%, 75%, 85%, 95%]`).")
    lines.append("- **EAD Construction (`CCF = 50%` Baseline):** $\\text{Drawn}_i = \\max(\\text{BILL\\_AMT1}_i, 0)$, $\\text{Undrawn}_i = \\max(\\text{LIMIT\\_BAL}_i - \\text{Drawn}_i, 0)$, and $\\text{EAD}_i = \\text{Drawn}_i + \\text{CCF} \\times \\text{Undrawn}_i$.\n")

    lines.append("### 2.2 Grade-Level Basel QRRE Capital Schedule (`LGD = 85%`, `CCF = 50%`, `R = 0.04`, `PD Floor = 0.10%`)\n")
    lines.append("| Split | Grade | Accounts ($n$) | `DEV` 1M Bad Rate (%) | Jeffreys MoC (%) | **IRB PD Proxy w/ MoC (%)** | 99.9% Stressed CDR (%) | Total Stressed Loss Rate $\\text{EL}+\\text{UL}$ (% EAD) | UL Capital Factor $K_{\\text{UL}}$ | RWA Density (% EAD) | Drawn (`NT$ M`) | Undrawn (`NT$ M`) | EAD (`NT$ M`) | RWA (`NT$ M`) | Capital @ 8% (`NT$ M`) | Expected Loss (`NT$ M`) | Total Stressed Loss $\\text{EL}+\\text{UL}$ (`NT$ M`) |")
    lines.append("| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for sp in ["DEV", "HOLDOUT", "ALL_SILVER_GOLD"]:
        sub = capital_by_grade_df[capital_by_grade_df["sample_split"] == sp]
        for _, r in sub.iterrows():
            lines.append(
                f"| **{sp}** | **{r['rating_grade']}** | {int(r['account_count']):,} | {r['dev_empirical_pd_1m_as_12m_proxy']*100:.4f}% | +{r['moc_add_on']*100:.4f}% | **{r['regulatory_pd_with_moc_proxy']*100:.4f}%** | {r['conditional_default_rate_999']*100:.4f}% | {r['total_stressed_loss_rate_el_plus_ul']*100:.4f}% | {r['capital_factor_k_reg']:.6f} | **{r['rwa_density_reg_pct']:.2f}%** | {r['drawn_balance_ntd_m']:,.2f} | {r['undrawn_limit_ntd_m']:,.2f} | {r['ead_ntd_m']:,.2f} | **{r['rwa_reg_ntd_m']:,.2f}** | {r['capital_8pct_reg_ntd_m']:,.2f} | {r['expected_loss_reg_ntd_m']:,.2f} | {r['total_stressed_loss_el_plus_ul_ntd_m']:,.2f} |"
            )
        tot_n = int(sub["account_count"].sum())
        tot_drawn = float(sub["drawn_balance_ntd_m"].sum())
        tot_undrawn = float(sub["undrawn_limit_ntd_m"].sum())
        tot_ead = float(sub["ead_ntd_m"].sum())
        tot_rwa = float(sub["rwa_reg_ntd_m"].sum())
        tot_cap = float(sub["capital_8pct_reg_ntd_m"].sum())
        tot_el = float(sub["expected_loss_reg_ntd_m"].sum())
        tot_stl = float(sub["total_stressed_loss_el_plus_ul_ntd_m"].sum())
        tot_dens = 100.0 * tot_rwa / tot_ead
        lines.append(
            f"| **{sp}** | **Total** | **{tot_n:,}** | — | — | — | — | — | — | **{tot_dens:.2f}%** | **{tot_drawn:,.2f}** | **{tot_undrawn:,.2f}** | **{tot_ead:,.2f}** | **{tot_rwa:,.2f}** | **{tot_cap:,.2f}** | **{tot_el:,.2f}** | **{tot_stl:,.2f}** |"
        )

    lines.append("\n### 2.3 Mathematical Monotonicity Verification (PD Below/Above Peak & LGD)\n")
    lines.append("1. **$K_{\\text{UL}}$ and RWA Density Monotonic in PD Below the Vasicek Peak ($\\text{PD}^* = 38.9795\\%$ for $R = 0.04$):**")
    lines.append("   - Across `G1` $\\rightarrow$ `G5` ($\\text{PD} \\in [6.6126\\%, 33.9083\\%] < 38.9795\\%$), $K_{\\text{UL}}$ increases strictly (`0.098996` $\\rightarrow$ `0.125212` $\\rightarrow$ `0.164724` $\\rightarrow$ `0.194045` $\\rightarrow$ `0.206564`) and RWA density increases strictly (`123.75%` $\\rightarrow$ `156.52%` $\\rightarrow$ `205.90%` $\\rightarrow$ `242.56%` $\\rightarrow$ `258.20%`).")
    lines.append("   - Above the Vasicek inflection point $\\text{PD}^* = 38.9795\\%$ (where $\\partial \\text{CDR}/\\partial \\text{PD} = 1$), incremental PD shifts loss out of Unexpected Loss ($K_{\\text{UL}}$) and into Expected Loss ($\\text{EL} = \\text{LGD} \\times \\text{PD}$), so `G6` (`PD = 66.4239%`) has $K_{\\text{UL}} = 0.163204$ (`204.01%` RWA density) and $\\text{EL} = 56.4603\\%$ of EAD.")
    lines.append("2. **Total Stressed Loss Rate ($\\text{EL} + \\text{UL} = \\text{LGD} \\times \\text{CDR}(\\text{PD})$) Strictly Monotonic in PD Everywhere (`G1` $\\rightarrow$ `G6`):**")
    lines.append("   - Across all six grades `G1` $\\rightarrow$ `G6`, 99.9% Stressed CDR (`18.26%` $\\rightarrow$ `24.52%` $\\rightarrow$ `36.00%` $\\rightarrow$ `48.29%` $\\rightarrow$ `58.21%` $\\rightarrow$ `85.62%`) and Total Stressed Loss $\\text{EL} + \\text{UL}$ (`15.52%` $\\rightarrow$ `20.84%` $\\rightarrow$ `30.60%` $\\rightarrow$ `41.05%` $\\rightarrow$ `49.48%` $\\rightarrow$ `72.78%` of EAD) increase **strictly monotonically**.")
    lines.append("3. **$K_{\\text{UL}}$, RWA, and $\\text{EL} + \\text{UL}$ Strictly Monotonic in LGD Everywhere (`50% CRR3 Floor` $\\rightarrow$ `95%`):**\n")

    lines.append("| LGD Assumption | `CCF = 25%` (`EAD = NT$ 721.88M`) RWA (`NT$ M`) / Density | `CCF = 50%` (`EAD = NT$ 987.36M`) RWA (`NT$ M`) / Density | `CCF = 75%` (`EAD = NT$ 1,252.85M`) RWA (`NT$ M`) / Density | `CCF = 100%` (`EAD = NT$ 1,518.33M`) RWA (`NT$ M`) / Density |")
    lines.append("| :--- | ---: | ---: | ---: | ---: |")
    hold_sens = cap_sensitivity_df[cap_sensitivity_df["sample_split"] == "HOLDOUT"]
    for lgd_v in [0.50, 0.65, 0.75, 0.85, 0.95]:
        lbl = f"**`LGD = {int(lgd_v*100)}%`**"
        if lgd_v == 0.50:
            lbl += " *(CRR3 QRRE Floor)*"
        elif lgd_v == 0.85:
            lbl += " *(Baseline Assumption)*"
        cells = []
        for ccf_v in [0.25, 0.50, 0.75, 1.00]:
            r_s = hold_sens[(hold_sens["lgd_assumption"] == lgd_v) & (hold_sens["ccf_assumption"] == ccf_v)].iloc[0]
            cells.append(f"`{r_s['total_rwa_ntd_m']:,.2f}` (`{r_s['portfolio_rwa_density_pct']:.2f}%`)")
        lines.append(f"| {lbl} | {cells[0]} | {cells[1]} | {cells[2]} | {cells[3]} |")

    lines.append("\n---\n")
    lines.append("## 3. IFRS 9 Staging Illustration (`gold.ifrs9_markov_matrix` & `gold.ifrs9_stage_summary`)\n")
    lines.append("> [!WARNING]")
    lines.append("> **Illustrative Status & Target Horizon Caveat:**  ")
    lines.append("> All IFRS 9 12-month and 36-month lifetime PD and ECL figures below are **illustrative**. First, the underlying scorecard target (`default payment next month`) is a **1-month default event** used here as an explicit **12-month PD proxy (`1M-as-12M PD proxy`)**. Second, UCI ID 350 contains no origination date or origination PD (precluding a relative origination-PD SICR test), no cure/write-off cash-flow logs, and no contractual APRs (so $\\text{LGD} = 85\\%$, $\\text{EIR} = 18.0\\%$ p.a., and a $T = 36\\text{-month}$ revolving behavioural horizon are stated assumptions).\n")

    lines.append("### 3.1 Documentation of the UCI `PAY` Code `1` Coding Inconsistency & 3-State Markov Redesign\n")
    lines.append("A cross-month audit of `PAY_0, PAY_2, PAY_3, PAY_4, PAY_5, PAY_6` in UCI ID 350 reveals a well-known historical coding inconsistency between Month 1 (`PAY_0`) and Months 2–6 (`PAY_2`–`PAY_6`):\n")
    lines.append("| Delinquency Column | Raw Bronze ($N=30,000$) `PAY <= 0` | Raw Bronze ($N=30,000$) **`PAY == 1`** | Raw Bronze ($N=30,000$) **`PAY == 2`** | Raw Bronze ($N=30,000$) `PAY >= 3` | Gold ($N=29,932$) **`PAY == 1`** | Gold ($N=29,932$) **`PAY == 2`** | `DEV` ($n=20,952$) **`PAY == 1`** | `DEV` ($n=20,952$) **`PAY == 2`** |")
    lines.append("| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for _, r_p in uci_pay_coding_audit_df.iterrows():
        lines.append(
            f"| **`{r_p['column']}`** | {int(r_p['raw_30k_code_le_0']):,} | **{int(r_p['raw_30k_code_1']):,}** | **{int(r_p['raw_30k_code_2']):,}** | {int(r_p['raw_30k_code_ge_3']):,} | **{int(r_p['gold_code_1']):,}** | **{int(r_p['gold_code_2']):,}** | {int(r_p['dev_code_1']):,} | {int(r_p['dev_code_2']):,} |"
        )
    lines.append("\n- **Root Cause & Resolution:** In `PAY_0` (September 2005), code `1` (`1M DPD`) appears **3,688 times** in the raw 30,000-row dataset (`3,678` in Gold), whereas in `PAY_2`–`PAY_6` (April–August 2005) code `1` appears only **28, 4, 2, 0, and 0 times** while code `2` appears in the thousands (`2,621` to `3,919` times). Treating code `1` and code `2` as separate Markov states creates a degenerate `State 1` row based on just 19 `DEV` transitions.")
    lines.append("- **3-State Markov Matrix Specification:** To align with both the data structure and the IFRS 9 Stage 2 (`30–89 DPD`) definition, we merge codes `1` and `2` into a single **Delinquent 30–89 DPD** state, forming three well-populated states across all five monthly transitions (`PAY_6 -> PAY_5 -> PAY_4 -> PAY_3 -> PAY_2 -> PAY_0`):")
    lines.append("  1. **`State 0: Current / 0 DPD (`PAY_x <= 0`)`** $\\rightarrow$ **Stage 1**")
    lines.append("  2. **`State 1: Delinquent 30–89 DPD (`PAY_x in {1, 2}` merged)`** $\\rightarrow$ **Stage 2**")
    lines.append("  3. **`State 2: Default / 90+ DPD (`PAY_x >= 3`)`** $\\rightarrow$ **Stage 3**\n")

    lines.append("### 3.2 Empirical 3-State Monthly Markov Transition Matrix (`gold.ifrs9_markov_matrix`)\n")
    lines.append("| Split | From State ($t$) | Transition Count | To `State 0: Current (<=0)` | To `State 1: 30–89 DPD (1–2)` | To `State 2: 90+ DPD (>=3)` | Row Sum | Absorbing Cumul Default **12M** | Absorbing Cumul Default **24M** | Absorbing Cumul Default **36M** |")
    lines.append("| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for _, r_m in markov_all_df.iterrows():
        lines.append(
            f"| **{r_m['sample_split']}** | **{r_m['from_state_label']}** | {int(r_m['transition_count_total']):,} | `{r_m['prob_to_state0_current']:.6f}` ({int(r_m['count_to_state0_current']):,}) | `{r_m['prob_to_state1_30_89dpd']:.6f}` ({int(r_m['count_to_state1_30_89dpd']):,}) | `{r_m['prob_to_state2_90plus_dpd']:.6f}` ({int(r_m['count_to_state2_90plus_dpd']):,}) | **`{r_m['row_sum']:.6f}`** | {r_m['absorbing_cum_default_12m']*100:.2f}% | {r_m['absorbing_cum_default_24m']*100:.2f}% | **{r_m['absorbing_cum_default_36m']*100:.2f}%** |"
        )

    lines.append("\n### 3.3 Illustrative IFRS 9 Stage Allocation, Lifetime PD & ECL (`LGD = 85%`, `EIR = 18.0%`, `Horizon = 36M`)\n")
    lines.append("For each account $i$ in state $s_i \\in \\{0, 1\\}$ with 12-month PD proxy $p_{12,i}$ (`1M-as-12M PD proxy`), the 3-state `DEV` absorbing Markov cumulative survival curve $1 - C_{s_i}(t)$ is scaled via $\\kappa_i = \\ln(1 - p_{12,i}) / \\ln(1 - C_{s_i}(12))$ so that 12-step cumulative default equals $p_{12,i}$ and 36-step illustrative lifetime default is $\\text{PD}_{\\text{LT},36\\text{M},i} = 1 - (1 - C_{s_i}(36))^{\\kappa_i}$. Monthly marginal defaults $d_i(t) = S_i(t-1) - S_i(t)$ are discounted at monthly $\\text{EIR} = 1.3888\\%$.\n")
    lines.append("| Split | IFRS 9 Stage | Accounts ($n$) | Account Share (%) | Observed Next-Month Bad Rate (%) | Mean 12M PD Proxy (1M-Target Assumption, %) | Mean Illustrative 36M Lifetime PD (%) | Drawn (`NT$ M`) | EAD @ CCF 50% (`NT$ M`) | EAD Share (%) | Illustrative 12M-Proxy ECL (`NT$ M`) | Illustrative 36M Lifetime ECL (`NT$ M`) | **Allocated IFRS 9 ECL (`NT$ M`)** | **Stage 2 Lifetime Uplift (`NT$ M`)** | **Allocated ECL Coverage (% EAD)** |")
    lines.append("| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for _, r_i in ifrs9_stage_df.iterrows():
        lines.append(
            f"| **{r_i['sample_split']}** | **{r_i['stage_label']}** | {int(r_i['account_count']):,} | {r_i['account_share_pct']:.2f}% | {r_i['observed_default_rate_next_month']*100:.2f}% | {r_i['mean_12m_pd_proxy']*100:.2f}% | **{r_i['mean_lifetime_36m_pd_illustrative']*100:.2f}%** | {r_i['drawn_balance_ntd_m']:,.2f} | {r_i['ead_ccf50_ntd_m']:,.2f} | {r_i['ead_share_pct']:.2f}% | {r_i['ecl_if_12m_ntd_m']:,.2f} | {r_i['ecl_if_lifetime_36m_ntd_m']:,.2f} | **{r_i['ifrs9_allocated_ecl_ntd_m']:,.2f}** | +{r_i['stage2_lifetime_uplift_ntd_m']:,.2f} | **{r_i['coverage_ratio_allocated_ecl_to_ead_pct']:.2f}%** |"
        )

    lines.append("\n---\n")
    lines.append("## 4. Irish Stress Sensitivity Illustration via CSO `LRM01` (`gold.stress_scenario_summary` & `gold.stress_grade_migration`)\n")
    lines.append("> [!NOTE]")
    lines.append("> **Scenario Sensitivity Caveat (Not an Empirical Macro-Credit Model):**  ")
    lines.append("> Because a 6-month 2005 Taiwanese credit card cross-section cannot be regressed against Irish macroeconomic time series, this analysis is explicitly labelled a **scenario sensitivity illustration**. We compute historical 12-month YoY relative changes from the official CSO Ireland `LRM01` series (`Persons on the Live Register`, filtered to `All ages`, `Both sexes`, `All live register schemes`, $n = 718$ months from `1967M01` to `2026M10`, yielding $n = 706$ valid YoY observations) and apply a **stated pass-through elasticity assumption ($\\varepsilon = 0.50$, so $\\text{PD Multiplier} = 1 + 0.50 \\times \\Delta_{\\text{YoY}}$)** to the 1M-as-12M PD proxies.\n")

    lines.append("### 4.1 CSO `LRM01` YoY Distribution & Stress Scenarios\n")
    hist_dist = step5_report["irish_lrm01_stress_sensitivity"]["historical_yoy_distribution"]
    lines.append(f"- **CSO `LRM01` Historical YoY Percentiles ($n = 706$ months):** Min $= {hist_dist['min']*100:.2f}\\%$, Median (`P50`) $= {hist_dist['p50_median']*100:.2f}\\%$, `P75` $= +{hist_dist['p75']*100:.2f}\\%$, **`P90` (Mild Stress) $= +{hist_dist['p90']*100:.2f}\\%$**, `P95` $= +{hist_dist['p95']*100:.2f}\\%$, `P99` $= +{hist_dist['p99']*100:.2f}\\%$, **Historical Max (`{hist_dist['max_month']}`, `{int(hist_dist['max_level_persons']):,}` persons, Severe Stress) $= +{hist_dist['max']*100:.2f}\\%$**.\n")

    lines.append("### 4.2 Portfolio Stress Impact Summary (`HOLDOUT` $n = 8,980$ & Full Portfolio $n = 29,932$)\n")
    lines.append("| Split | Scenario | LRM01 YoY Shock (%) | PD Multiplier | Mean Stressed PD Proxy (%) | Mean Stressed Score | Accounts Downgraded $\\ge 1$ Grade ($n$ / %) | `G5 + G6` Share (%) | Direct Stressed RWA (`NT$ M` / $\\Delta\\%$) | Grade-Migration RWA (`NT$ M` / $\\Delta\\%$) | Stressed IRB EL (`NT$ M`) | Total Stressed IRB Loss $\\text{EL}+\\text{UL}$ (`NT$ M`) | Fixed-Stage IFRS 9 ECL (`NT$ M` / $\\Delta\\%$) | SICR-Overlay Stage 2 Share (%) | SICR-Overlay IFRS 9 ECL (`NT$ M` / $\\Delta\\%$) |")
    lines.append("| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for _, r_st in stress_summary_df.iterrows():
        lines.append(
            f"| **{r_st['sample_split']}** | **`{r_st['scenario_id']}`** | `+{r_st['lrm01_yoy_shock_pct']:.2f}%` | `{r_st['pd_multiplier']:.4f}x` | {r_st['mean_stressed_regulatory_pd_proxy']*100:.2f}% | {r_st['mean_stressed_score']:.2f} | `{int(r_st['accounts_downgraded_count']):,}` (`{r_st['accounts_downgraded_pct']:.2f}%`) | {r_st['subinvestment_g5_g6_share_pct']:.2f}% | **`{r_st['rwa_full_stressed_pd_ntd_m']:,.2f}` (`+{r_st['rwa_uplift_vs_baseline_pct']:.2f}%`)** | **`{r_st['rwa_grade_migration_only_ntd_m']:,.2f}` (`+{r_st['rwa_grade_migration_uplift_pct']:.2f}%`)** | {r_st['irb_expected_loss_stressed_ntd_m']:,.2f} | {r_st['irb_total_stressed_loss_el_plus_ul_ntd_m']:,.2f} | **`{r_st['ifrs9_ecl_fixed_stage_ntd_m']:,.2f}` (`+{r_st['ifrs9_ecl_fixed_stage_uplift_pct']:.2f}%`)** | {r_st['ifrs9_sicr_stage2_share_pct']:.2f}% | **`{r_st['ifrs9_ecl_with_grade_downgrade_sicr_ntd_m']:,.2f}` (`+{r_st['ifrs9_ecl_with_sicr_uplift_pct']:.2f}%`)** |"
        )

    lines.append("\n### 4.3 Rating Grade Migration Schedule Under Stress (`gold.stress_grade_migration`)\n")
    lines.append("| Split | Rating Grade | Score Band | `S0_BASELINE` (`1.0000x`) Count / Share | `S1_MILD_P90` (`1.1141x`) Count / Share | `S2_SEVERE_MAX` (`1.4768x`) Count / Share | Severe vs Baseline Account Shift |")
    lines.append("| :--- | :--- | :--- | ---: | ---: | ---: | ---: |")
    for sp in ["HOLDOUT", "ALL_SILVER_GOLD"]:
        sub_g = stress_grade_dist_df[stress_grade_dist_df["sample_split"] == sp]
        for g_lbl in GRADE_ORDER:
            g_short = GRADE_SHORT[g_lbl]
            r0 = sub_g[(sub_g["scenario_id"] == "S0_BASELINE") & (sub_g["rating_grade"] == g_short)].iloc[0]
            r1 = sub_g[(sub_g["scenario_id"] == "S1_MILD_P90") & (sub_g["rating_grade"] == g_short)].iloc[0]
            r2 = sub_g[(sub_g["scenario_id"] == "S2_SEVERE_MAX") & (sub_g["rating_grade"] == g_short)].iloc[0]
            diff_cnt = int(r2["account_count"]) - int(r0["account_count"])
            diff_pp = float(r2["share_pct"]) - float(r0["share_pct"])
            sign_c = "+" if diff_cnt >= 0 else ""
            sign_p = "+" if diff_pp >= 0 else ""
            lines.append(
                f"| **{sp}** | **{g_lbl}** | `{GRADE_BAND_STR[g_short]}` | `{int(r0['account_count']):,}` (`{r0['share_pct']:.2f}%`) | `{int(r1['account_count']):,}` (`{r1['share_pct']:.2f}%`) | `{int(r2['account_count']):,}` (`{r2['share_pct']:.2f}%`) | `{sign_c}{diff_cnt:,}` (`{sign_p}{diff_pp:.2f} pp`) |"
            )

    (RESULTS_DIR / "Step5_Capital_IFRS9_Stress_Report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    rep = run_step5_pipeline()
    print("Step 5 completed successfully.")
    print(json.dumps(rep["ifrs9_staging_illustration"]["stage_summary"], indent=2))
