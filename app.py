"""
ScoreGuard-IRB Streamlit app (Step 6).

Three pages, all reading from the committed results/ folder. No numbers are hard-coded here:
every figure on screen comes from results/*.csv, results/*.json or results/audit_summary.json.

Run:  streamlit run app.py
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

RESULTS = Path(__file__).resolve().parent / "results"

st.set_page_config(page_title="ScoreGuard-IRB", layout="wide")


# ---------------------------------------------------------------- loaders
@st.cache_data
def csv(name: str) -> pd.DataFrame:
    return pd.read_csv(RESULTS / name)


@st.cache_data
def js(name: str) -> dict:
    with open(RESULTS / name, encoding="utf-8") as f:
        return json.load(f)


audit = js("audit_summary.json")
M = audit["readme_metrics"]
ASM = audit["assumptions"]

TARGET_CAVEAT = (
    "**One-month target.** The model predicts default in the next month (October 2005), not over 12 months. "
    "Capital and IFRS 9 figures use it as a 12-month PD proxy, which is a stated assumption."
)


def pct_cols(df: pd.DataFrame, cols: list[str], dp: int = 2) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        out[c] = (out[c] * 100).round(dp)
    return out


# ---------------------------------------------------------------- page 1
def page_governance() -> None:
    st.title("1. Data governance and segmentation")
    rc = audit["row_counts"]
    c = st.columns(5)
    c[0].metric("Raw accounts", f"{rc['bronze.credit_card_raw']:,}")
    c[1].metric("Quarantined", f"{rc['quarantine.credit_card_quarantine']:,}")
    c[2].metric("Modelling (gold)", f"{rc['gold.credit_card_modelling']:,}")
    c[3].metric("Development", f"{rc['dev_rows']:,}")
    c[4].metric("Holdout", f"{rc['holdout_rows']:,}")

    st.subheader("Data quality rules")
    dq = csv("dq_rule_summary.csv")
    st.dataframe(pct_cols(dq, ["flagged_default_rate"]), hide_index=True, width="stretch")
    st.caption(
        "DQ09 duplicates are flagged and retained (mostly all-zero dormant accounts with unique IDs). "
        "DQ10 is a documented mapping note for undocumented codes, not a violation."
    )

    st.subheader("Pseudonymisation")
    st.markdown(
        "Account IDs are replaced at ingestion by `SHA256(salt::id)`. The salt is read only from the "
        "`SCOREGUARD_SALT` environment variable and the code fails if it is missing. Because the source IDs are "
        "sequential, a leaked salt would make the hashes reversible (Challenge Log CL-15)."
    )

    st.subheader("Behavioural segments")
    seg = csv("step5_behavioural_segments.csv")
    split = st.radio("Sample", ["DEV", "HOLDOUT", "ALL_SILVER_GOLD"], horizontal=True, key="seg_split")
    s = seg[seg["sample_split"] == split]
    fig = px.bar(s, x="segment", y="observed_default_rate_pct", text="observed_default_rate_pct",
                 labels={"segment": "", "observed_default_rate_pct": "Observed next-month default rate (%)"})
    fig.update_traces(texttemplate="%{text:.2f}%", textposition="outside")
    st.plotly_chart(fig, width="stretch")
    st.dataframe(s.drop(columns=["sample_split"]), hide_index=True, width="stretch")

    st.subheader("Rating grade risk-driver profiles")
    gp = csv("step5_grade_profiles.csv")
    st.dataframe(gp[gp["sample_split"] == split].drop(columns=["sample_split"]), hide_index=True, width="stretch")


# ---------------------------------------------------------------- page 2
def page_models() -> None:
    st.title("2. Scorecard versus challenger, and validation")
    c = st.columns(4)
    c[0].metric("Champion holdout AUROC", M["champion_holdout_auroc"])
    c[1].metric("Champion holdout Gini", M["champion_holdout_gini"])
    c[2].metric("Challenger holdout Gini", M["challenger_holdout_gini"], f"+{M['gini_gap_points']} pts")
    c[3].metric("Review trigger (Gini pts)", M["gini_trigger_points"])
    st.info(
        f"Production decision: the champion scorecard stays in production. The challenger's holdout Gini gain "
        f"({M['gini_gap_points']} points) is below the {M['gini_trigger_points']}-point review trigger."
    )

    st.subheader("Model comparison")
    st.dataframe(csv("step3_champion_vs_challenger.csv"), hide_index=True, width="stretch")

    st.subheader("Champion scorecard: WoE bins and points")
    bins = csv("step2_woe_bin_tables.csv")
    feat = st.selectbox("Feature", bins["feature"].unique())
    b = bins[bins["feature"] == feat]
    fig = px.bar(b, x="bin_label", y="dev_bad_rate", text="scorecard_points_rounded",
                 labels={"bin_label": "Bin", "dev_bad_rate": "DEV bad rate"})
    fig.update_traces(texttemplate="%{text} pts", textposition="outside")
    fig.update_yaxes(tickformat=".0%")
    st.plotly_chart(fig, width="stretch")
    st.dataframe(b, hide_index=True, width="stretch")

    st.subheader("Information value of candidate features")
    st.dataframe(csv("step2_iv_summary.csv"), hide_index=True, width="stretch")

    st.subheader("Challenger SHAP importance")
    sh = csv("step3_shap_importance_expanded.csv").sort_values("mean_abs_shap_dev")
    fig = px.bar(sh, x="mean_abs_shap_dev", y="feature", orientation="h",
                 labels={"mean_abs_shap_dev": "Mean |SHAP| (DEV)", "feature": ""})
    st.plotly_chart(fig, width="stretch")

    st.subheader("Calibration")
    st.markdown(
        f"Hosmer-Lemeshow on the holdout: champion chi-square {M['champion_hl_chi2']} (p = {M['champion_hl_p']}), "
        f"challenger p = {M['challenger_hl_p']}. The test is very sensitive at this sample size."
    )
    c1, c2 = st.columns(2)
    c1.image(str(RESULTS / "step4_calibration_by_decile.png"), caption="Predicted vs observed by decile")
    c2.image(str(RESULTS / "step4_calibration_by_grade.png"), caption="Predicted vs observed by grade")

    st.subheader("Per-grade binomial backtest (holdout, champion)")
    bt = csv("step4_grade_binomial_backtest.csv")
    bt = bt[(bt["model"].str.startswith("Champion")) & (bt["sample_split"] == "HOLDOUT")]
    cols = ["rating_grade", "count", "observed_defaults", "observed_default_rate", "dev_empirical_grade_pd",
            "mean_predicted_pd", "jeffreys_95_lower", "jeffreys_95_upper", "one_sided_p_value_vs_dev_emp_pd",
            "traffic_light_vs_dev_emp_pd", "two_sided_p_value", "two_sided_calibration_status_vs_pred"]
    st.dataframe(bt[cols], hide_index=True, width="stretch")
    st.caption(
        f"G1 holdout uplift ({M['g1_observed_rate']} vs {M['g1_dev_pd']}, {M['g1_light']}) is unexplained and monitored. "
        f"G5 is over-predicted ({M['g5_predicted_pd']} vs {M['g5_observed_rate']}, two-sided p = {M['g5_p']}). "
        "Traffic light: green below 95% confidence, amber 95% to 99.9%, red above 99.9%."
    )

    st.subheader("Stability (PSI / CSI)")
    st.warning("Random 70/30 split, not an out-of-time sample. Low PSI is expected and is not evidence of stability over time.")
    st.dataframe(csv("step4_psi_csi_summary.csv"), hide_index=True, width="stretch")

    st.subheader("Fair-lending check")
    st.markdown(
        "Sex and marital status are excluded from both models. Approval rule: grades G1 to G4. "
        "The four-fifths rule is shown as a US heuristic, not an EU legal test."
    )
    st.dataframe(csv("step4_fair_lending_approval.csv"), hide_index=True, width="stretch")
    st.dataframe(csv("step4_fair_lending_calibration.csv"), hide_index=True, width="stretch")
    st.caption(f"{M['n_group_tests']} group-level calibration tests were run, so some significance is expected by chance.")


# ---------------------------------------------------------------- page 3
def page_capital() -> None:
    st.title("3. Capital, IFRS 9 illustration and stress sensitivity")
    st.warning(TARGET_CAVEAT)

    st.header("Basel QRRE capital (illustrative)")
    st.markdown(
        f"Asset correlation R = {ASM['asset_correlation_r']}, PD floor {M['pd_floor']} (CRR3 QRRE revolver), "
        f"LGD {M['lgd']} (assumption; CRR3 QRRE input floor {ASM['lgd_crr3_qrre_input_floor']:.0%}), CCF {ASM['ccf']:.0%}. "
        f"Grade PD = DEV grade default rate + 95% Jeffreys margin of conservatism."
    )
    cap = csv("step5_basel_capital_by_grade.csv")
    split = st.radio("Sample", ["HOLDOUT", "DEV", "ALL_SILVER_GOLD"], horizontal=True, key="cap_split")
    cs = cap[cap["sample_split"] == split]
    c = st.columns(4)
    c[0].metric("EAD (NT$ m)", f"{cs['ead_ntd_m'].sum():,.2f}")
    c[1].metric("RWA (NT$ m)", f"{cs['rwa_reg_ntd_m'].sum():,.2f}")
    c[2].metric("RWA density", f"{cs['rwa_reg_ntd_m'].sum() / cs['ead_ntd_m'].sum():.2%}")
    c[3].metric("Capital at 8% (NT$ m)", f"{cs['capital_8pct_reg_ntd_m'].sum():,.2f}")
    st.dataframe(cs.drop(columns=["sample_split"]), hide_index=True, width="stretch")
    st.caption("K rises with PD only below about 39% for R = 0.04; above that, loss shifts into expected loss. EL + UL rises with PD everywhere.")

    sens = csv("step5_basel_capital_sensitivity.csv")
    ss = sens[sens["sample_split"] == split]
    piv = ss.pivot(index="lgd_assumption", columns="ccf_assumption", values="total_rwa_ntd_m")
    fig = px.imshow(piv, text_auto=".0f", aspect="auto",
                    labels={"x": "CCF", "y": "LGD", "color": "RWA (NT$ m)"})
    st.plotly_chart(fig, width="stretch")

    st.header("IFRS 9 staging (illustrative: assumptions stated)")
    st.markdown(
        f"Stage 1: current (PAY_0 at most 0). Stage 2: 30 to 89 days past due (PAY_0 = 1 or 2). Stage 3: 90+ (PAY_0 at least 3). "
        f"No origination data, so there is no origination-based SICR test. EIR {ASM['eir_annual']:.0%} a year, "
        f"{ASM['ifrs9_lifetime_horizon_months']}-month lifetime, LGD {M['lgd']}."
    )
    mk = csv("step5_ifrs9_markov_matrix.csv")
    st.subheader("Monthly transition matrix (3 states)")
    st.dataframe(mk[mk["sample_split"] == split].drop(columns=["sample_split"]), hide_index=True, width="stretch")
    st.caption("PAY code 1 almost never appears after the first month in this dataset, so codes 1 and 2 are merged (Challenge Log CL-06).")
    stg = csv("step5_ifrs9_stage_summary.csv")
    st.subheader("Stage allocation and ECL")
    st.dataframe(stg[stg["sample_split"] == split].drop(columns=["sample_split"]), hide_index=True, width="stretch")
    st.info(
        f"Holdout mean PDs: {M['mean_reg_pd']} is the mean grade-level regulatory PD (every account at its grade PD), used for capital "
        f"and as the stress baseline. {M['mean_ifrs9_pd']} is the mean IFRS 9 12-month PD proxy with Stage 3 set to 100%."
    )

    st.header("Irish stress (scenario sensitivity, not a calibrated model)")
    st.markdown(
        f"Shocks come from year-on-year changes in CSO LRM01 (Persons on the Live Register): mild = 90th percentile "
        f"({M['mild_shock']}), severe = historical maximum ({M['severe_shock']}, May 2009). "
        f"PD multiplier = 1 + {ASM['stress_elasticity_epsilon']} x shock (assumption). SICR overlay: {ASM['stress_sicr_overlay_rule']}"
    )
    stz = csv("step5_stress_scenario_summary.csv")
    ssplit = st.radio("Sample", ["HOLDOUT", "ALL_SILVER_GOLD"], horizontal=True, key="stress_split")
    st.dataframe(stz[stz["sample_split"] == ssplit].drop(columns=["sample_split"]), hide_index=True, width="stretch")
    mig = csv("step5_stress_grade_migration.csv")
    mg = mig[mig["sample_split"] == ssplit]
    fig = px.bar(mg, x="rating_grade", y="share_pct", color="scenario_id", barmode="group",
                 labels={"rating_grade": "Grade", "share_pct": "Share of accounts (%)", "scenario_id": "Scenario"})
    st.plotly_chart(fig, width="stretch")


PAGES = {
    "1. Data governance and segmentation": page_governance,
    "2. Scorecard vs challenger and validation": page_models,
    "3. Capital, IFRS 9 and stress": page_capital,
}

st.sidebar.title("ScoreGuard-IRB")
st.sidebar.caption("Independent portfolio project. Real public data (UCI 350, CSO LRM01, both CC BY 4.0). Not a production model.")
choice = st.sidebar.radio("Page", list(PAGES))
PAGES[choice]()
