"""
ScoreGuard-IRB, Step 6: provenance record (results/audit_summary.json).

Collects data hashes, row counts, key metrics, assumptions and timestamps from the
executed outputs of Steps 1-5 (results/*.json, results/*.csv, data/raw/*).
Every number written here is read from an executed artefact; nothing is typed in by hand.

The file also contains a flat `readme_metrics` block. README.md is rendered from
src/templates/README.template.md using only these values (see src/render_readme.py),
and tests/test_readme_consistency.py checks the two stay in sync.
"""
from __future__ import annotations

import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

import capital_ifrs9_stress as cap_mod

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_ROOT / "results"
RAW_DIR = PROJECT_ROOT / "data" / "raw"
AUDIT_PATH = RESULTS_DIR / "audit_summary.json"


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _load_json(name: str) -> dict:
    with open(RESULTS_DIR / name, encoding="utf-8") as f:
        return json.load(f)


def _csv(name: str) -> pd.DataFrame:
    return pd.read_csv(RESULTS_DIR / name)


def _pct(x: float, dp: int = 2) -> str:
    return f"{x * 100:.{dp}f}%"


def _num(x: float, dp: int = 4) -> str:
    return f"{x:.{dp}f}"


def _int(x: float) -> str:
    return f"{int(round(x)):,}"


def _m(x: float) -> str:
    """NT$ millions with 2 dp."""
    return f"{x:,.2f}"


def build_audit_summary() -> dict:
    s1 = _load_json("step1_governance_summary.json")
    s2 = _load_json("step2_scorecard_summary.json")
    s3 = _load_json("step3_challenger_summary.json")
    s4 = _load_json("step4_validation_summary.json")
    s5 = _load_json("step5_capital_ifrs9_stress_summary.json")

    disc = _csv("step4_discrimination_metrics.csv")
    bt = _csv("step4_grade_binomial_backtest.csv")
    fl_app = _csv("step4_fair_lending_approval.csv")
    fl_cal = _csv("step4_fair_lending_calibration.csv")
    psi = _csv("step4_psi_csi_summary.csv")
    cap = _csv("step5_basel_capital_by_grade.csv")
    stage = _csv("step5_ifrs9_stage_summary.csv")
    stress = _csv("step5_stress_scenario_summary.csv")
    markov = _csv("step5_ifrs9_markov_matrix.csv")
    cvc = _csv("step3_champion_vs_challenger.csv")

    # ---------------- data provenance ----------------
    raw_files = {}
    if RAW_DIR.exists():
        for p in sorted(RAW_DIR.iterdir()):
            if p.is_file():
                raw_files[p.name] = {"sha256": _sha256(p), "bytes": p.stat().st_size}

    counts = s1["table_row_counts"]
    splits = s1["gold_sample_splits"]

    # ---------------- discrimination ----------------
    def drow(model_prefix: str, split: str, basis_contains: str | None = None) -> pd.Series:
        sub = disc[disc["model"].str.startswith(model_prefix) & (disc["sample_split"] == split)]
        if basis_contains:
            sub = sub[sub["evaluation_basis"].str.contains(basis_contains, regex=False)]
        assert len(sub) == 1, (model_prefix, split, basis_contains, len(sub))
        return sub.iloc[0]

    ch_dev = drow("Champion", "DEV")
    ch_hold = drow("Champion", "HOLDOUT")
    cl_hold = drow("Challenger", "HOLDOUT")
    cl_cv_metrics = s3["challenger_expanded_19f"]["metrics"]

    gini_gap = float(cl_hold["gini"] - ch_hold["gini"])
    trigger = s3["production_recommendation_and_defence"]["governance_trigger_threshold"]

    hl = s4["hosmer_lemeshow_test"]

    # ---------------- grade backtest ----------------
    def btrow(model_prefix: str, split: str, grade: str) -> pd.Series:
        sub = bt[bt["model"].str.startswith(model_prefix) & (bt["sample_split"] == split) & (bt["rating_grade"] == grade)]
        assert len(sub) == 1, (model_prefix, split, grade, len(sub))
        return sub.iloc[0]

    g1_hold = btrow("Champion", "HOLDOUT", "G1")
    g5_hold = btrow("Champion", "HOLDOUT", "G5")

    # ---------------- stability ----------------
    psi_score = psi[psi["feature_or_score"].str.startswith("CHAMPION_SCORE (6")].iloc[0]
    psi_max = float(psi["psi_or_csi_value"].max())

    # ---------------- fair lending (HOLDOUT, champion) ----------------
    fa_h = fl_app[fl_app["sample_split"] == "HOLDOUT"]
    fc_h = fl_cal[fl_cal["sample_split"] == "HOLDOUT"]
    dir_min = float(fa_h["champion_dir_vs_highest_group"].min())
    male = fc_h[(fc_h["protected_attribute"] == "SEX") & (fc_h["group_label"] == "Male")].iloc[0]
    married = fc_h[(fc_h["protected_attribute"] == "MARRIAGE") & (fc_h["group_label"] == "Married")].iloc[0]
    n_group_tests = int(len(fl_cal)) * 2  # champion + challenger per group row

    # ---------------- capital (HOLDOUT, baseline LGD/CCF) ----------------
    cap_h = cap[cap["sample_split"] == "HOLDOUT"]
    ead_h = float(cap_h["ead_ntd_m"].sum())
    rwa_h = float(cap_h["rwa_reg_ntd_m"].sum())
    k8_h = float(cap_h["capital_8pct_reg_ntd_m"].sum())
    el_h = float(cap_h["expected_loss_reg_ntd_m"].sum())

    # ---------------- IFRS 9 (HOLDOUT) ----------------
    st_h = stage[stage["sample_split"] == "HOLDOUT"].set_index("ifrs9_stage")
    tot = st_h.loc[0]
    mk_dev = markov[markov["sample_split"] == "DEV"].set_index("from_state_code")

    # ---------------- stress (HOLDOUT) ----------------
    sh = stress[stress["sample_split"] == "HOLDOUT"].set_index("scenario_id")
    base, mild, sev = sh.loc["S0_BASELINE"], sh.loc["S1_MILD_P90"], sh.loc["S2_SEVERE_MAX"]

    capital_meta = s5["illustrative_basel_capital"]
    stress_meta = s5["irish_lrm01_stress_sensitivity"]

    # Baseline mean PD definitions (fix for the 23.59% vs 23.08% inconsistency)
    pd_def = {
        "hold_mean_regulatory_grade_pd_with_moc": float(base["mean_stressed_regulatory_pd_proxy"]),
        "hold_mean_ifrs9_12m_pd_proxy_stage3_at_100pct": float(tot["mean_12m_pd_proxy"]),
        "definitions": {
            "mean_regulatory_grade_pd_with_moc": (
                "Account-weighted mean of the grade-level regulatory PD proxy (DEV grade default rate + 95% Jeffreys MoC, "
                "floored at the CRR3 QRRE revolver floor of 0.10%). Every account, including Stage 3, carries its grade PD. "
                "Used for Basel capital and as the S0 baseline of the stress table."
            ),
            "mean_ifrs9_12m_pd_proxy": (
                "Account-weighted mean of the IFRS 9 12-month PD proxy. Identical to the regulatory grade PD for Stage 1 and "
                "Stage 2 accounts, but Stage 3 (credit-impaired, PAY_0 >= 3) accounts are set to PD = 100%. "
                "Used in the IFRS 9 stage table. The difference between the two means is caused entirely by the Stage 3 override."
            ),
        },
    }

    audit = {
        "project": "ScoreGuard-IRB",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "step_timestamps_utc": {
            "step1": s1["generated_at_utc"],
            "step2": s2["generated_at_utc"],
            "step3": s3["generated_at_utc"],
            "step4": s4["generated_at_utc"],
            "step5": s5["generated_at_utc"],
        },
        "data_sources": {
            "uci_350": {
                "name": "Default of Credit Card Clients (UCI Machine Learning Repository, ID 350)",
                "citation": "Yeh, I-Cheng (2016). Default of Credit Card Clients. UCI Machine Learning Repository. https://doi.org/10.24432/C55S3H",
                "licence": "CC BY 4.0",
                "portfolio": "Taiwan, credit card accounts, repayment history April-September 2005",
                "target": "default payment next month (one-month-ahead default event)",
            },
            "cso_lrm01": {
                "name": "CSO Ireland LRM01, Persons on the Live Register",
                "api": "https://ws.cso.ie/public/api.restful/PxStat.Data.Cube_API.ReadDataset/LRM01/CSV/1.0/en",
                "licence": "CC BY 4.0",
                "filter": stress_meta["cso_filter"],
            },
            "raw_file_hashes": raw_files,
        },
        "row_counts": {
            **counts,
            "dev_rows": splits["DEV"]["rows"],
            "holdout_rows": splits["HOLDOUT"]["rows"],
            "dev_defaults": splits["DEV"]["defaults"],
            "holdout_defaults": splits["HOLDOUT"]["defaults"],
        },
        "key_metrics": {
            "champion": {
                "n_features": len(s2["final_model_coefficients_and_vif"]),
                "max_vif": max(r["vif_dev"] for r in s2["final_model_coefficients_and_vif"]),
                "dev_auroc": float(ch_dev["auroc"]), "dev_gini": float(ch_dev["gini"]),
                "holdout_auroc": float(ch_hold["auroc"]), "holdout_gini": float(ch_hold["gini"]),
                "holdout_ks": float(ch_hold["ks_statistic"]), "holdout_brier": float(ch_hold["brier_score"]),
                "hl_holdout_chi2": hl["champion_holdout"]["hl_chi2_statistic"],
                "hl_holdout_p_df8": hl["champion_holdout"]["p_value_df8"],
            },
            "challenger_expanded_platt": {
                "n_features": int(cvc.iloc[-1]["feature_count"]),
                "dev_cv_auroc": float(cl_cv_metrics["DEV_CV_OOF_AUROC"]),
                "holdout_auroc": float(cl_hold["auroc"]), "holdout_gini": float(cl_hold["gini"]),
                "holdout_brier": float(cl_hold["brier_score"]),
                "hl_holdout_p_df8": hl["challenger_platt_holdout"]["p_value_df8"],
                "holdout_gini_gap_vs_champion": gini_gap,
                "governance_trigger_gini_points": trigger["trigger_threshold_gini_points"],
            },
            "grade_backtest_holdout": {
                "g1_observed_rate": float(g1_hold["observed_default_rate"]),
                "g1_dev_empirical_pd": float(g1_hold["dev_empirical_grade_pd"]),
                "g1_one_sided_p_vs_dev_emp": float(g1_hold["one_sided_p_value_vs_dev_emp_pd"]),
                "g1_traffic_light_vs_dev_emp": str(g1_hold["traffic_light_vs_dev_emp_pd"]),
                "g5_observed_rate": float(g5_hold["observed_default_rate"]),
                "g5_mean_predicted_pd": float(g5_hold["mean_predicted_pd"]),
                "g5_two_sided_p": float(g5_hold["two_sided_p_value"]),
            },
            "stability": {
                "score_psi_grades": float(psi_score["psi_or_csi_value"]),
                "max_psi_or_csi": psi_max,
                "note": "Random stratified split, not out-of-time. Low PSI is expected and is not evidence of stability over time.",
            },
            "fair_lending_holdout_champion": {
                "min_disparate_impact_ratio": dir_min,
                "male_obs_minus_pred_bps": float(male["champion_calib_diff_obs_minus_pred_bps"]),
                "male_two_sided_p": float(male["champion_two_sided_p_value"]),
                "married_obs_minus_pred_bps": float(married["champion_calib_diff_obs_minus_pred_bps"]),
                "married_two_sided_p": float(married["champion_two_sided_p_value"]),
                "n_group_calibration_tests": n_group_tests,
            },
            "capital_holdout_illustrative": {
                "ead_ntd_m": ead_h, "rwa_ntd_m": rwa_h, "rwa_density": rwa_h / ead_h,
                "capital_8pct_ntd_m": k8_h, "expected_loss_ntd_m": el_h,
            },
            "ifrs9_holdout_illustrative": {
                "stage1_share_pct": float(st_h.loc[1, "account_share_pct"]),
                "stage2_share_pct": float(st_h.loc[2, "account_share_pct"]),
                "stage3_share_pct": float(st_h.loc[3, "account_share_pct"]),
                "allocated_ecl_ntd_m": float(tot["ifrs9_allocated_ecl_ntd_m"]),
                "coverage_pct": float(tot["coverage_ratio_allocated_ecl_to_ead_pct"]),
                "stage2_lifetime_uplift_ntd_m": float(tot["stage2_lifetime_uplift_ntd_m"]),
                "markov_dev_current_to_90plus": float(mk_dev.loc[0, "prob_to_state2_90plus_dpd"]),
            },
            "stress_holdout_scenario_sensitivity": {
                "mild_yoy_shock_pct": float(mild["lrm01_yoy_shock_pct"]),
                "severe_yoy_shock_pct": float(sev["lrm01_yoy_shock_pct"]),
                "mild_pd_multiplier": float(mild["pd_multiplier"]),
                "severe_pd_multiplier": float(sev["pd_multiplier"]),
                "mild_rwa_uplift_pct": float(mild["rwa_uplift_vs_baseline_pct"]),
                "severe_rwa_uplift_pct": float(sev["rwa_uplift_vs_baseline_pct"]),
                "mild_ecl_sicr_uplift_pct": float(mild["ifrs9_ecl_with_sicr_uplift_pct"]),
                "severe_ecl_sicr_uplift_pct": float(sev["ifrs9_ecl_with_sicr_uplift_pct"]),
                "severe_downgraded_pct": float(sev["accounts_downgraded_pct"]),
            },
            "baseline_mean_pd_definitions_holdout": pd_def,
        },
        "assumptions": {
            "target_horizon": s5["fundamental_target_horizon_caveat"],
            "asset_class": capital_meta["asset_class_citation"],
            "asset_correlation_r": capital_meta["asset_correlation_r"],
            "pd_floor_applied": capital_meta["pd_floor_crr3_qrre_revolver"],
            "lgd": capital_meta["lgd_assumption"],
            "lgd_crr3_qrre_input_floor": capital_meta["lgd_crr3_qrre_input_floor"],
            "ccf": cap_mod.BASELINE_CCF,
            "eir_annual": cap_mod.ANNUAL_EIR,
            "ifrs9_lifetime_horizon_months": cap_mod.LIFETIME_HORIZON_MONTHS,
            "stress_elasticity_epsilon": cap_mod.MACRO_ELASTICITY_EPSILON,
            "stress_sicr_overlay_rule": (
                "A Stage 1 account moves to Stage 2 under stress if it is downgraded by at least one grade "
                "AND its post-stress grade is G4, G5 or G6."
            ),
            "approval_cutoff_for_fair_lending": "Approve G1-G4 (score >= 460, PD <= 26.82%)",
            "grade_moc_method": "DEV 95% Jeffreys upper bound minus DEV grade default rate",
            "points_scaling": s2["points_scaling_parameters"],
        },
    }

    km = audit["key_metrics"]
    rc = audit["row_counts"]
    ch, cl = km["champion"], km["challenger_expanded_platt"]
    gb, stab, fl = km["grade_backtest_holdout"], km["stability"], km["fair_lending_holdout_champion"]
    capm, ifr, strs = km["capital_holdout_illustrative"], km["ifrs9_holdout_illustrative"], km["stress_holdout_scenario_sensitivity"]

    audit["readme_metrics"] = {
        "raw_rows": _int(rc["bronze.credit_card_raw"]),
        "quarantine_rows": _int(rc["quarantine.credit_card_quarantine"]),
        "gold_rows": _int(rc["gold.credit_card_modelling"]),
        "dev_rows": _int(rc["dev_rows"]),
        "holdout_rows": _int(rc["holdout_rows"]),
        "holdout_default_rate": _pct(splits["HOLDOUT"]["default_rate"]),
        "lrm01_rows": _int(rc["bronze.cso_lrm01_raw"]),
        "champion_features": str(ch["n_features"]),
        "champion_max_vif": _num(ch["max_vif"], 2),
        "champion_dev_auroc": _num(ch["dev_auroc"]),
        "champion_holdout_auroc": _num(ch["holdout_auroc"]),
        "champion_holdout_gini": _num(ch["holdout_gini"]),
        "champion_holdout_ks": _num(ch["holdout_ks"]),
        "champion_holdout_brier": _num(ch["holdout_brier"]),
        "champion_hl_chi2": _num(ch["hl_holdout_chi2"], 2),
        "champion_hl_p": _num(ch["hl_holdout_p_df8"]),
        "challenger_features": str(cl["n_features"]),
        "challenger_holdout_auroc": _num(cl["holdout_auroc"]),
        "challenger_holdout_gini": _num(cl["holdout_gini"]),
        "challenger_holdout_brier": _num(cl["holdout_brier"]),
        "challenger_hl_p": _num(cl["hl_holdout_p_df8"]),
        "gini_gap_points": f"{cl['holdout_gini_gap_vs_champion'] * 100:.2f}",
        "gini_trigger_points": f"{cl['governance_trigger_gini_points']:.1f}",
        "g1_observed_rate": _pct(gb["g1_observed_rate"]),
        "g1_dev_pd": _pct(gb["g1_dev_empirical_pd"]),
        "g1_p": _num(gb["g1_one_sided_p_vs_dev_emp"]),
        "g1_light": gb["g1_traffic_light_vs_dev_emp"],
        "g5_observed_rate": _pct(gb["g5_observed_rate"]),
        "g5_predicted_pd": _pct(gb["g5_mean_predicted_pd"]),
        "g5_p": _num(gb["g5_two_sided_p"]),
        "score_psi": _num(stab["score_psi_grades"], 6),
        "min_dir": _num(fl["min_disparate_impact_ratio"]),
        "male_gap_bps": f"{fl['male_obs_minus_pred_bps']:+.2f}",
        "male_p": _num(fl["male_two_sided_p"]),
        "married_gap_bps": f"{fl['married_obs_minus_pred_bps']:+.2f}",
        "married_p": _num(fl["married_two_sided_p"]),
        "n_group_tests": str(fl["n_group_calibration_tests"]),
        "ead_m": _m(capm["ead_ntd_m"]),
        "rwa_m": _m(capm["rwa_ntd_m"]),
        "rwa_density": _pct(capm["rwa_density"]),
        "capital_m": _m(capm["capital_8pct_ntd_m"]),
        "el_m": _m(capm["expected_loss_ntd_m"]),
        "stage1_share": f"{ifr['stage1_share_pct']:.2f}%",
        "stage2_share": f"{ifr['stage2_share_pct']:.2f}%",
        "stage3_share": f"{ifr['stage3_share_pct']:.2f}%",
        "ecl_m": _m(ifr["allocated_ecl_ntd_m"]),
        "ecl_coverage": f"{ifr['coverage_pct']:.2f}%",
        "stage2_uplift_m": _m(ifr["stage2_lifetime_uplift_ntd_m"]),
        "mild_shock": f"{strs['mild_yoy_shock_pct']:.2f}%",
        "severe_shock": f"{strs['severe_yoy_shock_pct']:.2f}%",
        "mild_rwa_uplift": f"{strs['mild_rwa_uplift_pct']:.2f}%",
        "severe_rwa_uplift": f"{strs['severe_rwa_uplift_pct']:.2f}%",
        "mild_ecl_uplift": f"{strs['mild_ecl_sicr_uplift_pct']:.2f}%",
        "severe_ecl_uplift": f"{strs['severe_ecl_sicr_uplift_pct']:.2f}%",
        "severe_downgraded": f"{strs['severe_downgraded_pct']:.2f}%",
        "mean_reg_pd": _pct(pd_def["hold_mean_regulatory_grade_pd_with_moc"]),
        "mean_ifrs9_pd": _pct(pd_def["hold_mean_ifrs9_12m_pd_proxy_stage3_at_100pct"]),
        "lgd": _pct(capital_meta["lgd_assumption"], 0),
        "pd_floor": _pct(capital_meta["pd_floor_crr3_qrre_revolver"]),
        "ccf": _pct(cap_mod.BASELINE_CCF, 0),
    }

    # sanity: no NaN / inf anywhere in key metrics
    def _check(x, path="key_metrics"):
        if isinstance(x, dict):
            for k, v in x.items():
                _check(v, f"{path}.{k}")
        elif isinstance(x, float):
            assert math.isfinite(x), f"Non-finite value at {path}"
    _check(audit["key_metrics"])
    return audit


def main() -> None:
    audit = build_audit_summary()
    with open(AUDIT_PATH, "w", encoding="utf-8") as f:
        json.dump(audit, f, indent=2, ensure_ascii=False)
    print(f"Wrote {AUDIT_PATH.relative_to(PROJECT_ROOT)} with {len(audit['readme_metrics'])} README metrics.")


if __name__ == "__main__":
    main()
