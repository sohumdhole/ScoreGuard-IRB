"""Step 6: provenance and documentation consistency.

- audit_summary.json must equal a fresh rebuild from results/ (ignoring its own timestamp).
- README.md and docs/ must equal a fresh render from audit_summary.json and results/.
- Every value quoted in the README must be one of the audit_summary readme_metrics.
- Where the DuckDB database is present, key audit figures are recomputed from raw rows.
"""
import json
import re

import numpy as np
import pytest

import build_audit_summary as bas
import render_docs as rd


def _strip(d: dict) -> dict:
    d = json.loads(json.dumps(d))
    d.pop("generated_at_utc", None)
    return d


def test_audit_summary_matches_fresh_rebuild(audit):
    fresh = bas.build_audit_summary()
    assert _strip(fresh) == _strip(audit), "results/audit_summary.json is stale: run src/build_audit_summary.py"


@pytest.mark.parametrize("name", list(rd.OUTPUTS))
def test_rendered_docs_match_disk(name):
    if not rd.OUTPUTS[name].exists() and "private" in rd.OUTPUTS[name].parts:
        pytest.skip(f"{name} is a private, git-ignored file (run src/render_docs.py to create it)")
    fresh = rd.render_all()[name]
    on_disk = rd.OUTPUTS[name].read_text(encoding="utf-8")
    assert on_disk == fresh, f"{name} has drifted from results: run src/render_docs.py"


def test_readme_numbers_all_trace_to_audit(audit):
    """Every number in the README's results sections must appear among the audit readme_metrics values
    (or be one of a small set of structural constants such as step numbers and stated parameters)."""
    readme = rd.OUTPUTS["README.md"].read_text(encoding="utf-8")
    start = readme.index("## Key results")
    end = readme.index("## Repository layout")
    section = readme[start:end]

    metric_values = set(audit["readme_metrics"].values())
    # numeric tokens like 0.7598, 1,761.22, 22.15%, +190.50, 5.0
    # numbers glued to letters (G5, R2) are labels, not results
    tokens = re.findall(r"(?<![A-Za-z_])[+-]?\d[\d,]*\.?\d*%?", section)
    # structural constants, not model results: stage/step numbers, "IFRS 9", HL groups and df (10, 8),
    # the Basel 8% capital ratio, QRRE correlation 0.04, 12-month horizon, Stage 3 = 100%, data years
    structural = {"1", "2", "3", "8", "9", "10", "12", "0.04", "8%", "100%", "2005", "2009"}
    unexplained = []
    for t in tokens:
        bare = t.lstrip("+")
        if t in metric_values or bare in metric_values or t in structural or bare in structural:
            continue
        unexplained.append(t)
    assert not unexplained, f"README numbers not traceable to audit_summary.readme_metrics: {sorted(set(unexplained))}"


def test_baseline_mean_pd_definitions(audit, read_csv):
    km = audit["key_metrics"]["baseline_mean_pd_definitions_holdout"]
    stress = read_csv("step5_stress_scenario_summary.csv")
    stage = read_csv("step5_ifrs9_stage_summary.csv")
    base = stress[(stress.sample_split == "HOLDOUT") & (stress.scenario_id == "S0_BASELINE")].iloc[0]
    tot = stage[(stage.sample_split == "HOLDOUT") & (stage.ifrs9_stage == 0)].iloc[0]
    assert km["hold_mean_regulatory_grade_pd_with_moc"] == pytest.approx(base["mean_stressed_regulatory_pd_proxy"])
    assert km["hold_mean_ifrs9_12m_pd_proxy_stage3_at_100pct"] == pytest.approx(tot["mean_12m_pd_proxy"])
    assert "Stage 3" in km["definitions"]["mean_ifrs9_12m_pd_proxy"]


# ------------------------------------------------------------------ independent recomputation from the database
def test_audit_metrics_recomputed_from_database(db, audit):
    from sklearn.metrics import roc_auc_score

    df = db.sql("""
        select sample_split, default_flag, champion_pd, irb_grade_pd_reg, ifrs9_pd_12m,
               ifrs9_ecl_allocated_ntd, ead_ccf50
        from gold.credit_card_modelling
    """).df()
    h = df[df.sample_split == "HOLDOUT"]
    km = audit["key_metrics"]

    assert len(h) == audit["row_counts"]["holdout_rows"]
    assert int(h.default_flag.sum()) == audit["row_counts"]["holdout_defaults"]
    assert roc_auc_score(h.default_flag, h.champion_pd) == pytest.approx(km["champion"]["holdout_auroc"], abs=1e-6)
    assert h.irb_grade_pd_reg.mean() == pytest.approx(km["baseline_mean_pd_definitions_holdout"]["hold_mean_regulatory_grade_pd_with_moc"], abs=1e-6)
    assert h.ifrs9_pd_12m.mean() == pytest.approx(km["baseline_mean_pd_definitions_holdout"]["hold_mean_ifrs9_12m_pd_proxy_stage3_at_100pct"], abs=1e-6)
    assert h.ifrs9_ecl_allocated_ntd.sum() / 1e6 == pytest.approx(km["ifrs9_holdout_illustrative"]["allocated_ecl_ntd_m"], abs=0.01)
    assert h.ead_ccf50.sum() / 1e6 == pytest.approx(km["capital_holdout_illustrative"]["ead_ntd_m"], abs=0.01)


def test_raw_file_hashes_match(audit):
    from pathlib import Path
    raw = Path(bas.RAW_DIR)
    if not raw.exists():
        pytest.skip("data/raw not present")
    for name, meta in audit["data_sources"]["raw_file_hashes"].items():
        assert bas._sha256(raw / name) == meta["sha256"], name
