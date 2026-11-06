"""Step 2: WoE/IV maths, scorecard points direction, grade boundaries."""
import math

import numpy as np
import pandas as pd
import pytest

import scorecard_champion as sc


# ------------------------------------------------------------------ WoE / IV maths
def test_woe_iv_on_known_example():
    # Three discrete levels with hand-computable counts:
    #   x=0: 400 goods, 100 bads ; x=1: 300 goods, 200 bads ; x=2: 100 goods, 400 bads
    x = np.repeat([0, 1, 2], 500)
    y = np.concatenate([
        np.r_[np.zeros(400), np.ones(100)],
        np.r_[np.zeros(300), np.ones(200)],
        np.r_[np.zeros(100), np.ones(400)],
    ]).astype(int)
    edges, table, iv = sc.fit_monotonic_woe_bins(pd.Series(x), y, max_bins=6, min_bin_pct=0.05, is_discrete=True)

    goods, bads = np.array([400, 300, 100]), np.array([100, 200, 400])
    dg, db = goods / goods.sum(), bads / bads.sum()
    woe = np.log(dg / db)
    iv_expected = float(np.sum((dg - db) * woe))

    assert len(table) == 3
    np.testing.assert_allclose(table["woe"].values, woe, rtol=1e-12)
    assert math.isclose(iv, iv_expected, rel_tol=1e-12)
    # WoE = ln(%good / %bad): riskier bins have lower WoE
    assert np.all(np.diff(table["woe"].values) < 0)


def test_woe_bins_merge_to_monotonic():
    # middle level breaks monotonicity, so the fitter must merge it away
    x = np.repeat([0, 1, 2], 1000)
    y = np.concatenate([
        np.r_[np.zeros(900), np.ones(100)],   # 10%
        np.r_[np.zeros(950), np.ones(50)],    # 5% (violates increasing pattern)
        np.r_[np.zeros(600), np.ones(400)],   # 40%
    ]).astype(int)
    _, table, _ = sc.fit_monotonic_woe_bins(pd.Series(x), y, is_discrete=True)
    br = table["bad_rate"].values
    assert np.all(np.diff(br) > 0) or np.all(np.diff(br) < 0)


def test_executed_bins_are_monotonic(read_csv):
    bins = read_csv("step2_woe_bin_tables.csv")
    for feat, b in bins.groupby("feature"):
        d = np.diff(b.sort_values("bin_index")["dev_bad_rate"].values)
        assert np.all(d > 0) or np.all(d < 0), feat


# ------------------------------------------------------------------ points direction
def test_points_fall_as_bin_risk_rises(read_csv):
    bins = read_csv("step2_woe_bin_tables.csv")
    for feat, b in bins.groupby("feature"):
        b = b.sort_values("dev_bad_rate")
        assert np.all(np.diff(b["scorecard_points"].values) < 0), feat


def test_all_coefficients_negative(results_dir):
    import json
    with open(results_dir / "step2_scorecard_summary.json", encoding="utf-8") as f:
        s2 = json.load(f)
    rows = s2["final_model_coefficients_and_vif"]
    assert [r["feature"] for r in rows] == sc.FINAL_SCORECARD_FEATURES
    for r in rows:
        assert r["coefficient_beta"] < 0, r["feature"]
        assert r["vif_dev"] < s2["vif_threshold_stated"], r["feature"]


def test_points_scaling_identity():
    factor = sc.PDO / math.log(2.0)
    offset = sc.BASE_POINTS - factor * math.log(sc.BASE_ODDS_GOOD_TO_BAD)
    # a PD of exactly 1/(1+19) must score the base points
    pd5 = 1.0 / (1.0 + sc.BASE_ODDS_GOOD_TO_BAD)
    assert math.isclose(offset - factor * math.log(pd5 / (1 - pd5)), sc.BASE_POINTS, rel_tol=1e-12)
    # doubling the good:bad odds adds PDO points
    pd_double = 1.0 / (1.0 + 2 * sc.BASE_ODDS_GOOD_TO_BAD)
    assert math.isclose(offset - factor * math.log(pd_double / (1 - pd_double)) - sc.BASE_POINTS, sc.PDO, rel_tol=1e-12)


# ------------------------------------------------------------------ grade boundaries
@pytest.mark.parametrize("score,grade", [
    (399.99, "G6"), (400.0, "G5"), (459.99, "G5"), (460.0, "G4"), (499.99, "G4"),
    (500.0, "G3"), (534.99, "G3"), (535.0, "G2"), (564.99, "G2"), (565.0, "G1"), (700.0, "G1"),
])
def test_grade_boundaries(score, grade):
    import capital_ifrs9_stress as cap
    assert str(sc.assign_rating_grade(np.array([score])).iloc[0]).startswith(grade)
    assert cap.assign_rating_grade_from_score(np.array([score]))[0].startswith(grade)


def test_grade_table_matches_stated_bands(read_csv):
    g = read_csv("step2_rating_grades.csv")
    assert list(g["score_band"]) == ["[565, +inf)", "[535, 565)", "[500, 535)", "[460, 500)", "[400, 460)", "(-inf, 400)"]
    # every adjacent grade is riskier on DEV (the reason G1/G2 were merged)
    assert np.all(np.diff(g["dev_observed_default_rate"].values) > 0)
