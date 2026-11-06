"""Step 5: Markov matrix, staging rules, ECL mechanics, Basel ASRF monotonicity."""
import numpy as np
import pytest

import capital_ifrs9_stress as cap

R = cap.QRRE_CORRELATION_R


# ------------------------------------------------------------------ Markov matrix
def test_markov_rows_sum_to_one(read_csv):
    mk = read_csv("step5_ifrs9_markov_matrix.csv")
    s = mk[["prob_to_state0_current", "prob_to_state1_30_89dpd", "prob_to_state2_90plus_dpd"]].sum(axis=1)
    np.testing.assert_allclose(s.values, 1.0, atol=1e-6)


def test_markov_counts_match_probabilities(read_csv):
    mk = read_csv("step5_ifrs9_markov_matrix.csv")
    counts = mk[["count_to_state0_current", "count_to_state1_30_89dpd", "count_to_state2_90plus_dpd"]].sum(axis=1)
    assert (counts == mk["transition_count_total"]).all()
    np.testing.assert_allclose(mk["count_to_state0_current"] / mk["transition_count_total"],
                               mk["prob_to_state0_current"], atol=1e-6)


def test_no_current_to_90plus_in_one_month(read_csv):
    mk = read_csv("step5_ifrs9_markov_matrix.csv")
    cur = mk[mk["from_state_code"] == 0]
    assert (cur["count_to_state2_90plus_dpd"] == 0).all()


def test_cumulative_default_curves_absorbing_and_nondecreasing():
    p = np.array([[0.94, 0.06, 0.0], [0.25, 0.68, 0.07], [0.1, 0.36, 0.54]])
    c = cap.compute_markov_cumulative_default_curves(p, horizon_months=36)
    assert c.shape == (3, 37)
    assert np.all(np.diff(c, axis=1) >= -1e-12)
    np.testing.assert_allclose(c[2], 1.0)          # default state is absorbing
    assert c[0, 0] == 0.0 and c[1, 0] == 0.0


# ------------------------------------------------------------------ staging rules
@pytest.mark.parametrize("pay,state", [(-2, 0), (-1, 0), (0, 0), (1, 1), (2, 1), (3, 2), (8, 2)])
def test_three_state_mapping(pay, state):
    assert cap.map_pay_to_ifrs9_state_3state(np.array([pay]))[0] == state


def test_stage_table_shares_and_rules(read_csv):
    st = read_csv("step5_ifrs9_stage_summary.csv")
    for split, g in st.groupby("sample_split"):
        parts = g[g["ifrs9_stage"].isin([1, 2, 3])]
        tot = g[g["ifrs9_stage"] == 0].iloc[0]
        assert parts["account_count"].sum() == tot["account_count"], split
        s3 = parts[parts["ifrs9_stage"] == 3].iloc[0]
        assert s3["mean_12m_pd_proxy"] == 1.0                    # Stage 3: PD = 100%
        s1 = parts[parts["ifrs9_stage"] == 1].iloc[0]
        assert abs(s1["ifrs9_allocated_ecl_ntd_m"] - s1["ecl_if_12m_ntd_m"]) < 1e-6        # Stage 1: 12-month ECL
        s2 = parts[parts["ifrs9_stage"] == 2].iloc[0]
        assert abs(s2["ifrs9_allocated_ecl_ntd_m"] - s2["ecl_if_lifetime_36m_ntd_m"]) < 1e-6  # Stage 2: lifetime ECL


def test_ecl_mechanics_on_known_inputs():
    p = np.array([[0.94, 0.06, 0.0], [0.25, 0.68, 0.07], [0.1, 0.36, 0.54]])
    c = cap.compute_markov_cumulative_default_curves(p, horizon_months=36)
    pd12 = np.array([0.05, 0.40, 0.70])
    stage = np.array([1, 2, 3])
    pd12_eff, pdlt, e12, elt, ealloc = cap.compute_discounted_ecl_per_unit_ead_markov3(pd12, stage, c, lgd=0.85)
    np.testing.assert_allclose(pd12_eff[:2], pd12[:2], rtol=1e-9)     # 12-month PD reproduced exactly
    assert pd12_eff[2] == 1.0 and pdlt[2] == 1.0                       # Stage 3 override
    assert e12[2] == pytest.approx(0.85) and elt[2] == pytest.approx(0.85)
    assert np.all(pdlt[:2] >= pd12_eff[:2]) and np.all(elt >= e12 - 1e-12)
    assert ealloc[0] == e12[0] and ealloc[1] == elt[1] and ealloc[2] == elt[2]


# ------------------------------------------------------------------ Basel ASRF
def _peak():
    grid = np.linspace(0.001, 0.99, 20000)
    k = cap.basel_asrf_k(grid, 1.0, R)
    return grid[np.argmax(k)]


def test_k_peak_location():
    assert _peak() == pytest.approx(cap.VASICEK_UL_PEAK_PD, abs=0.002)


def test_k_increases_with_pd_below_peak():
    grid = np.linspace(0.001, _peak() - 0.005, 2000)
    assert np.all(np.diff(cap.basel_asrf_k(grid, 0.85, R)) > 0)


def test_k_decreases_with_pd_above_peak():
    grid = np.linspace(_peak() + 0.005, 0.99, 2000)
    assert np.all(np.diff(cap.basel_asrf_k(grid, 0.85, R)) < 0)


def test_el_plus_ul_increases_with_pd_everywhere():
    grid = np.linspace(0.001, 0.99, 5000)
    assert np.all(np.diff(cap.basel_total_stressed_loss_rate(grid, 0.85, R)) > 0)


@pytest.mark.parametrize("pd_val", [0.01, 0.1, 0.3, 0.6])
def test_k_rwa_and_el_ul_increase_with_lgd(pd_val):
    lgds = np.linspace(0.5, 0.95, 10)
    k = np.array([float(cap.basel_asrf_k(pd_val, l, R)) for l in lgds])
    loss = np.array([float(cap.basel_total_stressed_loss_rate(pd_val, l, R)) for l in lgds])
    assert np.all(np.diff(k) > 0)
    assert np.all(np.diff(12.5 * k * 1000.0) > 0)
    assert np.all(np.diff(loss) > 0)


def test_pd_floor_applied():
    assert float(cap.basel_asrf_k(0.0, 0.85, R)) == pytest.approx(float(cap.basel_asrf_k(cap.PD_REGULATORY_FLOOR, 0.85, R)))
    assert cap.PD_REGULATORY_FLOOR == 0.001


def test_capital_table_reproduces_formula(read_csv):
    capt = read_csv("step5_basel_capital_by_grade.csv")
    k = cap.basel_asrf_k(capt["regulatory_pd_with_moc_proxy"].values, cap.BASELINE_LGD, R)
    np.testing.assert_allclose(k, capt["capital_factor_k_reg"].values, atol=1e-5)
    np.testing.assert_allclose(12.5 * capt["capital_factor_k_reg"] * capt["ead_ntd_m"], capt["rwa_reg_ntd_m"], rtol=1e-3)


def test_sensitivity_grid_monotonic(read_csv):
    s = read_csv("step5_basel_capital_sensitivity.csv")
    for split, g in s.groupby("sample_split"):
        piv = g.pivot(index="lgd_assumption", columns="ccf_assumption", values="total_rwa_ntd_m").sort_index().sort_index(axis=1)
        assert np.all(np.diff(piv.values, axis=0) > 0), split   # RWA rises with LGD
        assert np.all(np.diff(piv.values, axis=1) > 0), split   # RWA rises with CCF
