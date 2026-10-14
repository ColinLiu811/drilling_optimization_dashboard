import math

import numpy as np
import pytest

from drillopt import hydraulics as hy


@pytest.fixture
def mud():
    return hy.Fluid.from_pv_yp(9.2, 14, 12)


@pytest.fixture
def well():
    return hy.Well(
        string=[hy.StringComponent("BHA", 6.5, 2.8125, 600),
                hy.StringComponent("HWDP", 5.0, 3.0, 900),
                hy.StringComponent("DP", 5.0, 4.276, 9487)],
        hole=[hy.HoleSection("Casing", 8.835, 6129), hy.HoleSection("OH", 8.75, 10987)],
        tvd=8560,
    )


def test_rheology_from_fann():
    f = hy.Fluid(10.0, theta600=40, theta300=26)
    assert f.pv == 14 and f.yp == 12
    assert f.n == pytest.approx(3.322 * math.log10(40 / 26))
    # Round trip through PV/YP.
    g = hy.Fluid.from_pv_yp(10.0, 14, 12)
    assert (g.theta600, g.theta300) == (40, 26)


def test_bit_pressure_drop_matches_textbook():
    # Applied Drilling Engineering Example 4.31: 12 ppg, 400 gpm, 3 x 12/32 jets.
    f = hy.Fluid.from_pv_yp(12.0, 20, 10)
    b = hy.bit_hydraulics(f, 400, hy.tfa_from_nozzles([12, 12, 12]), 8.5)
    assert b["pressure_drop"] == pytest.approx(1611, rel=0.005)
    assert b["jet_velocity"] == pytest.approx(387, rel=0.005)


def test_tfa():
    assert hy.tfa_from_nozzles([32]) == pytest.approx(math.pi / 4)
    assert hy.tfa_from_nozzles([12, 12, 12]) == pytest.approx(0.3313, abs=1e-4)


@pytest.mark.parametrize("model", ["power_law", "bingham"])
def test_losses_increase_with_flow(model, mud):
    grad = hy.MODELS[model]
    for d_out, d_in in [(4.276, 0.0), (8.75, 5.0)]:
        g = [grad(mud, q, d_out, d_in).gradient for q in (200, 400, 600, 800)]
        assert np.all(np.diff(g) > 0)


@pytest.mark.parametrize("model", ["power_law", "bingham"])
def test_regime_switches_to_turbulent(model, mud):
    grad = hy.MODELS[model]
    assert grad(mud, 50, 4.276).regime == "laminar"
    assert grad(mud, 900, 2.5).regime == "turbulent"


def test_annular_intervals_cover_hole(well):
    ivs = well.annular_intervals()
    assert sum(b - t for _, t, b, _, _ in ivs) == pytest.approx(well.bit_depth)
    # BHA is inside open hole, top drill pipe inside casing.
    assert any("BHA" in n and d == 8.75 for n, _, _, d, _ in ivs)
    assert any("DP" in n and d == 8.835 for n, _, _, d, _ in ivs)


def test_circulate_balances(well, mud):
    r = hy.circulate(well, mud, 650, hy.tfa_from_nozzles([13] * 6), extra_losses=300, rop=60)
    assert r.standpipe_pressure == pytest.approx(sum(r.breakdown().values()))
    assert r.ecd > mud.mud_weight
    assert r.ecd_with_cuttings > r.ecd
    # ECD from annular loss.
    assert r.ecd == pytest.approx(mud.mud_weight + r.annulus_loss / (0.052 * 8560))


def test_cuttings_concentration_limits():
    assert hy.cuttings_concentration(0, 8.75, 600) == 0
    assert 0 < hy.cuttings_concentration(100, 8.75, 600) < 0.1


def test_select_nozzles_close_to_target():
    target = hy.tfa_from_nozzles([12, 12, 13, 13, 13])
    combo = hy.select_nozzles(target, 5)
    assert hy.tfa_from_nozzles(combo) == pytest.approx(target, rel=1e-6)
    assert max(combo) - min(combo) <= 1


def test_optimum_flow_criteria():
    c, m = 0.01, 1.75
    hhp = hy.optimum_flow(4000, c, m, 100, 2000, "hhp")
    imp = hy.optimum_flow(4000, c, m, 100, 2000, "impact")
    assert hhp["parasitic_fraction"] == pytest.approx(1 / (m + 1))
    assert imp["parasitic_fraction"] == pytest.approx(2 / (m + 2))
    assert imp["q"] > hhp["q"]
    clamped = hy.optimum_flow(4000, c, m, 100, 300, "hhp")
    assert clamped["q"] == 300


def test_design_nozzles(well, mud):
    d = hy.design_nozzles(well, mud, 4500, 450, 850, 6)
    assert 450 <= d["q"] <= 850
    assert 1.4 < d["m"] < 2.0  # turbulent-dominated parasitic losses
    assert len(d["nozzles"]) == 6
    assert d["result"].standpipe_pressure == pytest.approx(4500, rel=0.05)


def test_design_nozzles_infeasible(well, mud):
    with pytest.raises(ValueError):
        hy.design_nozzles(well, mud, 500, 600, 800, 6)
