import numpy as np
import pandas as pd
import pytest

from drillopt import optimize as opt
from drillopt import rop
from drillopt.data import PROCESSED_PATH, bin_by_depth, filter_drilling, load_processed


def synthetic(n=1500, seed=0):
    """Power-law ROP with known exponents plus a WOB founder above 45 klbf."""
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        "depth": np.arange(n, dtype=float) + 5000,
        "wob": rng.uniform(10, 40, n),
        "rpm": rng.uniform(40, 160, n),
        "flow": rng.uniform(400, 800, n),
        "torque": rng.uniform(2000, 8000, n),
        "bit_size": 8.75,
    })
    df["rop"] = 0.5 * df.wob ** 0.8 * df.rpm ** 0.5 * df.flow ** 0.2 * rng.lognormal(0, 0.05, n)
    return df


def test_power_law_recovers_exponents():
    df = synthetic()
    m = rop.PowerLawROP().fit(df[rop.CONTROL_FEATURES], df["rop"])
    np.testing.assert_allclose(m.exponents_, [0.8, 0.5, 0.2], atol=0.03)
    assert m.K_ == pytest.approx(0.5, rel=0.2)


def test_ridge_shrinks_exponents():
    df = synthetic()
    plain = rop.PowerLawROP().fit(df[rop.CONTROL_FEATURES], df["rop"])
    ridge = rop.PowerLawROP(alpha=500).fit(df[rop.CONTROL_FEATURES], df["rop"])
    assert np.abs(ridge.exponents_).sum() < np.abs(plain.exponents_).sum()


def test_mse_teale():
    # Thrust term only when torque is zero: 4W/(pi d^2).
    assert rop.mechanical_specific_energy(10, 100, 0, 50, 8) == pytest.approx(4 * 10000 / (np.pi * 64))
    # Rotary term dominates at low ROP.
    hi = rop.mechanical_specific_energy(30, 100, 8000, 5, 8.75)
    lo = rop.mechanical_specific_energy(30, 100, 8000, 100, 8.75)
    assert hi > lo


def test_depth_blocks_contiguous():
    depth = np.random.default_rng(1).permutation(np.arange(100.0))
    blocks = rop.depth_blocks(depth, 4)
    for b in range(4):
        d = np.sort(depth[blocks == b])
        assert np.all(np.diff(d) == 1)


@pytest.mark.parametrize("kind", list(rop.MODEL_NAMES))
def test_models_fit_synthetic(kind):
    m = rop.fit_rop_model(synthetic(), kind, n_blocks=3)
    assert m.test_metrics["r2"] > 0.6
    assert len(m.cv) == 3


def test_trailing_features_do_not_leak():
    df = rop.add_drilling_features(synthetic(), window=10)
    # rop_trail at row i only uses rows < i.
    i = 50
    expected = df["rop"].iloc[i - 10:i].median()
    assert df["rop_trail"].iloc[i] == pytest.approx(expected)
    assert "mse" in df and "mse_trail" in df


def test_optimizer_finds_upper_bounds_for_monotone_model():
    df = synthetic()
    m = rop.fit_rop_model(df, "power_law", run_cv=False)
    lim = opt.Limits((10, 40), (40, 160), (400, 800))
    res = opt.optimize(m, df.iloc[0].to_dict(), lim, n=9)
    assert (res["wob"], res["rpm"], res["flow"]) == (40, 160, 800)
    assert res["gain_pct"] >= 0
    sl = opt.slice_2d(res["grid"], "wob", "rpm", "flow", 800)
    assert sl.shape == (9, 9)


def test_limits_from_data():
    lim = opt.Limits.from_data(synthetic())
    assert 10 < lim.wob[0] < lim.wob[1] < 40


def test_filter_and_bin():
    raw = pd.DataFrame({
        "time": pd.date_range("2020-01-01", periods=6, freq="10s"),
        "depth": [100.2, 100.6, 101.1, 101.5, 50.0, 102.3],
        "on_bottom": [1, 1, 1, 1, 1, 0],
        "wob": [20, 22, 25, 25, 25, 25], "rpm": [60] * 6, "flow": [600] * 6,
        "rop": [50, 55, 60, 65, 70, 75], "spp": [3000] * 6, "bit_size": [8.75] * 6,
    })
    f = filter_drilling(raw)
    assert len(f) == 4  # off-bottom row and the depth step backwards removed
    b = bin_by_depth(f)
    assert list(b["depth"]) == [100.0, 101.0]
    assert b["rop"].tolist() == [52.5, 62.5]


@pytest.mark.skipif(not PROCESSED_PATH.exists(), reason="processed dataset not present")
def test_bundled_dataset():
    df = load_processed()
    assert len(df) > 9000
    assert {"wob", "rpm", "flow", "rop", "spp", "section"} <= set(df.columns)
    assert df["rop"].between(0.5, 400).all()
