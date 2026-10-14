import numpy as np
import pandas as pd

from drillopt.realtime import AlertConfig, RealtimeMonitor, ReplayStream


def steady_well(n=300, seed=0):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        "depth": np.arange(n, dtype=float) + 1000,
        "wob": rng.uniform(25, 35, n), "rpm": rng.uniform(50, 70, n),
        "flow": 600 + rng.normal(0, 3, n), "spp": 3000 + rng.normal(0, 20, n),
        "torque": 5000 + rng.normal(0, 200, n), "bit_size": 8.75,
    })
    df["rop"] = 2.0 * df.wob ** 0.7 * df.rpm ** 0.4 * rng.lognormal(0, 0.05, n)
    return df


def run(df, **kw):
    mon = RealtimeMonitor(**kw)
    for rec in ReplayStream(df):
        mon.update(rec)
    return mon


def test_stream_batches_and_seek():
    s = ReplayStream(steady_well(50))
    assert len(s.next_batch(10)) == 10 and s.cursor == 10
    s.seek_depth(1040)
    assert s.next_batch(1)[0]["depth"] == 1040
    s.next_batch(100)
    assert s.done and s.progress == 1.0


def test_quiet_well_raises_no_alerts():
    mon = run(steady_well())
    assert mon.alerts_frame().empty
    assert mon.model is not None
    assert len(mon.buffer) == mon.window


def test_washout_triggers_spp_alert():
    df = steady_well()
    df.loc[df.index >= 250, "spp"] *= 0.8  # 20 % pressure loss at constant flow
    alerts = run(df).alerts_frame()
    spp = alerts[alerts["kind"] == "SPP_DEVIATION"]
    assert len(spp) >= 1
    assert spp["depth"].iloc[0] >= 1250
    assert "washout" in spp["message"].iloc[0]


def test_rop_drop_and_mse_spike():
    df = steady_well()
    df.loc[df.index >= 200, "rop"] *= 0.3  # e.g. balled bit
    alerts = run(df).alerts_frame()
    kinds = set(alerts["kind"])
    assert {"ROP_BELOW_MODEL", "MSE_SPIKE"} <= kinds
    assert alerts["depth"].min() >= 1200


def test_persistence_filters_single_spikes():
    df = steady_well()
    df.loc[220, "spp"] *= 0.7  # one bad sample
    alerts = run(df, config=AlertConfig(persistence=3)).alerts_frame()
    assert (alerts["kind"] != "SPP_DEVIATION").all()
