"""Real-time drilling data handling.

:class:`ReplayStream` replays a historical well as if it were a live WITS/EDR
feed. :class:`RealtimeMonitor` consumes records one at a time and keeps a
bounded rolling buffer, recomputes MSE, refits a local ROP model every few
feet and raises alerts for common drilling dysfunctions:

* ``ROP_BELOW_MODEL`` - measured ROP well below what the local model expects
  for the current WOB/RPM/flow (bit balling, dull bit, formation change).
* ``MSE_SPIKE`` - MSE jumps relative to its recent baseline (vibration,
  founder, inefficient energy transfer).
* ``SPP_DEVIATION`` - standpipe pressure departs from the trend at roughly
  constant flow (pressure drop -> possible washout, rise -> plugged jet or
  packing off).
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Iterator

import numpy as np
import pandas as pd

from .rop import PowerLawROP, mechanical_specific_energy


class ReplayStream:
    """Replays a dataframe row by row, ordered by depth (or time)."""

    def __init__(self, df: pd.DataFrame, order_by: str = "depth", start: int = 0):
        self.df = df.sort_values(order_by).reset_index(drop=True)
        self.cursor = start

    def __len__(self) -> int:
        return len(self.df)

    @property
    def done(self) -> bool:
        return self.cursor >= len(self.df)

    @property
    def progress(self) -> float:
        return self.cursor / max(len(self.df), 1)

    def next_batch(self, n: int = 1) -> list[dict]:
        batch = self.df.iloc[self.cursor:self.cursor + n].to_dict("records")
        self.cursor += len(batch)
        return batch

    def seek_depth(self, depth: float) -> None:
        self.cursor = int(np.searchsorted(self.df["depth"].to_numpy(), depth))

    def __iter__(self) -> Iterator[dict]:
        while not self.done:
            yield self.next_batch(1)[0]


@dataclass
class AlertConfig:
    rop_tolerance: float = 0.40      # alert when ROP < (1 - tol) * expected
    mse_z: float = 3.0               # robust z-score threshold
    spp_tolerance: float = 0.10      # fractional SPP change at constant flow
    flow_tolerance: float = 0.05     # flow considered constant within +/-5 %
    persistence: int = 3             # consecutive samples before alerting


@dataclass
class Alert:
    depth: float
    kind: str
    message: str
    severity: str = "warning"


@dataclass
class RealtimeMonitor:
    """Stateful processor for a stream of drilling records."""

    window: int = 150          # samples kept in the rolling buffer
    refit_every: int = 25      # refit the local ROP model every N samples
    min_fit_samples: int = 40
    config: AlertConfig = field(default_factory=AlertConfig)

    def __post_init__(self):
        self.buffer: deque[dict] = deque(maxlen=self.window)
        self.history: list[dict] = []
        self.alerts: list[Alert] = []
        self.model: PowerLawROP | None = None
        self._since_fit = 0
        self._streak: dict[str, int] = {}

    # -- model -------------------------------------------------------------

    def _refit(self) -> None:
        buf = pd.DataFrame(self.buffer).dropna(subset=["wob", "rpm", "flow", "rop"])
        if len(buf) < self.min_fit_samples:
            return
        self.model = PowerLawROP().fit(buf[["wob", "rpm", "flow"]], buf["rop"])
        self._rop_cap = float(buf["rop"].max())
        self._since_fit = 0

    def expected_rop(self, rec: dict) -> float | None:
        if self.model is None:
            return None
        x = np.array([[rec["wob"], rec["rpm"], rec["flow"]]])
        # Cap extrapolation at the fastest ROP seen in the buffer.
        return float(min(self.model.predict(x)[0], self._rop_cap))

    # -- alerts ------------------------------------------------------------

    def _check(self, kind: str, condition: bool, rec: dict, message: str, severity="warning"):
        self._streak[kind] = self._streak.get(kind, 0) + 1 if condition else 0
        if self._streak[kind] == self.config.persistence:
            alert = Alert(rec["depth"], kind, message, severity)
            self.alerts.append(alert)
            return alert
        return None

    def _baseline(self, key: str, n: int = 50) -> np.ndarray:
        vals = [r.get(key) for r in list(self.buffer)[-n - 1:-1]]
        return np.array([v for v in vals if v is not None and np.isfinite(v)])

    # -- main entry point --------------------------------------------------

    def update(self, rec: dict) -> dict:
        """Ingest one record; returns it enriched with derived values and alerts."""
        rec = dict(rec)
        if {"torque", "bit_size"} <= rec.keys():
            rec["mse"] = float(mechanical_specific_energy(
                rec["wob"], rec["rpm"], rec["torque"], rec["rop"], rec["bit_size"])) / 1000.0
        rec["rop_expected"] = self.expected_rop(rec)
        self.buffer.append(rec)

        new_alerts = []
        cfg = self.config
        if rec["rop_expected"]:
            a = self._check("ROP_BELOW_MODEL", rec["rop"] < (1 - cfg.rop_tolerance) * rec["rop_expected"],
                            rec, f"ROP {rec['rop']:.0f} ft/hr vs {rec['rop_expected']:.0f} expected")
            new_alerts.append(a)

        mse_hist = self._baseline("mse")
        if len(mse_hist) >= 20 and rec.get("mse") is not None:
            med = np.median(mse_hist)
            mad = np.median(np.abs(mse_hist - med)) * 1.4826 or 1e-6
            z = (rec["mse"] - med) / mad
            rec["mse_z"] = float(z)
            new_alerts.append(self._check("MSE_SPIKE", z > cfg.mse_z, rec,
                                          f"MSE {rec['mse']:.0f} kpsi, {z:.1f} sigma above baseline"))

        spp_hist, flow_hist = self._baseline("spp", 20), self._baseline("flow", 20)
        if len(spp_hist) >= 10 and rec.get("spp"):
            flow_ref, spp_ref = np.median(flow_hist), np.median(spp_hist)
            steady = abs(rec["flow"] / flow_ref - 1) < cfg.flow_tolerance
            change = rec["spp"] / spp_ref - 1
            rec["spp_change"] = float(change)
            direction = "drop - check for washout" if change < 0 else "rise - check for plugged jet / pack-off"
            new_alerts.append(self._check(
                "SPP_DEVIATION", steady and abs(change) > cfg.spp_tolerance, rec,
                f"SPP {change:+.0%} at steady flow ({direction})", "critical"))

        self._since_fit += 1
        if self.model is None or self._since_fit >= self.refit_every:
            self._refit()

        rec["alerts"] = [a.kind for a in new_alerts if a]
        self.history.append(rec)
        return rec

    def frame(self) -> pd.DataFrame:
        return pd.DataFrame(self.history)

    def alerts_frame(self) -> pd.DataFrame:
        return pd.DataFrame([a.__dict__ for a in self.alerts],
                            columns=["depth", "kind", "message", "severity"])
