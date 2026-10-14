"""Drilling parameter optimization on top of a fitted ROP model.

Given the current context (depth, inclination, recent drillability), search
the controllable parameters - WOB, RPM and flow rate - for the combination
that maximises predicted ROP within operating limits.

The search is an exhaustive grid rather than a gradient method because the
tree-based models are piecewise constant, and because the full response
surface is what the dashboard plots anyway.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .rop import ROPModel


@dataclass
class Limits:
    wob: tuple[float, float]
    rpm: tuple[float, float]
    flow: tuple[float, float]

    @classmethod
    def from_data(cls, df: pd.DataFrame, lo: float = 0.05, hi: float = 0.95) -> "Limits":
        """Limits spanning the observed operating envelope.

        Restricting the search to where the model has data avoids
        recommending extrapolated set-points (a power law with a positive
        WOB exponent would otherwise always push to maximum weight).
        """
        q = df[["wob", "rpm", "flow"]].quantile([lo, hi])
        return cls(*(tuple(q[c]) for c in ("wob", "rpm", "flow")))


def response_grid(model: ROPModel, context: dict, limits: Limits, n: int = 25) -> pd.DataFrame:
    """Predicted ROP over a WOB x RPM x flow grid at fixed context."""
    axes = {k: np.linspace(*getattr(limits, k), n) for k in ("wob", "rpm", "flow")}
    mesh = np.meshgrid(axes["wob"], axes["rpm"], axes["flow"], indexing="ij")
    grid = pd.DataFrame({k: m.ravel() for k, m in zip(axes, mesh)})
    for f in model.features:
        if f not in grid:
            grid[f] = context[f]
    grid["rop_pred"] = model.predict(grid)
    return grid


def optimize(model: ROPModel, context: dict, limits: Limits, n: int = 25) -> dict:
    """Best set-point on the grid, plus the predicted gain over ``context``."""
    grid = response_grid(model, context, limits, n)
    best = grid.loc[grid["rop_pred"].idxmax()]
    current = pd.DataFrame([{f: context[f] for f in model.features}])
    current_rop = float(model.predict(current)[0])
    if best["rop_pred"] <= current_rop:
        # Current set-point already beats everything inside the limits.
        best = pd.Series({"wob": context["wob"], "rpm": context["rpm"],
                          "flow": context["flow"], "rop_pred": current_rop})
    return {
        "wob": float(best["wob"]),
        "rpm": float(best["rpm"]),
        "flow": float(best["flow"]),
        "rop_pred": float(best["rop_pred"]),
        "current_rop_pred": current_rop,
        "gain_pct": float(100.0 * (best["rop_pred"] / max(current_rop, 1e-6) - 1.0)),
        "grid": grid,
    }


def slice_2d(grid: pd.DataFrame, x: str, y: str, fixed: str, value: float) -> pd.DataFrame:
    """Pivot a 2-D slice of the response grid at the grid level nearest ``value``."""
    levels = np.sort(grid[fixed].unique())
    nearest = levels[np.abs(levels - value).argmin()]
    sl = grid[np.isclose(grid[fixed], nearest)]
    return sl.pivot_table(index=y, columns=x, values="rop_pred")


def partial_dependence(model: ROPModel, data: pd.DataFrame, feature: str, n: int = 40) -> pd.DataFrame:
    """Average predicted ROP as one feature is swept over its observed range."""
    lo, hi = data[feature].quantile([0.02, 0.98])
    xs = np.linspace(lo, hi, n)
    sample = data[model.features].dropna().sample(min(len(data), 400), random_state=0)
    out = []
    for x in xs:
        s = sample.copy()
        s[feature] = x
        out.append(model.predict(s).mean())
    return pd.DataFrame({feature: xs, "rop_pred": out})
