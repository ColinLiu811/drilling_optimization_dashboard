"""Drilling hydraulics in oilfield units.

Frictional pressure losses, equivalent circulating density and bit
hydraulics/nozzle selection following Bourgoyne et al., *Applied Drilling
Engineering* (SPE Textbook Vol. 2), Chapter 4.

Units throughout: density ppg, flow gal/min, diameters in, lengths ft,
pressure psi, plastic viscosity cP, yield point lbf/100 ft^2, power-law
consistency index K in equivalent cP.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import brentq

CD_DEFAULT = 0.95  # nozzle discharge coefficient
NOZZLE_SIZES_32 = list(range(7, 25))  # standard jet sizes, 32nds of an inch


# --------------------------------------------------------------------------
# Rheology
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class Fluid:
    """Drilling fluid described by its density and Fann viscometer readings."""

    mud_weight: float  # ppg
    theta600: float
    theta300: float

    @property
    def pv(self) -> float:
        """Bingham plastic viscosity, cP."""
        return self.theta600 - self.theta300

    @property
    def yp(self) -> float:
        """Bingham yield point, lbf/100 ft^2."""
        return self.theta300 - self.pv

    @property
    def n(self) -> float:
        """Power-law flow behaviour index."""
        return 3.322 * math.log10(self.theta600 / self.theta300)

    @property
    def K(self) -> float:
        """Power-law consistency index, equivalent cP."""
        return 510.0 * self.theta300 / 511.0 ** self.n

    @classmethod
    def from_pv_yp(cls, mud_weight: float, pv: float, yp: float) -> "Fluid":
        return cls(mud_weight, theta600=2 * pv + yp, theta300=pv + yp)


# --------------------------------------------------------------------------
# Frictional pressure gradients
# --------------------------------------------------------------------------

@dataclass
class FlowResult:
    velocity: float      # ft/s
    reynolds: float
    regime: str          # "laminar" | "turbulent"
    gradient: float      # psi/ft
    length: float = 0.0  # ft
    name: str = ""

    @property
    def loss(self) -> float:
        return self.gradient * self.length


def _dodge_metzner(re: float, n: float) -> float:
    """Fanning friction factor for turbulent power-law flow (Dodge & Metzner)."""
    def g(f):
        return 1 / math.sqrt(f) - (4.0 / n ** 0.75) * math.log10(re * f ** (1 - n / 2)) + 0.395 / n ** 1.2
    return brentq(g, 1e-5, 1.0)


def critical_reynolds(n: float) -> float:
    """Laminar/turbulent transition for power-law fluids (API RP 13D)."""
    return 3470.0 - 1370.0 * n


def power_law_gradient(fluid: Fluid, q: float, d_out: float, d_in: float = 0.0) -> FlowResult:
    """Frictional gradient for a power-law fluid in a pipe (``d_in`` = 0) or annulus."""
    n, K, rho = fluid.n, fluid.K, fluid.mud_weight
    if d_in <= 0:
        v = q / (2.448 * d_out ** 2)
        re = 89100.0 * rho * v ** (2 - n) / K * (0.0416 * d_out / (3 + 1 / n)) ** n
        if re < critical_reynolds(n):
            grad = K * v ** n * ((3 + 1 / n) / 0.0416) ** n / (144000.0 * d_out ** (1 + n))
            return FlowResult(v, re, "laminar", grad)
        f = _dodge_metzner(re, n)
        return FlowResult(v, re, "turbulent", f * rho * v ** 2 / (25.8 * d_out))

    gap = d_out - d_in
    v = q / (2.448 * (d_out ** 2 - d_in ** 2))
    re = 109000.0 * rho * v ** (2 - n) / K * (0.0208 * gap / (2 + 1 / n)) ** n
    if re < critical_reynolds(n):
        grad = K * v ** n * ((2 + 1 / n) / 0.0208) ** n / (144000.0 * gap ** (1 + n))
        return FlowResult(v, re, "laminar", grad)
    f = _dodge_metzner(re, n)
    return FlowResult(v, re, "turbulent", f * rho * v ** 2 / (21.1 * gap))


def bingham_gradient(fluid: Fluid, q: float, d_out: float, d_in: float = 0.0) -> FlowResult:
    """Frictional gradient for a Bingham plastic fluid in a pipe or annulus."""
    pv, yp, rho = fluid.pv, fluid.yp, fluid.mud_weight
    if d_in <= 0:
        v = q / (2.448 * d_out ** 2)
        mu_a = pv + 6.66 * yp * d_out / v
        re = 928.0 * rho * v * d_out / mu_a
        if re < 2100:
            return FlowResult(v, re, "laminar", pv * v / (1500 * d_out ** 2) + yp / (225 * d_out))
        return FlowResult(v, re, "turbulent",
                          rho ** 0.75 * v ** 1.75 * pv ** 0.25 / (1800 * d_out ** 1.25))
    gap = d_out - d_in
    v = q / (2.448 * (d_out ** 2 - d_in ** 2))
    mu_a = pv + 5.0 * yp * gap / v
    re = 757.0 * rho * v * gap / mu_a
    if re < 2100:
        return FlowResult(v, re, "laminar", pv * v / (1000 * gap ** 2) + yp / (200 * gap))
    return FlowResult(v, re, "turbulent",
                      rho ** 0.75 * v ** 1.75 * pv ** 0.25 / (1396 * gap ** 1.25))


MODELS = {"power_law": power_law_gradient, "bingham": bingham_gradient}


# --------------------------------------------------------------------------
# Wellbore geometry and circulating system
# --------------------------------------------------------------------------

@dataclass
class StringComponent:
    name: str
    od: float      # in
    id: float      # in
    length: float  # ft


@dataclass
class HoleSection:
    name: str
    diameter: float   # casing ID or open-hole/bit size, in
    bottom_md: float  # ft


@dataclass
class Well:
    """Drillstring (listed bit-up) inside hole sections (listed top-down)."""

    string: list[StringComponent]
    hole: list[HoleSection]
    tvd: float | None = None  # ft; defaults to bit depth (vertical well)

    @property
    def bit_depth(self) -> float:
        return sum(c.length for c in self.string)

    @property
    def bit_size(self) -> float:
        return self.hole[-1].diameter

    def annular_intervals(self) -> list[tuple[str, float, float, float, float]]:
        """Split the annulus into (name, top, bottom, hole_d, pipe_od) intervals."""
        # String component depth ranges, top-down.
        comps, bottom = [], self.bit_depth
        for c in self.string:
            comps.append((c, bottom - c.length, bottom))
            bottom -= c.length
        comps.reverse()

        holes, top = [], 0.0
        for h in self.hole:
            holes.append((h, top, h.bottom_md))
            top = h.bottom_md

        out = []
        for c, ct, cb in comps:
            for h, ht, hb in holes:
                t, b = max(ct, ht), min(cb, hb, self.bit_depth)
                if b > t:
                    out.append((f"{c.name} x {h.name}", t, b, h.diameter, c.od))
        return out


@dataclass
class HydraulicsResult:
    q: float
    pipe: list[FlowResult]
    annulus: list[FlowResult]
    surface_loss: float
    bit: dict
    ecd: float
    ecd_with_cuttings: float | None
    extra_losses: float = 0.0
    tvd: float = 0.0
    profile: list[dict] = field(default_factory=list)

    @property
    def pipe_loss(self) -> float:
        return sum(r.loss for r in self.pipe)

    @property
    def annulus_loss(self) -> float:
        return sum(r.loss for r in self.annulus)

    @property
    def parasitic_loss(self) -> float:
        return self.surface_loss + self.pipe_loss + self.annulus_loss + self.extra_losses

    @property
    def standpipe_pressure(self) -> float:
        return self.parasitic_loss + self.bit["pressure_drop"]

    def breakdown(self) -> dict[str, float]:
        return {
            "Surface equipment": self.surface_loss,
            "Drillstring": self.pipe_loss,
            "Motor / MWD": self.extra_losses,
            "Bit": self.bit["pressure_drop"],
            "Annulus": self.annulus_loss,
        }


# Surface equipment as an equivalent length of 3.826 in ID drill pipe
# (IADC surface equipment cases 1-4).
SURFACE_CASES = {1: 2600.0, 2: 946.0, 3: 610.0, 4: 424.0}


def bit_hydraulics(fluid: Fluid, q: float, tfa: float, bit_size: float, cd: float = CD_DEFAULT) -> dict:
    """Pressure drop, jet velocity, hydraulic horsepower and impact force at the bit."""
    dp = 8.311e-5 * fluid.mud_weight * q ** 2 / (cd ** 2 * tfa ** 2)
    hhp = dp * q / 1714.0
    return {
        "pressure_drop": dp,
        "jet_velocity": 0.3208 * q / tfa,
        "hhp": hhp,
        "hsi": hhp / (math.pi / 4 * bit_size ** 2),
        "impact_force": 0.01823 * cd * q * math.sqrt(fluid.mud_weight * dp),
        "tfa": tfa,
    }


def cuttings_concentration(rop: float, bit_size: float, q: float) -> float:
    """Volume fraction of cuttings in the annulus (no-slip), dimensionless.

    Rock volume rate = ROP * pi/4 * d^2 ; ft/hr x in^2 -> gal/min factor is
    7.48 / (144 * 60) so C = ROP d^2 / (1471 Q) / (1 + ROP d^2 / (1471 Q)).
    """
    r = rop * bit_size ** 2 / (1471.0 * q)
    return r / (1.0 + r)


def circulate(
    well: Well,
    fluid: Fluid,
    q: float,
    tfa: float,
    model: str = "power_law",
    surface_case: int = 3,
    extra_losses: float = 0.0,
    rop: float | None = None,
    cuttings_sg: float = 2.6,
) -> HydraulicsResult:
    """Pressure losses around the full circulating system at flow rate ``q``."""
    grad = MODELS[model]
    surface = grad(fluid, q, 3.826).gradient * SURFACE_CASES[surface_case]

    pipe = []
    for c in well.string:
        r = grad(fluid, q, c.id)
        r.length, r.name = c.length, c.name
        pipe.append(r)

    annulus, profile, cum = [], [], 0.0
    for name, top, bottom, hole_d, od in sorted(well.annular_intervals(), key=lambda x: x[1]):
        r = grad(fluid, q, hole_d, od)
        r.length, r.name = bottom - top, name
        annulus.append(r)
        cum += r.loss
        profile.append({"md": bottom, "annular_loss": cum, "velocity": r.velocity, "regime": r.regime})

    tvd = well.tvd or well.bit_depth
    ann_loss = sum(r.loss for r in annulus)
    ecd = fluid.mud_weight + ann_loss / (0.052 * tvd)
    ecd_c = None
    if rop:
        ca = cuttings_concentration(rop, well.bit_size, q)
        loaded = fluid.mud_weight * (1 - ca) + cuttings_sg * 8.33 * ca
        ecd_c = loaded + ann_loss / (0.052 * tvd)

    # Convert cumulative annular loss to an ECD profile (vertical-well approximation
    # scales TVD with MD when the well's TVD is given).
    tvd_ratio = tvd / well.bit_depth
    for p in profile:
        p["ecd"] = fluid.mud_weight + p["annular_loss"] / (0.052 * p["md"] * tvd_ratio)

    return HydraulicsResult(
        q=q, pipe=pipe, annulus=annulus, surface_loss=surface,
        bit=bit_hydraulics(fluid, q, tfa, well.bit_size), ecd=ecd, ecd_with_cuttings=ecd_c,
        extra_losses=extra_losses, tvd=tvd, profile=profile,
    )


# --------------------------------------------------------------------------
# Nozzle selection and flow-rate optimisation
# --------------------------------------------------------------------------

def tfa_from_nozzles(sizes_32nds: list[int]) -> float:
    """Total flow area, in^2, of nozzles given in 32nds of an inch."""
    return sum(math.pi / 4 * (s / 32.0) ** 2 for s in sizes_32nds)


def required_tfa(fluid: Fluid, q: float, bit_dp: float, cd: float = CD_DEFAULT) -> float:
    """TFA that produces ``bit_dp`` psi across the bit at flow ``q``."""
    return math.sqrt(8.311e-5 * fluid.mud_weight * q ** 2 / (cd ** 2 * bit_dp))


def select_nozzles(target_tfa: float, count: int, sizes: list[int] | None = None,
                   max_spread: int = 1) -> list[int]:
    """Nozzle combination whose TFA is closest to ``target_tfa``.

    Sizes within a set differ by at most ``max_spread`` 32nds, which is how
    jets are dressed in practice (e.g. 3x12 + 2x13 rather than 11s and 16s).
    """
    sizes = sizes or NOZZLE_SIZES_32
    best, best_err = None, float("inf")
    for combo in itertools.combinations_with_replacement(sizes, count):
        if combo[-1] - combo[0] > max_spread:
            continue
        err = abs(tfa_from_nozzles(combo) - target_tfa)
        if err < best_err:
            best, best_err = list(combo), err
    return best


def fit_parasitic(qs: np.ndarray, losses: np.ndarray) -> tuple[float, float]:
    """Fit parasitic loss = c * Q^m (log-log least squares). Returns (c, m)."""
    m, logc = np.polyfit(np.log(qs), np.log(losses), 1)
    return float(np.exp(logc)), float(m)


CRITERIA = {"hhp": "Max bit hydraulic horsepower", "impact": "Max jet impact force"}


def optimum_flow(p_max: float, c: float, m: float, q_min: float, q_max: float,
                 criterion: str = "hhp") -> dict:
    """Optimum flow and bit pressure drop for a pump pressure limit.

    With parasitic losses ``dPd = c Q^m`` the classic results are:
    maximum bit HHP when ``dPd = p_max / (m + 1)`` and maximum impact force
    when ``dPd = 2 p_max / (m + 2)``. The optimum flow is clamped to the
    hole-cleaning minimum and the pump's maximum.
    """
    frac = 1.0 / (m + 1.0) if criterion == "hhp" else 2.0 / (m + 2.0)
    dpd_opt = frac * p_max
    q = (dpd_opt / c) ** (1.0 / m)
    q = float(min(max(q, q_min), q_max))
    dpd = c * q ** m
    return {"q": q, "parasitic": dpd, "bit_dp": max(p_max - dpd, 0.0),
            "parasitic_fraction": dpd / p_max}


def design_nozzles(well: Well, fluid: Fluid, p_max: float, q_min: float, q_max: float,
                   n_nozzles: int, criterion: str = "hhp", model: str = "power_law",
                   surface_case: int = 3, extra_losses: float = 0.0) -> dict:
    """End-to-end bit hydraulics design: optimum flow, TFA and jet sizes."""
    qs = np.linspace(q_min, q_max, 8)
    dummy_tfa = 1.0  # parasitic losses do not depend on TFA
    losses = np.array([circulate(well, fluid, q, dummy_tfa, model, surface_case,
                                 extra_losses).parasitic_loss for q in qs])
    c, m = fit_parasitic(qs, losses)
    opt = optimum_flow(p_max, c, m, q_min, q_max, criterion)
    if opt["bit_dp"] <= 0:
        raise ValueError("parasitic losses exceed the pump pressure limit at the minimum flow")
    tfa = required_tfa(fluid, opt["q"], opt["bit_dp"])
    nozzles = select_nozzles(tfa, n_nozzles)
    actual = circulate(well, fluid, opt["q"], tfa_from_nozzles(nozzles), model,
                       surface_case, extra_losses)
    return {**opt, "c": c, "m": m, "target_tfa": tfa, "nozzles": nozzles,
            "tfa": tfa_from_nozzles(nozzles), "result": actual,
            "curve": {"q": qs, "parasitic": losses}}
