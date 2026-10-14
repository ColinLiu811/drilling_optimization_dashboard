"""Loading and cleaning surface drilling data.

The bundled dataset comes from the Utah FORGE well 16A(78)-32 rig sensor
export (10 s intervals, Pason EDR) published on the DOE Geothermal Data
Repository. The raw file is ~120 MB, so ``scripts/prepare_forge.py`` reduces
it to on-bottom rotary drilling samples and bins them by measured depth.

Any other well (Volve, an operator's own EDR export, ...) can be used by
passing a column mapping to :func:`standardize`.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

FORGE_URL = (
    "https://gdr.openei.org/files/1283/16A%2878%29-32_time_data_10s_intervals.csv"
)
NULL_VALUE = -999.25

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
PROCESSED_PATH = DATA_DIR / "forge_16a78_32_depth_binned.csv"

# Standard column names used throughout the package, with units.
COLUMNS = {
    "depth": "ft",
    "tvd": "ft",
    "inc": "deg",
    "bit_size": "in",
    "wob": "klbf",
    "rpm": "rev/min",
    "bit_rpm": "rev/min",
    "flow": "gal/min",
    "spp": "psi",
    "diff_press": "psi",
    "torque": "ft-lbf",
    "rop": "ft/hr",
    "mud_temp": "degF",
}

FORGE_COLUMN_MAP = {
    "Time": "time",
    "Total Depth": "depth",
    "Depth Hole TVD": "tvd",
    "MWD Inclination": "inc",
    "Bit Diameter": "bit_size",
    "Weight on Bit": "wob",
    "Top Drive RPM": "rpm",
    "Bit RPM": "bit_rpm",
    "Flow In": "flow",
    "Pump Pressure": "spp",
    "Differential Pressure": "diff_press",
    "Top Drive Torque (ft-lbs)": "torque",
    "ROP Depth/Hour": "rop",
    "MUD TEMP": "mud_temp",
    "On Bottom": "on_bottom",
}

# Common column names in the Volve real-time drilling exports.
VOLVE_COLUMN_MAP = {
    "TIME": "time",
    "DEPTH": "depth",
    "Hole Depth": "depth",
    "WOB": "wob",
    "Weight on Bit": "wob",
    "SURF_RPM": "rpm",
    "Rotary Speed": "rpm",
    "FLOWIN": "flow",
    "Mud Flow In": "flow",
    "SPPA": "spp",
    "Standpipe Pressure": "spp",
    "TQA": "torque",
    "ROP": "rop",
    "Rate of Penetration": "rop",
}

# Physically plausible ranges for on-bottom rotary drilling; samples outside
# these are sensor spikes or connection/off-bottom artefacts.
VALID_RANGES = {
    "wob": (1.0, 80.0),
    "rpm": (5.0, 200.0),
    "flow": (150.0, 1300.0),
    "rop": (0.5, 400.0),
    "spp": (100.0, 7500.0),
    "torque": (0.0, 40000.0),
    "bit_rpm": (0.0, 600.0),
    "depth": (0.0, 40000.0),
}


def standardize(df: pd.DataFrame, column_map: dict[str, str]) -> pd.DataFrame:
    """Rename source columns to the package's standard names.

    Columns not in the mapping are dropped. Null sentinels are replaced by NaN.
    """
    keep = {src: dst for src, dst in column_map.items() if src in df.columns}
    out = df[list(keep)].rename(columns=keep)
    out = out.loc[:, ~out.columns.duplicated()]
    return out.replace(NULL_VALUE, np.nan)


def read_forge_raw(path: str | Path) -> pd.DataFrame:
    """Read the raw FORGE 10 s export and map it to standard column names."""
    raw = pd.read_csv(path, na_values=[NULL_VALUE], low_memory=False)
    df = standardize(raw, FORGE_COLUMN_MAP)
    df["time"] = pd.to_datetime(df["time"], format="%m/%d/%y %H:%M", errors="coerce")
    return df


def filter_drilling(df: pd.DataFrame) -> pd.DataFrame:
    """Keep on-bottom rotary drilling samples within plausible sensor ranges."""
    mask = pd.Series(True, index=df.index)
    if "on_bottom" in df:
        mask &= df["on_bottom"] == 1
    for col, (lo, hi) in VALID_RANGES.items():
        if col not in df:
            continue
        if col in ("bit_rpm", "torque"):
            # Optional channels: drop bad values but keep the row.
            df.loc[(df[col] < lo) | (df[col] > hi), col] = np.nan
        else:
            mask &= df[col].between(lo, hi)
    out = df[mask].copy()
    # Depth must advance while drilling; drop samples recorded behind the bit
    # (e.g. reaming back to bottom after a connection).
    out = out.sort_values("time") if "time" in out else out
    out = out[out["depth"] >= out["depth"].cummax() - 1.0]
    return out


def bin_by_depth(df: pd.DataFrame, interval_ft: float = 1.0) -> pd.DataFrame:
    """Median-aggregate drilling samples into fixed measured-depth bins.

    Depth binning removes the bias time-based sampling puts on slow
    intervals (a 5 ft/hr section produces many more 10 s samples per foot
    than a 150 ft/hr one).
    """
    df = df.copy()
    df["depth_bin"] = (df["depth"] // interval_ft) * interval_ft
    numeric = df.select_dtypes("number").columns.drop(["depth_bin", "on_bottom"], errors="ignore")
    agg = df.groupby("depth_bin")[numeric.drop("depth", errors="ignore")].median()
    if "time" in df:
        agg["time"] = df.groupby("depth_bin")["time"].min()
    agg["n_samples"] = df.groupby("depth_bin").size()
    agg = agg.reset_index().rename(columns={"depth_bin": "depth"})
    if "bit_size" in agg:
        # Bit size is a hole-section property; fill sensor gaps from neighbours.
        agg["bit_size"] = agg["bit_size"].where(agg["bit_size"] > 5).ffill().bfill()
    return agg


def add_hole_section(df: pd.DataFrame) -> pd.DataFrame:
    """Label each row with its hole section, e.g. ``12.00 in``."""
    df = df.copy()
    if "bit_size" in df:
        df["section"] = df["bit_size"].map(lambda s: f"{s:.2f} in" if pd.notna(s) else "unknown")
    else:
        df["section"] = "all"
    return df


def load_processed(path: str | Path | None = None) -> pd.DataFrame:
    """Load the depth-binned dataset shipped with the repo."""
    path = Path(path) if path else PROCESSED_PATH
    df = pd.read_csv(path, parse_dates=["time"])
    return add_hole_section(df)


def prepare(raw_path: str | Path, interval_ft: float = 1.0) -> pd.DataFrame:
    """Full raw -> depth-binned pipeline."""
    df = read_forge_raw(raw_path)
    df = filter_drilling(df)
    return bin_by_depth(df, interval_ft)
