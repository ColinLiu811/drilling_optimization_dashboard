"""Drilling Optimization Dashboard - Streamlit entry point.

Run with:  streamlit run app/streamlit_app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "app"))

import streamlit as st  # noqa: E402

from drillopt.data import load_processed  # noqa: E402
from drillopt.rop import add_drilling_features  # noqa: E402

st.set_page_config(page_title="Drilling Optimization", page_icon="🛢️", layout="wide")


@st.cache_data(show_spinner="Loading drilling data...")
def get_data(uploaded: bytes | None = None):
    import io
    import pandas as pd

    if uploaded is not None:
        df = pd.read_csv(io.BytesIO(uploaded), parse_dates=["time"])
        from drillopt.data import add_hole_section
        df = add_hole_section(df)
    else:
        df = load_processed()
    return add_drilling_features(df)


def sidebar_filters(df):
    st.sidebar.header("Data")
    up = st.sidebar.file_uploader(
        "Depth-binned CSV (optional)", type="csv",
        help="Same columns as data/forge_16a78_32_depth_binned.csv. "
             "Use scripts/prepare_forge.py or drillopt.data.standardize to build one.")
    if up is not None:
        df = get_data(up.getvalue())

    sections = sorted(df["section"].unique(), key=lambda s: -float(s.split()[0]) if s[0].isdigit() else 0)
    chosen = st.sidebar.multiselect("Hole sections", sections, default=sections)
    lo, hi = int(df["depth"].min()), int(df["depth"].max())
    depth = st.sidebar.slider("Depth range (ft MD)", lo, hi, (lo, hi), step=50)
    filtered = df[df["section"].isin(chosen) & df["depth"].between(*depth)]
    st.sidebar.caption(f"{len(filtered):,} of {len(df):,} one-foot intervals selected")
    st.sidebar.divider()
    st.sidebar.caption(
        "Data: Utah FORGE well 16A(78)-32, rig sensor data at 10 s intervals "
        "(DOE Geothermal Data Repository, submission 1283).")
    return filtered


df_all = get_data()
st.session_state["df_all"] = df_all
st.session_state["df"] = sidebar_filters(df_all)

from views import overview, rop_view  # noqa: E402

pages = [
    st.Page(overview.render, title="Overview", icon="📊", url_path="overview", default=True),
    st.Page(rop_view.render, title="ROP Model & Optimizer", icon="⛏️", url_path="rop"),
]
st.navigation(pages).run()
