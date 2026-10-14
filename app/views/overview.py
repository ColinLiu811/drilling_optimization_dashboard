"""Overview page: well summary, depth tracks and parameter cross-plots."""

from __future__ import annotations

import plotly.graph_objects as go
import streamlit as st

from charts import SECTION_COLORS, SERIES, depth_tracks, style


def render():
    df = st.session_state["df"]
    st.title("Drilling Overview")
    st.caption("Utah FORGE 16A(78)-32 - on-bottom rotary drilling, median per 1 ft of hole")
    if df.empty:
        st.warning("No data in the selected filter.")
        return

    days = (df["time"].max() - df["time"].min()).total_seconds() / 86400
    c = st.columns(5)
    c[0].metric("Footage", f"{df['depth'].max() - df['depth'].min():,.0f} ft")
    c[1].metric("Median ROP", f"{df['rop'].median():.0f} ft/hr")
    c[2].metric("Median WOB", f"{df['wob'].median():.0f} klbf")
    c[3].metric("Median MSE", f"{df['mse'].median():.0f} kpsi")
    c[4].metric("Elapsed", f"{days:.1f} days")

    st.subheader("Depth tracks")
    tracks = [("rop", "ROP (ft/hr)"), ("wob", "WOB (klbf)"), ("rpm", "Surface RPM"),
              ("flow", "Flow (gpm)"), ("spp", "SPP (psi)"), ("mse", "MSE (kpsi)")]
    st.plotly_chart(depth_tracks(df, tracks, color_by="section"), width="stretch")

    left, right = st.columns(2)
    with left:
        st.subheader("ROP vs drilling parameter")
        x = st.selectbox("Parameter", ["wob", "rpm", "flow", "torque", "mse"],
                         format_func=lambda s: {"wob": "WOB (klbf)", "rpm": "Surface RPM",
                                                "flow": "Flow (gpm)", "torque": "Torque (ft-lbf)",
                                                "mse": "MSE (kpsi)"}[s])
        fig = go.Figure()
        for i, (sec, g) in enumerate(df.groupby("section")):
            fig.add_trace(go.Scatter(
                x=g[x], y=g["rop"], mode="markers", name=sec,
                marker=dict(size=5, opacity=0.45, color=SECTION_COLORS.get(sec, SERIES[i])),
                hovertemplate=f"{x}: %{{x:.1f}}<br>ROP: %{{y:.0f}} ft/hr<br>%{{customdata:,.0f}} ft<extra>{sec}</extra>",
                customdata=g["depth"]))
        fig.update_layout(xaxis_title=x, yaxis_title="ROP (ft/hr)")
        if x == "mse":
            fig.update_xaxes(type="log")
        st.plotly_chart(style(fig), width="stretch")

    with right:
        st.subheader("Depth vs days")
        fig = go.Figure()
        for i, (sec, g) in enumerate(df.groupby("section")):
            t = (g["time"] - df["time"].min()).dt.total_seconds() / 86400
            fig.add_trace(go.Scatter(
                x=t, y=g["depth"], mode="markers", name=sec,
                marker=dict(size=4, color=SECTION_COLORS.get(sec, SERIES[i])),
                hovertemplate="Day %{x:.1f}<br>%{y:,.0f} ft<extra>" + sec + "</extra>"))
        fig.update_layout(xaxis_title="Days since first on-bottom sample", yaxis_title="Measured depth (ft)")
        fig.update_yaxes(autorange="reversed")
        st.plotly_chart(style(fig), width="stretch")

    with st.expander("Section statistics"):
        cols = ["rop", "wob", "rpm", "flow", "spp", "torque", "mse"]
        st.dataframe(df.groupby("section")[cols].median().round(1), width="stretch")
