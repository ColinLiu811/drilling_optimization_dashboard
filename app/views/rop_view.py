"""ROP modeling and parameter optimization page."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from charts import MUTED, SEQUENTIAL, SERIES, style
from drillopt import optimize as opt
from drillopt.rop import CONTEXT_FEATURES, CONTROL_FEATURES, MODEL_NAMES, compare_models, fit_rop_model

FEATURE_LABELS = {
    "wob": "WOB (klbf)", "rpm": "Surface RPM", "flow": "Flow (gpm)", "depth": "Depth (ft)",
    "inc": "Inclination (deg)", "bit_size": "Bit size (in)",
    "rop_trail": "Trailing ROP, 30 ft", "mse_trail": "Trailing MSE, 30 ft",
}


@st.cache_resource(show_spinner="Fitting model...")
def _fit(key: tuple, _df: pd.DataFrame, kind: str, features: tuple, test_fraction: float):
    return fit_rop_model(_df, kind, list(features), test_fraction)


@st.cache_data(show_spinner="Cross-validating all models...")
def _compare(key: tuple, _df: pd.DataFrame, features: tuple):
    return compare_models(_df, list(features))


def render():
    df = st.session_state["df"]
    st.title("ROP Model & Optimizer")
    st.caption("Regress rate of penetration on WOB, RPM and flow rate; then search for the "
               "set-point that maximises predicted ROP inside the observed operating envelope.")
    if len(df) < 200:
        st.warning("Select at least a few hundred feet of data in the sidebar.")
        return

    c1, c2, c3 = st.columns([1.2, 2, 1])
    sections = sorted(df["section"].unique())
    section = c1.selectbox("Hole section", ["All selected"] + sections,
                           index=1 + sections.index("8.75 in") if "8.75 in" in sections else 0)
    kind = c1.selectbox("Model", list(MODEL_NAMES), index=3, format_func=MODEL_NAMES.get)
    context = c2.multiselect(
        "Context features (not controllable)", CONTEXT_FEATURES, default=["rop_trail"],
        format_func=FEATURE_LABELS.get,
        help="Trailing ROP/MSE use only footage already drilled, so they are available in "
             "real time and do not leak the target. They stand in for rock strength.")
    test_fraction = c3.slider("Hold-out (deepest %)", 10, 40, 20, 5) / 100

    data = df if section == "All selected" else df[df["section"] == section]
    features = tuple(CONTROL_FEATURES + context)
    key = (section, len(data), float(data["depth"].min()), float(data["depth"].max()))
    model = _fit(key + (kind, features, test_fraction), data, kind, features, test_fraction)

    st.subheader("Model skill")
    m = st.columns(5)
    m[0].metric("Hold-out R²", f"{model.test_metrics.get('r2', np.nan):.2f}")
    m[1].metric("Hold-out RMSE", f"{model.test_metrics.get('rmse', np.nan):.1f} ft/hr")
    m[2].metric("Hold-out MAE", f"{model.test_metrics.get('mae', np.nan):.1f} ft/hr")
    m[3].metric("Blocked-CV R²", f"{model.cv['r2'].mean():.2f}")
    m[4].metric("Train R²", f"{model.train_metrics['r2']:.2f}",
                help="Large gap to hold-out/CV skill = overfitting")

    left, right = st.columns(2)
    test = model.test_frame
    with left:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=test["rop"], y=test["rop_pred"], mode="markers", name="Hold-out",
                                 marker=dict(size=6, color=SERIES[0], opacity=0.6),
                                 hovertemplate="Actual %{x:.0f}<br>Predicted %{y:.0f}<extra></extra>"))
        hi = float(max(test["rop"].max(), test["rop_pred"].max()))
        fig.add_trace(go.Scatter(x=[0, hi], y=[0, hi], mode="lines", name="1:1",
                                 line=dict(color=MUTED, dash="dash", width=1), hoverinfo="skip"))
        fig.update_layout(title="Predicted vs actual (hold-out)", xaxis_title="Actual ROP (ft/hr)",
                          yaxis_title="Predicted ROP (ft/hr)")
        st.plotly_chart(style(fig, legend=False), width="stretch")
    with right:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=test["depth"], y=test["rop"], mode="lines", name="Measured",
                                 line=dict(color=SERIES[0], width=1.5)))
        fig.add_trace(go.Scatter(x=test["depth"], y=test["rop_pred"], mode="lines", name="Predicted",
                                 line=dict(color=SERIES[1], width=2)))
        fig.update_layout(title="Hold-out interval", xaxis_title="Measured depth (ft)",
                          yaxis_title="ROP (ft/hr)", hovermode="x unified")
        st.plotly_chart(style(fig), width="stretch")

    left, right = st.columns(2)
    with left:
        imp = model.importance()
        if imp is not None:
            title = "Fitted exponents (ROP ∝ Πxᵢ^aᵢ)" if kind == "power_law" else "Feature importance"
            fig = go.Figure(go.Bar(x=imp.values, y=[FEATURE_LABELS.get(i, i) for i in imp.index],
                                   orientation="h", marker_color=SERIES[0],
                                   hovertemplate="%{y}: %{x:.3f}<extra></extra>"))
            fig.update_layout(title=title, showlegend=False)
            if kind == "power_law":
                st.caption(f"K = {model.estimator.K_:.3g}. An exponent near 1 means ROP still scales "
                           "with that parameter; near 0 means it no longer helps (founder).")
            st.plotly_chart(style(fig, 320, legend=False), width="stretch")
    with right:
        cv = model.cv.copy()
        cv["interval"] = cv.apply(lambda r: f"{r.depth_from:,.0f}-{r.depth_to:,.0f} ft", axis=1)
        st.markdown("**Depth-blocked cross-validation** (each block predicted by a model trained "
                    "on the others)")
        st.dataframe(cv[["interval", "n", "r2", "rmse", "mae"]].round(2), hide_index=True,
                     width="stretch")
        if st.button("Compare all models"):
            st.dataframe(_compare(key, data, features).round(2), hide_index=True,
                         width="stretch")

    st.subheader("Parameter response")
    fig = make_subplots(rows=1, cols=3, shared_yaxes=True,
                        subplot_titles=[FEATURE_LABELS[f] for f in CONTROL_FEATURES])
    for i, f in enumerate(CONTROL_FEATURES, start=1):
        pdp = opt.partial_dependence(model, data.dropna(subset=list(features)), f)
        fig.add_trace(go.Scatter(x=pdp[f], y=pdp["rop_pred"], mode="lines",
                                 line=dict(color=SERIES[0], width=2), showlegend=False,
                                 hovertemplate=f"{f}: %{{x:.1f}}<br>ROP: %{{y:.0f}}<extra></extra>"),
                      row=1, col=i)
    fig.update_yaxes(title_text="Mean predicted ROP (ft/hr)", row=1, col=1)
    st.plotly_chart(style(fig, 300), width="stretch")
    st.caption("Partial dependence from field data mixes cause and effect: drillers add weight "
               "in harder rock, so a falling WOB curve usually reflects formation, not a "
               "physical penalty for weight. Adding trailing ROP/MSE context reduces this bias.")

    _optimizer(model, data, features)


def _optimizer(model, data, features):
    st.subheader("Set-point optimizer")
    usable = data.dropna(subset=list(features)).sort_values("depth")
    depth = st.select_slider("Optimise at depth (ft)", options=usable["depth"].astype(int).tolist(),
                             value=int(usable["depth"].iloc[int(len(usable) * 0.9)]))
    idx = min(int(np.searchsorted(usable["depth"].to_numpy(), depth)), len(usable) - 1)
    ctx = usable.iloc[idx].to_dict()

    lim = opt.Limits.from_data(usable)
    c = st.columns(3)
    wob = c[0].slider("WOB limits (klbf)", 0.0, 80.0, (float(round(lim.wob[0], 1)), float(round(lim.wob[1], 1))))
    rpm = c[1].slider("RPM limits", 0.0, 200.0, (float(round(lim.rpm[0])), float(round(lim.rpm[1]))))
    flow = c[2].slider("Flow limits (gpm)", 100.0, 1300.0, (float(round(lim.flow[0])), float(round(lim.flow[1]))))
    res = opt.optimize(model, ctx, opt.Limits(wob, rpm, flow), n=21)

    k = st.columns(4)
    k[0].metric("Recommended WOB", f"{res['wob']:.1f} klbf", f"{res['wob'] - ctx['wob']:+.1f}")
    k[1].metric("Recommended RPM", f"{res['rpm']:.0f}", f"{res['rpm'] - ctx['rpm']:+.0f}")
    k[2].metric("Recommended flow", f"{res['flow']:.0f} gpm", f"{res['flow'] - ctx['flow']:+.0f}")
    k[3].metric("Predicted ROP", f"{res['rop_pred']:.0f} ft/hr", f"{res['gain_pct']:+.0f}% vs current")
    st.caption(f"Current at {ctx['depth']:,.0f} ft: WOB {ctx['wob']:.1f} klbf, RPM {ctx['rpm']:.0f}, "
               f"flow {ctx['flow']:.0f} gpm, measured ROP {ctx['rop']:.0f} ft/hr "
               f"(model {res['current_rop_pred']:.0f}).")

    sl = opt.slice_2d(res["grid"], "wob", "rpm", "flow", res["flow"])
    fig = go.Figure(go.Heatmap(
        x=sl.columns, y=sl.index, z=sl.values, colorscale=SEQUENTIAL,
        colorbar=dict(title="ROP<br>ft/hr"),
        hovertemplate="WOB %{x:.1f}<br>RPM %{y:.0f}<br>ROP %{z:.0f}<extra></extra>"))
    fig.add_trace(go.Scatter(x=[ctx["wob"]], y=[ctx["rpm"]], mode="markers+text", text=["current"],
                             textposition="top center", name="Current",
                             marker=dict(size=11, color="white", line=dict(color="black", width=2))))
    fig.add_trace(go.Scatter(x=[res["wob"]], y=[res["rpm"]], mode="markers+text", text=["optimum"],
                             textposition="top center", name="Optimum",
                             marker=dict(size=13, symbol="star", color=SERIES[1],
                                         line=dict(color="white", width=2))))
    fig.update_layout(title=f"Predicted ROP surface at {res['flow']:.0f} gpm",
                      xaxis_title="WOB (klbf)", yaxis_title="Surface RPM")
    st.plotly_chart(style(fig, 450, legend=False), width="stretch")
