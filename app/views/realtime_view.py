"""Real-time monitor: replays the well as a live feed with rolling analytics and alerts."""

from __future__ import annotations

from datetime import timedelta

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from charts import MUTED, SERIES, STATUS, style
from drillopt.realtime import AlertConfig, RealtimeMonitor, ReplayStream

ALERT_STYLE = {
    "ROP_BELOW_MODEL": (STATUS["warning"], "ROP below model"),
    "MSE_SPIKE": (STATUS["serious"], "MSE spike"),
    "SPP_DEVIATION": (STATUS["critical"], "SPP deviation"),
}


def _reset(df, start_depth, cfg):
    stream = ReplayStream(df)
    stream.seek_depth(start_depth)
    st.session_state.rt_stream = stream
    st.session_state.rt_monitor = RealtimeMonitor(config=cfg)
    st.session_state.rt_running = False


def render():
    df = st.session_state["df_all"]
    st.title("Real-time Monitor")
    st.caption("Replays the FORGE well foot by foot as if it were a live rig feed. Each new record "
               "updates a rolling buffer, recomputes MSE, refits a local power-law ROP model every "
               "25 ft and checks for drilling dysfunction.")

    with st.expander("Stream and alert settings", expanded="rt_stream" not in st.session_state):
        c = st.columns(4)
        start = c[0].number_input("Start depth (ft)", int(df["depth"].min()), int(df["depth"].max()),
                                  6200, 100)
        speed = c[1].slider("Feet per tick", 1, 50, 10)
        interval = c[2].select_slider("Tick interval (s)", [0.5, 1.0, 2.0, 5.0], value=1.0)
        window = c[3].slider("Display window (ft)", 100, 1500, 400, 50)
        a = st.columns(3)
        cfg = AlertConfig(
            rop_tolerance=a[0].slider("ROP shortfall to alert", 0.1, 0.8, 0.4, 0.05),
            mse_z=a[1].slider("MSE spike threshold (robust σ)", 1.5, 6.0, 3.0, 0.5),
            spp_tolerance=a[2].slider("SPP change to alert", 0.03, 0.3, 0.10, 0.01),
        )

    if "rt_stream" not in st.session_state:
        _reset(df, start, cfg)

    b = st.columns([1, 1, 1, 4])
    if b[0].button("Start" if not st.session_state.rt_running else "Pause", type="primary"):
        st.session_state.rt_running = not st.session_state.rt_running
        st.rerun()
    if b[1].button("Step"):
        _advance(speed)
    if b[2].button("Reset"):
        _reset(df, start, cfg)
        st.rerun()
    st.session_state.rt_monitor.config = cfg

    run_every = timedelta(seconds=interval) if st.session_state.rt_running else None
    _live(speed, window, run_every=run_every)


def _advance(n: int):
    stream, mon = st.session_state.rt_stream, st.session_state.rt_monitor
    for rec in stream.next_batch(n):
        mon.update(rec)
    if stream.done:
        st.session_state.rt_running = False


def _live_body(speed: int, window: int):
    if st.session_state.rt_running:
        _advance(speed)
        if not st.session_state.rt_running:
            st.rerun()  # replay finished: stop the timer and refresh the controls
    mon, stream = st.session_state.rt_monitor, st.session_state.rt_stream
    hist = mon.frame()
    st.progress(stream.progress, text=f"Replay {stream.progress:.0%} - "
                f"{stream.cursor:,} of {len(stream):,} records")
    if hist.empty:
        st.info("Press **Start** to begin streaming.")
        return

    last = hist.iloc[-1]
    k = st.columns(3) + st.columns(3)
    k[0].metric("Bit depth", f"{last['depth']:,.0f} ft")
    exp = last.get("rop_expected")
    k[1].metric("ROP", f"{last['rop']:.0f} ft/hr",
                f"{last['rop'] - exp:+.0f} vs model" if pd.notna(exp) else None)
    k[2].metric("WOB / RPM", f"{last['wob']:.0f} / {last['rpm']:.0f}")
    k[3].metric("Flow", f"{last['flow']:.0f} gpm")
    k[4].metric("SPP", f"{last['spp']:,.0f} psi")
    k[5].metric("MSE", f"{last.get('mse', float('nan')):.0f} kpsi")

    view = hist[hist["depth"] >= last["depth"] - window]
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.06,
                        subplot_titles=["ROP (ft/hr)", "MSE (kpsi, log)", "Standpipe pressure (psi)"])
    fig.add_trace(go.Scatter(x=view["depth"], y=view["rop"], name="Measured ROP", mode="lines",
                             line=dict(color=SERIES[0], width=1.5)), row=1, col=1)
    if "rop_expected" in view:
        fig.add_trace(go.Scatter(x=view["depth"], y=view["rop_expected"], name="Model expected",
                                 mode="lines", line=dict(color=SERIES[1], width=2)), row=1, col=1)
    if "mse" in view:
        fig.add_trace(go.Scatter(x=view["depth"], y=view["mse"], name="MSE", mode="lines",
                                 line=dict(color=SERIES[2], width=1.5), showlegend=False), row=2, col=1)
        fig.update_yaxes(type="log", row=2, col=1)
    fig.add_trace(go.Scatter(x=view["depth"], y=view["spp"], name="SPP", mode="lines",
                             line=dict(color=SERIES[0], width=1.5), showlegend=False), row=3, col=1)

    alerts = mon.alerts_frame()
    alerts = alerts[alerts["depth"] >= last["depth"] - window]
    for kind, (color, label) in ALERT_STYLE.items():
        a = alerts[alerts["kind"] == kind]
        if a.empty:
            continue
        row = {"ROP_BELOW_MODEL": 1, "MSE_SPIKE": 2, "SPP_DEVIATION": 3}[kind]
        col = {1: "rop", 2: "mse", 3: "spp"}[row]
        y = view.set_index("depth").reindex(a["depth"])[col]
        fig.add_trace(go.Scatter(x=a["depth"], y=y, mode="markers", name=label,
                                 marker=dict(size=10, color=color, symbol="diamond",
                                             line=dict(color="white", width=1.5)),
                                 text=a["message"], hovertemplate="%{x:,.0f} ft<br>%{text}<extra></extra>"),
                      row=row, col=1)
    fig.update_xaxes(title_text="Measured depth (ft)", row=3, col=1)
    fig.update_layout(hovermode="x unified")
    st.plotly_chart(style(fig, 620), width="stretch")

    left, right = st.columns([2, 1])
    with left:
        st.markdown("**Alert log**")
        log = mon.alerts_frame().iloc[::-1].head(50)
        if log.empty:
            st.caption("No alerts yet.")
        else:
            log["type"] = log["kind"].map(lambda k: ALERT_STYLE[k][1])
            st.dataframe(log[["depth", "type", "message"]], hide_index=True,
                         width="stretch", height=260)
    with right:
        st.markdown("**Alerts per 100 ft drilled**")
        counts = mon.alerts_frame()["kind"].value_counts()
        footage = max(hist["depth"].iloc[-1] - hist["depth"].iloc[0], 1)
        fig = go.Figure(go.Bar(
            x=[100 * counts.get(k, 0) / footage for k in ALERT_STYLE],
            y=[v[1] for v in ALERT_STYLE.values()], orientation="h",
            marker_color=[v[0] for v in ALERT_STYLE.values()],
            hovertemplate="%{y}: %{x:.2f} per 100 ft<extra></extra>"))
        st.plotly_chart(style(fig, 260, legend=False), width="stretch")
        if mon.model is not None:
            e = mon.model.exponents_
            st.caption(f"Local model (last {len(mon.buffer)} ft): ROP ∝ WOB^{e[0]:.2f} · "
                       f"RPM^{e[1]:.2f} · Q^{e[2]:.2f}")


def _live(speed: int, window: int, run_every):
    # st.fragment(run_every=...) reruns only this block on a timer, so the
    # feed advances without re-executing the whole page.
    st.fragment(_live_body, run_every=run_every)(speed, window)
