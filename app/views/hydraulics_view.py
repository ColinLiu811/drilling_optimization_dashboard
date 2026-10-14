"""Hydraulics calculator page: pressure losses, ECD and bit nozzle sizing."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from charts import MUTED, SERIES, STATUS, style
from drillopt import hydraulics as hy


def _inputs():
    with st.expander("Mud, wellbore and drillstring", expanded=True):
        c = st.columns(4)
        c[0].markdown("**Mud**")
        mw = c[0].number_input("Mud weight (ppg)", 8.3, 20.0, 9.2, 0.1)
        rheo_mode = c[0].radio("Rheology input", ["PV / YP", "Fann 600 / 300"], horizontal=True)
        if rheo_mode == "PV / YP":
            pv = c[0].number_input("PV (cP)", 1.0, 80.0, 14.0, 1.0)
            yp = c[0].number_input("YP (lbf/100 ft²)", 0.0, 60.0, 12.0, 1.0)
            fluid = hy.Fluid.from_pv_yp(mw, pv, yp)
        else:
            t600 = c[0].number_input("θ600", 2.0, 250.0, 40.0, 1.0)
            t300 = c[0].number_input("θ300", 1.0, 200.0, 26.0, 1.0)
            fluid = hy.Fluid(mw, t600, t300)
        model = c[0].selectbox("Rheology model", ["power_law", "bingham"],
                               format_func={"power_law": "Power law", "bingham": "Bingham plastic"}.get)

        c[1].markdown("**Hole**")
        bit_depth = c[1].number_input("Bit depth (ft MD)", 1000.0, 40000.0, 10987.0, 100.0)
        tvd = c[1].number_input("TVD (ft)", 1000.0, 40000.0, 8560.0, 100.0)
        shoe = c[1].number_input("Casing shoe (ft MD)", 0.0, 40000.0, 6129.0, 100.0)
        csg_id = c[1].number_input("Casing ID (in)", 4.0, 20.0, 8.835, 0.001, format="%.3f")
        bit = c[1].number_input("Bit / open hole (in)", 3.0, 26.0, 8.75, 0.125, format="%.3f")

        c[2].markdown("**Drillstring**")
        dp_od = c[2].number_input("DP OD (in)", 2.375, 6.625, 5.0, 0.125, format="%.3f")
        dp_id = c[2].number_input("DP ID (in)", 1.5, 6.0, 4.276, 0.001, format="%.3f")
        hw_len = c[2].number_input("HWDP length (ft)", 0.0, 3000.0, 900.0, 30.0)
        hw_id = c[2].number_input("HWDP ID (in)", 1.5, 5.0, 3.0, 0.125)
        bha_od = c[2].number_input("BHA / collar OD (in)", 3.0, 11.0, 6.5, 0.25)
        bha_id = c[2].number_input("BHA ID (in)", 1.0, 4.0, 2.8125, 0.0625, format="%.4f")
        bha_len = c[2].number_input("BHA length (ft)", 30.0, 2000.0, 600.0, 10.0)

        c[3].markdown("**Operating**")
        q = c[3].number_input("Flow rate (gpm)", 50.0, 1500.0, 650.0, 10.0)
        n_noz = c[3].number_input("Number of nozzles", 1, 9, 6)
        noz = c[3].number_input("Nozzle size (1/32 in)", 7, 32, 13)
        extra = c[3].number_input("Motor / MWD / RSS loss (psi)", 0.0, 3000.0, 0.0, 50.0)
        surf = c[3].selectbox("Surface equipment case", [1, 2, 3, 4], index=2)
        rop = c[3].number_input("ROP for cuttings load (ft/hr)", 0.0, 500.0, 60.0, 5.0)

    dp_len = bit_depth - hw_len - bha_len
    if dp_len <= 0:
        st.error("BHA + HWDP is longer than the bit depth.")
        st.stop()
    well = hy.Well(
        string=[hy.StringComponent("BHA", bha_od, bha_id, bha_len),
                hy.StringComponent("HWDP", dp_od, hw_id, hw_len),
                hy.StringComponent("Drill pipe", dp_od, dp_id, dp_len)],
        hole=[hy.HoleSection("Casing", csg_id, min(shoe, bit_depth)),
              hy.HoleSection("Open hole", bit, bit_depth)],
        tvd=tvd,
    )
    return dict(fluid=fluid, model=model, well=well, q=q, nozzles=[int(noz)] * int(n_noz),
                extra=extra, surface_case=surf, rop=rop)


def render():
    st.title("Drilling Hydraulics")
    st.caption("Pressure losses, equivalent circulating density and bit hydraulics "
               "(Applied Drilling Engineering, ch. 4). Defaults approximate the FORGE "
               "16A(78)-32 8¾ in section at TD; mud properties are assumed.")
    p = _inputs()
    fluid = p["fluid"]
    st.caption(f"Rheology: PV {fluid.pv:.0f} cP, YP {fluid.yp:.0f} lbf/100 ft², "
               f"n = {fluid.n:.3f}, K = {fluid.K:.0f} eq. cP")

    tabs = st.tabs(["Circulating system", "Nozzle design", "Calibrate vs FORGE SPP"])
    with tabs[0]:
        _system(p)
    with tabs[1]:
        _nozzles(p)
    with tabs[2]:
        _calibrate(p)


def _system(p):
    tfa = hy.tfa_from_nozzles(p["nozzles"])
    r = hy.circulate(p["well"], p["fluid"], p["q"], tfa, p["model"], p["surface_case"],
                     p["extra"], rop=p["rop"] or None)
    b = r.bit
    k = st.columns(3) + st.columns(3)
    k[0].metric("Standpipe pressure", f"{r.standpipe_pressure:,.0f} psi")
    k[1].metric("ECD at TD", f"{r.ecd:.2f} ppg", f"+{r.ecd - p['fluid'].mud_weight:.2f}", delta_color="off")
    if r.ecd_with_cuttings:
        k[2].metric("ECD with cuttings", f"{r.ecd_with_cuttings:.2f} ppg")
    k[3].metric("Bit ΔP", f"{b['pressure_drop']:,.0f} psi",
                f"{100 * b['pressure_drop'] / r.standpipe_pressure:.0f}% of SPP", delta_color="off")
    k[4].metric("HSI", f"{b['hsi']:.2f} hp/in²")
    k[5].metric("Jet velocity", f"{b['jet_velocity']:.0f} ft/s")
    st.caption(f"TFA {tfa:.3f} in² ({len(p['nozzles'])}×{p['nozzles'][0]}/32) · bit HHP "
               f"{b['hhp']:.0f} · impact force {b['impact_force']:,.0f} lbf")

    left, right = st.columns(2)
    with left:
        bd = r.breakdown()
        fig = go.Figure(go.Bar(x=list(bd.values()), y=list(bd.keys()), orientation="h",
                               marker_color=[SERIES[0] if k != "Bit" else SERIES[1] for k in bd],
                               text=[f"{v:,.0f}" for v in bd.values()], textposition="outside",
                               hovertemplate="%{y}: %{x:,.0f} psi<extra></extra>"))
        fig.update_layout(title="Pressure loss breakdown", xaxis_title="psi",
                          xaxis_range=[0, max(bd.values()) * 1.2])
        fig.update_yaxes(autorange="reversed")
        st.plotly_chart(style(fig, 360, legend=False), width="stretch")
    with right:
        prof = pd.DataFrame(r.profile)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=prof["ecd"], y=prof["md"], mode="lines+markers", name="ECD",
                                 line=dict(color=SERIES[0], width=2),
                                 hovertemplate="%{y:,.0f} ft: %{x:.2f} ppg<extra></extra>"))
        fig.add_vline(x=p["fluid"].mud_weight, line=dict(color=MUTED, dash="dash"),
                      annotation_text="static MW")
        fig.update_layout(title="ECD at the bottom of each annular interval",
                          xaxis_title="ECD (ppg)", yaxis_title="MD (ft)")
        fig.update_yaxes(autorange="reversed")
        st.plotly_chart(style(fig, 360, legend=False), width="stretch")

    # SPP vs flow rate, with each component.
    qs = np.linspace(max(100.0, p["q"] * 0.4), p["q"] * 1.4, 30)
    rows = []
    for qi in qs:
        ri = hy.circulate(p["well"], p["fluid"], qi, tfa, p["model"], p["surface_case"], p["extra"])
        rows.append({"q": qi, "Total SPP": ri.standpipe_pressure, "Bit": ri.bit["pressure_drop"],
                     "Parasitic": ri.parasitic_loss})
    curve = pd.DataFrame(rows)
    fig = go.Figure()
    for i, col in enumerate(["Total SPP", "Parasitic", "Bit"]):
        fig.add_trace(go.Scatter(x=curve["q"], y=curve[col], mode="lines", name=col,
                                 line=dict(color=SERIES[i], width=2)))
    fig.add_vline(x=p["q"], line=dict(color=MUTED, dash="dot"), annotation_text="current")
    fig.update_layout(title="Pressure vs flow rate", xaxis_title="Flow (gpm)", yaxis_title="psi",
                      hovermode="x unified")
    st.plotly_chart(style(fig, 360), width="stretch")

    rows = [{"Section": a.name, "Length (ft)": a.length, "Velocity (ft/s)": a.velocity,
             "Reynolds": a.reynolds, "Regime": a.regime, "Loss (psi)": a.loss}
            for a in r.pipe + r.annulus]
    st.dataframe(pd.DataFrame(rows).round(2), hide_index=True, width="stretch")
    min_ann_v = min(a.velocity for a in r.annulus) * 60
    if min_ann_v < 100:
        st.warning(f"Lowest annular velocity is {min_ann_v:.0f} ft/min - below ~100 ft/min hole "
                   "cleaning in high-angle wells is usually poor.")


def _nozzles(p):
    c = st.columns(5)
    p_max = c[0].number_input("Max surface pressure (psi)", 1000.0, 10000.0, 4500.0, 100.0)
    q_min = c[1].number_input("Min flow, hole cleaning (gpm)", 50.0, 1500.0, 450.0, 10.0)
    q_max = c[2].number_input("Max flow, pump limit (gpm)", 100.0, 2000.0, 850.0, 10.0)
    n = c[3].number_input("Nozzles", 1, 9, len(p["nozzles"]), key="nn")
    crit = c[4].radio("Criterion", list(hy.CRITERIA), format_func=hy.CRITERIA.get)
    if q_max <= q_min:
        st.error("Max flow must exceed min flow.")
        return
    try:
        d = hy.design_nozzles(p["well"], p["fluid"], p_max, q_min, q_max, int(n), crit,
                              p["model"], p["surface_case"], p["extra"])
    except ValueError as e:
        st.error(str(e))
        return
    sizes = d["nozzles"]
    combo = " + ".join(f"{sizes.count(s)}×{s}/32" for s in sorted(set(sizes)))
    k = st.columns(3) + st.columns(2)
    k[0].metric("Optimum flow", f"{d['q']:.0f} gpm")
    k[1].metric("Bit ΔP", f"{d['bit_dp']:,.0f} psi", f"{100 * (1 - d['parasitic_fraction']):.0f}% of max",
                delta_color="off")
    k[2].metric("Required TFA", f"{d['target_tfa']:.3f} in²")
    k[3].metric("Nozzles", combo, f"TFA {d['tfa']:.3f} in²", delta_color="off")
    k[4].metric("Resulting HSI", f"{d['result'].bit['hsi']:.2f}")
    st.caption(f"Parasitic losses fit as ΔPd = {d['c']:.3g}·Q^{d['m']:.2f}. Optimum when "
               f"ΔPd = {'P/(m+1)' if crit == 'hhp' else '2P/(m+2)'} = {d['parasitic_fraction']:.0%} "
               "of the pressure limit, clamped to the flow-rate window.")

    qs = np.linspace(q_min, q_max, 40)
    para = d["c"] * qs ** d["m"]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=qs, y=para, mode="lines", name="Parasitic loss",
                             line=dict(color=SERIES[0], width=2)))
    fig.add_trace(go.Scatter(x=qs, y=np.maximum(p_max - para, 0), mode="lines",
                             name="Available for bit", line=dict(color=SERIES[1], width=2)))
    fig.add_hline(y=p_max, line=dict(color=STATUS["critical"], dash="dash"),
                  annotation_text="pressure limit")
    fig.add_trace(go.Scatter(x=[d["q"]], y=[d["bit_dp"]], mode="markers", name="Design point",
                             marker=dict(size=12, symbol="star", color=SERIES[1],
                                         line=dict(color="white", width=2))))
    fig.update_layout(title="Flow-rate design window", xaxis_title="Flow (gpm)", yaxis_title="psi",
                      hovermode="x unified")
    st.plotly_chart(style(fig, 380), width="stretch")


@st.cache_data(show_spinner=False)
def _section_samples(_df, key):
    return _df[_df["bit_size"].between(8.5, 9.0)].iloc[::40]


def _calibrate(p):
    st.markdown("Model the circulating system at each depth of the FORGE 8¾ in section using the "
                "**measured flow rate** and **motor differential pressure**, and compare with the "
                "**measured standpipe pressure**. The constant residual is pressure lost in tools "
                "not in the surface data (motor/RSS/MWD off-bottom loss).")
    df = st.session_state["df_all"]
    s = _section_samples(df, len(df))
    if s.empty:
        return
    w0 = p["well"]
    tfa = hy.tfa_from_nozzles(p["nozzles"])
    rows = []
    for _, r in s.iterrows():
        bha, hw = w0.string[0], w0.string[1]
        dp_len = r["depth"] - bha.length - hw.length
        if dp_len <= 0:
            continue
        well = hy.Well([bha, hw, hy.StringComponent("Drill pipe", w0.string[2].od, w0.string[2].id, dp_len)],
                       [hy.HoleSection("Casing", w0.hole[0].diameter, min(w0.hole[0].bottom_md, r["depth"])),
                        hy.HoleSection("Open hole", w0.hole[1].diameter, r["depth"])])
        res = hy.circulate(well, p["fluid"], r["flow"], tfa, p["model"], p["surface_case"],
                           max(r.get("diff_press", 0.0), 0.0))
        rows.append({"depth": r["depth"], "measured": r["spp"], "model": res.standpipe_pressure})
    cal = pd.DataFrame(rows)
    offset = float((cal["measured"] - cal["model"]).median())
    cal["calibrated"] = cal["model"] + offset
    resid = cal["measured"] - cal["calibrated"]

    k = st.columns(3)
    k[0].metric("Unmodelled BHA loss", f"{offset:,.0f} psi")
    k[1].metric("Residual std after offset", f"{resid.std():.0f} psi")
    k[2].metric("Mean abs error", f"{resid.abs().mean():.0f} psi",
                f"{100 * resid.abs().mean() / cal['measured'].mean():.1f}% of SPP", delta_color="off")

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=cal["depth"], y=cal["measured"], mode="markers", name="Measured SPP",
                             marker=dict(size=6, color=SERIES[0])))
    fig.add_trace(go.Scatter(x=cal["depth"], y=cal["model"], mode="lines", name="Model (no BHA loss)",
                             line=dict(color=MUTED, width=1.5, dash="dot")))
    fig.add_trace(go.Scatter(x=cal["depth"], y=cal["calibrated"], mode="lines",
                             name="Model + constant BHA loss", line=dict(color=SERIES[1], width=2)))
    fig.update_layout(title="Standpipe pressure along the 8¾ in section", xaxis_title="MD (ft)",
                      yaxis_title="psi", hovermode="x unified")
    st.plotly_chart(style(fig, 400), width="stretch")
