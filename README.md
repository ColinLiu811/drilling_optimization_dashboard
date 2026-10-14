# Drilling Optimization Dashboard

An interactive Streamlit dashboard for drilling engineers built on real rig sensor
data from the **Utah FORGE 16A(78)-32** geothermal well. It does three things:

1. **ROP modeling and optimization.** Regress rate of penetration on weight on bit,
   rotary speed and flow rate, then search for the set-point that maximises
   predicted ROP inside the observed operating envelope.
2. **Hydraulics calculator.** Pressure losses through the circulating system,
   ECD, bit hydraulics and nozzle sizing. Power-law and Bingham plastic
   rheology are both supported, and the model is calibrated against the
   standpipe pressure the rig measured.
3. **Real-time monitoring.** Replays the well foot by foot as a live feed. A
   rolling model refits as it goes and flags drilling dysfunction (ROP shortfall,
   MSE spikes, standpipe-pressure deviations).

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
streamlit run app/streamlit_app.py
```

## Data

| | |
|---|---|
| Well | Utah FORGE 16A(78)-32, Milford, Utah (deviated, ~65° below 6,000 ft) |
| Source | [DOE Geothermal Data Repository, submission 1283](https://gdr.openei.org/submissions/1283): Pason EDR export at 10 s intervals |
| Raw | 608,676 rows × 36 channels, 118 MB, 25 Oct 2020 to 5 Jan 2021 |
| Bundled | `data/forge_16a78_32_depth_binned.csv`: 9,303 one-foot bins, 587 to 10,987 ft MD, 1 MB |

`scripts/prepare_forge.py` downloads the raw file and reduces it in three steps:

* It keeps **on-bottom rotary drilling** samples only (`On Bottom == 1`) and
  applies sanity ranges to WOB, RPM, flow, ROP and SPP. It also drops samples
  recorded behind the deepest depth reached so far (reaming after connections).
* It takes the **median within each 1 ft depth bin**. Time-based sampling puts
  many more rows on slow footage than fast footage, and depth binning removes
  that bias.
* It labels each bin with its **hole section** from the bit diameter channel
  (12¼ in, 12 in, 8¾ in).

To use another well (Volve, or an operator's own EDR export), map its columns
with `drillopt.data.standardize(df, column_map)`. A starter mapping for the Volve
real-time logs is in `VOLVE_COLUMN_MAP`. A CSV with the standard columns can
also be uploaded from the dashboard sidebar.

## ROP modeling

`drillopt/rop.py` provides four regressors with one interface:

| Model | Form |
|---|---|
| Power law (Bingham-type) | `ROP = K · WOB^a · RPM^b · Q^c`, least squares in log space with a smearing back-transform. Optional ridge shrinkage |
| Polynomial ridge | Degree-2 interactions on standardised inputs |
| Random forest | 200 trees, min leaf 10 |
| Gradient boosting | 300 stages, depth 3, subsample 0.8 |

**Validation uses depth-blocked cross-validation.** Neighbouring feet of hole
are nearly identical, so a shuffled K-fold lets the model memorise local
conditions and overstates its skill. In the 8¾ in section, a random forest scores
R² = 0.75 under shuffled 5-fold CV but only **0.33** when each 1,000 ft block is
predicted from the others. All figures below are blocked.

**Context matters more than the model.** WOB, RPM and flow alone cannot explain
ROP across a formation change. The 12 in section crosses the basin-fill to
granite contact, and ROP there drops several-fold with no change in parameters.
`rop_trail` (median ROP over the previous 30 ft, lagged so it never sees its own
target) is a real-time proxy for rock drillability:

| 5-block CV R² | 12 in, WOB/RPM/Q | 12 in, + `rop_trail` | 8¾ in, WOB/RPM/Q | 8¾ in, + `rop_trail` |
|---|---|---|---|---|
| Power law | -6.19 | **0.24** | 0.04 | 0.19 |
| Polynomial ridge | -3.52 | 0.18 | 0.32 | **0.42** |
| Random forest | -9.77 | -0.08 | 0.19 | 0.33 |
| Gradient boosting | -7.85 | -0.87 | 0.16 | 0.33 |

The tree models do badly in the 12 in section because a held-out depth block
sits outside the range of the training data, and trees cannot extrapolate.

**About the parameter responses.** Field data mixes cause and effect. Drillers
run more weight in harder rock, so a raw ROP-vs-WOB trend often slopes
*downward*. The dashboard shows partial dependence curves with that caveat, and
adding trailing context features reduces the bias.

### Optimizer

`drillopt/optimize.py` evaluates the model over a WOB × RPM × flow grid with
the context fixed (depth, recent drillability) and returns the best set-point
and the predicted gain. By default the search limits are the 5th to 95th
percentiles of the section's own data, so the model is never asked to
extrapolate. If no grid point beats the current parameters, it recommends
holding them. The dashboard plots the WOB × RPM response surface at the optimum
flow rate.

## Hydraulics

`drillopt/hydraulics.py` uses oilfield units and follows Bourgoyne et al.,
*Applied Drilling Engineering*, chapter 4.

* **Rheology** comes from PV/YP or Fann θ600/θ300. Power-law `n` and `K` are
  derived from the readings.
* **Frictional losses** are computed for pipe and concentric annulus.
  Laminar/turbulent switching uses Reynolds number. Turbulent friction uses
  Dodge–Metzner for power-law fluids and the Blasius form for Bingham plastic.
  Surface equipment follows IADC cases 1–4.
* **ECD** comes from cumulative annular loss, optionally with the cuttings load
  implied by the current ROP. It is reported per annular interval.
* **Bit hydraulics** covers ΔP, jet velocity, hydraulic horsepower, HSI and
  impact force.
* **Nozzle sizing** fits parasitic loss as `ΔPd = c·Q^m` from the full model,
  then solves the classic optimum: max bit HHP at `ΔPd = P/(m+1)`, or max jet
  impact at `ΔPd = 2P/(m+2)`. The flow rate is clamped to the window between
  the hole-cleaning minimum and the pump maximum. The result is matched to a
  practical jet set (sizes within 1/32 in of each other).

**Checks.** Applied Drilling Engineering Example 4.31 (12 ppg, 400 gpm,
3 × 12/32 jets) reproduces to within 0.5%: 1,611 psi bit ΔP and 387 ft/s jet
velocity. The calibration tab replays the FORGE 8¾ in section using the measured
flow rate and motor differential pressure. The pure model sits about 1,460 psi
under the measured standpipe pressure. That gap is pressure lost through the
motor, RSS and MWD, which the surface data does not record. Once it is taken as
a constant BHA loss, the model tracks measured SPP from 6,100 to 11,000 ft with a
**mean absolute error of 242 psi (6.7%)**.

The default mud properties (9.2 ppg, PV 14, YP 12) and string geometry are
reasonable assumptions for this well, not published values.

## Real-time monitor

`drillopt/realtime.py` contains:

* `ReplayStream`, which serves records in depth order with batching and seeking.
  It plays the role a WITS/WITSML feed would on a rig.
* `RealtimeMonitor`, which consumes one record at a time. It keeps a bounded
  150 ft buffer, computes Teale MSE, and refits a ridge-regularised power-law
  ROP model every 25 ft. It raises alerts once a condition has held for three
  consecutive records:
  * `ROP_BELOW_MODEL`: measured ROP is more than 40% below what the local model
    expects at the current parameters.
  * `MSE_SPIKE`: MSE is more than 3 robust σ (median/MAD) above its recent
    baseline.
  * `SPP_DEVIATION`: standpipe pressure changes by more than 10% while flow
    holds steady. A drop suggests a washout; a rise suggests a plugged jet or
    pack-off.

In the dashboard, the live panel runs inside `st.fragment(run_every=…)`, so
only that panel reruns on each tick. Over the full well the monitor raises
about 2.3 alerts per 100 ft (239 over 10,400 ft), and every threshold can be adjusted in the UI.

## Layout

```
drillopt/
  data.py          raw EDR -> filtered, depth-binned dataset
  rop.py           MSE, ROP regressors, blocked CV
  optimize.py      set-point search, response grids, partial dependence
  hydraulics.py    rheology, pressure losses, ECD, bit hydraulics, nozzles
  realtime.py      replay stream, rolling monitor, alerts
app/
  streamlit_app.py entry point and sidebar filters
  views/           overview, ROP model, hydraulics, real-time pages
scripts/
  prepare_forge.py download and preprocess the FORGE data
tests/             pytest suite (hydraulics, ROP, real-time)
```

## Tests

```bash
pytest -q
```

The suite covers the textbook bit-hydraulics check, monotonicity and regime
switching of the friction models, the pressure balance of the circulating
system, nozzle selection, exponent recovery on synthetic power-law data, the
no-leakage guarantee of the trailing features, optimizer bounds, and alert
behaviour on injected washouts and balled-bit events.

## Data citation

Utah FORGE: Well 16A(78)-32 Drilling Data. Energy & Geoscience Institute at the
University of Utah, Geothermal Data Repository, submission 1283.
https://gdr.openei.org/submissions/1283
