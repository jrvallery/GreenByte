# GreenByte

Personalized greenhouse climate predictor using a simulation-to-ML pipeline. Calibrates the [GreenLight](https://github.com/davkat1/GreenLight) physics simulator to a specific residential greenhouse in Longmont, CO, generates a 500-sample synthetic dataset via Latin Hypercube Sampling, and trains a Neural Network surrogate that predicts indoor temperature, relative humidity, and operating cost in microseconds instead of the ~18 seconds per GreenLight ODE solve.

**Authors:** Alexis Marez · James Vallery — University of Colorado Boulder  
**Status (2026-04-14):** Full pipeline complete. Calibration → sweep → NN surrogate trained.

---

## Pipeline Overview

```
Real outdoor weather (TMYx / station)
        │
        ▼
GreenLight ODE simulator
  └─ calibrated to Longmont greenhouse physics
        │
        ▼
500-sample LHS parameter sweep  →  data/training_data.csv
        │
        ▼
Neural Network surrogate  →  models/nn_surrogate.pkl
        │
        ▼
Iris planner (plan scoring / setpoint optimization)
```

**Why this matters:** Small greenhouse owners lack months of labeled sensor data. By calibrating a physics simulator to their specific structure, we generate a rich synthetic dataset and train a personalized model — no manual data collection required.

---

## Greenhouse: Longmont, CO

Residential polycarbonate structure attached to the north wall of a house. Concrete slab floor, two staged electric/gas heaters, two exhaust fans, one-zone fog/mister humidification, 49× Barrina T8 LED grow lights.

| Parameter | Value | Notes |
|---|---|---|
| Floor area (`aFlr`) | 34.10 m² | |
| Cover area (`aCov`) | 74.8 m² | North wall excluded — attached to house |
| Mean height (`hGh`) | 3.00 m | |
| Elevation | 1551 m | Longmont, CO |
| Cover type | 6 mm twin-wall opal polycarbonate | |
| Roof vents | None | `aRoof = 0` |
| Fans | 2 × 2450 CFM exhaust | `phiVentForced = 2.31 m³/s` |
| Heater stage 1 | 1,500 W electric resistance | Fires at < 66.5°F |
| Heater stage 2 | 75,000 BTU/hr gas furnace | Fires rarely at < ~63.5°F |
| Grow lights | 49× Barrina T8, 1446 W total | `thetaLampMax = 42.4 W/m²` |

Model definition: `greenlight/models/katzin_2021/definition/longmont_greenhouse.json`

---

## Calibration

### Key Finding: Operational Parameters Dominate

The dominant calibration levers are **control parameters**, not envelope physics:

1. **Setpoint mismatch** — GreenLight default `tSpDay = 14.4°C (58°F)`. Real thermostat: **19.17°C (66.5°F)**. With the wrong setpoint the heater never fires and any envelope search is fitting the wrong mode.

2. **Heater staging** — GreenLight default `pBlow = 23,478 W` fires a 14× oversized burst heater. Real greenhouse runs stage 1 electric (1,500 W) continuously. **Fix: `pBlow = 1500`**.

3. **Fan dead zone** — Default `heatDeadZone = 13.4°C` means fans engage at 90.6°F — never in spring/fall, causing 7–8°F solar overshoot. **Fix: 5°C spring/winter, 8°C shoulder**.

4. **Envelope params** (`cLeakage`, `lambdaRf`) — Contribute < 1% MAE improvement over defaults once operational params are correct. **Default values are effectively optimal.**

### Calibration Results

All windows: `tSpDay=19.17°C`, `tSpNight=17.17°C`, `pBlow=1500W`, `cLeakage=3e-4`, `lambdaRf=0.021`

| Window | Dates | MAE | Bias | Fan% real/sim | Rating |
|---|---|---|---|---|---|
| `jan_cold` | Jan 13–20, 2026 | **3.86°F** | +0.80°F | ~0 / ~0 | **7/10** |
| `spring_apr` | Apr 6–13, 2026 | **3.36°F** | +2.28°F | 18.5 / 14.9 | **7.5/10** |
| `oct_shoulder` | Oct 6–13, 2025 | **5.97°F** | +5.97°F systematic | 3.0 / 3.6 | **4/10** |
| `aug_summer` | Aug 6–13, 2025 | 13.70°F | -4.78°F | 27.8 / **100** | N/A — excluded |

The summer window is excluded: GreenLight's ventilation model runs fans at 100% continuously in pure cooling season, overcooling the sim. Two fixes required before summer calibration is possible (see §Structural Gaps below).

### Calibrated Parameter Table

| Parameter | Value | Notes |
|---|---|---|
| `tSpDay` | 19.17°C (66.5°F) | Fixed across all windows |
| `tSpNight` | 17.17°C (62.9°F) | Fixed across all windows |
| `pBlow` | 1,500 W | Electric stage 1 only |
| `heatDeadZone` | 5°C (spring/winter) · 8°C (shoulder) | Window-specific |
| `cLeakage` | 3e-4 (default) | No improvement from tuning |
| `lambdaRf` | 0.021 W/m·K (default) | No improvement from tuning |

---

## Parameter Sweep

Script: `scripts/parameter_sweep.py`  
Output: `data/training_data.csv` — **500 rows × 24 columns**

| Setting | Value |
|---|---|
| Samples | 500 (Latin Hypercube) |
| Simulation length | 30 days per sample |
| Start days | LHS-distributed 0–335 (all seasons) |
| Workers | 10 parallel |
| `pBlow` | Fixed 1500 W (electric) |

**Input parameters swept:**

| Parameter | Range | Meaning |
|---|---|---|
| `tSpDay` | 10–22°C | Day heating setpoint |
| `tSpNight` | 10–22°C | Night setback (clamped ≤ tSpDay) |
| `thetaLampMax` | 0–42.4 W/m² | Supplemental lamp intensity |
| `heatDeadZone` | 5–20°C | Dead zone before fan engagement |
| `rhMax` | 70–95% | RH threshold for exhaust fan |

**Output columns:** mean air temp, min air temp, mean RH, mean canopy temp, heating/lighting/water/total energy, heating/lighting/water/total cost (USD/m²), fruit dry matter, yield.

**Cost pricing (Longmont, CO — April 2026):** Electricity $0.111/kWh ($0.0308/MJ) · Water $0.00484/gal ($0.00128/L)

> **Note on yield:** All 500 runs show `out_yield_kg_m2 ≈ 0`. The GreenLight tomato model needs 3–4 months to accumulate harvestable fruit from a cold start. Use `out_final_cFruit` (fruit dry matter) as the crop proxy for NN training.

---

## Neural Network Surrogate

Script: `scripts/train_nn.py`  
Model: `models/nn_surrogate.pkl` (270 KB sklearn Pipeline)  
Metadata: `models/nn_surrogate_meta.json`

**Architecture:**

| Setting | Value |
|---|---|
| Inputs | 6: tSpDay, tSpNight, thetaLampMax, heatDeadZone, rhMax, start_month |
| Targets | 3: mean air temp (°C), mean RH (%), total cost ($/m²) |
| Hidden layers | 128 → 64 (ReLU) |
| Solver | Adam, lr=5e-4, early stopping (1693 iters) |
| Train/test split | 425 / 75 (85/15) |

**Performance:**

| Target | 5-fold CV R² | Test MAE |
|---|---|---|
| Mean air temp | **0.92 ± 0.02** | 1.48°C |
| Mean RH | **0.80 ± 0.02** | 4.7% |
| Total cost | **0.82 ± 0.01** | $0.14/m² |

The surrogate predicts in microseconds vs ~18 s per GreenLight ODE solve — enabling real-time plan scoring and Pareto frontier generation for the Iris planner.

---

## Structural Physics Gaps

Residual errors that no combination of calibration parameters can fix:

| Gap | Est. Error | Season | Mitigation |
|---|---|---|---|
| East tree shading | +2–4°F morning hot bias | Spring/fall/summer | Solar attenuation factor by time-of-day |
| Slab thermal mass | +1–2°F overnight lag | All | Tune `cPFlr`/`rhoFlr`/`hFlr` for concrete |
| North-wall house buffer | +0.5–1°F | Winter | Reduce effective `aCov` on north face |
| Patio door airflow | Variable, large | Any | Flag in context CSV, exclude from scoring |
| Single-zone model | 3–5°F zone divergence | All | Use `temp_avg`; zone diagnostics separate |
| Humidity (`phiFog=0`) | 13–32% RH MAE | All | Phase 3: fog nozzle model |
| Summer ventilation | 13.7°F cold bias | Summer | Add `tOut < tAir` guard on forced vent |

---

## Running the Pipeline

### Calibrate a window

```bash
# Score at calibrated params — fast, 1 sim (~3 min)
python scripts/calibrate.py <csv> --window <label> --tSpDay 66.5 --pBlow 1500 --heatDeadZone 5

# Compare against saved calibration file
python scripts/calibrate.py <csv> --window <label> --tSpDay 66.5 --pBlow 1500 \
    --heatDeadZone 5 --params-from calibration/params_jan_cold_elec.json

# Full Nelder-Mead optimize (low value for this greenhouse — landscape is flat)
# NOTE: xatol was changed from 1e-5 → 1e-3 on 2026-04-14 to prevent infinite runtime
python scripts/calibrate.py <csv> --window <label> --tSpDay 66.5 --pBlow 1500 \
    --heatDeadZone 5 --optimize
```

**heatDeadZone by window:**
- `jan_cold` → `--heatDeadZone 5`
- `spring_apr` → `--heatDeadZone 5`
- `oct_shoulder` → `--heatDeadZone 8`

**Interpreting MAE:**
- < 3°F: excellent — trust for planning
- 3–4°F: good — usable with known bias correction
- 4–6°F: structural gap — check solar or slab issues
- > 6°F: setpoint, pBlow, or weather alignment problem

### Generate training data

```bash
python scripts/parameter_sweep.py
# output: data/training_data.csv (500 rows × 24 cols, ~90 min, 10 workers)
```

### Train NN surrogate

```bash
python scripts/train_nn.py
# output: models/nn_surrogate.pkl, models/nn_surrogate_meta.json
```

---

## Open Items

| Item | Priority | Status |
|---|---|---|
| NN analysis script (`analyze_nn.py`) | High | ⬜ Pending — Pareto frontier, monthly optima, speed benchmark |
| Update paper parameters table | Medium | ⬜ Pending — use calibrated values |
| Solar attenuation correction (morning tree shade) | Medium | ⬜ Pending — biggest oct accuracy gain |
| Slab thermal mass tuning (`cPFlr`, `rhoFlr`, `hFlr`) | Medium | ⬜ Pending — 1–2°F improvement |
| Summer calibration window | Medium | ❌ Blocked — needs ventilation guard + real Aug weather |
| Humidity calibration (Phase 3) | Low | ⬜ Pending — after thermal MAE < 2°F |
| Fix xatol in calibrate.py | High | ✅ Done 2026-04-14 — changed 1e-5 → 1e-3 |
| Train NN on training_data.csv | High | ✅ Done 2026-04-14 — CV R²: temp=0.92, RH=0.80, cost=0.82 |
| 500-sample LHS sweep | High | ✅ Done 2026-04-14 |
| Three calibration windows scored | High | ✅ Done 2026-04-14 |

---

## Files Reference

```
greenbyte/
  scripts/
    calibrate.py              Nelder-Mead calibration (--tSpDay, --pBlow, --heatDeadZone)
    parameter_sweep.py        LHS training data generator (500 samples, 10 workers)
    train_nn.py               NN surrogate trainer
    sensitivity.py            OAT sensitivity analysis (69/69 complete)
    prepare_longmont_weather.py
    test_longmont.py

  calibration/
    params_jan_cold_elec.json       Jan — MAE 3.86°F
    params_spring_apr_cal.json      Spring — MAE 3.36°F (best)
    params_oct_shoulder_cal.json    Oct shoulder — MAE 5.97°F (structural gap)
    params_aug_summer_cal.json      Aug summer — MAE 13.70°F (excluded)
    sensitivity_report.json         OAT results (cLeakage, aCov, lambdaRf × 3 seasons)

  models/
    nn_surrogate.pkl                Trained sklearn Pipeline (StandardScaler + MLP 128→64)
    nn_surrogate_meta.json          Training metadata and CV metrics

  data/
    training_data.csv               500 rows × 24 cols, LHS sweep, electric pricing

  james-csv-files-2026-04-13/
    jan_cold_week.csv               Jan 13–20, 2026 (4206 rows)
    spring_apr_2026.csv             Apr 6–13, 2026 (9847 rows, ~1-min res)
    oct_shoulder.csv                Oct 6–13, 2025 (4248 rows)
    aug_summer.csv                  Aug 6–13, 2025 (3252 rows; ws_* cols filled from TMYx)

  greenlight/
    models/katzin_2021/definition/longmont_greenhouse.json   Physical model definition

  CALIBRATION_RESEARCH.md     Full calibration notes, optimizer lessons, structural gap analysis
  paper.tex                   NeurIPS 2026 paper draft
```

---

## References

- Katzin, D. et al. (2021). GreenLight — An open source model for greenhouses with supplemental lighting. *Biosystems Engineering*, 194, 486–508.
- Vanthoor, B. et al. (2011). A methodology for model-based greenhouse design. *Biosystems Engineering*, 110(4), 363–377.
