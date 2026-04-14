# GreenLight Calibration — Research & Status

*Longmont, CO residential greenhouse. Last updated: 2026-04-14. Full pipeline complete: calibration → sweep → NN surrogate trained.*

---

## 1. What We Are Trying to Do

GreenLight is a physics simulator (Katzin 2021, built on Vanthoor 2011) that models
greenhouse climate using ODEs for heat, humidity, CO₂, and crop growth. Out of the box
it is parameterized for a Dutch commercial glass greenhouse. Our greenhouse is a
residential polycarbonate structure in Longmont, CO, attached to the north wall of a
house, with a concrete slab, asymmetric airflow, and one-zone misters.

The goal of calibration is to tune the model's physical constants so that when you feed
it real outdoor weather, the simulated indoor temperature matches your actual Verdify
telemetry. Once the physics fit, the model can:

1. Score proposed climate plans before committing them (Iris as planner, GreenLight as
   physics oracle)
2. Replay historical windows to diagnose what went wrong
3. Predict energy use and stress hours under different setpoint strategies

**Current phase:** Full pipeline complete as of 2026-04-14.
- ✅ Three calibration windows scored (jan, spring, oct)
- ✅ Summer window attempted — excluded, structural model failure documented (§4.4)
- ✅ 500-sample LHS parameter sweep generated (`data/training_data.csv`)
- ✅ NN surrogate trained (`models/nn_surrogate.pkl`) — CV R²: temp=0.92, RH=0.80, cost=0.82
- ⬜ NN analysis script not yet written (optimization, Pareto frontier, inference benchmark)
- ⬜ Paper parameters table not yet updated with calibrated values

---

## 2. Greenhouse Physical Model (Longmont Overrides)

File: `greenlight/models/katzin_2021/definition/longmont_greenhouse.json`

| Parameter | Value | Notes |
|---|---|---|
| `aFlr` | 34.10 m² | Floor area |
| `aCov` | 74.8 m² | Cover area (north wall excluded — attached to house) |
| `hGh` | 3.00 m | Mean height |
| `hAir` | 2.90 m | Main compartment height |
| `hElevation` | 1551 m | Longmont, CO |
| `psi` | 30° | Roof panel angle |
| `lambdaRf` | 0.021 W/m·K | 6 mm twin-wall opal polycarbonate U-value proxy |
| `hRf` | 6e-3 m | Cover thickness |
| `cLeakage` | 3e-4 | Infiltration coefficient (calibration target — see §5) |
| `tauRfPar` | 0.57 | PAR transmission |
| `tauRfNir` | 0.55 | NIR transmission |
| `aRoof` | 0 | No roof vents |
| `phiVentForced` | 2.31 m³/s | 2 × 2450 CFM fans |
| `pBlow` | **1,500 W** | Electric stage 1 only — calibrated, see §4 |
| `tSpDay` | **19.17°C (66.5°F)** | Real thermostat setpoint — calibrated, see §3 |
| `thetaLampMax` | 42.4 W/m² | 49× Barrina T8 LED, 1446 W total |
| `lPipe` | 0 | No hot-water heating pipes |
| `pBoil` | 0 | No boiler |

---

## 3. Critical Finding: Operational Parameters Dominate

**The entire calibration is in operational parameters, not envelope physics.**

After running 3 calibration windows with a Nelder-Mead optimizer over cLeakage and
lambdaRf, the result in every case was: default envelope params are at or near optimal.
The optimizer improved MAE by 0–1.3% vs baseline. Nudging cLeakage or lambdaRf from
their defaults made things negligibly better or marginally worse.

The real calibration levers, in order of impact:

### 3.1 Setpoint Mismatch (biggest lever)

GreenLight default `tSpDay = 14.4°C (58°F)`. The real thermostat is set to 66.5°F
(19.17°C) year-round. With the wrong setpoint the heater never fires, the sim runs
cold, and any subsequent parameter search is fitting a fundamentally wrong mode.

**Fix:** Always pass `--tSpDay 66.5` to `calibrate.py`.

### 3.2 Heater Staging Mismatch (second biggest lever)

The real greenhouse has two staged heaters:

| Stage | Device | Power | Fires When |
|---|---|---|---|
| Stage 1 | Electric resistance | 1,500 W | Indoor < 66.5°F |
| Stage 2 | Gas furnace (75,000 BTU/hr) | 21,978 W | Indoor < ~63.5°F (rarely) |

GreenLight's default `pBlow = 23,478 W` (combined) causes it to fire a 14× oversized
heater in short bursts — low duty cycle, large temperature spikes. The real greenhouse
runs stage 1 (electric) continuously at low power. Stage 2 fires rarely.

**Fix:** Override `pBlow = 1500` in all runs. Gas rate only applies if modeling deep-
cold events where stage 2 actually fired.

**Cost implication:** Heat cost uses electric pricing ($0.0308/MJ), not gas
($0.00787/MJ). Electric is 3.9× more expensive per MJ.

### 3.3 Fan Dead Zone (heatDeadZone) — seasonal, important for fan runtime

GreenLight controls exhaust fans via `tSpDay + heatDeadZone`. The default dead zone is
13.4°C, meaning fans engage at 66.5°F + 24.1°F = 90.6°F — fans never fire in spring
or fall, causing a 7–8°F solar gain overshoot in those seasons.

Calibrated dead zone values:

| Window | heatDeadZone | Fan threshold | Real fan % | Sim fan % |
|---|---|---|---|---|
| jan_cold | 5°C (default OK) | 76°F | ~0% | ~0% |
| spring_apr | **5°C** | 76°F | 18.5% | 14.9% |
| oct_shoulder | **8°C** | 81°F | 3.0% | 3.6% |

**Fix:** Sweep heatDeadZone per window. Start at 5°C for spring/summer, 8°C for
shoulder seasons. Default 13.4°C is wrong for every season.

### 3.4 Envelope Parameters (cLeakage, lambdaRf) — negligible effect

Despite being the original calibration targets, these parameters contribute <1% MAE
improvement over defaults after the operational parameters are set correctly.

**Why:** The dominant error sources are structural (§8), not envelope. The greenhouse's
structural gaps — tree shading, slab thermal mass, north-wall buffering — are not
representable by leakage/conductivity adjustments. The flat objective landscape means
Nelder-Mead ran ~34 evaluations over 2+ hours with no convergence. Default values are
effectively optimal.

---

## 4. Calibration Results by Window

All windows: tSpDay=19.17°C, tSpNight=17.17°C, pBlow=1500W, cLeakage=3e-4, lambdaRf=0.021

### 4.1 jan_cold — **3.86°F MAE — Best thermal fit**

```
Window:   2026-01-13T06:00Z → 2026-01-20T06:00Z
CSV:      james-csv-files-2026-04-13/jan_cold_week.csv
File:     calibration/params_jan_cold_elec.json
```

| Metric | Value |
|---|---|
| MAE temp | **3.86°F** |
| Bias | +0.80°F (slight hot) |
| MAE RH | 31.7% |
| Heater real | 37.5% |
| Heater sim | 45.2% |
| Fan real | ~0% |
| Fan sim | ~0% |

Best result for thermal accuracy. Low bias. Heating dynamics well-captured. RH error
is large but expected — misters are unmodeled.

**heatDeadZone=5** (fans on at 76°F — no fan activity in January, so dead zone
doesn't matter here, but set consistently).

### 4.2 spring_apr — **3.36°F MAE — Overall best**

```
Window:   2026-04-06T06:00Z → 2026-04-13T06:00Z
CSV:      james-csv-files-2026-04-13/spring_apr_2026.csv
File:     calibration/params_spring_apr_cal.json
```

| Metric | Value |
|---|---|
| MAE temp | **3.36°F** |
| Bias | +2.28°F (moderate hot) |
| MAE RH | 13.8% |
| Heater real | 44.6% |
| Heater sim | 32.7% |
| Fan real | 18.5% |
| Fan sim | 14.9% |

Lowest MAE of all three windows. Fan runtimes within 4 pp. The +2.28°F hot bias
persists — solar gain overshoot in afternoon hours despite heatDeadZone=5 engaging
fans at 76°F. Likely east tree shade in morning limiting effective solar cooling.

**heatDeadZone=5** critical here — without it fans would never fire and MAE would
be ~7–8°F.

### 4.3 oct_shoulder — **5.97°F MAE — Structural gap**

```
Window:   2025-10-06T06:14Z → 2025-10-13T06:00Z
CSV:      james-csv-files-2026-04-13/oct_shoulder.csv
File:     calibration/params_oct_shoulder_cal.json
```

| Metric | Value |
|---|---|
| MAE temp | **5.97°F** |
| Bias | **+5.97°F (pure systematic hot)** |
| MAE RH | 16.5% |
| Heater real | 37.5% |
| Heater sim | 42.3% |
| Fan real | 3.0% |
| Fan sim | 3.6% |

The +5.97°F bias is completely systematic — the sim is always too hot by the same
amount. Fan and heater runtime fractions are close to real. This is not a parameter
problem; it's a structural model gap (see §8). No combination of cLeakage/lambdaRf
moves the mean temperature down by 6°F without destroying heater runtime accuracy.

**heatDeadZone=8** chosen to match 3.0% fan runtime. Lower dead zones overshoot
fan runtime (heatDeadZone=5 gave 10.1% sim vs 3.0% real).

### 4.4 aug_summer — **13.70°F MAE — Ventilation model failure (excluded)**

```
Window:   2025-08-06T06:00Z → 2025-08-13T00:00Z
CSV:      james-csv-files-2026-04-13/aug_summer.csv
File:     calibration/params_aug_summer_cal.json
Weather:  TMYx Boulder (substituted — station not installed Aug 2025)
```

| Metric | Value |
|---|---|
| MAE temp | **13.70°F** |
| Bias | **-4.78°F (sim runs cold)** |
| MAE RH | 23.2% |
| Heater real | 0.6% |
| Heater sim | 0.0% |
| Fan real | 27.8% |
| Fan sim | **100.0%** |

**Root cause — ventilation model breaks in pure cooling season.**
GreenLight's `ventHeat` control opens fans whenever `tAir > tSpDay + heatDeadZone` (75.5°F at heatDeadZone=5). Colorado August afternoons routinely exceed this, so the sim runs fans at 100% continuously. The ODE overcools the greenhouse, producing the -4.78°F cold bias. Increasing heatDeadZone to 15°C (raising fan threshold to 93.5°F) produced identical results — the ODE finds the same steady state because any reduction in fan speed lets temp immediately spike above heatMax.

**Compounding factor: TMYx weather substitution.** The weather station was not yet installed in August 2025. TMYx typical-year data was used instead. Unknown error in outdoor temperature and solar forcing adds to the structural mismatch.

**Two fixes needed before summer calibration is possible:**
1. Real weather station data for that week (not available retrospectively)
2. A modified ventilation guard: only open fans when `tOut < tAir` (prevents venting hot outdoor air into a cooler greenhouse)

**This window is excluded from the calibration suite.** It is documented here as a known limitation.

---

## 5. Summary Calibration Table

| Parameter | Calibrated Value | All Windows |
|---|---|---|
| `tSpDay` | 19.17°C (66.5°F) | Fixed across all |
| `tSpNight` | 17.17°C (62.9°F) | Fixed across all |
| `pBlow` | 1,500 W | Fixed across all |
| `heatDeadZone` | 5°C (spring/winter), 8°C (shoulder) | Window-specific |
| `cLeakage` | 3e-4 (default) | No improvement from tuning |
| `lambdaRf` | 0.021 (default) | No improvement from tuning |

**Key insight:** This greenhouse is operationally calibrated, not physically calibrated.
The GreenLight glass-greenhouse envelope physics is "good enough" once you fix the
control parameters. The error floor is set by structural gaps, not by the envelope.

---

## 6. Accuracy Assessment

| Window | MAE | Bias | Rating | Limiting Factor |
|---|---|---|---|---|
| jan_cold | 3.86°F | +0.80°F | **7/10** | Slab thermal lag overnight |
| spring_apr | 3.36°F | +2.28°F | **7.5/10** | East tree morning shade, slab |
| oct_shoulder | 5.97°F | +5.97°F | **4/10** | Structural gap, not fixable with params |
| aug_summer | 13.70°F | -4.78°F | **N/A** | Ventilation model failure + no real weather |
| **Overall (3 valid windows)** | — | — | **~6/10** | |

The summer window is excluded from the accuracy rating. It reveals a structural limitation of GreenLight in pure cooling season, not a calibration problem.

**For NN training data**: 7/10. The sim captures the correct shape of the response
surface (heat vs. lamps vs. season tradeoffs). Absolute temperatures are off by 3–6°F
but relative comparisons across parameter settings are valid. Training data was generated
across all seasons (including summer) using TMYx weather — the surrogate learns the
simulator's response surface, not ground truth, which is appropriate for plan scoring.

**For digital shadow (replay past behavior)**: 6/10 winter/spring, 4/10 fall.

**For planning copilot (optimize future setpoints)**: 5/10 winter, 3/10 fall until
structural gaps are addressed.

---

## 7. Optimizer Performance and Lessons Learned

### What We Used: Nelder-Mead (scipy.optimize.minimize)

The Nelder-Mead optimizer was implemented in `calibrate.py` with these settings:
- xatol=1e-5 (parameter tolerance)
- fatol=0.05°F (function value tolerance)
- maxiter=300

### Problem: xatol=1e-5 Caused Near-Infinite Runtime

On a near-flat objective landscape, the Nelder-Mead simplex shrinks toward xatol
rather than converging via fatol. With lambdaRf starting at 0.021 and an initial
step of ~0.001, reaching xatol=1e-5 requires 100× reduction in simplex size —
dozens of shrink steps that take 2+ hours of wall clock time.

**Spring and oct calibrations were killed at ~34 evals after 2+ hours, still not
converged.** Scored runs at default params confirmed defaults were near-optimal anyway.

**Fix for future runs:** Change `xatol=1e-5` to `xatol=1e-3` in `calibrate.py:633`.
Or remove xatol entirely and rely on `fatol=0.05` alone — on a smooth physics
landscape, function-value convergence is the right criterion.

### Why the Landscape is Flat

With operational params correctly set, the objective surface over (cLeakage, lambdaRf)
is very shallow. Both params control heat loss, which is modest on mild calibration
windows. The global minimum is at or near the defaults. The optimizer can't distinguish
signal from the ~0.1°F ODE solver resolution at this scale.

This confirms that **envelope parameter optimization is not the right lever for this
greenhouse**. Phase 2 should focus on structural corrections (§9) or accept the current
accuracy floor.

### Comparison of Methods for Future Work

| Method | Best For | Wall-clock (2 params) | Parallelism |
|---|---|---|---|
| Nelder-Mead (current) | Single smooth minimum | 1.5–3+ hrs | None |
| Grid search + Pool | Mapping the landscape | ~13 min (7×7, 4 workers) | Full |
| Bayesian (skopt) | 4+ params | 30–60 min | Partial (batch) |
| Differential Evolution | Multi-modal landscapes | 1–2 hrs | Full |

**Recommendation:** For the current 2-param problem, a 7×7 grid search using the
`multiprocessing.Pool` pattern from `sensitivity.py` would complete in ~13 minutes
(7-day sims) vs 2+ hours for Nelder-Mead. But given the flat landscape, it would
just confirm that defaults are optimal — so the value is in the visualization, not
the optimization.

If future phases add slab mass or solar correction factors (§9), switch to Bayesian
optimization (skopt) once you have 3+ parameters.

---

## 8. Structural Physics Gaps — What Parameters Can't Fix

These are residual errors that no combination of cLeakage/lambdaRf/heatDeadZone can
correct. They represent real physics that GreenLight doesn't model for this greenhouse.

| Gap | Estimated Error | Season | Mitigation |
|---|---|---|---|
| **East tree shading** | +2–4°F morning hot bias | Spring, fall, summer | Add solar attenuation factor by time-of-day; hardest to model |
| **Slab thermal mass** | +1–2°F overnight lag | All | Phase 2: tune cPFlr/rhoFlr/hFlr |
| **North-wall house buffer** | +0.5–1°F | Winter | Reduce effective aCov on north face or add tOut correction |
| **Patio door airflow** | Variable, large | Any | Flag in context CSV, exclude from scoring |
| **Single-zone model** | Zone divergence ~3–5°F | All | Use mean temp_avg; zone diagnostics separate |
| **Humidity (phiFog=0)** | 13–32% RH MAE | All | Phase 3: add fog nozzle model |

The oct_shoulder +5.97°F bias is most likely explained by **east tree shading** (low
sun angle in October amplifies morning shadow) combined with **slab thermal mass**
(warm slab from summer still releasing heat overnight). This is the primary reason oct
accuracy is 4/10 vs 7–7.5/10 for jan and spring.

---

## 9. What Would Actually Improve Accuracy

In priority order:

### 9.1 Solar attenuation correction (biggest potential gain)

The east-facing trees block morning direct solar in spring through fall. GreenLight
uses TMYx weather which assumes a clear horizon. Adding a multiplier that reduces
`isOut` (outdoor solar irradiance) by 20–40% for the first 2–3 hours after sunrise
from April–October would likely reduce oct bias from +5.97°F to +2–3°F.

**Implementation:** Preprocess the weather CSV to attenuate morning solar, or add a
time-of-day solar correction factor as a GreenLight parameter.

### 9.2 Slab thermal mass tuning

The concrete slab (34 m², ~10 cm thick) stores significant heat. GreenLight has
`cPFlr`, `rhoFlr`, `hFlr` for floor thermal mass but they're set for a generic
greenhouse soil, not concrete.

Concrete values:
- `cPFlr` = 840 J/kg·K (vs ~1000 for soil)
- `rhoFlr` = 2300 kg/m³ (vs ~1700 for soil)
- `hFlr` = 0.10 m (10 cm slab — vs ~0.05 m default)

Increasing effective floor heat capacity would slow nighttime cooling and reduce
overnight undershoot. Worth 1–2°F MAE improvement.

### 9.3 Fix xatol in calibrate.py ✅ Done 2026-04-14

`calibrate.py:633` changed from `"xatol": 1e-5` to `"xatol": 1e-3`.

This caused spring and oct calibration runs to execute ~34 Nelder-Mead evaluations over
2+ hours without converging. The fix allows convergence in ~15 evals on the flat
cLeakage/lambdaRf landscape. Confirmed: default envelope params are near-optimal and
no further optimization over these params is needed.

### 9.4 Summer calibration window — blocked on two structural issues

Attempted 2026-04-14. Aug 6–13, 2025 window scored at MAE=13.70°F. **Not usable.**

Two fixes are required before summer calibration is possible:

**Fix 1 — Ventilation outdoor temperature guard.** Add a condition to `uVentForced` in
`longmont_greenhouse.json` that prevents forced ventilation when outdoor temperature
exceeds indoor temperature:

```json
"uVentForced": {
  "definition": "min(ventCold, max(ventHeat, ventRh)) * smoothHeaviside(tAir - tOut, 2)"
}
```

This suppresses fan opening when outdoor air is hotter than indoor air — the correct
physical behavior for a residential greenhouse without an evaporative cooling system.

**Fix 2 — Real weather data.** The Aug 2025 weather station was not yet installed.
TMYx typical-year data was substituted but introduces unknown forcing error.
Summer calibration requires actual measured outdoor temp, solar, and wind for that week.

### 9.5 Humidity calibration (Phase 3)

Currently phiFog=0. MAE RH is 13–32% across windows. Requires:
1. MAE temp < 2°F first (Phase 2 gets there)
2. Windows with known misting activity and gallons consumed
3. 30-second resolution CSV (current 2.5-min smooths out misting cycles)

---

## 10. Parameter Sweep for NN Training

Script: `scripts/parameter_sweep.py`
Completed: **2026-04-14. 500 rows × 24 columns.**
Output: `data/training_data.csv`

### Sweep Configuration

| Setting | Value |
|---|---|
| N_SAMPLES | 500 |
| SIMULATION_DAYS | 30 |
| Sampling | Latin Hypercube (LHS) |
| Workers | 10 parallel |
| Start days | LHS-sampled 0–335 (all seasons) |
| pBlow fixed | 1500 W (electric) |

### Parameter Space

| Parameter | Min | Max | Physical Meaning |
|---|---|---|---|
| `tSpDay` | 10°C | 22°C | Day heating setpoint |
| `tSpNight` | 10°C | 22°C | Night setback (clamped ≤ tSpDay) |
| `thetaLampMax` | 0 | 42.4 W/m² | Supplemental lamp intensity |
| `heatDeadZone` | 5°C | 20°C | Dead zone before fan engagement |
| `rhMax` | 70% | 95% | RH threshold for exhaust fan |

### Output Columns

**Inputs:** `sim_id`, `simulation_days`, `start_day`, `start_month`, `in_tSpDay`,
`in_tSpNight`, `in_thetaLampMax`, `in_heatDeadZone`, `in_rhMax`

**Physical outputs:** `out_yield_kg_m2`, `out_energy_heat_MJ_m2`,
`out_energy_light_MJ_m2`, `out_energy_total_MJ_m2`, `out_water_L_m2`,
`out_mean_tAir_C`, `out_min_tAir_C`, `out_mean_rh_pct`, `out_mean_tCan_C`,
`out_final_cFruit`

**Cost outputs (electric pricing throughout):**
`out_cost_heat_usd_m2`, `out_cost_light_usd_m2`, `out_cost_water_usd_m2`,
`out_cost_total_usd_m2`, `out_cost_per_kg_yield`

### Cost Pricing (Longmont, CO — April 2026)

| Resource | Rate | Per MJ |
|---|---|---|
| Electricity | $0.111/kWh | $0.0308/MJ |
| Water | $0.00484/gallon | $0.00128/L |

Note: Gas pricing ($0.00787/MJ) is defined but not used — pBlow=1500W is
electric-only. An earlier draft used gas pricing for heat; this was corrected
before the final 500-sample run.

### Known Limitation: Yield is Near-Zero

All 500 runs show `out_yield_kg_m2 ≈ 0`. The GreenLight tomato crop model needs
3–4 months to accumulate harvestable fruit from a cold start. Use `out_final_cFruit`
(fruit dry matter at end of run) as the crop proxy for NN training.

---

## 11. NN Surrogate Model Results

Script: `scripts/train_nn.py`
Completed: **2026-04-14.**
Model: `models/nn_surrogate.pkl` (270 KB)
Metadata: `models/nn_surrogate_meta.json`

### Architecture

| Setting | Value |
|---|---|
| Input features | 6: in_tSpDay, in_tSpNight, in_thetaLampMax, in_heatDeadZone, in_rhMax, start_month |
| Targets | 3: out_mean_tAir_C, out_mean_rh_pct, out_cost_total_usd_m2 |
| Hidden layers | 128 → 64 (ReLU) |
| Solver | Adam, lr=5e-4, early stopping |
| Converged | Yes — 1693 iterations |
| Train / test split | 425 / 75 (85/15) |

### Performance

| Target | 5-fold CV R² | Test MAE |
|---|---|---|
| Mean air temp (°C) | **0.92 ± 0.02** | 1.48°C |
| Mean RH (%) | **0.80 ± 0.02** | 4.7% |
| Total cost ($/m²) | **0.82 ± 0.01** | $0.14/m² |

The surrogate learns the simulator's response surface, not ground truth. Temperature
R²=0.92 means the NN captures 92% of variance in what GreenLight would simulate.
Given the calibrated GreenLight is ~6/10 accurate vs reality, the NN introduces a
second layer of approximation. For plan ranking and relative comparison it is
sufficient; for absolute setpoint predictions it should carry uncertainty bounds.

### What the NN Can Do

- Predict mean climate and monthly cost in microseconds (vs ~18 seconds per GreenLight sim)
- Score control parameter candidates for Iris planner without running the full ODE
- Generate Pareto frontiers (comfort vs cost vs humidity) across parameter space

### What Is Still Missing

- No analysis script yet — the surrogate exists but doesn't *do* anything in code
- No inference speed benchmark (claim: orders of magnitude faster than GreenLight)
- No Pareto frontier or optimal setpoint curves
- No uncertainty quantification

---

## 12. Open Items

| Item | Priority | Status | Notes |
|---|---|---|---|
| Fix xatol=1e-5 in calibrate.py | High | ✅ Done 2026-04-14 | Changed to 1e-3 |
| Train NN on training_data.csv | High | ✅ Done 2026-04-14 | CV R²: temp=0.92, RH=0.80, cost=0.82 |
| Summer calibration window | Medium | ❌ Blocked | MAE=13.70°F — needs ventilation guard + real weather (§9.4) |
| NN analysis script | High | ⬜ Pending | Pareto frontier, monthly optima, speed benchmark — paper payoff |
| Solar attenuation correction | Medium | ⬜ Pending | Biggest potential accuracy gain for oct (§9.1) |
| Slab thermal mass tuning | Medium | ⬜ Pending | 1–2°F improvement, all windows (§9.2) |
| Update paper parameters table | Medium | ⬜ Pending | Use calibrated values from §5 |
| Humidity calibration (Phase 3) | Low | ⬜ Pending | After thermal MAE < 2°F (§9.5) |

---

## 13. Next Steps for the Paper

The calibration and surrogate pipeline is complete. What the paper still needs:

### Immediate (required for paper completeness)

**1. NN analysis script (`scripts/analyze_nn.py`)**
The trained surrogate needs to demonstrate value. Minimum viable analysis:
- Monthly optimal setpoints: for each month, sweep (tSpDay, heatDeadZone) and find
  the Pareto-efficient frontier of mean temp vs cost
- Speed benchmark: time 10,000 NN predictions vs 1 GreenLight sim — show the
  speedup (expected: ~10,000×)
- Prediction grid: 2D heatmaps of temp/cost over (tSpDay, start_month) to
  visually validate the surrogate's learned surface

**2. Paper parameters table**
Fill in the calibrated values from §5 and §4 results. Reference the calibration
JSON files. Note the seasonal heatDeadZone variation.

### Optional (strengthens accuracy claims)

**3. Solar attenuation correction**
Attenuate morning TMYx solar by 20–40% for 2–3 hrs post-sunrise, Apr–Oct.
Re-score oct window. If oct MAE drops below 4°F, the model is defensible year-round
for the heating season (Sep–May). This is the one change most likely to move the
overall accuracy rating from 6/10 to 7/10.

**4. Slab thermal mass tuning**
Set cPFlr=840, rhoFlr=2300, hFlr=0.10 in longmont_greenhouse.json. Re-score all
three windows. Expected 1–2°F improvement in overnight accuracy.

### Blocked (cannot proceed without new data or code changes)

**5. Summer calibration**
Requires (a) real weather station data for a summer week and (b) ventilation outdoor
temperature guard in the model definition (see §9.4). Both conditions are hard to meet
retroactively. Document as a known limitation.

**6. Humidity calibration**
Requires temp MAE < 2°F first. Currently at 3.4–6°F. Phase 3 only after Phase 2
structural corrections close the gap.

---

## 14. Files Reference

```
verdify/GreenLight/
  scripts/
    calibrate.py             ← calibration (Nelder-Mead, --tSpDay, --pBlow, --heatDeadZone)
    sensitivity.py           ← OAT sensitivity analysis (complete, 69/69)
    parameter_sweep.py       ← ML training data generator (LHS, 500-sample run complete)
    train_nn.py              ← NN surrogate trainer (complete 2026-04-14)
    prepare_longmont_weather.py
    test_longmont.py

  calibration/
    sensitivity_report.json          ← OAT results for cLeakage, aCov, lambdaRf × 3 seasons
    params_jan_cold_elec.json        ← Jan cold, electric pBlow — MAE 3.86°F
    params_spring_apr_cal.json       ← Spring Apr — MAE 3.36°F (best)
    params_oct_shoulder_cal.json     ← Oct shoulder — MAE 5.97°F (structural gap)
    params_aug_summer_cal.json       ← Aug summer — MAE 13.70°F (excluded, ventilation model failure)

  models/
    nn_surrogate.pkl                 ← trained sklearn Pipeline (StandardScaler + MLP 128→64)
    nn_surrogate_meta.json           ← training metadata and CV metrics

  data/
    training_data.csv                ← 500 rows × 24 cols, LHS sweep, correct electric pricing

  james-csv-files-2026-04-13/
    jan_cold_week.csv       ← Jan 13–20, 2026 (4206 rows)
    oct_shoulder.csv        ← Oct 6–13, 2025 (4248 rows)
    spring_apr_2026.csv     ← Apr 6–13, 2026 (9847 rows, ~1-min res)
    aug_summer.csv          ← Aug 6–13, 2025 (3252 rows; ws_* cols filled from TMYx)
```

---

## 15. Quick Reference: Running Calibration

```bash
# Score at default params — fast, 1 sim (~3 min)
python scripts/calibrate.py <csv> --window <label> --tSpDay 66.5 --pBlow 1500 --heatDeadZone 5

# Score and compare against previous calibration
python scripts/calibrate.py <csv> --window <label> --tSpDay 66.5 --pBlow 1500 \
    --heatDeadZone 5 --params-from calibration/params_jan_cold_elec.json

# Full optimize (envelope params only — low value for this greenhouse)
# NOTE: fix xatol to 1e-3 before running or it will never converge
python scripts/calibrate.py <csv> --window <label> --tSpDay 66.5 --pBlow 1500 \
    --heatDeadZone 5 --optimize
```

**Window → heatDeadZone map:**
- `jan_cold` → `--heatDeadZone 5` (fans don't fire in winter anyway)
- `spring_apr` → `--heatDeadZone 5` (fans on at 76°F matches 18.5% → 14.9% runtime)
- `oct_shoulder` → `--heatDeadZone 8` (fans on at 81°F matches 3.0% → 3.6% runtime)

**Interpreting MAE results for this greenhouse:**
- < 3°F: excellent — trust the model for planning
- 3–4°F: good — usable for plan scoring with known bias correction
- 4–6°F: structural gap likely — check for solar or slab issues
- > 6°F: something is wrong with setpoint, pBlow, or weather alignment
