"""
GreenLight/scripts/parameter_sweep.py

Generate a training dataset for ML by running many greenhouse simulations
with varied control parameters against the Longmont, CO residential greenhouse
model and Boulder TMYx real weather data.

Each row in the output CSV represents one simulation run:
    input parameters + aggregated output metrics.

Usage:
    python scripts/parameter_sweep.py
    (run scripts/prepare_longmont_weather.py first if weather CSV is missing)

Configuration:
    Edit the constants below (N_SAMPLES, SIMULATION_DAYS, PARAM_SPACE) to
    control sweep size and parameter ranges.
"""

import os
import sys
import uuid
import warnings
from multiprocessing import Pool

import numpy as np
import pandas as pd

# ── Path setup ────────────────────────────────────────────────────────────────
project_dir = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, project_dir)

import greenlight  # noqa: E402

# ── Configuration ─────────────────────────────────────────────────────────────
SIMULATION_DAYS = 30    # Days per run. Up to 365 with the Boulder TMYx data.
N_SAMPLES = 500         # Number of random parameter combinations.
RANDOM_SEED = 42        # For reproducibility.
N_WORKERS = max(1, (os.cpu_count() or 4) - 2)  # Leave 2 cores free for the OS
OUTPUT_CSV = os.path.join(project_dir, "data", "training_data.csv")

# GreenLight model paths — Longmont residential greenhouse + Boulder real weather
BASE_PATH = os.path.join(project_dir, "greenlight", "models")
MODEL = os.path.join("katzin_2021", "definition", "main_longmont.json")
WEATHER_FILE = os.path.abspath(
    os.path.join(BASE_PATH, "katzin_2021", "input_data", "longmont",
                 "longmont_weather_from_jan_01_000000.csv")
)

if not os.path.exists(WEATHER_FILE):
    raise FileNotFoundError(
        f"Boulder weather file not found:\n  {WEATHER_FILE}\n"
        "Run  python scripts/prepare_longmont_weather.py  first."
    )

# Temporary output directory (relative to BASE_PATH, as required by GreenLight)
TEMP_OUTPUT_SUBDIR = os.path.join("katzin_2021", "output", "_sweep_temp")

# ── Utility rates (Longmont, CO — April 2026) ─────────────────────────────────
ELECTRIC_USD_PER_KWH  = 0.111    # $/kWh
GAS_USD_PER_THERM     = 0.83     # $/therm  (1 therm ≈ 1 CCF natural gas)
WATER_USD_PER_GALLON  = 0.00484  # $/gallon ($4.84/1000 gal)

# Derived cost-per-physical-unit used in metrics below
_ELEC_USD_PER_MJ  = ELECTRIC_USD_PER_KWH / 3.6      # 1 kWh = 3.6 MJ  → $0.0308/MJ
_GAS_USD_PER_MJ   = GAS_USD_PER_THERM    / 105.5    # 1 therm = 105.5 MJ → $0.00787/MJ
_WATER_USD_PER_L  = WATER_USD_PER_GALLON / 3.785    # 1 gal = 3.785 L  → $0.00128/L
# Electric is 3.9× more expensive per MJ than gas — critical for lamp-vs-heat tradeoffs

# ── Fixed overrides (calibrated operational parameters) ──────────────────────
# These are held constant for all sweep runs, matching the calibrated greenhouse.
# Calibration finding (jan_cold, 7-day window): the real greenhouse runs the
# 1500 W electric stage only (gas stage fires rarely / not during calibration windows).
# Default pBlow=23478 W (combined electric+gas) produces wildly over-powered heating
# and wrong energy estimates for this greenhouse.
FIXED_OVERRIDES = {
    "pBlow": 1500.0,   # 1500 W electric stage — calibrated from jan_cold_7day_v2
}

# ── Parameter space ───────────────────────────────────────────────────────────
# Each entry: parameter_name -> (min, max, default)
# Longmont-specific ranges — no CO2 injection, no thermal screen, LED lamps capped at 42.4 W/m².
PARAM_SPACE = {
    # Heating setpoints (°C) — installed thermostat range 10–22 °C
    "tSpDay":       (10.0, 22.0, 14.4),  # Day heating setpoint (default: 58 °F = 14.4 °C)
    "tSpNight":     (10.0, 22.0, 14.4),  # Night setback — sampled then clamped to ≤ tSpDay
    # Supplemental lamp intensity (W/m²) — 0 = off, 42.4 = all 49 fixtures on
    "thetaLampMax": (0.0,  42.4, 42.4),
    # Dead zone between heat setpoint and fan-on temperature (°C)
    # Fans engage at tSpDay + heatDeadZone.
    # Calibrated value ≈ 5 °C (fans on at ~75 °F / 24 °C for this greenhouse).
    # GreenLight default 13.4 °C (fan-on at 90.6 °F) causes 7–8 °F hot bias in spring/fall
    # because fans never fire in the sim despite running 18% of the time in reality.
    "heatDeadZone": (5.0,  20.0, 5.0),
    # RH threshold for exhaust fan activation (%)
    "rhMax":        (70.0, 95.0, 81.0),
}

# ── Simulation start-day range ────────────────────────────────────────────────
# Each run gets a random start day so the dataset covers all seasons.
# Valid range: day 0 to day (365 - SIMULATION_DAYS) so the window stays within the weather file.
_MAX_START_DAY = 365 - SIMULATION_DAYS   # = 335 for 30-day runs


def sample_params(n_samples: int) -> list[dict]:
    """
    Draw n_samples parameter combinations using Latin Hypercube Sampling (LHS).

    LHS divides each parameter's range into n_samples equal-width intervals and
    guarantees exactly one sample per interval per dimension, then shuffles columns
    independently. This gives much better coverage than pure random with the same
    number of runs — no clustering, no gaps.

    Also samples a random simulation start day (0–335) per run so the dataset
    spans all seasons, not just January.

    Enforces tSpNight ≤ tSpDay (night setback can't exceed day setpoint).

    Returns a list of dicts {param_name: value, "t_start_s": seconds_offset}.
    """
    rng = np.random.default_rng(seed=RANDOM_SEED)
    n_params = len(PARAM_SPACE)

    # LHS: place one point per interval per dimension, then shuffle columns
    intervals = (np.arange(n_samples)[:, None] + rng.random((n_samples, n_params))) / n_samples
    for col in range(n_params):
        rng.shuffle(intervals[:, col])

    # Also LHS-sample the start day across the year
    start_day_intervals = (np.arange(n_samples) + rng.random(n_samples)) / n_samples
    rng.shuffle(start_day_intervals)
    start_days = (start_day_intervals * _MAX_START_DAY).astype(int)

    samples = []
    for i in range(n_samples):
        sample = {}
        for j, (name, (lo, hi, _)) in enumerate(PARAM_SPACE.items()):
            sample[name] = float(lo + intervals[i, j] * (hi - lo))

        # Night setpoint must not exceed day setpoint (clamp, don't discard)
        sample["tSpNight"] = min(sample["tSpNight"], sample["tSpDay"])

        # Simulation window: random start day, fixed length
        sample["t_start_s"] = int(start_days[i]) * 86400
        samples.append(sample)

    return samples


def run_simulation(params: dict, sim_idx: int) -> dict | None:
    """
    Run one GreenLight simulation with the given parameter overrides.
    Returns a flat dict of input params + aggregated output metrics, or None on failure.
    """
    # Unique output filename to avoid collisions during parallel future use
    output_filename = f"sweep_{sim_idx:04d}_{uuid.uuid4().hex[:6]}.csv"
    output_rel_path = os.path.join(TEMP_OUTPUT_SUBDIR, output_filename)
    output_abs_path = os.path.join(BASE_PATH, output_rel_path)

    os.makedirs(os.path.dirname(output_abs_path), exist_ok=True)

    # Per-run simulation window (t_start randomised across the year)
    t_start_s = int(params.get("t_start_s", 0))
    t_end_s   = t_start_s + SIMULATION_DAYS * 86400
    options   = {"options": {"t_start": str(t_start_s), "t_end": str(t_end_s)}}

    # Parameter overrides — fixed calibrated overrides first, then swept params
    param_mods = [
        {name: {"definition": f"{value:.6g}"}}
        for name, value in FIXED_OVERRIDES.items()
    ] + [
        {name: {"definition": f"{value:.6g}"}}
        for name, value in params.items()
        if name != "t_start_s"
    ]

    # Full input: model + options + parameter overrides + weather data
    input_arg = [MODEL, options] + param_mods + [WEATHER_FILE]

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mdl = greenlight.GreenLight(
                base_path=BASE_PATH,
                input_prompt=input_arg,
                output_path=output_rel_path,
            )
            mdl.run()

        # ── Load and aggregate output ─────────────────────────────────────
        raw = pd.read_csv(output_abs_path, header=None, low_memory=False)
        col_names = raw.iloc[0]
        data = raw.iloc[3:].reset_index(drop=True).apply(pd.to_numeric, errors="coerce")
        data.columns = col_names

        if len(data) < 2:
            print(f"  [sim {sim_idx}] Too few output rows, skipping.")
            return None

        dt_s = float(data["Time"].iloc[1]) - float(data["Time"].iloc[0])  # time step in seconds
        dmc  = 0.06  # fruit dry matter content (fraction)

        # ── Aggregated physical quantities ────────────────────────────────
        heat_MJ_m2  = dt_s * data["hBlowAir"].sum() * 1e-6
        light_MJ_m2 = dt_s * (data["qLampIn"] + data["qIntLampIn"]).sum() * 1e-6
        water_L_m2  = dt_s * 1.1 * data["mvCanAir"].sum()
        yield_kg_m2 = dt_s * data["mcFruitHar"].sum() * 1e-6 / dmc

        # ── Cost in USD per m² of floor area ──────────────────────────────
        # pBlow is fixed at 1500 W electric (stage 1) — use electric rate for heat.
        # Gas rate would apply only if the gas furnace (stage 2, 21978 W) were modeled.
        cost_heat_usd_m2  = heat_MJ_m2  * _ELEC_USD_PER_MJ
        cost_light_usd_m2 = light_MJ_m2 * _ELEC_USD_PER_MJ
        cost_water_usd_m2 = water_L_m2  * _WATER_USD_PER_L
        cost_total_usd_m2 = cost_heat_usd_m2 + cost_light_usd_m2 + cost_water_usd_m2

        # ── Cost per kg of yield (NaN-safe) ──────────────────────────────
        cost_per_kg = cost_total_usd_m2 / yield_kg_m2 if yield_kg_m2 > 1e-6 else float("nan")

        # ── Start month (1=Jan … 12=Dec) for easy seasonal filtering ─────
        start_day   = t_start_s // 86400
        start_month = min(12, start_day // 30 + 1)

        result = {
            "sim_id":          sim_idx,
            "simulation_days": SIMULATION_DAYS,
            "start_day":       start_day,
            "start_month":     start_month,
            # ── Input parameters ──────────────────────────────────────────
            **{f"in_{k}": v for k, v in params.items() if k != "t_start_s"},
            # ── Physical outputs ──────────────────────────────────────────
            # Note: yield is near-zero for 30-day cold-start runs — the crop model needs
            # months to accumulate harvestable fruit. Use out_final_cFruit as the crop proxy.
            "out_yield_kg_m2":        yield_kg_m2,
            "out_energy_heat_MJ_m2":  heat_MJ_m2,
            "out_energy_light_MJ_m2": light_MJ_m2,
            "out_energy_total_MJ_m2": heat_MJ_m2 + light_MJ_m2,
            "out_water_L_m2":         water_L_m2,
            "out_mean_tAir_C":        float(data["tAir"].mean()),
            "out_min_tAir_C":         float(data["tAir"].min()),
            "out_mean_rh_pct":        float(data["rhIn"].mean()),
            "out_mean_tCan_C":        float(data["tCan"].mean()),
            "out_final_cFruit":       float(data["cFruit"].iloc[-1]),
            # ── Cost outputs (USD per m² of floor area) ───────────────────
            "out_cost_heat_usd_m2":   cost_heat_usd_m2,   # electric (pBlow=1500W)
            "out_cost_light_usd_m2":  cost_light_usd_m2,  # LED lamps (electric)
            "out_cost_water_usd_m2":  cost_water_usd_m2,
            "out_cost_total_usd_m2":  cost_total_usd_m2,
            "out_cost_per_kg_yield":  cost_per_kg,         # $/kg — primary optimisation target
        }
        return result

    except Exception as exc:
        print(f"  [sim {sim_idx}] FAILED: {exc}")
        return None

    finally:
        # Clean up temporary output file
        if os.path.exists(output_abs_path):
            os.remove(output_abs_path)
        # Also remove the model struct log if present
        log_path = output_abs_path.replace(".csv", "_model_struct_log.json")
        if os.path.exists(log_path):
            os.remove(log_path)
        sim_log_path = output_abs_path.replace(".csv", "_simulation_log.txt")
        if os.path.exists(sim_log_path):
            os.remove(sim_log_path)


def _run_sim_worker(args: tuple) -> dict | None:
    """Top-level wrapper so multiprocessing.Pool can pickle it."""
    params, sim_idx = args
    return run_simulation(params, sim_idx)


def main():
    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)

    print(f"Parameter sweep: {N_SAMPLES} simulations × {SIMULATION_DAYS} days each")
    print(f"Workers: {N_WORKERS} parallel (of {os.cpu_count()} CPU cores)")
    print(f"Weather data: {WEATHER_FILE}")
    print(f"Output: {OUTPUT_CSV}\n")

    params_list = sample_params(N_SAMPLES)
    results = []
    completed = 0

    args = [(params, i) for i, params in enumerate(params_list)]

    with Pool(processes=N_WORKERS) as pool:
        for result in pool.imap_unordered(_run_sim_worker, args):
            completed += 1
            if result is not None:
                results.append(result)
                print(
                    f"[{completed:>3}/{N_SAMPLES}] sim_id={result['sim_id']}  "
                    f"yield={result['out_yield_kg_m2']:.3f} kg/m²  "
                    f"cost=${result['out_cost_total_usd_m2']:.2f}/m²  "
                    f"(heat=${result['out_cost_heat_usd_m2']:.2f}  "
                    f"light=${result['out_cost_light_usd_m2']:.2f})  "
                    f"tAir_min={result['out_min_tAir_C']:.1f}°C  "
                    f"month={result['start_month']}"
                )
            else:
                print(f"[{completed:>3}/{N_SAMPLES}] (skipped)")

            # Save incrementally every 10 completions so progress isn't lost
            if completed % 10 == 0 and results:
                pd.DataFrame(results).to_csv(OUTPUT_CSV, index=False)
                print(f"  -- Checkpoint: {len(results)} rows saved to {OUTPUT_CSV}\n")

    # Final save
    df_out = pd.DataFrame(results)
    df_out.to_csv(OUTPUT_CSV, index=False)
    print(f"\nDone. {len(results)}/{N_SAMPLES} successful simulations.")
    print(f"Training data saved to: {OUTPUT_CSV}")
    print(f"\nColumns: {list(df_out.columns)}")
    print(df_out.describe())


if __name__ == "__main__":
    main()
