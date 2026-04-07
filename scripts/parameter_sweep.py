"""
GreenLight/scripts/parameter_sweep.py

Generate a training dataset for ML by running many greenhouse simulations
with varied control parameters. Each row in the output CSV represents one
simulation run: input parameters + aggregated output metrics.

Usage:
    python scripts/parameter_sweep.py

Configuration:
    Edit the constants below (N_SAMPLES, SIMULATION_DAYS, PARAM_SPACE) to
    control sweep size and parameter ranges.
"""

import os
import sys
import uuid
import warnings

import numpy as np
import pandas as pd

# ── Path setup ────────────────────────────────────────────────────────────────
project_dir = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, project_dir)

import greenlight  # noqa: E402

# ── Configuration ─────────────────────────────────────────────────────────────
SIMULATION_DAYS = 30    # Days per run. Max 111 with the built-in test data.
N_SAMPLES = 50          # Number of random parameter combinations.
RANDOM_SEED = 42        # For reproducibility.
OUTPUT_CSV = os.path.join(project_dir, "data", "training_data.csv")

# GreenLight model paths
BASE_PATH = os.path.join(project_dir, "greenlight", "models")
MODEL = os.path.join("katzin_2021", "definition", "main_katzin_2021.json")
WEATHER_FILE = os.path.abspath(
    os.path.join(BASE_PATH, "katzin_2021", "input_data", "test_data", "Bleiswijk_from_20091020.csv")
)

# Temporary output directory (relative to BASE_PATH, as required by GreenLight)
TEMP_OUTPUT_SUBDIR = os.path.join("katzin_2021", "output", "_sweep_temp")

# ── Parameter space ───────────────────────────────────────────────────────────
# Each entry: parameter_name -> (min, max, default)
# These are all "const" parameters in the Katzin 2021 model definition.
PARAM_SPACE = {
    # Heating setpoints (°C)
    "tSpDay":        (17.0,  24.0,  19.5),  # Day heating setpoint
    "tSpNight":      (14.0,  21.0,  18.5),  # Night heating setpoint
    # Lamp intensity (W/m²) — 0 = no supplemental lighting
    "thetaLampMax":  (0.0,   200.0, 120.0), # Maximum lamp intensity
    # CO2 injection setpoint during the day (ppm)
    "co2SpDay":      (400.0, 1500.0, 1000.0),
    # Ventilation dead zone between heating setpoint and vent opening (°C)
    "heatDeadZone":  (2.0,   10.0,  5.0),
    # Maximum allowable relative humidity before venting (%)
    "rhMax":         (70.0,  95.0,  87.0),
    # Thermal screen closes at night when outdoor temp is below this (°C)
    "thScrSpNight":  (5.0,   15.0,  10.0),
}


def sample_params(n_samples: int) -> list[dict]:
    """
    Draw n_samples random parameter combinations using uniform sampling.
    Returns a list of dicts {param_name: value}.
    """
    rng = np.random.default_rng(seed=RANDOM_SEED)
    samples = []
    for _ in range(n_samples):
        sample = {
            name: float(rng.uniform(low, high))
            for name, (low, high, _) in PARAM_SPACE.items()
        }
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

    # Simulation time option
    t_end = SIMULATION_DAYS * 24 * 3600
    options = {"options": {"t_end": str(t_end)}}

    # Parameter overrides — each param is a model constant redefinition
    param_mods = [
        {name: {"definition": f"{value:.6g}"}}
        for name, value in params.items()
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
        dmc = 0.06  # fruit dry matter content (fraction)

        result = {
            "sim_id": sim_idx,
            "simulation_days": SIMULATION_DAYS,
            # ── Input parameters ──────────────────────────────────────────
            **{f"in_{k}": v for k, v in params.items()},
            # ── Output metrics ────────────────────────────────────────────
            # Yield: total fresh-weight fruit harvest (kg/m²)
            "out_yield_kg_m2":        dt_s * data["mcFruitHar"].sum() * 1e-6 / dmc,
            # Heating energy from boiler (MJ/m²)
            "out_energy_heat_MJ_m2":  dt_s * (data["hBoilPipe"] + data["hBoilGroPipe"]).sum() * 1e-6,
            # Lighting energy (MJ/m²)
            "out_energy_light_MJ_m2": dt_s * (data["qLampIn"] + data["qIntLampIn"]).sum() * 1e-6,
            # Total energy (heat + light)
            "out_energy_total_MJ_m2": dt_s * (
                data["hBoilPipe"] + data["hBoilGroPipe"] + data["qLampIn"] + data["qIntLampIn"]
            ).sum() * 1e-6,
            # CO2 injected (kg/m²)
            "out_co2_kg_m2":          dt_s * data["mcExtAir"].sum() * 1e-6,
            # Irrigation water (liters/m²), estimated from canopy transpiration
            "out_water_L_m2":         dt_s * 1.1 * data["mvCanAir"].sum(),
            # Mean indoor air temperature (°C)
            "out_mean_tAir_C":        float(data["tAir"].mean()),
            # Mean indoor CO2 concentration (ppm)
            "out_mean_co2_ppm":       float(data["co2InPpm"].mean()),
            # Mean canopy temperature (°C)
            "out_mean_tCan_C":        float(data["tCan"].mean()),
            # Final fruit carbohydrate level (kg/m²) — proxy for crop state
            "out_final_cFruit":       float(data["cFruit"].iloc[-1]),
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


def main():
    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)

    print(f"Parameter sweep: {N_SAMPLES} simulations × {SIMULATION_DAYS} days each")
    print(f"Weather data: {WEATHER_FILE}")
    print(f"Output: {OUTPUT_CSV}\n")

    params_list = sample_params(N_SAMPLES)
    results = []

    for i, params in enumerate(params_list):
        param_str = ", ".join(f"{k.replace('in_', '')}={v:.2f}" for k, v in params.items())
        print(f"[{i + 1:>3}/{N_SAMPLES}] {param_str}")

        result = run_simulation(params, i)
        if result is not None:
            results.append(result)
            print(
                f"         yield={result['out_yield_kg_m2']:.3f} kg/m²  "
                f"heat={result['out_energy_heat_MJ_m2']:.1f} MJ/m²  "
                f"light={result['out_energy_light_MJ_m2']:.1f} MJ/m²  "
                f"CO2={result['out_co2_kg_m2']:.3f} kg/m²"
            )
        else:
            print("         (skipped)")

        # Save incrementally every 10 runs so progress isn't lost
        if (i + 1) % 10 == 0 and results:
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
