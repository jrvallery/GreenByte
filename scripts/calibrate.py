"""
GreenLight/scripts/calibrate.py

Calibrate GreenLight's thermal envelope to match actual Verdify greenhouse
telemetry, then report the parameters that best reproduce your physical
greenhouse's temperature and humidity response.

── How it works ────────────────────────────────────────────────────────────────

1. Load a Verdify CSV export (outdoor weather + indoor climate + equipment states)
2. Convert the outdoor columns to GreenLight's weather-input format
3. Run a baseline open-loop simulation (GreenLight uses its own control logic
   against the real outdoor weather)
4. Score the error: sim tAir (°C→°F) vs real temp_avg (°F)
5. Optionally run Nelder-Mead optimisation over the three most impactful
   thermal-envelope parameters, saving best-fit overrides to calibration/

── Why open-loop, not constrained? ─────────────────────────────────────────────
In open-loop mode GreenLight runs its own thermostat against the real outdoor
weather.  Residual error (sim vs real) comes from wrong *thermal physics*
(leakage, U-value, cover area), not from control-logic mismatches.  Once the
physics fits you can add constrained-control replays for richer diagnostics.

── Export SQL ───────────────────────────────────────────────────────────────────
Run this on your Verdify TimescaleDB to get the CSV this script expects:

    COPY (
      SELECT
        c.ts,
        w.temp_f            AS ws_temp_f,
        w.rh_pct            AS ws_rh_pct,
        w.solar_irradiance_w_m2  AS ws_solar_w_m2,
        w.wind_speed_avg_mph AS ws_wind_mph,
        c.temp_avg, c.rh_avg, c.vpd_avg,
        c.temp_north, c.temp_east, c.temp_west, c.temp_south,
        c.heat1, c.heat2, c.fan1, c.fan2, c.vent
      FROM v_greenhouse_state c
      LEFT JOIN LATERAL (
          SELECT * FROM weather_station
          WHERE ts <= c.ts ORDER BY ts DESC LIMIT 1
      ) w ON true
      WHERE c.ts BETWEEN '<start_iso>' AND '<end_iso>'
      ORDER BY c.ts
    ) TO STDOUT WITH CSV HEADER;

    docker exec verdify-timescaledb psql -U verdify -d verdify -c "<query>" > export.csv

── Usage ────────────────────────────────────────────────────────────────────────
    # Score baseline only (fast — one sim run):
    python scripts/calibrate.py export.csv --window spring_dry_vpd

    # Score + optimise thermal envelope (slow — ~100 sim runs, ~5 min):
    python scripts/calibrate.py export.csv --window spring_dry_vpd --optimize
"""

import argparse
import io
import json
import os
import sys
import warnings

import numpy as np
import pandas as pd
from scipy.optimize import minimize

# ── Path setup ────────────────────────────────────────────────────────────────
project_dir = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, project_dir)
import greenlight  # noqa: E402

BASE_PATH   = os.path.join(project_dir, "greenlight", "models")
MODEL       = os.path.join("katzin_2021", "definition", "main_longmont.json")
TEMP_SUBDIR = os.path.join("katzin_2021", "output", "_cal_temp")
CALIB_DIR   = os.path.join(project_dir, "calibration")

ELEVATION_M = 1551.0   # Longmont, CO
CO2_PPM     = 420

# ── Calibration parameter space ───────────────────────────────────────────────
# (default, lo, hi) — only physics parameters, not control setpoints
#
# aCov (cover area) was dropped after sensitivity analysis showed only 0.2–0.4 °F
# swing across its full 55–78 m² range — not worth optimizer iterations.
# cLeakage and lambdaRf each move mean tAir by ~2.5–3.0 °F and are the dominant
# thermal levers for this polycarbonate residential greenhouse.
CAL_PARAMS = {
    # Infiltration leakage coefficient. Residential poly is leakier than glass.
    # Sensitivity: ~2.3 °F winter, ~4.1 °F spring across range.
    "cLeakage":  (3e-4,  1e-4,  4.0e-3),

    # Effective thermal conductivity of the polycarbonate cover (W/m·K).
    # lambdaRf / hRf = U-value.  Sensitivity: ~3.0 °F consistent across seasons.
    "lambdaRf":  (0.021, 0.010, 0.060),
}


# ── Thermodynamic helpers ─────────────────────────────────────────────────────

def sat_vp(temp_c):
    """Saturation vapor pressure (Pa) at temp_c (°C)."""
    t = np.asarray(temp_c, dtype=float)
    return 610.78 * np.exp(17.2694 * t / (t + 238.3))


def co2_ppm_to_mgm3(temp_c, ppm=CO2_PPM):
    """Convert CO₂ ppm → mg/m³ at ambient temperature."""
    t = np.asarray(temp_c, dtype=float)
    return 101325 * 1e-6 * ppm * 44.01e-3 / (8.314 * (t + 273.15)) * 1e6


def sky_temp_proxy(tOut_c, iGlob_Wm2):
    """
    Approximate apparent sky temperature (°C).
    Colorado rule of thumb: clear-sky day → tOut − 25; night/overcast → tOut − 12.
    """
    return np.where(iGlob_Wm2 > 50, tOut_c - 25.0, tOut_c - 12.0)


def soil_temp_annual(day_of_year, mean_c=10.0, amp_c=8.0, peak_day=240):
    """
    Annual sinusoidal soil temperature at 2 m depth.
    Longmont defaults: mean 10 °C, amplitude 8 °C, peaks late August (day 240).
    """
    doy = np.asarray(day_of_year, dtype=float)
    return mean_c + amp_c * np.sin(2 * np.pi * (doy - peak_day) / 365.0)


# ── Data loading ──────────────────────────────────────────────────────────────

def load_verdify_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["ts"] = pd.to_datetime(df["ts"], utc=True)
    df = df.sort_values("ts").reset_index(drop=True)

    required = ["ts", "ws_temp_f", "ws_rh_pct", "ws_solar_w_m2", "ws_wind_mph",
                "temp_avg", "rh_avg"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        print("\nERROR — missing columns:", missing)
        print("Available columns:", list(df.columns))
        print("\nCheck the export SQL at the top of this file.")
        sys.exit(1)

    # Weather-station data arrives every ~60 s; climate rows every ~5 min.
    # Forward-fill then back-fill so every climate row has outdoor values.
    for col in ["ws_temp_f", "ws_rh_pct", "ws_solar_w_m2", "ws_wind_mph"]:
        df[col] = df[col].ffill().bfill()

    # PostgreSQL exports booleans as 't'/'f' strings — coerce to Python bool.
    _bool_map = {"t": True, "f": False, True: True, False: False}
    for col in ["heat1", "heat2", "fan1", "fan2", "vent", "fog",
                "mist_south", "mist_west", "mist_center"]:
        if col in df.columns:
            df[col] = df[col].map(_bool_map)

    n       = len(df)
    n_temp  = df["temp_avg"].notna().sum()
    n_solar = (df["ws_solar_w_m2"].fillna(0) > 0).sum()
    print(f"  {n} rows  |  temp_avg valid: {n_temp}  |  solar > 0: {n_solar}")
    return df


def infer_tSpDay_f(real_df: pd.DataFrame) -> float | None:
    """
    Estimate the daytime heating setpoint (°F) from equipment data.

    Logic: find rising edges where heat1 just turned ON — the indoor temperature
    at those moments should be approximately at the thermostat setpoint.
    Returns the median of those temperatures, or None if data is insufficient.
    """
    if "heat1" not in real_df.columns or "temp_avg" not in real_df.columns:
        return None
    heat    = real_df["heat1"].ffill().fillna(False).astype(bool)
    # Rising edge: current row is ON, previous row was OFF
    rising  = heat & ~heat.shift(1, fill_value=False)
    temps_f = real_df.loc[rising, "temp_avg"].dropna()
    if len(temps_f) < 3:
        return None
    return float(temps_f.median())


# ── Weather CSV builder ───────────────────────────────────────────────────────

def build_weather_csv(df: pd.DataFrame):
    """
    Convert Verdify export to GreenLight's 3-row-header weather format.
    Returns (csv_string, t_start_ts, t_end_seconds).
    """
    ts     = df["ts"]
    time_s = (ts - ts.iloc[0]).dt.total_seconds().values

    tOut_c = (df["ws_temp_f"].values - 32.0) * 5.0 / 9.0
    rh_out = np.clip(df["ws_rh_pct"].values, 1.0, 100.0) / 100.0
    vpOut  = rh_out * sat_vp(tOut_c)
    iGlob  = np.clip(df["ws_solar_w_m2"].values, 0.0, None)
    wind   = df["ws_wind_mph"].values * 0.44704
    tSky   = sky_temp_proxy(tOut_c, iGlob)
    tSoOut = soil_temp_annual(ts.dt.dayofyear.values)
    co2Out = co2_ppm_to_mgm3(tOut_c)

    # Daily radiation sum (MJ/m²) — trapezoidal integration per calendar day
    dates      = ts.dt.date.values
    dayRadSum  = np.zeros(len(df))
    for d in np.unique(dates):
        mask = dates == d
        dt_arr = np.diff(time_s[mask], prepend=time_s[mask][0])
        dayRadSum[mask] = float(np.sum(iGlob[mask] * dt_arr) * 1e-6)

    isDay = (iGlob > 1.0).astype(float)

    out = pd.DataFrame({
        "Time":         time_s,
        "iGlob":        iGlob,
        "tOut":         tOut_c,
        "vpOut":        vpOut,
        "co2Out":       co2Out,
        "wind":         wind,
        "tSky":         tSky,
        "tSoOut":       tSoOut,
        "dayRadSum":    dayRadSum,
        "isDay":        isDay,
        "isDaySmooth":  isDay,
        "hElevation":   ELEVATION_M,
    })

    cols  = list(out.columns)
    units = ["s", "W m**-2", "°C", "Pa", "mg m**-3", "m s**-1",
             "°C", "°C", "MJ m**-2", "-", "-", "m above sea level"]
    descs = [
        "Time since start of data",      "Outdoor global solar radiation",
        "Outdoor temperature",            "Outdoor vapor pressure",
        "Outdoor CO2 concentration",      "Outdoor wind speed",
        "Apparent sky temperature",       "Soil temperature at 2 m depth",
        "Daily sum of outdoor global solar radiation",
        "Switch determining if it is day or night",
        "Smooth switch determining if it is day or night",
        "Elevation at location",
    ]

    buf = io.StringIO()
    buf.write(",".join(cols) + "\n")
    buf.write(",".join(descs) + "\n")
    buf.write(",".join(units) + "\n")
    out.to_csv(buf, header=False, index=False)

    return buf.getvalue(), ts.iloc[0], float(time_s[-1])


# ── Simulation runner ─────────────────────────────────────────────────────────

_sim_count = 0
_run_id    = f"{os.getpid()}"   # unique per process — prevents temp file collisions
                                 # when multiple windows are calibrated in parallel


def run_sim(weather_csv_path: str, t_end_s: float,
            param_overrides: dict | None = None) -> pd.DataFrame:
    """
    Run one GreenLight simulation.  Returns parsed output DataFrame.
    Cleans up all temp files regardless of success/failure.
    """
    global _sim_count
    _sim_count += 1
    out_rel = os.path.join(TEMP_SUBDIR, f"cal_{_run_id}_{_sim_count:05d}.csv")
    out_abs = os.path.join(BASE_PATH, out_rel)
    os.makedirs(os.path.dirname(out_abs), exist_ok=True)

    overrides = []
    if param_overrides:
        for name, val in param_overrides.items():
            overrides.append({name: {"definition": f"{float(val):.8g}"}})

    options      = {"options": {"t_end": str(int(t_end_s))}}
    input_prompt = [MODEL, options] + overrides + [weather_csv_path]

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mdl = greenlight.GreenLight(
                base_path=BASE_PATH,
                input_prompt=input_prompt,
                output_path=out_rel,
            )
            mdl.run()

        raw      = pd.read_csv(out_abs, header=None, low_memory=False)
        col_names = raw.iloc[0]
        data     = raw.iloc[3:].reset_index(drop=True).apply(pd.to_numeric, errors="coerce")
        data.columns = col_names
        return data

    finally:
        for path in [out_abs,
                     out_abs.replace(".csv", "_model_struct_log.json"),
                     out_abs.replace(".csv", "_simulation_log.txt")]:
            if os.path.exists(path):
                os.remove(path)


# ── Scoring ───────────────────────────────────────────────────────────────────

def score_sim(sim_df: pd.DataFrame, real_df: pd.DataFrame) -> dict:
    """
    Align sim output to real Verdify timestamps and compute error metrics.

    GreenLight tAir is °C; Verdify temp_avg is °F.
    GreenLight rhIn is %; Verdify rh_avg is %.
    """
    t_start   = real_df["ts"].iloc[0]
    sim_ts    = t_start + pd.to_timedelta(sim_df["Time"].values.astype(float), unit="s")
    sim_index = pd.DatetimeIndex(sim_ts)

    real_ix   = real_df.set_index("ts").sort_index()

    def align(col_name):
        return (real_ix[col_name]
                .reindex(sim_index, method="nearest", tolerance=pd.Timedelta("10min"))
                if col_name in real_ix.columns else pd.Series(np.nan, index=sim_index))

    # ── Temperature ────────────────────────────────────────────────────────────
    real_temp_f = align("temp_avg")
    sim_tair_f  = sim_df["tAir"].values * 9.0 / 5.0 + 32.0
    valid_t     = real_temp_f.notna().values & np.isfinite(sim_tair_f)
    n_t         = valid_t.sum()

    if n_t < 10:
        print("  WARNING: fewer than 10 aligned temperature rows — check timestamp timezone")

    mae_t  = float(np.abs(sim_tair_f[valid_t] - real_temp_f.values[valid_t]).mean()) if n_t else np.nan
    rmse_t = float(np.sqrt(((sim_tair_f[valid_t] - real_temp_f.values[valid_t])**2).mean())) if n_t else np.nan
    bias_t = float((sim_tair_f[valid_t] - real_temp_f.values[valid_t]).mean()) if n_t else np.nan

    # ── Relative humidity ──────────────────────────────────────────────────────
    real_rh  = align("rh_avg")
    sim_rh   = sim_df["rhIn"].values if "rhIn" in sim_df.columns else np.full(len(sim_df), np.nan)
    valid_rh = real_rh.notna().values & np.isfinite(sim_rh)
    mae_rh   = float(np.abs(sim_rh[valid_rh] - real_rh.values[valid_rh]).mean()) if valid_rh.sum() else np.nan

    # ── Heater runtime fraction ────────────────────────────────────────────────
    real_heat_frac = sim_heat_frac = np.nan
    if "heat1" in real_ix.columns and "uBlow" in sim_df.columns:
        real_heat      = align("heat1").fillna(False).astype(float)
        real_heat_frac = float(real_heat.mean())
        sim_heat_frac  = float((sim_df["uBlow"] > 0.1).mean())

    # ── Fan runtime fraction ───────────────────────────────────────────────────
    real_fan_frac = sim_fan_frac = np.nan
    if "fan1" in real_ix.columns and "uVentForced" in sim_df.columns:
        real_fan      = align("fan1").fillna(False).astype(float)
        real_fan_frac = float(real_fan.mean())
        sim_fan_frac  = float((sim_df["uVentForced"] > 0.1).mean())

    return {
        "mae_temp_f":      mae_t,
        "rmse_temp_f":     rmse_t,
        "bias_temp_f":     bias_t,
        "mae_rh_pct":      mae_rh,
        "real_heat_frac":  real_heat_frac,
        "sim_heat_frac":   sim_heat_frac,
        "real_fan_frac":   real_fan_frac,
        "sim_fan_frac":    sim_fan_frac,
        "n_aligned":       int(n_t),
    }


# ── Objective function ────────────────────────────────────────────────────────

def _objective(x, weather_csv_path, t_end_s, real_df, verbose, fixed_overrides=None):
    """Weighted MAE: temperature (primary) + RH/10 (secondary).

    fixed_overrides: dict of GreenLight params held constant across all evaluations
    (e.g. {"tSpDay": 18.3} to set the heating setpoint).  These are merged with
    the physics params being optimised — fixed values are never moved by Nelder-Mead.
    """
    names  = list(CAL_PARAMS.keys())
    bounds = [(CAL_PARAMS[n][1], CAL_PARAMS[n][2]) for n in names]

    # Clip params to bounds instead of returning a penalty.
    # Nelder-Mead doesn't enforce bounds natively.  A fixed penalty causes
    # fatol termination to fail because out-of-bounds vertices inflate the
    # simplex function-value spread >> fatol=0.05.  Clipping evaluates at
    # the boundary instead, so all simplex vertices return real objective
    # values and fatol/xatol terminate correctly.
    x = [max(lo, min(hi, v)) for v, (lo, hi) in zip(x, bounds)]

    overrides = {**(fixed_overrides or {}), **dict(zip(names, x))}
    try:
        sim_df = run_sim(weather_csv_path, t_end_s, overrides)
        m      = score_sim(sim_df, real_df)
        # Optimise on temperature MAE only.  RH is dominated by fog/mist physics
        # that GreenLight does not model — including it contaminates the thermal
        # envelope fit by trading worse temperature for better humidity.
        obj    = m["mae_temp_f"]
        if verbose:
            phys_str = "  ".join(f"{n}={v:.5g}" for n, v in zip(names, x))
            print(f"    {phys_str}  → MAE_T={m['mae_temp_f']:.2f}°F"
                  f"  MAE_RH={m['mae_rh_pct']:.1f}%  obj={obj:.3f}")
        return float(obj) if np.isfinite(obj) else 1e6
    except Exception as exc:
        if verbose:
            print(f"    FAILED: {exc}")
        return 1e6


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description="Calibrate GreenLight thermal envelope to Verdify telemetry")
    ap.add_argument("csv",      help="Path to Verdify export CSV")
    ap.add_argument("--window", default="default",
                    help="Label for this calibration window (used in output filename)")
    ap.add_argument("--optimize", action="store_true",
                    help="Run Nelder-Mead optimisation (adds ~5–10 min)")
    ap.add_argument("--tSpDay", type=float, default=None, metavar="TEMP_F",
                    help="Daytime heating setpoint in °F (e.g. 65).  "
                         "GreenLight default is 58 °F — override this when your "
                         "real thermostat is set higher, otherwise the sim heater "
                         "never fires and the optimizer fits the wrong physics.  "
                         "Query your real value with: "
                         "SELECT ts, value FROM setpoint_changes "
                         "WHERE parameter='temp_low' ORDER BY ts;")
    ap.add_argument("--tSpNight", type=float, default=None, metavar="TEMP_F",
                    help="Night-time heating setpoint in °F.  "
                         "Defaults to --tSpDay if not given separately.")
    ap.add_argument("--pBlow", type=float, default=None, metavar="WATTS",
                    help="Heater output power in watts.  Default: 23478 W (combined "
                         "1500 W electric + 21978 W gas).  Use 21978 to model gas-only "
                         "(what runs most of the time in winter) or 1500 for electric-only "
                         "(stage-1 mild-cold).  Useful for diagnosing whether the staged "
                         "heating model is warping envelope parameter estimates.")
    ap.add_argument("--heatDeadZone", type=float, default=None, metavar="DELTA_C",
                    help="Dead zone between heating setpoint and fan-on temperature (°C). "
                         "Default: 13.4 °C (fans turn on at tSpDay+13.4 ≈ 91 °F for this "
                         "greenhouse).  In spring/fall the real fans turn on much earlier — "
                         "use ~5 °C (fans on at ~76 °F) to match observed fan runtime.")
    ap.add_argument("--days", type=float, default=None, metavar="N",
                    help="Truncate the CSV to the first N days before calibrating.  "
                         "3–4 days is enough to identify cLeakage and lambdaRf and "
                         "cuts sim time from ~2.5 min to ~1 min per eval.  "
                         "Default: use the full CSV.")
    ap.add_argument("--params-from", default=None, metavar="JSON",
                    help="Path to a previous calibration JSON (e.g. calibration/params_jan_cold.json).  "
                         "Seeds the optimizer at those parameter values instead of the "
                         "GreenLight defaults, skipping the slow initial simplex phase and "
                         "letting you test whether parameters are consistent across seasons.")
    args = ap.parse_args()

    print(f"\n{'='*62}")
    print(f"  GreenLight calibration  —  window: {args.window}")
    print(f"{'='*62}\n")

    # ── 1. Load Verdify data ───────────────────────────────────────────────────
    print("Loading Verdify CSV...")
    real_df = load_verdify_csv(args.csv)

    if args.days is not None:
        cutoff = real_df["ts"].iloc[0] + pd.Timedelta(days=args.days)
        real_df = real_df[real_df["ts"] <= cutoff].copy()
        print(f"  Truncated to {args.days} days  ({len(real_df)} rows)")

    t_lo = real_df["ts"].iloc[0].isoformat()
    t_hi = real_df["ts"].iloc[-1].isoformat()
    print(f"  Window    : {t_lo}  →  {t_hi}")
    print(f"  Temp range: {real_df['temp_avg'].min():.1f} – {real_df['temp_avg'].max():.1f} °F")
    print(f"  Solar max : {real_df['ws_solar_w_m2'].max():.0f} W/m²")

    # ── 1b. Resolve heating setpoint ──────────────────────────────────────────
    # GreenLight default tSpDay = 14.4 °C (58 °F).  If your real greenhouse
    # minimum indoor temp is above that, the sim heater never fires and the
    # optimizer compensates by tightening the envelope — wrong physics.
    fixed_overrides: dict[str, float] = {}

    if args.tSpDay is not None:
        tSpDay_c = (args.tSpDay - 32.0) * 5.0 / 9.0
        fixed_overrides["tSpDay"] = tSpDay_c
        tSpNight_c = ((args.tSpNight - 32.0) * 5.0 / 9.0
                      if args.tSpNight is not None else tSpDay_c - 2.0)
        fixed_overrides["tSpNight"] = tSpNight_c
        print(f"\n  Setpoint override  : tSpDay={args.tSpDay:.1f}°F "
              f"({tSpDay_c:.2f}°C)  tSpNight={tSpNight_c*9/5+32:.1f}°F "
              f"({tSpNight_c:.2f}°C)")

    if args.pBlow is not None:
        fixed_overrides["pBlow"] = args.pBlow
        print(f"  Heater override    : pBlow={args.pBlow:.0f} W  "
              f"(default 23478 W combined)")

    if args.heatDeadZone is not None:
        fixed_overrides["heatDeadZone"] = args.heatDeadZone
        fan_on_f = (fixed_overrides.get("tSpDay", 14.4) + args.heatDeadZone) * 9/5 + 32
        print(f"  Fan dead zone      : heatDeadZone={args.heatDeadZone:.1f}°C  "
              f"(fans on above {fan_on_f:.0f}°F)")

    else:
        # Auto-detect from rising edges of heat1 and warn if suspicious
        inferred_f = infer_tSpDay_f(real_df)
        gl_default_f = 14.4 * 9 / 5 + 32   # 58 °F
        real_min_f   = real_df["temp_avg"].min()
        if real_min_f > gl_default_f + 2:
            print(f"\n  ⚠  SETPOINT WARNING")
            print(f"     GreenLight default heat setpoint : {gl_default_f:.0f} °F")
            print(f"     Real indoor minimum this window  : {real_min_f:.1f} °F")
            if inferred_f is not None:
                print(f"     Auto-detected setpoint (rising edge median): {inferred_f:.1f} °F")
                print(f"     → Consider re-running with --tSpDay {inferred_f:.0f}")
            else:
                print(f"     → Query setpoint_changes for temp_low and re-run with --tSpDay <value>")
            print(f"     Without this fix the sim heater will not fire and calibration")
            print(f"     will fit wrong thermal-envelope parameters.\n")

    # ── 2. Build GreenLight weather input ─────────────────────────────────────
    print("\nBuilding GreenLight weather input...")
    weather_str, _, t_end_s = build_weather_csv(real_df)

    weather_dir = os.path.join(BASE_PATH, "katzin_2021", "input_data", "longmont")
    os.makedirs(weather_dir, exist_ok=True)
    weather_path = os.path.join(weather_dir, f"_cal_{args.window}.csv")
    with open(weather_path, "w", encoding="utf-8") as wf:
        wf.write(weather_str)

    n_rows = len(weather_str.splitlines()) - 3   # subtract 3 header rows
    print(f"  {n_rows} rows  |  {t_end_s/86400:.1f} days  |  saved: {os.path.basename(weather_path)}")

    # ── 3. Baseline simulation ─────────────────────────────────────────────────
    print("\n── Baseline simulation (default parameters) ────────────────────")
    base_sim = run_sim(weather_path, t_end_s, fixed_overrides or None)
    base_m   = score_sim(base_sim, real_df)

    print(f"  MAE  temp  : {base_m['mae_temp_f']:.2f} °F")
    print(f"  RMSE temp  : {base_m['rmse_temp_f']:.2f} °F")
    print(f"  Bias temp  : {base_m['bias_temp_f']:+.2f} °F  "
          f"({'sim runs hot' if base_m['bias_temp_f'] > 0 else 'sim runs cold'})")
    print(f"  MAE  RH    : {base_m['mae_rh_pct']:.1f} %")
    print(f"  Heater on  : real={base_m['real_heat_frac']*100:.1f}%  "
          f"sim={base_m['sim_heat_frac']*100:.1f}%")
    print(f"  Fan on     : real={base_m['real_fan_frac']*100:.1f}%  "
          f"sim={base_m['sim_fan_frac']*100:.1f}%")
    print(f"  Aligned pts: {base_m['n_aligned']}")

    # Interpretation hint
    heater_gap = (base_m['real_heat_frac'] - base_m['sim_heat_frac'])
    if not np.isnan(heater_gap) and heater_gap > 0.30 and not fixed_overrides:
        print(f"\n  *** HEATER MISMATCH: real={base_m['real_heat_frac']*100:.0f}%  "
              f"sim={base_m['sim_heat_frac']*100:.0f}%")
        inferred_f = infer_tSpDay_f(real_df)
        hint_val   = f"{inferred_f:.0f}" if inferred_f else "<your_setpoint>"
        print(f"  *** The sim heater never fires — GreenLight's default setpoint (58°F) is")
        print(f"  *** below your real indoor minimum ({real_df['temp_avg'].min():.1f}°F).")
        print(f"  *** Re-run with:  --tSpDay {hint_val}")
        print(f"  *** (query exact value: SELECT value FROM setpoint_changes")
        print(f"  ***  WHERE parameter='temp_low' ORDER BY ts DESC LIMIT 1;)")
    elif abs(base_m['bias_temp_f']) > 5:
        if base_m['bias_temp_f'] > 0:
            print("\n  Hint: sim runs hot — try INCREASING cLeakage (more infiltration cooling)")
        else:
            print("\n  Hint: sim runs cold — try DECREASING cLeakage or lambdaRf")

    # ── 3b. Score a loaded parameter set (--params-from without --optimize) ─────
    if args.params_from and not args.optimize:
        try:
            with open(args.params_from, encoding="utf-8") as pf:
                prev = json.load(pf)
            loaded = {n: prev["params"].get(n, CAL_PARAMS[n][0]) for n in CAL_PARAMS}
            print(f"\n── Calibrated simulation  ({os.path.basename(args.params_from)}) ──")
            for n, v in loaded.items():
                print(f"  {n}: {CAL_PARAMS[n][0]:.4g} (default) → {v:.6g}")
            overrides_all = {**(fixed_overrides or {}), **loaded}
            cal_sim = run_sim(weather_path, t_end_s, overrides_all)
            cal_m   = score_sim(cal_sim, real_df)
            print(f"  MAE  temp  : {cal_m['mae_temp_f']:.2f} °F  "
                  f"(baseline: {base_m['mae_temp_f']:.2f} °F)")
            print(f"  RMSE temp  : {cal_m['rmse_temp_f']:.2f} °F")
            print(f"  Bias temp  : {cal_m['bias_temp_f']:+.2f} °F  "
                  f"({'sim runs hot' if cal_m['bias_temp_f'] > 0 else 'sim runs cold'})")
            print(f"  MAE  RH    : {cal_m['mae_rh_pct']:.1f} %")
            print(f"  Heater on  : real={cal_m['real_heat_frac']*100:.1f}%  "
                  f"sim={cal_m['sim_heat_frac']*100:.1f}%")
            print(f"  Fan on     : real={cal_m['real_fan_frac']*100:.1f}%  "
                  f"sim={cal_m['sim_fan_frac']*100:.1f}%")
            print(f"  Aligned pts: {cal_m['n_aligned']}")
            if base_m['mae_temp_f'] > 0:
                imp = (base_m['mae_temp_f'] - cal_m['mae_temp_f']) / base_m['mae_temp_f'] * 100
                print(f"\n  Improvement: {imp:+.1f}% vs baseline MAE")
        except Exception as exc:
            print(f"  WARNING: could not score --params-from ({exc})")
        _cleanup(weather_path)
        return

    if not args.optimize:
        print("\nTip: run with --optimize to search for better thermal-envelope parameters.")
        _cleanup(weather_path)
        return

    # ── 4. Nelder-Mead optimisation ───────────────────────────────────────────
    print(f"\n── Nelder-Mead optimisation ────────────────────────────────────")
    print(f"  Tuning : {', '.join(CAL_PARAMS.keys())}")
    print(f"  Expect : 80–200 sim evaluations  (~5–15 min)\n")

    x0 = [CAL_PARAMS[n][0] for n in CAL_PARAMS]
    if args.params_from:
        try:
            with open(args.params_from, encoding="utf-8") as f:
                prev = json.load(f)
            seeded = [prev["params"].get(n, CAL_PARAMS[n][0]) for n in CAL_PARAMS]
            print(f"  Seeding optimizer from: {args.params_from}")
            for n, v in zip(CAL_PARAMS.keys(), seeded):
                print(f"    {n}: {CAL_PARAMS[n][0]:.6g} (default) → {v:.6g} (seeded)")
            x0 = seeded
        except Exception as exc:
            print(f"  WARNING: could not load --params-from ({exc}), using defaults")

    _cache  = {}

    def _cached_objective(x, weather_csv_path, t_end_s, real_df, verbose, fixed_overrides=None):
        key = tuple(round(v, 10) for v in x)
        if key not in _cache:
            _cache[key] = _objective(x, weather_csv_path, t_end_s, real_df,
                                     verbose, fixed_overrides)
        return _cache[key]

    obj0 = _cached_objective(x0, weather_path, t_end_s, real_df, verbose=True,
                             fixed_overrides=fixed_overrides or None)
    print(f"\n  Baseline objective: {obj0:.4f}\n")

    result = minimize(
        _cached_objective,
        x0=x0,
        args=(weather_path, t_end_s, real_df, True, fixed_overrides or None),
        method="Nelder-Mead",
        options={
            "xatol":   1e-3,   # 1e-5 caused hours-long non-convergence on flat cLeakage/lambdaRf landscape
            "fatol":   0.05,      # stop when Δobjective < 0.05 °F — good enough
            "maxiter": 300,
            "adaptive": True,     # scale simplex to parameter magnitudes
        },
    )

    # ── 5. Report results ──────────────────────────────────────────────────────
    print(f"\n── Optimisation result ─────────────────────────────────────────")
    print(f"  Converged : {result.success}  |  Evaluations: {result.nfev}  "
          f"|  Iterations: {result.nit}")

    # Clip result.x to bounds — the clipping objective lets internal simplex
    # vertices drift outside the feasible region; result.x may be unphysical.
    names_list = list(CAL_PARAMS.keys())
    clipped_x  = [max(CAL_PARAMS[n][1], min(CAL_PARAMS[n][2], v))
                  for n, v in zip(names_list, result.x)]
    best = dict(zip(names_list, clipped_x))
    print("\n  Best-fit parameters:")
    for name, val in best.items():
        raw_val  = result.x[names_list.index(name)]
        base_val = CAL_PARAMS[name][0]
        pct_chg  = (val - base_val) / base_val * 100
        clipped_note = f"  [clipped from {raw_val:.4g}]" if abs(raw_val - val) > 1e-10 else ""
        print(f"    {name:20s}  {val:.6g}   (default: {base_val:.6g},  Δ = {pct_chg:+.1f}%){clipped_note}")

    best_with_fixed = {**(fixed_overrides or {}), **best}
    final_sim = run_sim(weather_path, t_end_s, best_with_fixed)
    final_m   = score_sim(final_sim, real_df)

    print(f"\n  Temperature error:")
    print(f"    Before : MAE={base_m['mae_temp_f']:.2f}°F  RMSE={base_m['rmse_temp_f']:.2f}°F"
          f"  bias={base_m['bias_temp_f']:+.2f}°F")
    print(f"    After  : MAE={final_m['mae_temp_f']:.2f}°F  RMSE={final_m['rmse_temp_f']:.2f}°F"
          f"  bias={final_m['bias_temp_f']:+.2f}°F")
    improv = (base_m['mae_temp_f'] - final_m['mae_temp_f']) / base_m['mae_temp_f'] * 100
    print(f"    Improvement: {improv:.1f}%")

    print(f"\n  Humidity error:")
    print(f"    Before : MAE_RH={base_m['mae_rh_pct']:.1f}%")
    print(f"    After  : MAE_RH={final_m['mae_rh_pct']:.1f}%")

    # ── 6. Save calibrated params ──────────────────────────────────────────────
    os.makedirs(CALIB_DIR, exist_ok=True)
    out_path = os.path.join(CALIB_DIR, f"params_{args.window}.json")
    output = {
        "window":               args.window,
        "t_start":              t_lo,
        "t_end":                t_hi,
        "baseline": {
            "mae_temp_f":   round(base_m["mae_temp_f"], 3),
            "bias_temp_f":  round(base_m["bias_temp_f"], 3),
            "mae_rh_pct":   round(base_m["mae_rh_pct"], 2),
        },
        "calibrated": {
            "mae_temp_f":   round(final_m["mae_temp_f"], 3),
            "bias_temp_f":  round(final_m["bias_temp_f"], 3),
            "mae_rh_pct":   round(final_m["mae_rh_pct"], 2),
        },
        "improvement_pct": round(improv, 1),
        # Physics params tuned by the optimizer
        "params": {k: float(v) for k, v in best.items()},
        # Control params held fixed during optimization (setpoints, etc.)
        "fixed_overrides": {k: float(v) for k, v in (fixed_overrides or {}).items()},
        # Ready-to-paste GreenLight override list (physics + fixed)
        "greenlight_overrides": [
            {k: {"definition": f"{float(v):.8g}"}} for k, v in best_with_fixed.items()
        ],
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)

    print(f"\nCalibrated parameters saved to: {out_path}")
    print("\nTo use in a simulation:")
    print(f"    import json, greenlight")
    print(f"    cal = json.load(open('{out_path}'))")
    print(f"    input_prompt = [MODEL, options] + cal['greenlight_overrides'] + [WEATHER]")
    print(f"    greenlight.GreenLight(base_path=BASE, input_prompt=input_prompt, ...).run()")

    _cleanup(weather_path)


def _cleanup(weather_path):
    if os.path.exists(weather_path):
        os.remove(weather_path)
    # Remove temp output dir if empty
    temp_dir = os.path.join(BASE_PATH, TEMP_SUBDIR)
    if os.path.isdir(temp_dir) and not os.listdir(temp_dir):
        os.rmdir(temp_dir)


if __name__ == "__main__":
    main()
