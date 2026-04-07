"""
GreenLight/scripts/prepare_longmont_weather.py

Download a Boulder, CO EPW weather file and convert it directly to GreenLight
format — no EnergyPlus desktop tool needed.

The Boulder Municipal Airport station (WMO 720533, lat 40.03°N, elev 1612 m)
is the closest high-quality TMYx station to Longmont (lat 40.17°N, 1551 m).

Usage:
    python scripts/prepare_longmont_weather.py

Output:
    greenlight/models/katzin_2021/input_data/longmont/
        boulder_tmyx.epw            — raw EPW (cached for reuse)
        longmont_weather_from_jan_01_000000.csv   — GreenLight-format CSV
"""

import datetime as dt
import io
import os
import urllib.request
import zipfile

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

# ── Config ─────────────────────────────────────────────────────────────────────
EPW_ZIP_URL = (
    "https://climate.onebuilding.org/WMO_Region_4_North_and_Central_America"
    "/USA_United_States_of_America/CO_Colorado"
    "/USA_CO_Boulder.Muni.AP.720533_TMYx.zip"
)
EPW_FILENAME = "USA_CO_Boulder.Muni.AP.720533_TMYx.epw"

project_dir  = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
OUTPUT_DIR   = os.path.join(project_dir, "greenlight", "models", "katzin_2021", "input_data", "longmont")
EPW_CACHE    = os.path.join(OUTPUT_DIR, "boulder_tmyx.epw")
OUTPUT_BASE  = os.path.join(OUTPUT_DIR, "longmont_weather.csv")

# Simulation window: full TMYx year
T_START = None   # None → use EPW's own year, Jan 1
T_END   = None   # None → full year (365 days)
CO2_PPM = 420    # current approximate outdoor CO2 (ppm)


# ── EPW download ───────────────────────────────────────────────────────────────
def download_epw(url: str, epw_filename: str, cache_path: str) -> str:
    if os.path.exists(cache_path):
        print(f"EPW already cached: {cache_path}")
        return cache_path
    print(f"Downloading {url} …")
    with urllib.request.urlopen(url) as resp:
        raw = resp.read()
    with zipfile.ZipFile(io.BytesIO(raw)) as zf:
        epw_bytes = zf.read(epw_filename)
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    with open(cache_path, "wb") as f:
        f.write(epw_bytes)
    print(f"Saved EPW to {cache_path}")
    return cache_path


# ── EPW parsing ────────────────────────────────────────────────────────────────
# EPW column indices (0-based) in the hourly data rows
_EPW_COL = {
    "year":   0,
    "month":  1,
    "day":    2,
    "hour":   3,        # 1-24 (EnergyPlus end-of-hour convention)
    "tDry":   6,        # Dry-bulb temperature (°C)
    "rh":     8,        # Relative humidity (%)
    "iGlob":  13,       # Global Horizontal Radiation (Wh/m²)
    "irSky":  12,       # Horizontal Infrared Radiation Intensity (Wh/m²)
    "wind":   21,       # Wind speed (m/s)
}


def parse_epw(epw_path: str):
    """
    Parse an EPW file and return:
        df        — pandas DataFrame with hourly data and columns matching GreenLight input
        elevation — site elevation in m
    """
    with open(epw_path, encoding="utf-8", errors="ignore") as f:
        lines = [line.rstrip("\n") for line in f]

    # ── Row 1: LOCATION → extract elevation ────────────────────────────────────
    loc_fields = lines[0].split(",")
    elevation  = float(loc_fields[9])
    ep_year    = None   # will be read from first data row

    # ── Row 4: GROUND TEMPERATURES → 2 m depth monthly temps ──────────────────
    gt_fields = lines[3].split(",")
    # Format: GROUND TEMPERATURES, n_depths, [depth, cond, dens, spheat, Jan..Dec], ...
    n_depths   = int(gt_fields[1])
    depths_raw = gt_fields[2:]   # everything after the count

    # Each depth block occupies 16 fields: depth + 3 physical props + 12 monthly temps
    ground_temps_2m = None
    idx = 0
    for _ in range(n_depths):
        block_depth = float(depths_raw[idx]) if depths_raw[idx] else 0.0
        monthly     = [float(x) for x in depths_raw[idx + 4 : idx + 16]]
        if block_depth >= 1.5:          # first depth ≥ 1.5 m → use as 2 m proxy
            ground_temps_2m = monthly
            break
        idx += 16
    if ground_temps_2m is None:
        # Fallback: use deepest available depth
        idx = (n_depths - 1) * 16
        ground_temps_2m = [float(x) for x in depths_raw[idx + 4 : idx + 16]]

    # ── Data rows: skip header lines, read 8760 hourly rows ───────────────────
    # EPW header is 8 lines (LOCATION … DATA PERIODS)
    data_start = 8
    rows = []
    for line in lines[data_start : data_start + 8760]:
        if not line:
            continue
        parts = line.split(",")
        rows.append(parts)

    if len(rows) < 8760:
        raise ValueError(f"EPW file has only {len(rows)} data rows (expected 8760)")

    ep_year = int(rows[0][_EPW_COL["year"]])

    # Build a pandas DataFrame from the raw rows
    arr = np.array(rows, dtype=object)
    df  = pd.DataFrame()
    df["year"]  = arr[:, _EPW_COL["year"]].astype(int)
    df["month"] = arr[:, _EPW_COL["month"]].astype(int)
    df["day"]   = arr[:, _EPW_COL["day"]].astype(int)
    df["hour"]  = arr[:, _EPW_COL["hour"]].astype(int)   # 1-24
    df["tDry"]  = arr[:, _EPW_COL["tDry"]].astype(float)
    df["rh"]    = arr[:, _EPW_COL["rh"]].astype(float)
    df["iGlob"] = arr[:, _EPW_COL["iGlob"]].astype(float)
    df["irSky"] = arr[:, _EPW_COL["irSky"]].astype(float)
    df["wind"]  = arr[:, _EPW_COL["wind"]].astype(float)

    # Convert EnergyPlus end-of-hour (hour=1 means 00:00-01:00) to start-of-hour
    # EPW hour 1 = 00:00 → 01:00, so start is 00:00
    # Use ep_year for all rows (EPW is a single representative year)
    datetimes = []
    for _, row in df.iterrows():
        # EPW hour 24 = 23:00 start
        h = int(row["hour"]) - 1   # shift: end-of-hour → start-of-hour
        try:
            datetimes.append(dt.datetime(ep_year, int(row["month"]), int(row["day"]), h, 0))
        except ValueError:
            # Feb 29 in a leap year EPW → skip
            datetimes.append(None)

    df["datetime"] = datetimes
    df = df[df["datetime"].notna()].copy()
    df["datetime"] = pd.to_datetime(df["datetime"])

    return df, elevation, ground_temps_2m, ep_year


# ── Ground temperature: fit sinusoid (same as energy_plus.py) ─────────────────
def _fit_hourly_soil_temp(monthly_temps: list) -> np.ndarray:
    y      = np.array(monthly_temps, dtype=float)
    y_max  = y.max()
    y_min  = y.min()
    y_range = y_max - y_min
    y_shift = y - y_max + y_range / 2
    y_mean  = y.mean()
    x       = np.arange(0.5, 12.5, 1.0)

    zeros_x = x[(y_shift * np.roll(y_shift, 1)) <= 0]
    period  = 2 * np.mean(np.diff(zeros_x)) if len(zeros_x) > 1 else 12.0

    def sinusoid(x, p0, p1, p2):
        return p0 * np.sin(2 * np.pi * x / period + 2 * np.pi / p1) + p2

    popt, _ = curve_fit(sinusoid, x, y, p0=[y_range, -1, y_mean], maxfev=5000)
    x_h = np.linspace(0, 12, 8760)
    return sinusoid(x_h, *popt)


# ── Thermodynamic helpers (same as energy_plus.py) ────────────────────────────
def _sat_vp(temp):
    p = [610.78, 238.3, 17.2694]
    return p[0] * np.exp(p[2] * temp / (temp + p[1]))


def _co2_ppm_to_mgm3(temp, co2_ppm):
    r, m_co2, p_air = 8.314, 44.01e-3, 101325.0
    return p_air * 1e-6 * co2_ppm * m_co2 / (r * (temp + 273.15)) * 1e6


# ── Main conversion ────────────────────────────────────────────────────────────
def convert_epw_to_greenlight(
    epw_path: str,
    output_base: str,
    t_start: dt.datetime = None,
    t_end: dt.datetime = None,
    co2_ppm: float = 420,
) -> str:
    """
    Parse EPW file and write a GreenLight-format weather CSV.
    Returns the path of the file written.
    """
    print(f"Parsing EPW: {epw_path}")
    df, elevation, monthly_soil, ep_year = parse_epw(epw_path)

    if t_start is None:
        t_start = dt.datetime(ep_year, 1, 1, 0, 0)
    if t_end is None:
        t_end = t_start + dt.timedelta(days=365, hours=-1)

    # Replicate year if t_end spans more than one year
    n_years = t_end.year - t_start.year + 1
    if n_years > 1:
        frames = []
        for yr_offset in range(n_years):
            tmp = df.copy()
            tmp["datetime"] = tmp["datetime"] + pd.DateOffset(years=yr_offset)
            frames.append(tmp)
        df = pd.concat(frames, ignore_index=True)

    # Trim to [t_start, t_end]
    df = df[(df["datetime"] >= t_start) & (df["datetime"] <= t_end)].copy()
    df = df.reset_index(drop=True)

    if len(df) < 24:
        raise ValueError(f"Only {len(df)} rows after trimming — check t_start/t_end")

    # Fit hourly soil temperature sinusoid
    soil_hourly = _fit_hourly_soil_temp(monthly_soil)
    # soil_hourly has 8760 values; tile/trim to match df length
    n_rows = len(df)
    soil_rep = np.tile(soil_hourly, (n_rows // 8760) + 1)[:n_rows]

    # Build output DataFrame
    out = pd.DataFrame()

    # Time in seconds since start
    out["Time"] = (df["datetime"].values.astype("int64") // 1_000_000_000
                   - df["datetime"].values[0].astype("int64") // 1_000_000_000)

    out["iGlob"]     = df["iGlob"].values
    out["tOut"]      = df["tDry"].values
    out["vpOut"]     = df["rh"].values / 100.0 * _sat_vp(df["tDry"].values)
    out["co2Out"]    = _co2_ppm_to_mgm3(df["tDry"].values, co2_ppm)
    out["wind"]      = df["wind"].values

    sigma = 5.6697e-8
    out["tSky"]      = (df["irSky"].values / sigma) ** 0.25 - 273.15

    out["tSoOut"]    = soil_rep
    out["dayRadSum"] = (
        (3600e-6 * df["iGlob"].values).reshape(-1, 24).sum(axis=1).repeat(24)[:n_rows]
        if n_rows % 24 == 0
        else np.nan   # partial day — fallback
    )
    out["isDay"]      = (df["iGlob"].values > 0).astype(float)
    out["isDaySmooth"] = out["isDay"]
    out["hElevation"] = elevation

    # Header rows (matching GreenLight format exactly)
    col_names = list(out.columns)
    units = ["s", "W m**-2", "°C", "Pa", "mg m**-3", "m s**-1", "°C", "°C",
             "MJ m**-2", "-", "-", "m above sea level"]
    descs = [
        "Time since start of data", "Outdoor global solar radiation",
        "Outdoor temperature", "Outdoor vapor pressure",
        "Outdoor CO2 concentration", "Outdoor wind speed",
        "Apparent sky temperature", "Soil temperature at 2 m depth",
        "Daily sum of outdoor global solar radiation",
        "Switch determining if it is day or night. Used for control purposes",
        "Smooth switch determining if it is day or night. Used for control purposes",
        "Elevation at location",
    ]

    # Build output filename with start-date suffix (same convention as convert_energy_plus)
    base, ext = os.path.splitext(output_base)
    outpath = f"{base}_from_{t_start.strftime('%b_%d_%H%M%S').lower()}{ext}"

    os.makedirs(os.path.dirname(outpath), exist_ok=True)
    vars_str = ",".join(col_names)
    desc_str = ",".join(descs)
    unit_str = ",".join(units)

    with open(outpath, "w", encoding="utf-8") as f:
        f.write(vars_str + "\n")
        f.write(desc_str + "\n")
        f.write(unit_str + "\n")
    out.to_csv(outpath, mode="a", header=False, index=False, encoding="utf-8")

    print(f"Saved GreenLight weather CSV: {outpath}")
    print(f"  Rows: {len(out)} ({len(out)/24:.0f} days)  |  Elevation: {elevation} m  |  CO₂: {co2_ppm} ppm")
    print(f"  tOut: mean={out['tOut'].mean():.1f} °C, range [{out['tOut'].min():.1f}, {out['tOut'].max():.1f}] °C")
    print(f"  vpOut: mean={out['vpOut'].mean():.0f} Pa   iGlob: mean={out['iGlob'].mean():.0f} W/m²")
    return outpath


# ── Entry point ────────────────────────────────────────────────────────────────
def main():
    epw_path = download_epw(EPW_ZIP_URL, EPW_FILENAME, EPW_CACHE)
    weather_csv = convert_epw_to_greenlight(
        epw_path=epw_path,
        output_base=OUTPUT_BASE,
        t_start=T_START,
        t_end=T_END,
        co2_ppm=CO2_PPM,
    )
    print(f"\nWeather file ready. Use in simulations:\n  {weather_csv}")
    print("\nExample Python usage:")
    print("  import greenlight, os")
    print("  BASE = 'greenlight/models'")
    print("  input_prompt = [")
    print("      'katzin_2021/definition/main_longmont.json',")
    print(f"      '{os.path.relpath(weather_csv, os.path.join(project_dir, 'greenlight', 'models'))}',")
    print("  ]")
    print("  greenlight.GreenLight(base_path=BASE, input_prompt=input_prompt, output_path='out.csv').run()")
    return weather_csv


if __name__ == "__main__":
    main()
