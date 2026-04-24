"""
scripts/make_graphs.py

Generate all GreenByte figures and save them to figures/.
Also moves any existing PNGs from output/ into figures/.

Sections:
  01  Pipeline overview
  02  Real greenhouse data windows (4 windows, climate + equipment)
  03  Calibration summary (MAE / bias / heater runtime)
  04  Envelope sensitivity analysis (OAT)
  05  LHS training-data coverage
  06  Output distributions by season
  07  Cost decomposition (heat / light / water)
  08  Input → output correlation heatmap
  09  Model comparison leaderboard
  10  Predicted vs actual (3 targets)
  11  Residuals analysis
  12  Feature importance
  13  Setpoint sweep (tSpDay across months)
  14  Lamp intensity tradeoffs
  15  Fan dead-zone sensitivity
  16  Pareto frontiers (temp-cost + RH-cost, combined)
  17  Monthly optimal settings
  18  Speed benchmark (XGBoost vs GreenLight ODE)
  19  Zone temperature divergence (real data)
  20  3-D scatter: inputs vs cost

Usage:
    python scripts/make_graphs.py
"""

import json
import os
import pickle
import shutil
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ── Paths ─────────────────────────────────────────────────────────────────────
project_dir = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
DATA_CSV    = os.path.join(project_dir, "data", "training_data.csv")
MODELS_DIR  = os.path.join(project_dir, "models")
OUTPUT_DIR  = os.path.join(project_dir, "output")
FIGURES_DIR = os.path.join(project_dir, "figures")
CSV_DIR     = os.path.join(project_dir, "james-csv-files-2026-04-13")
CAL_DIR     = os.path.join(project_dir, "calibration")

os.makedirs(FIGURES_DIR, exist_ok=True)

# ── Global style ──────────────────────────────────────────────────────────────
plt.rcParams.update({
    "figure.dpi":       150,
    "font.family":      "sans-serif",
    "axes.spines.top":  False,
    "axes.spines.right":False,
    "axes.grid":        True,
    "grid.alpha":       0.3,
    "figure.facecolor": "white",
})

PALETTE = {
    "heat":   "#EF5350",
    "cool":   "#42A5F5",
    "rh":     "#7E57C2",
    "cost":   "#66BB6A",
    "solar":  "#FFA726",
    "accent": "#26C6DA",
    "gray":   "#78909C",
    "dark":   "#37474F",
}

SEASON_COLORS = {
    "jan_cold":   "#42A5F5",
    "spring_apr": "#66BB6A",
    "aug_summer": "#FFA726",
    "oct_shoulder":"#AB47BC",
}

def savefig(fig, name):
    path = os.path.join(FIGURES_DIR, name)
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  → figures/{name}")


def c_to_f(c): return c * 9/5 + 32


# ── Load shared data ──────────────────────────────────────────────────────────

def load_training_data():
    df = pd.read_csv(DATA_CSV)
    df["month_sin"] = np.sin(2 * np.pi * df["start_month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["start_month"] / 12)
    return df


def load_model():
    pkl = os.path.join(MODELS_DIR, "nn_surrogate.pkl")
    with open(pkl, "rb") as f:
        return pickle.load(f)


def load_meta():
    with open(os.path.join(MODELS_DIR, "nn_surrogate_meta.json")) as f:
        return json.load(f)


def load_window_csv(name):
    path = os.path.join(CSV_DIR, name)
    df = pd.read_csv(path, parse_dates=["ts"])
    df["ts"] = pd.to_datetime(df["ts"], utc=True).dt.tz_convert("America/Denver")
    for col in ["heat1","heat2","fan1","fan2","vent"]:
        if col in df.columns:
            df[col] = df[col].map({"t": 1, "f": 0, True: 1, False: 0}).fillna(0)
    return df


# ═══════════════════════════════════════════════════════════════════════════════
# 01  Pipeline overview
# ═══════════════════════════════════════════════════════════════════════════════

def fig_01_pipeline():
    fig, ax = plt.subplots(figsize=(13, 5))
    ax.set_xlim(0, 13); ax.set_ylim(0, 5)
    ax.axis("off")
    ax.set_facecolor("white")

    boxes = [
        (0.6,  2.5, "Real Greenhouse\nLongmont, CO",        "#BBDEFB", "Polycarbonate\n34 m² slab\nElec + gas heat"),
        (2.9,  2.5, "Calibration\n(Verdify data)",          "#C8E6C9", "Nelder-Mead vs\n4 seasonal windows\nMAE 3.4–6.0°F"),
        (5.2,  2.5, "GreenLight\nODE Simulator",            "#FFE0B2", "Physics-based\nLongmont model\n~18 s / run"),
        (7.5,  2.5, "500-sample\nLHS Sweep",                "#E1BEE7", "All seasons\n6 control params\n30-day runs"),
        (9.8,  2.5, "XGBoost\nSurrogate",                   "#B2EBF2", "Temp · RH · Cost\nCV R²=0.90–0.95\n~1.6 µs / pred"),
        (12.1, 2.5, "Iris Planner\n(future)",               "#FCE4EC", "Plan scoring\nSetpoint opt.\nPareto fronts"),
    ]

    box_w, box_h = 1.85, 1.7
    for (x, y, title, color, sub) in boxes:
        rect = mpatches.FancyBboxPatch(
            (x - box_w/2, y - box_h/2), box_w, box_h,
            boxstyle="round,pad=0.1", linewidth=1.5,
            edgecolor="#90A4AE", facecolor=color, zorder=2)
        ax.add_patch(rect)
        ax.text(x, y + 0.35, title, ha="center", va="center",
                fontsize=8.5, fontweight="bold", zorder=3)
        ax.text(x, y - 0.35, sub, ha="center", va="center",
                fontsize=6.5, color="#546E7A", zorder=3)

    # Arrows
    for i in range(len(boxes) - 1):
        x1 = boxes[i][0]   + box_w/2
        x2 = boxes[i+1][0] - box_w/2
        ax.annotate("", xy=(x2, 2.5), xytext=(x1, 2.5),
                    arrowprops=dict(arrowstyle="->", color="#546E7A", lw=1.8))

    ax.text(6.5, 4.5, "GreenByte Pipeline", ha="center", fontsize=14,
            fontweight="bold", color=PALETTE["dark"])
    ax.text(6.5, 0.3, "Simulation-to-ML: bypass months of manual data collection",
            ha="center", fontsize=9, color=PALETTE["gray"], style="italic")

    savefig(fig, "01_pipeline_overview.png")


# ═══════════════════════════════════════════════════════════════════════════════
# 02  Real greenhouse data — all 4 windows
# ═══════════════════════════════════════════════════════════════════════════════

def fig_02_greenhouse_windows():
    windows = [
        ("jan_cold_week.csv",  "Jan 13–20, 2026",  "jan_cold",    False),
        ("spring_apr_2026.csv","Apr 6–13, 2026",   "spring_apr",  False),
        ("oct_shoulder.csv",   "Oct 6–13, 2025",   "oct_shoulder",False),
        ("aug_summer.csv",     "Aug 6–13, 2025",   "aug_summer",  True),
    ]

    fig = plt.figure(figsize=(18, 14))
    fig.suptitle("Real Greenhouse Data — Four Calibration Windows", fontsize=14, fontweight="bold", y=0.98)
    outer = gridspec.GridSpec(2, 2, figure=fig, hspace=0.45, wspace=0.35)

    for idx, (csv, title, key, excluded) in enumerate(windows):
        df = load_window_csv(csv)
        df = df.resample("15min", on="ts").mean(numeric_only=True).reset_index()
        color = SEASON_COLORS[key]
        inner = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=outer[idx], hspace=0.08)

        # Temperature panel
        ax_t = fig.add_subplot(inner[0])
        ax_t.plot(df["ts"], df["ws_temp_f"],   color=PALETTE["cool"], lw=1.2,  label="Outdoor", alpha=0.8)
        ax_t.plot(df["ts"], df["temp_avg"],    color=PALETTE["heat"], lw=1.8,  label="Indoor avg")
        for zone_col, zone_lbl, za in [("temp_north","N","#81C784"),("temp_east","E","#64B5F6"),
                                        ("temp_west","W","#FFB74D"),("temp_south","S","#F06292")]:
            if zone_col in df.columns:
                ax_t.plot(df["ts"], df[zone_col], lw=0.7, alpha=0.5, color=za, label=zone_lbl)
        ax_t.set_ylabel("Temp (°F)")
        ax_t.legend(loc="upper right", fontsize=6.5, ncol=3)
        excl_note = "  ⚠ excluded — ventilation model failure" if excluded else ""
        ax_t.set_title(f"{title}{excl_note}", fontsize=9, fontweight="bold", color=color)
        ax_t.tick_params(labelbottom=False)

        # Equipment panel
        ax_e = fig.add_subplot(inner[1], sharex=ax_t)
        t = df["ts"]
        offset = 0
        for col, lbl, ec in [("heat1","Heat 1","#EF5350"),("heat2","Heat 2","#FF8A65"),
                               ("fan1","Fan 1","#42A5F5"),("fan2","Fan 2","#26C6DA")]:
            if col in df.columns:
                ax_e.fill_between(t, offset, offset + df[col]*0.8, step="post",
                                  alpha=0.7, color=ec, label=lbl)
                offset += 1
        ax_e.set_ylabel("Equipment")
        ax_e.set_yticks([])
        ax_e.legend(loc="upper right", fontsize=6.5, ncol=2)
        ax_e.xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%m/%d"))

    savefig(fig, "02_greenhouse_windows.png")


# ═══════════════════════════════════════════════════════════════════════════════
# 03  Calibration summary — MAE / bias / heater runtime
# ═══════════════════════════════════════════════════════════════════════════════

def fig_03_calibration_summary():
    windows = ["Jan cold", "Spring Apr", "Oct shoulder", "Aug summer"]
    mae     = [3.86,  3.36,  5.97, 13.70]
    bias    = [0.80,  2.28,  5.97, -4.78]
    heat_r  = [67.3,  44.6,  37.5,  27.8]
    heat_s  = [69.6,  32.7,  42.3, 100.0]
    fan_r   = [ 0.0,  18.5,   3.0,  27.8]
    fan_s   = [ 0.0,  14.9,   3.6, 100.0]
    colors  = [SEASON_COLORS[k] for k in ["jan_cold","spring_apr","oct_shoulder","aug_summer"]]

    fig, axes = plt.subplots(3, 1, figsize=(8, 13))
    fig.suptitle("Calibration Results Across Windows", fontsize=13, fontweight="bold")

    x = np.arange(len(windows)); w = 0.35

    # MAE
    bars = axes[0].bar(x, mae, color=colors, edgecolor="white", linewidth=1.2)
    axes[0].axhline(3.0, color="green", ls="--", lw=1.2, label="< 3°F excellent")
    axes[0].axhline(6.0, color="orange", ls="--", lw=1.2, label="> 6°F structural gap")
    axes[0].bar(x[-1], mae[-1], color=colors[-1], edgecolor="#B71C1C", linewidth=2,
                hatch="//", label="Excluded (summer)")
    for bar, v in zip(bars, mae):
        axes[0].text(bar.get_x()+bar.get_width()/2, v+0.2, f"{v:.2f}°F",
                     ha="center", fontsize=8)
    axes[0].set_title("Temperature MAE (°F)")
    axes[0].set_xticks(x); axes[0].set_xticklabels(windows)
    axes[0].legend(fontsize=7.5)
    axes[0].set_ylabel("MAE (°F)")

    # Bias
    axes[1].bar(x, bias, color=[PALETTE["heat"] if b > 0 else PALETTE["cool"] for b in bias],
                edgecolor="white", linewidth=1.2)
    axes[1].axhline(0, color="black", lw=0.8)
    for i, v in enumerate(bias):
        axes[1].text(i, v + (0.2 if v >= 0 else -0.5), f"{v:+.2f}°F",
                     ha="center", fontsize=8)
    axes[1].set_title("Systematic Bias (°F)  [sim − real]")
    axes[1].set_xticks(x); axes[1].set_xticklabels(windows)
    axes[1].set_ylabel("Bias (°F)")

    # Heater / Fan runtime
    axes[2].bar(x - w/2, heat_r, w, color=PALETTE["heat"], label="Heater real %", alpha=0.85)
    axes[2].bar(x + w/2, heat_s, w, color=PALETTE["heat"], label="Heater sim %",
                alpha=0.4, hatch="//")
    axes[2].bar(x - w/2, fan_r, w, color=PALETTE["cool"], label="Fan real %",
                alpha=0.85, bottom=0, linewidth=0)
    axes[2].bar(x + w/2, fan_s, w, color=PALETTE["cool"], label="Fan sim %",
                alpha=0.4, hatch="//", bottom=0)
    axes[2].set_title("Heater & Fan Runtime  (solid=real, hatch=sim)")
    axes[2].set_xticks(x); axes[2].set_xticklabels(windows)
    axes[2].set_ylabel("Runtime (%)")
    axes[2].legend(fontsize=7)

    fig.tight_layout()
    savefig(fig, "03_calibration_summary.png")


# ═══════════════════════════════════════════════════════════════════════════════
# 04  Envelope sensitivity analysis (OAT)
# ═══════════════════════════════════════════════════════════════════════════════

def fig_04_sensitivity():
    with open(os.path.join(CAL_DIR, "sensitivity_report.json")) as f:
        sens = json.load(f)

    params  = list(sens["sensitivity"].keys())   # cLeakage, aCov, lambdaRf
    seasons = sens["seasons"]                     # Jan_winter, Apr_spring, Jul_summer

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Envelope Parameter Sensitivity (OAT)\n"
                 "Key finding: operational params dominate; envelope params are near-flat",
                 fontsize=11, fontweight="bold")

    x = np.arange(len(params)); w = 0.25
    s_colors = [PALETTE["cool"], PALETTE["cost"], PALETTE["solar"]]
    param_labels = ["cLeakage\n(infiltration)", "aCov\n(cover area)", "λRf\n(conductivity)"]

    # Temperature swing
    for i, (season, col) in enumerate(zip(seasons, s_colors)):
        swings = [abs(sens["sensitivity"][p][season]["tAir_swing_f"]) for p in params]
        axes[0].bar(x + i*w - w, swings, w, color=col, label=season.replace("_"," "), alpha=0.85)
    axes[0].set_title("Temperature Swing (°F) across full param range")
    axes[0].set_xticks(x); axes[0].set_xticklabels(param_labels)
    axes[0].set_ylabel("|ΔT| (°F)")
    axes[0].legend(fontsize=8)
    axes[0].axhline(3.86, color="red", ls="--", lw=1, label="Jan MAE = 3.86°F")

    # Heater runtime swing
    for i, (season, col) in enumerate(zip(seasons, s_colors)):
        swings = [abs(sens["sensitivity"][p][season]["heat_swing_pct"]) for p in params]
        axes[1].bar(x + i*w - w, swings, w, color=col, label=season.replace("_"," "), alpha=0.85)
    axes[1].set_title("Heater Runtime Swing (%) across full param range")
    axes[1].set_xticks(x); axes[1].set_xticklabels(param_labels)
    axes[1].set_ylabel("Δruntime (%)")
    axes[1].legend(fontsize=8)

    fig.tight_layout()
    savefig(fig, "04_sensitivity_analysis.png")


# ═══════════════════════════════════════════════════════════════════════════════
# 05  LHS training data coverage
# ═══════════════════════════════════════════════════════════════════════════════

def fig_05_lhs_coverage(df):
    params = ["in_tSpDay","in_tSpNight","in_thetaLampMax","in_heatDeadZone","in_rhMax"]
    labels = ["tSpDay (°C)","tSpNight (°C)","Lamp (W/m²)","HeatDeadZone (°C)","rhMax (%)"]
    n = len(params)

    fig, axes = plt.subplots(n, n, figsize=(13, 13))
    fig.suptitle(f"LHS Input Coverage — {len(df)} Samples × 5 Parameters",
                 fontsize=12, fontweight="bold")

    month_colors = plt.cm.hsv(df["start_month"] / 12)

    for i in range(n):
        for j in range(n):
            ax = axes[i][j]
            if i == j:
                ax.hist(df[params[i]], bins=20, color=PALETTE["accent"], alpha=0.7, edgecolor="white")
                ax.set_ylabel("Count" if j == 0 else "")
            elif i > j:
                ax.scatter(df[params[j]], df[params[i]], c=month_colors,
                           s=5, alpha=0.5, rasterized=True)
            else:
                ax.axis("off")
            if i == n-1: ax.set_xlabel(labels[j], fontsize=7)
            if j == 0:   ax.set_ylabel(labels[i], fontsize=7)
            ax.tick_params(labelsize=6)

    # Month legend
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize
    sm = ScalarMappable(cmap="hsv", norm=Normalize(1, 12))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes, orientation="vertical", fraction=0.015, pad=0.04)
    cbar.set_label("Start month", fontsize=9)
    cbar.set_ticks(range(1, 13))
    cbar.set_ticklabels(["J","F","M","A","M","J","J","A","S","O","N","D"])

    savefig(fig, "05_lhs_coverage.png")


# ═══════════════════════════════════════════════════════════════════════════════
# 06  Output distributions by season
# ═══════════════════════════════════════════════════════════════════════════════

def fig_06_output_distributions(df):
    season_map = {1:"Winter",2:"Winter",3:"Spring",4:"Spring",5:"Spring",
                  6:"Summer",7:"Summer",8:"Summer",9:"Fall",10:"Fall",11:"Fall",12:"Winter"}
    df = df.copy()
    df["season"] = df["start_month"].map(season_map)
    seasons     = ["Winter","Spring","Summer","Fall"]
    s_colors    = [PALETTE["cool"],PALETTE["cost"],PALETTE["solar"],PALETTE["rh"]]

    targets = [
        ("out_mean_tAir_C", "Mean air temp", "°C",  lambda v: v),
        ("out_mean_rh_pct", "Mean RH",       "%",   lambda v: v),
        ("out_cost_total_usd_m2","Monthly cost","$/m²", lambda v: v),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("Output Distributions by Season  (500 LHS simulations)", fontsize=12, fontweight="bold")

    for ax, (col, label, unit, _) in zip(axes, targets):
        data  = [df[df["season"]==s][col].dropna().values for s in seasons]
        bp = ax.boxplot(data, patch_artist=True, widths=0.6, medianprops={"color":"white","lw":2})
        for patch, color in zip(bp["boxes"], s_colors):
            patch.set_facecolor(color); patch.set_alpha(0.8)
        ax.set_xticklabels(seasons)
        ax.set_ylabel(f"{label} ({unit})")
        ax.set_title(label)
        # Convert temp axis to F on right
        if col == "out_mean_tAir_C":
            ax2 = ax.twinx()
            ax2.set_ylim(c_to_f(ax.get_ylim()[0]), c_to_f(ax.get_ylim()[1]))
            ax2.set_ylabel("°F", fontsize=8)

    fig.tight_layout()
    savefig(fig, "06_output_distributions.png")


# ═══════════════════════════════════════════════════════════════════════════════
# 07  Cost decomposition by month
# ═══════════════════════════════════════════════════════════════════════════════

def fig_07_cost_decomposition(df):
    monthly = df.groupby("start_month")[
        ["out_cost_heat_usd_m2","out_cost_light_usd_m2","out_cost_water_usd_m2"]
    ].median()

    months = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
    x = np.arange(1, 13)

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(x, monthly["out_cost_heat_usd_m2"],  color=PALETTE["heat"],  label="Heating (electric)")
    ax.bar(x, monthly["out_cost_light_usd_m2"], color=PALETTE["solar"], label="Lighting (LED)",
           bottom=monthly["out_cost_heat_usd_m2"])
    ax.bar(x, monthly["out_cost_water_usd_m2"], color=PALETTE["cool"],  label="Water (mist/fog)",
           bottom=monthly["out_cost_heat_usd_m2"]+monthly["out_cost_light_usd_m2"])

    ax.set_xticks(x); ax.set_xticklabels(months)
    ax.set_ylabel("Median cost ($/m²/month)")
    ax.set_title("Monthly Operating Cost Decomposition — Median over 500 LHS Simulations\n"
                 "Longmont, CO  ·  Electric $0.111/kWh  ·  Water $0.00484/gal",
                 fontsize=10)
    ax.legend(fontsize=9)

    # Annotate summer
    ax.annotate("Summer: heater off,\nlights off → near-zero cost",
                xy=(7, 0.015), xytext=(8.5, 0.25),
                arrowprops=dict(arrowstyle="->", color="gray"), fontsize=8)

    fig.tight_layout()
    savefig(fig, "07_cost_decomposition.png")


# ═══════════════════════════════════════════════════════════════════════════════
# 08  Input → output correlation heatmap
# ═══════════════════════════════════════════════════════════════════════════════

def fig_08_correlations(df):
    inputs  = ["in_tSpDay","in_tSpNight","in_thetaLampMax","in_heatDeadZone","in_rhMax","start_month"]
    outputs = ["out_mean_tAir_C","out_mean_rh_pct","out_cost_total_usd_m2",
               "out_energy_heat_MJ_m2","out_energy_light_MJ_m2","out_final_cFruit"]
    in_labels  = ["tSpDay","tSpNight","Lamp","HeatDead","rhMax","Month"]
    out_labels = ["Temp (°C)","RH (%)","Cost ($/m²)","Heat energy","Light energy","cFruit"]

    corr = df[inputs + outputs].corr().loc[inputs, outputs]

    fig, ax = plt.subplots(figsize=(10, 5))
    cmap = LinearSegmentedColormap.from_list("rdb", ["#1565C0","#FFFFFF","#C62828"])
    im = ax.imshow(corr.values, cmap=cmap, vmin=-1, vmax=1, aspect="auto")
    plt.colorbar(im, ax=ax, label="Pearson r")

    ax.set_xticks(range(len(out_labels))); ax.set_xticklabels(out_labels, rotation=30, ha="right")
    ax.set_yticks(range(len(in_labels)));  ax.set_yticklabels(in_labels)

    for i in range(len(in_labels)):
        for j in range(len(out_labels)):
            v = corr.values[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                    fontsize=8, color="white" if abs(v) > 0.5 else "black")

    ax.set_title("Input → Output Pearson Correlation (500 LHS samples)", fontsize=11, fontweight="bold")
    fig.tight_layout()
    savefig(fig, "08_correlations.png")


# ═══════════════════════════════════════════════════════════════════════════════
# 09  Model comparison
# ═══════════════════════════════════════════════════════════════════════════════

def fig_09_model_comparison():
    path = os.path.join(OUTPUT_DIR, "model_comparison.json")
    if not os.path.exists(path):
        print("  (skip 09 — model_comparison.json not found, run compare_models.py first)")
        return

    with open(path) as f:
        data = json.load(f)
    lb = pd.DataFrame(data["leaderboard"]).sort_values("avg_cv_r2", ascending=False)

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle("ML Model Comparison — 5-fold CV R²  (GreenByte surrogate candidates)",
                 fontsize=12, fontweight="bold")

    models = lb["model"].tolist()
    colors = ["#EF5350" if m == "MLP_128x64" else
              "#2E7D32" if m in ("XGBoost","GradientBoosting","LightGBM") else
              "#1565C0" for m in models]
    y = np.arange(len(models))

    # R² grouped bars
    w = 0.25
    axes[0].barh(y + w,   lb["temp_cv_r2"],  w, color=PALETTE["heat"],  label="Temp (°C)",   alpha=0.85)
    axes[0].barh(y,       lb["rh_cv_r2"],    w, color=PALETTE["rh"],    label="RH (%)",      alpha=0.85)
    axes[0].barh(y - w,   lb["cost_cv_r2"],  w, color=PALETTE["cost"],  label="Cost ($/m²)", alpha=0.85)
    axes[0].axvline(0.9, color="gray", ls=":", lw=1.5, label="R²=0.90")
    axes[0].set_yticks(y); axes[0].set_yticklabels(models, fontsize=8)
    axes[0].set_xlabel("5-fold CV R²")
    axes[0].set_title("Accuracy per target")
    axes[0].legend(fontsize=8)
    axes[0].invert_yaxis()

    # Speed (log scale)
    axes[1].barh(y, lb["infer_us"], color=colors, alpha=0.85, edgecolor="white")
    axes[1].set_xscale("log")
    axes[1].set_yticks(y); axes[1].set_yticklabels(models, fontsize=8)
    axes[1].set_xlabel("Inference time (µs / prediction)  — log scale")
    axes[1].set_title("Speed (batch of 1000)")
    axes[1].invert_yaxis()
    for i, (_, row) in enumerate(lb.iterrows()):
        axes[1].text(row["infer_us"] * 1.1, i, f"{row['infer_us']:.1f}µs",
                     va="center", fontsize=7)

    red_patch  = mpatches.Patch(color="#EF5350", label="Previous surrogate (MLP)")
    green_patch= mpatches.Patch(color="#2E7D32", label="Gradient boosting")
    blue_patch = mpatches.Patch(color="#1565C0", label="Other")
    axes[1].legend(handles=[red_patch, green_patch, blue_patch], fontsize=7.5, loc="lower right")

    fig.tight_layout()
    savefig(fig, "09_model_comparison.png")


# ═══════════════════════════════════════════════════════════════════════════════
# 10  Predicted vs actual (XGBoost surrogate)
# ═══════════════════════════════════════════════════════════════════════════════

def fig_10_predicted_vs_actual(df, model):
    features = ["in_tSpDay","in_tSpNight","in_thetaLampMax","in_heatDeadZone","in_rhMax","start_month"]
    targets  = ["out_mean_tAir_C","out_mean_rh_pct","out_cost_total_usd_m2"]
    labels   = ["Mean air temp (°C)", "Mean RH (%)", "Total cost ($/m²)"]
    units    = ["°C", "%", "$/m²"]

    X = df[features].copy()
    X["month_sin"] = np.sin(2 * np.pi * X["start_month"] / 12)
    X["month_cos"] = np.cos(2 * np.pi * X["start_month"] / 12)
    X = X.drop(columns=["start_month"])
    Y_pred = model.predict(X)

    month_colors = plt.cm.hsv(df["start_month"] / 12)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("XGBoost Surrogate — Predicted vs Actual  (500 training samples)",
                 fontsize=12, fontweight="bold")

    for ax, col, label, unit, i in zip(axes, targets, labels, units, range(3)):
        y_true = df[col].values
        y_pred = Y_pred[:, i]
        ax.scatter(y_true, y_pred, c=month_colors, s=12, alpha=0.6, rasterized=True)
        lo = min(y_true.min(), y_pred.min())
        hi = max(y_true.max(), y_pred.max())
        ax.plot([lo, hi], [lo, hi], "k--", lw=1, label="Perfect fit")
        from sklearn.metrics import r2_score, mean_absolute_error
        r2  = r2_score(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred)
        ax.set_xlabel(f"Actual ({unit})"); ax.set_ylabel(f"Predicted ({unit})")
        ax.set_title(label)
        ax.text(0.05, 0.93, f"R²={r2:.3f}\nMAE={mae:.3f}{unit}",
                transform=ax.transAxes, fontsize=8.5,
                bbox=dict(facecolor="white", edgecolor="gray", alpha=0.8))

    fig.tight_layout()
    savefig(fig, "10_predicted_vs_actual.png")


# ═══════════════════════════════════════════════════════════════════════════════
# 11  Residuals analysis
# ═══════════════════════════════════════════════════════════════════════════════

def fig_11_residuals(df, model):
    features = ["in_tSpDay","in_tSpNight","in_thetaLampMax","in_heatDeadZone","in_rhMax","start_month"]
    targets  = ["out_mean_tAir_C","out_mean_rh_pct","out_cost_total_usd_m2"]
    labels   = ["Mean temp (°C)", "Mean RH (%)", "Cost ($/m²)"]

    X = df[features].copy()
    X["month_sin"] = np.sin(2 * np.pi * X["start_month"] / 12)
    X["month_cos"] = np.cos(2 * np.pi * X["start_month"] / 12)
    X = X.drop(columns=["start_month"])
    Y_pred = model.predict(X)

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    fig.suptitle("Residuals Analysis — XGBoost Surrogate", fontsize=12, fontweight="bold")

    for col_idx, (col, label) in enumerate(zip(targets, labels)):
        y_true = df[col].values
        resid  = Y_pred[:, col_idx] - y_true

        # Residuals vs predicted
        ax_top = axes[0][col_idx]
        ax_top.scatter(Y_pred[:, col_idx], resid, s=8, alpha=0.5,
                       c=df["start_month"], cmap="hsv", vmin=1, vmax=12)
        ax_top.axhline(0, color="black", lw=1)
        ax_top.set_xlabel(f"Predicted {label}"); ax_top.set_ylabel("Residual")
        ax_top.set_title(f"{label} — residuals vs predicted")

        # Residual histogram
        ax_bot = axes[1][col_idx]
        ax_bot.hist(resid, bins=30, color=PALETTE["accent"], edgecolor="white", alpha=0.8)
        ax_bot.axvline(0, color="black", lw=1)
        ax_bot.axvline(resid.mean(), color="red", lw=1.5, ls="--", label=f"mean={resid.mean():.3f}")
        ax_bot.axvline(resid.std(),  color="orange", lw=1.2, ls=":", label=f"σ={resid.std():.3f}")
        ax_bot.axvline(-resid.std(), color="orange", lw=1.2, ls=":")
        ax_bot.set_xlabel("Residual"); ax_bot.set_ylabel("Count")
        ax_bot.set_title(f"{label} — residual distribution")
        ax_bot.legend(fontsize=7.5)

    fig.tight_layout()
    savefig(fig, "11_residuals.png")


# ═══════════════════════════════════════════════════════════════════════════════
# 12  Feature importance
# ═══════════════════════════════════════════════════════════════════════════════

def fig_12_feature_importance(meta):
    imp = meta["feature_importances"]
    labels_map = {
        "in_tSpDay":       "Day setpoint (tSpDay)",
        "in_tSpNight":     "Night setpoint (tSpNight)",
        "in_thetaLampMax": "Max lamp power (θLamp)",
        "in_heatDeadZone": "Fan dead zone (ΔTdead)",
        "in_rhMax":        "RH cap (rhMax)",
        "month_sin":       "Month · sin (seasonal)",
        "month_cos":       "Month · cos (seasonal)",
    }
    keys   = list(imp.keys())
    vals   = [imp[k] for k in keys]
    labels = [labels_map.get(k, k) for k in keys]

    # Sort by importance
    order  = np.argsort(vals)
    vals   = [vals[i] for i in order]
    labels = [labels[i] for i in order]
    colors = [PALETTE["solar"] if "seasonal" in l else PALETTE["accent"] for l in labels]

    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.barh(range(len(vals)), vals, color=colors, edgecolor="white", height=0.65)
    ax.set_yticks(range(len(labels))); ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel("Mean feature importance (avg across 3 XGBoost targets)")
    ax.set_title("Feature Importance — XGBoost Surrogate\n"
                 "Seasonal encoding (month cos/sin) dominates — thermal forcing drives ~85% of variance",
                 fontsize=10, fontweight="bold")
    for bar, v in zip(bars, vals):
        ax.text(v + 0.003, bar.get_y() + bar.get_height()/2, f"{v:.3f}",
                va="center", fontsize=8)

    orange_patch = mpatches.Patch(color=PALETTE["solar"], label="Seasonal (month encoding)")
    blue_patch   = mpatches.Patch(color=PALETTE["accent"], label="Control parameter")
    ax.legend(handles=[orange_patch, blue_patch], fontsize=8.5)
    fig.tight_layout()
    savefig(fig, "12_feature_importance.png")


# ═══════════════════════════════════════════════════════════════════════════════
# 13  Setpoint sweep — tSpDay effect across months
# ═══════════════════════════════════════════════════════════════════════════════

def fig_13_setpoint_sweep(model):
    months = [1, 4, 7, 10]
    month_names = {1:"January", 4:"April", 7:"July", 10:"October"}
    m_colors    = {1: PALETTE["cool"], 4: PALETTE["cost"], 7: PALETTE["solar"], 10: PALETTE["rh"]}
    tSpDay_vals = np.linspace(10, 22, 60)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("Effect of Day Setpoint (tSpDay) Across Seasons\n"
                 "Fixed: tSpNight=17°C, lamp=0, heatDeadZone=5, rhMax=80%",
                 fontsize=11, fontweight="bold")

    targets_plot = ["out_mean_tAir_C","out_mean_rh_pct","out_cost_total_usd_m2"]
    ylabels = ["Mean air temp (°C)", "Mean RH (%)", "Monthly cost ($/m²)"]

    for m in months:
        rows = []
        for t in tSpDay_vals:
            rows.append({
                "in_tSpDay":       t,
                "in_tSpNight":     min(17.0, t),
                "in_thetaLampMax": 0.0,
                "in_heatDeadZone": 5.0,
                "in_rhMax":        80.0,
                "month_sin":       np.sin(2 * np.pi * m / 12),
                "month_cos":       np.cos(2 * np.pi * m / 12),
            })
        X = pd.DataFrame(rows)
        preds = model.predict(X)

        for ax_idx, (target, ylabel) in enumerate(zip(targets_plot, ylabels)):
            ax = axes[ax_idx]
            y = preds[:, ax_idx]
            ax.plot(tSpDay_vals, y, lw=2, color=m_colors[m], label=month_names[m])
            ax.set_xlabel("Day setpoint tSpDay (°C)")
            ax.set_ylabel(ylabel)
            ax.set_title(ylabel)

    for ax in axes:
        ax.axvline(19.17, color="black", ls="--", lw=1, alpha=0.5)
        ax.text(19.4, ax.get_ylim()[0] + (ax.get_ylim()[1]-ax.get_ylim()[0])*0.05,
                "66.5°F\n(calibrated)", fontsize=7, color="black", alpha=0.6)
        ax.legend(fontsize=8)

    fig.tight_layout()
    savefig(fig, "13_setpoint_sweep.png")


# ═══════════════════════════════════════════════════════════════════════════════
# 14  Lamp intensity tradeoffs
# ═══════════════════════════════════════════════════════════════════════════════

def fig_14_lamp_tradeoffs(model):
    months = [1, 4, 7, 10]
    month_names = {1:"January", 4:"April", 7:"July", 10:"October"}
    m_colors    = {1: PALETTE["cool"], 4: PALETTE["cost"], 7: PALETTE["solar"], 10: PALETTE["rh"]}
    lamp_vals = np.linspace(0, 42.4, 60)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Effect of Supplemental Lighting (thetaLampMax) Across Seasons\n"
                 "Fixed: tSpDay=19.17°C, heatDeadZone=5, rhMax=80%",
                 fontsize=11, fontweight="bold")

    for m in months:
        rows = []
        for lamp in lamp_vals:
            rows.append({
                "in_tSpDay":       19.17,
                "in_tSpNight":     17.17,
                "in_thetaLampMax": lamp,
                "in_heatDeadZone": 5.0,
                "in_rhMax":        80.0,
                "month_sin":       np.sin(2 * np.pi * m / 12),
                "month_cos":       np.cos(2 * np.pi * m / 12),
            })
        X = pd.DataFrame(rows)
        preds = model.predict(X)
        temp = preds[:, 0]; cost = preds[:, 2]

        axes[0].plot(lamp_vals, temp, lw=2, color=m_colors[m], label=month_names[m])
        axes[1].plot(lamp_vals, cost, lw=2, color=m_colors[m], label=month_names[m])

    axes[0].set_xlabel("Max lamp intensity (W/m²)"); axes[0].set_ylabel("Predicted mean temp (°C)")
    axes[0].set_title("Lamp intensity → temperature")
    ax2 = axes[0].twinx()
    lo, hi = axes[0].get_ylim()
    ax2.set_ylim(c_to_f(lo), c_to_f(hi)); ax2.set_ylabel("°F", fontsize=8)

    axes[1].set_xlabel("Max lamp intensity (W/m²)"); axes[1].set_ylabel("Predicted cost ($/m²)")
    axes[1].set_title("Lamp intensity → cost")
    axes[1].axvline(42.4, color="gray", ls=":", lw=1, label="Max (49 Barrina T8)")

    for ax in axes:
        ax.legend(fontsize=8)
    fig.tight_layout()
    savefig(fig, "14_lamp_tradeoffs.png")


# ═══════════════════════════════════════════════════════════════════════════════
# 15  Fan dead-zone effect
# ═══════════════════════════════════════════════════════════════════════════════

def fig_15_heatdeadzone(model):
    dead_vals = np.linspace(5, 20, 60)
    months = [1, 4, 7, 10]
    month_names = {1:"January", 4:"April", 7:"July", 10:"October"}
    m_colors    = {1: PALETTE["cool"], 4: PALETTE["cost"], 7: PALETTE["solar"], 10: PALETTE["rh"]}

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Fan Dead-Zone Effect on Predicted Climate\n"
                 "heatDeadZone: °C above tSpDay before fans engage  "
                 "(default GreenLight=13.4°C causes 7-8°F solar overshoot)",
                 fontsize=10, fontweight="bold")

    for m in months:
        rows = [{"in_tSpDay":19.17,"in_tSpNight":17.17,"in_thetaLampMax":0,
                 "in_heatDeadZone":d,"in_rhMax":80,
                 "month_sin":np.sin(2*np.pi*m/12),"month_cos":np.cos(2*np.pi*m/12)}
                for d in dead_vals]
        X = pd.DataFrame(rows)
        preds = model.predict(X)
        temp  = preds[:, 0]; cost = preds[:, 2]

        axes[0].plot(dead_vals, temp, lw=2, color=m_colors[m], label=month_names[m])
        axes[1].plot(dead_vals, cost, lw=2, color=m_colors[m], label=month_names[m])

    for ax in axes:
        ax.axvline(5.0, color="green", ls="--", lw=1.5, label="Calibrated spring/winter=5")
        ax.axvline(8.0, color="#AB47BC", ls="--", lw=1.5, label="Calibrated shoulder=8")
        ax.axvline(13.4, color="red", ls=":", lw=1.5, label="GreenLight default=13.4")
        ax.legend(fontsize=7.5)

    axes[0].set_xlabel("heatDeadZone (°C)"); axes[0].set_ylabel("Predicted mean temp (°C)")
    axes[0].set_title("Dead zone → temperature  (larger=hotter)")
    ax2 = axes[0].twinx()
    lo, hi = axes[0].get_ylim()
    ax2.set_ylim(c_to_f(lo), c_to_f(hi)); ax2.set_ylabel("°F", fontsize=8)

    axes[1].set_xlabel("heatDeadZone (°C)"); axes[1].set_ylabel("Predicted cost ($/m²)")
    axes[1].set_title("Dead zone → cost")

    fig.tight_layout()
    savefig(fig, "15_heatdeadzone_effect.png")


# ═══════════════════════════════════════════════════════════════════════════════
# 16  Pareto frontiers — combined
# ═══════════════════════════════════════════════════════════════════════════════

def fig_16_pareto(model):
    rng  = np.random.default_rng(1)
    n    = 25_000
    params = {
        "in_tSpDay":       rng.uniform(10, 22, n),
        "in_tSpNight":     rng.uniform(10, 22, n),
        "in_thetaLampMax": rng.uniform(0, 42.4, n),
        "in_heatDeadZone": rng.uniform(5, 20,  n),
        "in_rhMax":        rng.uniform(70, 95, n),
    }
    params["in_tSpNight"] = np.minimum(params["in_tSpNight"], params["in_tSpDay"])
    m = rng.integers(1, 13, n).astype(float)
    params["month_sin"] = np.sin(2 * np.pi * m / 12)
    params["month_cos"] = np.cos(2 * np.pi * m / 12)
    X = pd.DataFrame(params)
    preds = model.predict(X)
    temp  = preds[:, 0]; rh = preds[:, 1]; cost = preds[:, 2]

    def pareto_mask(x_max, y_min):
        n = len(x_max)
        dominated = np.zeros(n, dtype=bool)
        for i in range(n):
            if np.any((x_max >= x_max[i]) & (y_min <= y_min[i]) &
                      ((x_max > x_max[i]) | (y_min < y_min[i]))):
                dominated[i] = True
        return ~dominated

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("Pareto Frontiers — XGBoost Surrogate  (25,000 random parameter combinations)",
                 fontsize=12, fontweight="bold")

    for ax, x_vals, x_label, x_unit, x_min, x_max, color in [
        (axes[0], temp, "Mean air temperature", "°C", 17, 27, PALETTE["heat"]),
        (axes[1], rh,   "Mean relative humidity", "%",  55, 90, PALETTE["rh"]),
    ]:
        mask = pareto_mask(x_vals, cost)
        # Background scatter
        ax.scatter(x_vals, cost, s=2, alpha=0.08, color=PALETTE["gray"], rasterized=True)
        # Feasible background
        feas = (x_vals >= x_min) & (x_vals <= x_max)
        ax.scatter(x_vals[feas], cost[feas], s=4, alpha=0.15, color=color, rasterized=True)
        # Pareto front
        front = pd.DataFrame({"x": x_vals[mask], "cost": cost[mask]}).sort_values("x")
        ax.scatter(front["x"], front["cost"], s=40, color=color, zorder=5,
                   edgecolors="white", linewidths=0.5, label="Pareto front")
        ax.step(front["x"], front["cost"], where="post", color=color, lw=2)

        ax.axvspan(x_min, x_max, alpha=0.07, color="green", label="Comfort zone")
        ax.axvline(x_min, color="green", ls="--", lw=1)
        ax.axvline(x_max, color="red",   ls="--", lw=1)
        ax.set_xlabel(f"{x_label} ({x_unit})"); ax.set_ylabel("Monthly cost ($/m²)")
        ax.set_title(f"{x_label} vs Cost")
        ax.legend(fontsize=8.5)

        if x_unit == "°C":
            ax2 = ax.twiny()
            lo, hi = ax.get_xlim()
            ax2.set_xlim(c_to_f(lo), c_to_f(hi)); ax2.set_xlabel("°F", fontsize=8)

    fig.tight_layout()
    savefig(fig, "16_pareto_frontiers.png")


# ═══════════════════════════════════════════════════════════════════════════════
# 17  Monthly optimal settings
# ═══════════════════════════════════════════════════════════════════════════════

def fig_17_monthly_optima():
    path = os.path.join(OUTPUT_DIR, "monthly_optima.csv")
    if not os.path.exists(path):
        print("  (skip 17 — run analyze_nn.py first)")
        return
    df = pd.read_csv(path)
    months = df["month_name"].tolist()
    x = np.arange(len(months))

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    fig.suptitle("Monthly Optimal Control Settings — XGBoost Surrogate\n"
                 "Cheapest settings per month within comfort bounds "
                 "(temp 63–81°F, RH 55–90%)",
                 fontsize=11, fontweight="bold")

    season_c = [PALETTE["cool"] if m in [12,1,2] else
                PALETTE["cost"] if m in [3,4,5] else
                PALETTE["solar"] if m in [6,7,8] else
                PALETTE["rh"] for m in df["month"]]

    # Optimal tSpDay
    axes[0,0].bar(x, df["in_tSpDay"], color=season_c, edgecolor="white")
    axes[0,0].axhline(19.17, color="black", ls="--", lw=1, label="Calibrated=19.17°C")
    ax2 = axes[0,0].twinx()
    ax2.set_ylim(c_to_f(axes[0,0].get_ylim()[0]), c_to_f(axes[0,0].get_ylim()[1]))
    ax2.set_ylabel("°F", fontsize=8)
    axes[0,0].set_title("Optimal day setpoint (tSpDay)"); axes[0,0].set_ylabel("°C")
    axes[0,0].set_xticks(x); axes[0,0].set_xticklabels(months, rotation=30, ha="right")
    axes[0,0].legend(fontsize=7.5)

    # Optimal lamp
    axes[0,1].bar(x, df["in_thetaLampMax"], color=PALETTE["solar"], edgecolor="white")
    axes[0,1].axhline(42.4, color="gray", ls=":", lw=1, label="Max (42.4 W/m²)")
    axes[0,1].set_title("Optimal lamp intensity (thetaLampMax)"); axes[0,1].set_ylabel("W/m²")
    axes[0,1].set_xticks(x); axes[0,1].set_xticklabels(months, rotation=30, ha="right")
    axes[0,1].legend(fontsize=7.5)

    # Predicted temp
    axes[1,0].bar(x, df["pred_tAir_F"], color=PALETTE["heat"], edgecolor="white")
    axes[1,0].axhline(63, color="green",  ls="--", lw=1, label="Min comfort 63°F")
    axes[1,0].axhline(81, color="red",    ls="--", lw=1, label="Max comfort 81°F")
    axes[1,0].set_title("Predicted mean temp"); axes[1,0].set_ylabel("°F")
    axes[1,0].set_xticks(x); axes[1,0].set_xticklabels(months, rotation=30, ha="right")
    axes[1,0].legend(fontsize=7.5)

    # Cost
    axes[1,1].bar(x, df["pred_cost_usd_m2"], color=PALETTE["cost"], edgecolor="white")
    for i, v in enumerate(df["pred_cost_usd_m2"]):
        axes[1,1].text(i, v + 0.003, f"${v:.2f}", ha="center", fontsize=6.5)
    axes[1,1].set_title("Predicted monthly cost"); axes[1,1].set_ylabel("$/m²")
    axes[1,1].set_xticks(x); axes[1,1].set_xticklabels(months, rotation=30, ha="right")

    fig.tight_layout()
    savefig(fig, "17_monthly_optima.png")


# ═══════════════════════════════════════════════════════════════════════════════
# 18  Speed benchmark
# ═══════════════════════════════════════════════════════════════════════════════

def fig_18_speed_benchmark(model):
    import time, pickle
    batch_sizes = [1, 10, 100, 1_000, 10_000]
    rng = np.random.default_rng(0)

    # XGBoost timings
    xgb_us = []
    for n in batch_sizes:
        X = pd.DataFrame({
            "in_tSpDay":       rng.uniform(10, 22, n),
            "in_tSpNight":     rng.uniform(10, 22, n),
            "in_thetaLampMax": rng.uniform(0, 42.4, n),
            "in_heatDeadZone": rng.uniform(5, 20,  n),
            "in_rhMax":        rng.uniform(70, 95, n),
            "month_sin":       rng.uniform(-1, 1,   n),
            "month_cos":       rng.uniform(-1, 1,   n),
        })
        _ = model.predict(X[:1])  # warm-up
        t0 = time.perf_counter()
        for _ in range(20): model.predict(X)
        xgb_us.append((time.perf_counter() - t0) / 20 / n * 1e6)

    greenlight_s = 18.0  # one ODE solve

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Inference Speed — XGBoost Surrogate vs GreenLight ODE",
                 fontsize=12, fontweight="bold")

    # Per-sample speed vs batch size
    axes[0].loglog(batch_sizes, xgb_us, "o-", color=PALETTE["accent"], lw=2, ms=8, label="XGBoost surrogate")
    axes[0].axhline(greenlight_s * 1e6, color="red", ls="--", lw=1.5, label="GreenLight ODE (~18 s)")
    axes[0].set_xlabel("Batch size"); axes[0].set_ylabel("Time per prediction (µs)  — log scale")
    axes[0].set_title("Per-sample inference time")
    axes[0].legend(fontsize=8.5)

    # Speedup
    speedups = [greenlight_s * 1e6 / u for u in xgb_us]
    axes[1].loglog(batch_sizes, speedups, "s-", color=PALETTE["cost"], lw=2, ms=8)
    axes[1].set_xlabel("Batch size"); axes[1].set_ylabel("Speedup factor vs GreenLight  — log scale")
    axes[1].set_title("Speedup over GreenLight ODE")
    for n, s in zip(batch_sizes, speedups):
        axes[1].annotate(f"{s:,.0f}×", (n, s), textcoords="offset points",
                         xytext=(5, 5), fontsize=8, color=PALETTE["dark"])

    fig.tight_layout()
    savefig(fig, "18_speed_benchmark.png")


# ═══════════════════════════════════════════════════════════════════════════════
# 19  Zone temperature divergence (real greenhouse data)
# ═══════════════════════════════════════════════════════════════════════════════

def fig_19_zone_temps():
    windows = [
        ("jan_cold_week.csv",   "Jan 13–20, 2026",   PALETTE["cool"]),
        ("spring_apr_2026.csv", "Apr 6–13, 2026",    PALETTE["cost"]),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Zone Temperature Divergence — Real Greenhouse\n"
                 "Single-zone model error: 3–5°F zone divergence driven by slab, patio door, and tree shading",
                 fontsize=11, fontweight="bold")

    for ax, (csv, title, color) in zip(axes, windows):
        df = load_window_csv(csv)
        df = df.resample("30min", on="ts").mean(numeric_only=True).reset_index()

        zones = {"North": ("temp_north","#81C784"), "East": ("temp_east","#64B5F6"),
                 "West":  ("temp_west","#FFB74D"),  "South":("temp_south","#F06292")}

        for zone_lbl, (col, zc) in zones.items():
            if col in df.columns and df[col].notna().sum() > 10:
                ax.plot(df["ts"], df[col], lw=1, alpha=0.8, color=zc, label=zone_lbl)

        ax.plot(df["ts"], df["temp_avg"], lw=2.5, color="black", ls="--", label="Avg (zone mean)", zorder=5)
        ax.set_title(title, fontsize=10, fontweight="bold", color=color)
        ax.set_ylabel("Temperature (°F)")
        ax.legend(fontsize=8)
        ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%m/%d"))

        # Annotate spread
        if "temp_north" in df.columns and "temp_south" in df.columns:
            spread = (df["temp_south"] - df["temp_north"]).abs().median()
            ax.text(0.02, 0.95, f"Median N-S spread: {spread:.1f}°F",
                    transform=ax.transAxes, fontsize=8.5,
                    bbox=dict(facecolor="white", edgecolor="gray", alpha=0.8))

    fig.tight_layout()
    savefig(fig, "19_zone_temperature_divergence.png")


# ═══════════════════════════════════════════════════════════════════════════════
# 20  3-D scatter: input parameters vs cost (two views)
# ═══════════════════════════════════════════════════════════════════════════════

def fig_20_3d_scatter(df):
    from mpl_toolkits.mplot3d import Axes3D  # noqa

    fig = plt.figure(figsize=(14, 6))
    fig.suptitle("GreenByte Training Data — 3-D Parameter Views",
                 fontsize=12, fontweight="bold")

    cost  = df["out_cost_total_usd_m2"].values
    norm  = (cost - cost.min()) / (cost.max() - cost.min())
    colors_map = plt.cm.RdYlGn_r(norm)

    # View 1: setpoint × lamp × cost
    ax1 = fig.add_subplot(121, projection="3d")
    sc1 = ax1.scatter(df["in_tSpDay"], df["in_thetaLampMax"], df["out_cost_total_usd_m2"],
                      c=df["start_month"], cmap="hsv", s=8, alpha=0.6, vmin=1, vmax=12)
    ax1.set_xlabel("tSpDay (°C)", labelpad=6)
    ax1.set_ylabel("Lamp (W/m²)", labelpad=6)
    ax1.set_zlabel("Cost ($/m²)", labelpad=6)
    ax1.set_title("Setpoint × Lamp × Cost\n(color = month)")
    fig.colorbar(sc1, ax=ax1, label="Month", shrink=0.6, pad=0.1)

    # View 2: dead-zone × rhMax × RH
    ax2 = fig.add_subplot(122, projection="3d")
    sc2 = ax2.scatter(df["in_heatDeadZone"], df["in_rhMax"], df["out_mean_rh_pct"],
                      c=df["out_cost_total_usd_m2"], cmap="viridis", s=8, alpha=0.6)
    ax2.set_xlabel("HeatDeadZone (°C)", labelpad=6)
    ax2.set_ylabel("rhMax (%)", labelpad=6)
    ax2.set_zlabel("Mean RH (%)", labelpad=6)
    ax2.set_title("Dead-Zone × rhMax → RH\n(color = cost)")
    fig.colorbar(sc2, ax=ax2, label="Cost ($/m²)", shrink=0.6, pad=0.1)

    fig.tight_layout()
    savefig(fig, "20_3d_scatter.png")


# ═══════════════════════════════════════════════════════════════════════════════
# Move existing PNGs from output/ → figures/
# ═══════════════════════════════════════════════════════════════════════════════

def move_existing_pngs():
    moved = []
    for fname in os.listdir(OUTPUT_DIR):
        if fname.lower().endswith(".png"):
            src = os.path.join(OUTPUT_DIR, fname)
            dst = os.path.join(FIGURES_DIR, fname)
            shutil.move(src, dst)
            moved.append(fname)
    if moved:
        print(f"\n  Moved {len(moved)} existing PNG(s) from output/ → figures/:")
        for f in moved:
            print(f"    {f}")


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    print("── GreenByte figure generator ───────────────────────────────────────")
    print(f"  Output: {FIGURES_DIR}\n")

    # Load shared resources once
    df    = load_training_data()
    model = load_model()
    meta  = load_meta()

    steps = [
        ("01  Pipeline overview",            lambda: fig_01_pipeline()),
        ("02  Greenhouse data windows",       lambda: fig_02_greenhouse_windows()),
        ("03  Calibration summary",           lambda: fig_03_calibration_summary()),
        ("04  Sensitivity analysis",          lambda: fig_04_sensitivity()),
        ("05  LHS coverage",                  lambda: fig_05_lhs_coverage(df)),
        ("06  Output distributions",          lambda: fig_06_output_distributions(df)),
        ("07  Cost decomposition",            lambda: fig_07_cost_decomposition(df)),
        ("08  Correlations heatmap",          lambda: fig_08_correlations(df)),
        ("09  Model comparison",              lambda: fig_09_model_comparison()),
        ("10  Predicted vs actual",           lambda: fig_10_predicted_vs_actual(df, model)),
        ("11  Residuals",                     lambda: fig_11_residuals(df, model)),
        ("12  Feature importance",            lambda: fig_12_feature_importance(meta)),
        ("13  Setpoint sweep",                lambda: fig_13_setpoint_sweep(model)),
        ("14  Lamp tradeoffs",                lambda: fig_14_lamp_tradeoffs(model)),
        ("15  Fan dead-zone effect",          lambda: fig_15_heatdeadzone(model)),
        ("16  Pareto frontiers",              lambda: fig_16_pareto(model)),
        ("17  Monthly optima",                lambda: fig_17_monthly_optima()),
        ("18  Speed benchmark",               lambda: fig_18_speed_benchmark(model)),
        ("19  Zone temp divergence",          lambda: fig_19_zone_temps()),
        ("20  3-D scatter",                   lambda: fig_20_3d_scatter(df)),
    ]

    for label, fn in steps:
        print(f"\n[{label}]")
        try:
            fn()
        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback; traceback.print_exc()

    move_existing_pngs()

    print(f"\n── Done — {len(os.listdir(FIGURES_DIR))} figures in figures/ ──────────────────────────")


if __name__ == "__main__":
    main()
