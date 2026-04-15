"""
scripts/analyze_nn.py

Post-training analysis of the NN surrogate:
  1. Pareto frontiers  — mean temp vs cost, mean RH vs cost
  2. Monthly optima    — cheapest control settings per month within comfort bounds
  3. Speed benchmark   — NN inference vs GreenLight ODE (~18 s/run)

Outputs (all in output/):
  pareto_temp_cost.csv      Pareto-optimal (temp, cost) points
  pareto_rh_cost.csv        Pareto-optimal (rh, cost) points
  monthly_optima.csv        Best settings per month
  analysis_report.json      All results as structured JSON
  pareto_temp_cost.png      Plot (if matplotlib available)
  pareto_rh_cost.png        Plot
  monthly_optima.png        Plot

Usage:
    python scripts/analyze_nn.py
"""

import json
import os
import pickle
import time
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ── Paths ─────────────────────────────────────────────────────────────────────
project_dir = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
MODEL_PKL   = os.path.join(project_dir, "models", "nn_surrogate.pkl")
META_JSON   = os.path.join(project_dir, "models", "nn_surrogate_meta.json")
OUTPUT_DIR  = os.path.join(project_dir, "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

FEATURES = [
    "in_tSpDay",
    "in_tSpNight",
    "in_thetaLampMax",
    "in_heatDeadZone",
    "in_rhMax",
    "month_sin",
    "month_cos",
]

TARGETS = ["out_mean_tAir_C", "out_mean_rh_pct", "out_cost_total_usd_m2"]

# ── Parameter ranges (matching the sweep in parameter_sweep.py) ───────────────
PARAM_RANGES = {
    "in_tSpDay":       (10.0, 22.0),
    "in_tSpNight":     (10.0, 22.0),
    "in_thetaLampMax": (0.0,  42.4),
    "in_heatDeadZone": (5.0,  20.0),
    "in_rhMax":        (70.0, 95.0),
}

# Comfort constraints for monthly optima
TEMP_MIN_C   = 17.0    # ~63°F — minimum acceptable mean temp
TEMP_MAX_C   = 27.0    # ~81°F — max before heat stress
RH_MIN_PCT   = 55.0    # dry threshold
RH_MAX_PCT   = 90.0    # disease risk threshold

GREENLIGHT_SIM_SECONDS = 18.0   # reference: single ODE solve on dev machine


# ── Helpers ───────────────────────────────────────────────────────────────────

def load_model():
    with open(MODEL_PKL, "rb") as f:
        return pickle.load(f)


def random_grid(n: int, seed: int = 0) -> pd.DataFrame:
    """Draw n random samples uniformly over the parameter space, all months."""
    rng = np.random.default_rng(seed)
    rows = {}
    for feat, (lo, hi) in PARAM_RANGES.items():
        rows[feat] = rng.uniform(lo, hi, n)

    # Clamp tSpNight ≤ tSpDay
    rows["in_tSpNight"] = np.minimum(rows["in_tSpNight"], rows["in_tSpDay"])

    # Cyclical month encoding
    m = rng.integers(1, 13, n).astype(float)
    rows["month_sin"] = np.sin(2 * np.pi * m / 12)
    rows["month_cos"] = np.cos(2 * np.pi * m / 12)
    return pd.DataFrame(rows)[FEATURES]


def predict_all(model, X: pd.DataFrame) -> pd.DataFrame:
    preds = model.predict(X)
    return pd.DataFrame(preds, columns=TARGETS)


def pareto_front(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Return boolean mask of Pareto-optimal points for:
      maximize x  AND  minimize y  (lower-left / upper-left front).

    A point is dominated if another point has x ≥ x_i AND y ≤ y_i (strict on at
    least one).  We want MAX x and MIN y, so we negate y for a standard
    "both-maximize" Pareto check.
    """
    n = len(x)
    dominated = np.zeros(n, dtype=bool)
    for i in range(n):
        # Any point that is at least as good on both dimensions and strictly
        # better on one dominates point i.
        better_x = x >= x[i]
        better_y = y <= y[i]
        strictly = (x > x[i]) | (y < y[i])
        if np.any(better_x & better_y & strictly & (np.arange(n) != i)):
            dominated[i] = True
    return ~dominated


# ── 1. Pareto frontiers ───────────────────────────────────────────────────────

def compute_pareto_frontiers(model, n_samples: int = 20_000) -> dict:
    print(f"\n[1/3] Pareto frontiers  ({n_samples:,} samples)...")
    X = random_grid(n_samples, seed=1)
    Y = predict_all(model, X)

    temp  = Y["out_mean_tAir_C"].values
    rh    = Y["out_mean_rh_pct"].values
    cost  = Y["out_cost_total_usd_m2"].values

    # ── temp vs cost ─────────────────────────────────────────────────────────
    mask_tc = pareto_front(temp, cost)
    df_tc = pd.concat([X[mask_tc].reset_index(drop=True),
                       Y[mask_tc].reset_index(drop=True)], axis=1)
    df_tc = df_tc.sort_values("out_mean_tAir_C").reset_index(drop=True)
    out_tc = os.path.join(OUTPUT_DIR, "pareto_temp_cost.csv")
    df_tc.to_csv(out_tc, index=False)
    print(f"  temp–cost front: {mask_tc.sum()} points  →  {out_tc}")

    # ── rh vs cost ────────────────────────────────────────────────────────────
    mask_rc = pareto_front(rh, cost)
    df_rc = pd.concat([X[mask_rc].reset_index(drop=True),
                       Y[mask_rc].reset_index(drop=True)], axis=1)
    df_rc = df_rc.sort_values("out_mean_rh_pct").reset_index(drop=True)
    out_rc = os.path.join(OUTPUT_DIR, "pareto_rh_cost.csv")
    df_rc.to_csv(out_rc, index=False)
    print(f"  RH–cost front:   {mask_rc.sum()} points  →  {out_rc}")

    return {
        "pareto_temp_cost": df_tc,
        "pareto_rh_cost":   df_rc,
        "n_samples":        n_samples,
    }


# ── 2. Monthly optima ─────────────────────────────────────────────────────────

def compute_monthly_optima(model, n_per_month: int = 5_000) -> pd.DataFrame:
    """
    For each calendar month, find the cheapest control settings that keep:
      mean temp in [TEMP_MIN_C, TEMP_MAX_C]  AND  mean RH in [RH_MIN_PCT, RH_MAX_PCT]
    """
    print(f"\n[2/3] Monthly optima  ({n_per_month:,} samples × 12 months)...")
    months = list(range(1, 13))
    month_names = ["Jan","Feb","Mar","Apr","May","Jun",
                   "Jul","Aug","Sep","Oct","Nov","Dec"]

    rows = []
    for m in months:
        rng = np.random.default_rng(seed=m * 100)
        X_m = {}
        for feat, (lo, hi) in PARAM_RANGES.items():
            X_m[feat] = rng.uniform(lo, hi, n_per_month)
        X_m["in_tSpNight"] = np.minimum(X_m["in_tSpNight"], X_m["in_tSpDay"])
        X_m["month_sin"] = np.full(n_per_month, np.sin(2 * np.pi * m / 12))
        X_m["month_cos"] = np.full(n_per_month, np.cos(2 * np.pi * m / 12))
        X_df = pd.DataFrame(X_m)[FEATURES]

        Y_df = predict_all(model, X_df)

        # Filter to feasible region
        feasible = (
            (Y_df["out_mean_tAir_C"] >= TEMP_MIN_C) &
            (Y_df["out_mean_tAir_C"] <= TEMP_MAX_C) &
            (Y_df["out_mean_rh_pct"] >= RH_MIN_PCT) &
            (Y_df["out_mean_rh_pct"] <= RH_MAX_PCT)
        )
        n_feasible = feasible.sum()

        if n_feasible == 0:
            print(f"  {month_names[m-1]:>3}: no feasible points — relaxing RH bound")
            feasible = (
                (Y_df["out_mean_tAir_C"] >= TEMP_MIN_C) &
                (Y_df["out_mean_tAir_C"] <= TEMP_MAX_C)
            )
            n_feasible = feasible.sum()

        X_feas = X_df[feasible].reset_index(drop=True)
        Y_feas = Y_df[feasible].reset_index(drop=True)

        if n_feasible == 0:
            print(f"  {month_names[m-1]:>3}: still no feasible points — skipping")
            continue

        best_idx = Y_feas["out_cost_total_usd_m2"].idxmin()
        best_x   = X_feas.loc[best_idx]
        best_y   = Y_feas.loc[best_idx]

        row = {
            "month":          m,
            "month_name":     month_names[m - 1],
            "n_feasible":     n_feasible,
            **{k: round(float(best_x[k]), 3) for k in FEATURES if k != "start_month"},
            "pred_tAir_C":    round(float(best_y["out_mean_tAir_C"]), 2),
            "pred_tAir_F":    round(float(best_y["out_mean_tAir_C"]) * 9/5 + 32, 2),
            "pred_rh_pct":    round(float(best_y["out_mean_rh_pct"]), 1),
            "pred_cost_usd_m2": round(float(best_y["out_cost_total_usd_m2"]), 4),
        }
        rows.append(row)
        print(f"  {month_names[m-1]:>3}: tSpDay={row['in_tSpDay']:.1f}°C  "
              f"lamp={row['in_thetaLampMax']:.1f}W/m²  "
              f"→ {row['pred_tAir_F']:.1f}°F / {row['pred_rh_pct']:.0f}% RH / "
              f"${row['pred_cost_usd_m2']:.3f}/m²")

    df = pd.DataFrame(rows)
    out_path = os.path.join(OUTPUT_DIR, "monthly_optima.csv")
    df.to_csv(out_path, index=False)
    print(f"  Saved → {out_path}")
    return df


# ── 3. Speed benchmark ────────────────────────────────────────────────────────

def benchmark_speed(model) -> dict:
    print("\n[3/3] Speed benchmark...")

    batch_sizes = [1, 10, 100, 1_000, 10_000]
    results = {}

    for n in batch_sizes:
        X = random_grid(n, seed=99)
        # Warm-up
        _ = model.predict(X[:1])
        # Time
        t0 = time.perf_counter()
        for _ in range(10):
            model.predict(X)
        elapsed = (time.perf_counter() - t0) / 10
        per_sample_us = elapsed / n * 1e6
        results[n] = {
            "batch_size":       n,
            "wall_s":           round(elapsed, 6),
            "per_sample_us":    round(per_sample_us, 3),
            "speedup_vs_greenlight": round(GREENLIGHT_SIM_SECONDS / elapsed, 0),
        }
        print(f"  n={n:>6}:  {elapsed*1000:.3f} ms total  |  "
              f"{per_sample_us:.2f} µs/sample  |  "
              f"{results[n]['speedup_vs_greenlight']:.0f}× vs GreenLight")

    return results


# ── 4. Plots (optional) ───────────────────────────────────────────────────────

def make_plots(pareto_data: dict, monthly_df: pd.DataFrame):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("\n  matplotlib not available — skipping plots")
        return

    print("\n  Generating plots...")

    # ── Pareto: temp vs cost ─────────────────────────────────────────────────
    df_tc = pareto_data["pareto_temp_cost"]
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(df_tc["out_mean_tAir_C"], df_tc["out_cost_total_usd_m2"],
               s=18, alpha=0.6, color="#2196F3", label="Pareto front")
    ax.axvline(TEMP_MIN_C, color="green", linestyle="--", linewidth=1,
               label=f"Min comfort ({TEMP_MIN_C}°C)")
    ax.axvline(TEMP_MAX_C, color="red", linestyle="--", linewidth=1,
               label=f"Max comfort ({TEMP_MAX_C}°C)")
    ax.set_xlabel("Mean air temperature (°C)")
    ax.set_ylabel("Monthly cost ($/m²)")
    ax.set_title("Pareto frontier: Temperature vs. Operating Cost")
    ax.legend(fontsize=9)

    # Secondary x-axis in °F
    ax2 = ax.twiny()
    x_min, x_max = ax.get_xlim()
    ax2.set_xlim(x_min * 9/5 + 32, x_max * 9/5 + 32)
    ax2.set_xlabel("Mean air temperature (°F)")

    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "pareto_temp_cost.png"), dpi=150)
    plt.close(fig)
    print(f"    → output/pareto_temp_cost.png")

    # ── Pareto: RH vs cost ───────────────────────────────────────────────────
    df_rc = pareto_data["pareto_rh_cost"]
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(df_rc["out_mean_rh_pct"], df_rc["out_cost_total_usd_m2"],
               s=18, alpha=0.6, color="#9C27B0", label="Pareto front")
    ax.axvline(RH_MIN_PCT, color="orange", linestyle="--", linewidth=1,
               label=f"Dry threshold ({RH_MIN_PCT}%)")
    ax.axvline(RH_MAX_PCT, color="red", linestyle="--", linewidth=1,
               label=f"Disease risk ({RH_MAX_PCT}%)")
    ax.set_xlabel("Mean relative humidity (%)")
    ax.set_ylabel("Monthly cost ($/m²)")
    ax.set_title("Pareto frontier: Humidity vs. Operating Cost")
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "pareto_rh_cost.png"), dpi=150)
    plt.close(fig)
    print(f"    → output/pareto_rh_cost.png")

    # ── Monthly optima bar chart ─────────────────────────────────────────────
    if len(monthly_df) > 0:
        fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True)

        axes[0].bar(monthly_df["month"], monthly_df["pred_tAir_F"],
                    color="#FF7043", edgecolor="white")
        axes[0].axhline(TEMP_MIN_C * 9/5 + 32, color="green", linestyle="--",
                        linewidth=1, label=f"{TEMP_MIN_C*9/5+32:.0f}°F min")
        axes[0].set_ylabel("Mean temp (°F)")
        axes[0].set_title("Monthly Optimal Settings — Predicted Climate & Cost")
        axes[0].legend(fontsize=8)

        axes[1].bar(monthly_df["month"], monthly_df["pred_rh_pct"],
                    color="#42A5F5", edgecolor="white")
        axes[1].axhline(RH_MIN_PCT, color="orange", linestyle="--",
                        linewidth=1, label=f"{RH_MIN_PCT}% floor")
        axes[1].axhline(RH_MAX_PCT, color="red", linestyle="--",
                        linewidth=1, label=f"{RH_MAX_PCT}% ceiling")
        axes[1].set_ylabel("Mean RH (%)")
        axes[1].legend(fontsize=8)

        axes[2].bar(monthly_df["month"], monthly_df["pred_cost_usd_m2"],
                    color="#66BB6A", edgecolor="white")
        axes[2].set_ylabel("Cost ($/m²)")
        axes[2].set_xlabel("Month")
        axes[2].set_xticks(range(1, 13))
        axes[2].set_xticklabels(["Jan","Feb","Mar","Apr","May","Jun",
                                  "Jul","Aug","Sep","Oct","Nov","Dec"])

        fig.tight_layout()
        fig.savefig(os.path.join(OUTPUT_DIR, "monthly_optima.png"), dpi=150)
        plt.close(fig)
        print(f"    → output/monthly_optima.png")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("── GreenByte NN surrogate analysis ─────────────────────────────────")

    if not os.path.exists(MODEL_PKL):
        raise FileNotFoundError(f"Model not found: {MODEL_PKL}\nRun scripts/train_nn.py first.")

    model = load_model()
    print(f"  Loaded: {MODEL_PKL}")

    with open(META_JSON) as f:
        meta = json.load(f)

    pareto_data  = compute_pareto_frontiers(model)
    monthly_df   = compute_monthly_optima(model)
    speed_data   = benchmark_speed(model)

    make_plots(pareto_data, monthly_df)

    # ── Summary report ────────────────────────────────────────────────────────
    report = {
        "model": {
            "path":       MODEL_PKL,
            "model_type": meta.get("model_type", "MLP"),
            "n_train":    meta["n_train"],
            "cv_r2": {
                "temp":  meta["cv_metrics"]["out_mean_tAir_C"]["cv_r2_mean"],
                "rh":    meta["cv_metrics"]["out_mean_rh_pct"]["cv_r2_mean"],
                "cost":  meta["cv_metrics"]["out_cost_total_usd_m2"]["cv_r2_mean"],
            },
        },
        "pareto": {
            "n_samples":          pareto_data["n_samples"],
            "temp_cost_front_n":  len(pareto_data["pareto_temp_cost"]),
            "rh_cost_front_n":    len(pareto_data["pareto_rh_cost"]),
            "cheapest_temp_comfortable": None,
            "cheapest_rh_comfortable":  None,
        },
        "monthly_optima": monthly_df.to_dict(orient="records") if len(monthly_df) else [],
        "speed_benchmark": speed_data,
        "comfort_constraints": {
            "temp_min_C": TEMP_MIN_C,
            "temp_max_C": TEMP_MAX_C,
            "rh_min_pct": RH_MIN_PCT,
            "rh_max_pct": RH_MAX_PCT,
        },
    }

    # Best single point on each Pareto front within comfort range
    df_tc = pareto_data["pareto_temp_cost"]
    feas_tc = df_tc[
        (df_tc["out_mean_tAir_C"] >= TEMP_MIN_C) &
        (df_tc["out_mean_tAir_C"] <= TEMP_MAX_C)
    ]
    if len(feas_tc):
        best_tc = feas_tc.loc[feas_tc["out_cost_total_usd_m2"].idxmin()]
        report["pareto"]["cheapest_temp_comfortable"] = {
            "tSpDay_C":      round(float(best_tc["in_tSpDay"]), 2),
            "tSpNight_C":    round(float(best_tc["in_tSpNight"]), 2),
            "lampMax_W_m2":  round(float(best_tc["in_thetaLampMax"]), 2),
            "heatDeadZone":  round(float(best_tc["in_heatDeadZone"]), 2),
            "rhMax":         round(float(best_tc["in_rhMax"]), 1),
            "pred_tAir_C":   round(float(best_tc["out_mean_tAir_C"]), 2),
            "pred_cost_usd_m2": round(float(best_tc["out_cost_total_usd_m2"]), 4),
        }

    df_rc = pareto_data["pareto_rh_cost"]
    feas_rc = df_rc[
        (df_rc["out_mean_rh_pct"] >= RH_MIN_PCT) &
        (df_rc["out_mean_rh_pct"] <= RH_MAX_PCT)
    ]
    if len(feas_rc):
        best_rc = feas_rc.loc[feas_rc["out_cost_total_usd_m2"].idxmin()]
        report["pareto"]["cheapest_rh_comfortable"] = {
            "tSpDay_C":      round(float(best_rc["in_tSpDay"]), 2),
            "tSpNight_C":    round(float(best_rc["in_tSpNight"]), 2),
            "lampMax_W_m2":  round(float(best_rc["in_thetaLampMax"]), 2),
            "heatDeadZone":  round(float(best_rc["in_heatDeadZone"]), 2),
            "rhMax":         round(float(best_rc["in_rhMax"]), 1),
            "pred_rh_pct":   round(float(best_rc["out_mean_rh_pct"]), 1),
            "pred_cost_usd_m2": round(float(best_rc["out_cost_total_usd_m2"]), 4),
        }

    report_path = os.path.join(OUTPUT_DIR, "analysis_report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    # ── Console summary ───────────────────────────────────────────────────────
    print("\n── Summary ──────────────────────────────────────────────────────────")
    if report["pareto"]["cheapest_temp_comfortable"]:
        best = report["pareto"]["cheapest_temp_comfortable"]
        print(f"  Cheapest comfortable temp setting:")
        print(f"    tSpDay={best['tSpDay_C']:.1f}°C  tSpNight={best['tSpNight_C']:.1f}°C  "
              f"lamp={best['lampMax_W_m2']:.1f} W/m²")
        print(f"    → {best['pred_tAir_C']:.1f}°C ({best['pred_tAir_C']*9/5+32:.1f}°F)  "
              f"${best['pred_cost_usd_m2']:.3f}/m²/month")

    if len(monthly_df):
        print(f"\n  Monthly cost range: "
              f"${monthly_df['pred_cost_usd_m2'].min():.3f} – "
              f"${monthly_df['pred_cost_usd_m2'].max():.3f}/m²")
        costliest = monthly_df.loc[monthly_df["pred_cost_usd_m2"].idxmax()]
        cheapest  = monthly_df.loc[monthly_df["pred_cost_usd_m2"].idxmin()]
        print(f"    Costliest: {costliest['month_name']}  "
              f"(${costliest['pred_cost_usd_m2']:.3f}/m²)")
        print(f"    Cheapest:  {cheapest['month_name']}   "
              f"(${cheapest['pred_cost_usd_m2']:.3f}/m²)")

    single_pred_us = speed_data[1]["per_sample_us"]
    speedup_1k     = speed_data[1_000]["speedup_vs_greenlight"]
    print(f"\n  Inference speed: {single_pred_us:.1f} µs/prediction  "
          f"({speedup_1k:.0f}× faster than GreenLight at n=1000)")

    print(f"\n  Report → {report_path}")
    print("── Done ─────────────────────────────────────────────────────────────")


if __name__ == "__main__":
    main()
