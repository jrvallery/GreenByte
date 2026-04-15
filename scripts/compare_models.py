"""
scripts/compare_models.py

Benchmark multiple ML approaches against the NN surrogate on the same
GreenLight training dataset (data/training_data.csv).

Models compared:
  Linear:        LinearRegression, Ridge, Lasso, ElasticNet
  Tree ensemble: RandomForest, ExtraTrees, GradientBoosting
  Boosting:      XGBoost, LightGBM
  Kernel/KNN:    SVR (RBF), KNeighborsRegressor
  Probabilistic: GaussianProcess (RBF kernel)
  Neural net:    MLPRegressor (current surrogate)

Feature engineering:
  - Cyclical month encoding: sin(2π·m/12) and cos(2π·m/12)
    replaces the raw 1-12 integer so Dec → Jan wrap-around is smooth.
  - StandardScaler applied where required (linear, SVR, GP, NN).

Evaluation:
  - 5-fold cross-validation R² (mean ± std) per target
  - Held-out test MAE per target
  - Inference speed (µs / prediction, batch of 1000)
  - Feature importances for tree-based models

Outputs (output/):
  model_comparison.csv        Leaderboard sorted by avg CV R²
  model_comparison.json       Full structured report
  feature_importance.csv      Importances for tree-based models
  model_comparison.png        Bar chart (if matplotlib available)

Usage:
    python scripts/compare_models.py
"""

import json
import os
import time
import warnings

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import (
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel
from sklearn.linear_model import ElasticNet, Lasso, LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import KFold, cross_val_predict, train_test_split
from sklearn.multioutput import MultiOutputRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

import xgboost as xgb
import lightgbm as lgb

warnings.filterwarnings("ignore")

# ── Paths ─────────────────────────────────────────────────────────────────────
project_dir = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
DATA_CSV    = os.path.join(project_dir, "data", "training_data.csv")
OUTPUT_DIR  = os.path.join(project_dir, "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

FEATURES = [
    "in_tSpDay",
    "in_tSpNight",
    "in_thetaLampMax",
    "in_heatDeadZone",
    "in_rhMax",
    "start_month",
]

TARGETS = ["out_mean_tAir_C", "out_mean_rh_pct", "out_cost_total_usd_m2"]

TARGET_SHORT = {
    "out_mean_tAir_C":       "temp_C",
    "out_mean_rh_pct":       "rh_pct",
    "out_cost_total_usd_m2": "cost_usd",
}

RANDOM_SEED = 42
N_CV_FOLDS  = 5


# ── Feature engineering ───────────────────────────────────────────────────────

def engineer_features(X: pd.DataFrame) -> pd.DataFrame:
    """
    Add cyclical month encoding (sin + cos) so Dec → Jan is a smooth wrap.
    The raw 1-12 integer treats month 1 and 12 as 11 apart; sin/cos makes
    them adjacent (distance ≈ 0.5 vs 10.5 in raw space).
    """
    X = X.copy()
    m = X["start_month"]
    X["month_sin"] = np.sin(2 * np.pi * m / 12)
    X["month_cos"] = np.cos(2 * np.pi * m / 12)
    X = X.drop(columns=["start_month"])
    return X


def load_data(cyclic_month: bool = True):
    df = pd.read_csv(DATA_CSV)
    X = df[FEATURES].copy()
    y = df[TARGETS].copy()

    if cyclic_month:
        X = engineer_features(X)

    # Drop rows with NaN/inf targets
    mask = y.replace([np.inf, -np.inf], np.nan).notna().all(axis=1)
    X, y = X[mask].reset_index(drop=True), y[mask].reset_index(drop=True)

    return train_test_split(X, y, test_size=0.15, random_state=RANDOM_SEED)


# ── Model definitions ─────────────────────────────────────────────────────────

def build_models() -> dict:
    """
    Returns an ordered dict of {name: (pipeline, needs_scaling, supports_multi_output)}.

    needs_scaling  — True → wrap in Pipeline([scaler, model])
    multi_output   — False → wrap in MultiOutputRegressor (handles 3 targets separately)
    """
    gp_kernel = ConstantKernel(1.0) * RBF(length_scale=1.0) + WhiteKernel(noise_level=0.1)

    models = {
        # ── Linear ────────────────────────────────────────────────────────────
        "LinearRegression": (LinearRegression(),                           True,  True),
        "Ridge":            (Ridge(alpha=1.0),                             True,  True),
        "Lasso":            (Lasso(alpha=0.01, max_iter=5000),             True,  True),
        "ElasticNet":       (ElasticNet(alpha=0.01, l1_ratio=0.5,
                                        max_iter=5000),                    True,  True),
        # ── Tree ensembles ────────────────────────────────────────────────────
        "RandomForest":     (RandomForestRegressor(n_estimators=300,
                                                    random_state=RANDOM_SEED,
                                                    n_jobs=-1),            False, True),
        "ExtraTrees":       (ExtraTreesRegressor(n_estimators=300,
                                                  random_state=RANDOM_SEED,
                                                  n_jobs=-1),              False, True),
        "GradientBoosting": (MultiOutputRegressor(
                                GradientBoostingRegressor(n_estimators=300,
                                                          learning_rate=0.05,
                                                          max_depth=4,
                                                          random_state=RANDOM_SEED)),
                                                                           False, True),
        # ── Gradient boosting libraries ───────────────────────────────────────
        "XGBoost":          (MultiOutputRegressor(
                                xgb.XGBRegressor(n_estimators=300,
                                                  learning_rate=0.05,
                                                  max_depth=4,
                                                  random_state=RANDOM_SEED,
                                                  verbosity=0,
                                                  n_jobs=-1)),             False, True),
        "LightGBM":         (MultiOutputRegressor(
                                lgb.LGBMRegressor(n_estimators=300,
                                                   learning_rate=0.05,
                                                   max_depth=4,
                                                   random_state=RANDOM_SEED,
                                                   verbosity=-1,
                                                   n_jobs=-1)),            False, True),
        # ── Kernel / KNN ──────────────────────────────────────────────────────
        "SVR_RBF":          (MultiOutputRegressor(SVR(kernel="rbf",
                                                       C=10, epsilon=0.1)), True, True),
        "KNeighbors":       (KNeighborsRegressor(n_neighbors=7,
                                                  weights="distance",
                                                  n_jobs=-1),              True,  True),
        # ── Gaussian Process ─────────────────────────────────────────────────
        # GP provides uncertainty estimates — valuable for plan scoring in Iris.
        # n_restarts_optimizer kept low for speed; increase for final model.
        "GaussianProcess":  (MultiOutputRegressor(
                                GaussianProcessRegressor(kernel=gp_kernel,
                                                          n_restarts_optimizer=3,
                                                          random_state=RANDOM_SEED,
                                                          normalize_y=True)), True, True),
        # ── Neural Network (current surrogate) ───────────────────────────────
        "MLP_128x64":       (MLPRegressor(hidden_layer_sizes=(128, 64),
                                           activation="relu",
                                           solver="adam",
                                           alpha=1e-4,
                                           learning_rate_init=5e-4,
                                           max_iter=5000,
                                           random_state=RANDOM_SEED,
                                           early_stopping=True,
                                           validation_fraction=0.1,
                                           n_iter_no_change=50,
                                           tol=1e-6),                      True,  True),
    }
    return models


def wrap_pipeline(estimator, needs_scaling: bool) -> Pipeline:
    if needs_scaling:
        return Pipeline([("scaler", StandardScaler()), ("model", estimator)])
    return Pipeline([("model", estimator)])


# ── Evaluation ────────────────────────────────────────────────────────────────

def evaluate_model(name: str, pipeline, X_train, y_train, X_test, y_test) -> dict:
    """Fit, CV, score. Returns metric dict."""
    t_fit = time.perf_counter()
    pipeline.fit(X_train, y_train)
    fit_s = time.perf_counter() - t_fit

    # Test set metrics
    y_pred = pipeline.predict(X_test)
    test_metrics = {}
    for i, col in enumerate(TARGETS):
        short = TARGET_SHORT[col]
        test_metrics[short] = {
            "mae": round(float(mean_absolute_error(y_test.iloc[:, i], y_pred[:, i])), 4),
            "r2":  round(float(r2_score(y_test.iloc[:, i], y_pred[:, i])), 4),
        }

    # 5-fold CV R² per target — use cross_val_predict so multi-output models
    # (wrapped in MultiOutputRegressor) get a full 2D y and we compute per-target
    # R² from the OOF predictions instead of calling cross_val_score with 1D y.
    kf = KFold(n_splits=N_CV_FOLDS, shuffle=True, random_state=RANDOM_SEED)
    pipe_cv = clone(pipeline)
    y_oof = cross_val_predict(pipe_cv, X_train, y_train, cv=kf)
    # y_oof may be 1D (single-target model) or 2D (multi-output)
    if y_oof.ndim == 1:
        y_oof = y_oof.reshape(-1, 1)

    # Compute per-fold R² by replaying the splits for std
    fold_r2s = {TARGET_SHORT[t]: [] for t in TARGETS}
    for fold_train_idx, fold_val_idx in kf.split(X_train):
        pipe_fold = clone(pipeline)
        pipe_fold.fit(X_train.iloc[fold_train_idx], y_train.iloc[fold_train_idx])
        y_fold_pred = pipe_fold.predict(X_train.iloc[fold_val_idx])
        if np.ndim(y_fold_pred) == 1:
            y_fold_pred = y_fold_pred.reshape(-1, 1)
        for i, col in enumerate(TARGETS):
            short = TARGET_SHORT[col]
            r2 = r2_score(y_train.iloc[fold_val_idx, i], y_fold_pred[:, i])
            fold_r2s[short].append(r2)

    cv_metrics = {}
    for col in TARGETS:
        short = TARGET_SHORT[col]
        scores = np.array(fold_r2s[short])
        cv_metrics[short] = {
            "cv_r2_mean": round(float(scores.mean()), 4),
            "cv_r2_std":  round(float(scores.std()), 4),
        }

    # Avg CV R² across all three targets (headline number)
    avg_cv_r2 = round(float(np.mean([cv_metrics[TARGET_SHORT[t]]["cv_r2_mean"]
                                      for t in TARGETS])), 4)

    # Inference speed (batch of 1000)
    X_bench = X_test.iloc[:min(len(X_test), 1)].copy()
    _ = pipeline.predict(X_bench)  # warm-up
    t0 = time.perf_counter()
    X_batch = pd.concat([X_test] * (1000 // len(X_test) + 1)).iloc[:1000].reset_index(drop=True)
    for _ in range(10):
        pipeline.predict(X_batch)
    infer_us = (time.perf_counter() - t0) / 10 / 1000 * 1e6

    return {
        "name":        name,
        "avg_cv_r2":   avg_cv_r2,
        "cv":          cv_metrics,
        "test":        test_metrics,
        "fit_s":       round(fit_s, 3),
        "infer_us":    round(infer_us, 3),
    }


# ── Feature importances ───────────────────────────────────────────────────────

def extract_importances(name: str, pipeline, feature_names: list) -> dict | None:
    model = pipeline.named_steps["model"]

    # Native multi-output trees
    if hasattr(model, "feature_importances_"):
        return dict(zip(feature_names, model.feature_importances_.tolist()))

    # MultiOutputRegressor wrapping a tree
    if hasattr(model, "estimators_"):
        try:
            imps = np.mean([e.feature_importances_ for e in model.estimators_], axis=0)
            return dict(zip(feature_names, imps.tolist()))
        except AttributeError:
            pass

    # Linear coefficients (averaged across targets for a rough importance signal)
    if hasattr(model, "coef_"):
        coef = np.abs(model.coef_)
        if coef.ndim == 2:
            coef = coef.mean(axis=0)
        # Normalize to sum to 1
        total = coef.sum()
        if total > 0:
            return dict(zip(feature_names, (coef / total).tolist()))

    return None


# ── Plots ─────────────────────────────────────────────────────────────────────

def make_plots(results: list[dict]):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    names     = [r["name"] for r in results]
    avg_cv    = [r["avg_cv_r2"] for r in results]
    temp_cv   = [r["cv"][TARGET_SHORT["out_mean_tAir_C"]]["cv_r2_mean"] for r in results]
    rh_cv     = [r["cv"][TARGET_SHORT["out_mean_rh_pct"]]["cv_r2_mean"] for r in results]
    cost_cv   = [r["cv"][TARGET_SHORT["out_cost_total_usd_m2"]]["cv_r2_mean"] for r in results]
    infer     = [r["infer_us"] for r in results]

    x = np.arange(len(names))
    w = 0.22

    fig, axes = plt.subplots(2, 1, figsize=(14, 10))

    # ── CV R² per target ─────────────────────────────────────────────────────
    ax = axes[0]
    bars_avg  = ax.bar(x - 1.5*w, avg_cv,  w, label="Avg (all targets)", color="#455A64")
    bars_temp = ax.bar(x - 0.5*w, temp_cv, w, label="Temp (°C)",         color="#EF5350")
    bars_rh   = ax.bar(x + 0.5*w, rh_cv,   w, label="RH (%)",            color="#42A5F5")
    bars_cost = ax.bar(x + 1.5*w, cost_cv, w, label="Cost ($/m²)",       color="#66BB6A")
    ax.axhline(0.9, color="gray", linestyle=":", linewidth=1, label="R²=0.9 target")
    ax.set_ylabel("5-fold CV R²")
    ax.set_title("Model Comparison — 5-fold Cross-Validation R²")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=30, ha="right", fontsize=9)
    ax.set_ylim(bottom=min(0, min(avg_cv) - 0.05))
    ax.legend(fontsize=8)

    # ── Inference speed ──────────────────────────────────────────────────────
    ax2 = axes[1]
    colors = ["#EF5350" if r["name"] == "MLP_128x64" else "#78909C" for r in results]
    ax2.bar(x, infer, color=colors, edgecolor="white")
    ax2.set_ylabel("Inference time (µs / prediction)")
    ax2.set_title("Inference Speed (batch of 1000)  — red = current surrogate")
    ax2.set_xticks(x)
    ax2.set_xticklabels(names, rotation=30, ha="right", fontsize=9)
    ax2.set_yscale("log")

    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "model_comparison.png"), dpi=150)
    plt.close(fig)
    print(f"  → output/model_comparison.png")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("── GreenByte ML model comparison ───────────────────────────────────")
    print(f"  Data: {DATA_CSV}")
    print(f"  Cyclical month encoding: yes (sin/cos of 2π·m/12)\n")

    X_train, X_test, y_train, y_test = load_data(cyclic_month=True)
    feature_names = list(X_train.columns)
    print(f"  Train: {len(X_train)}  Test: {len(X_test)}  Features: {feature_names}\n")

    model_defs = build_models()
    results    = []
    importances = {}

    col_w = 18
    header = (f"{'Model':<{col_w}}  {'Avg CV R²':>9}  "
              f"{'temp CV':>8}  {'rh CV':>7}  {'cost CV':>8}  "
              f"{'temp MAE':>9}  {'rh MAE':>7}  {'cost MAE':>9}  "
              f"{'µs/pred':>8}  {'fit(s)':>6}")
    print(header)
    print("─" * len(header))

    for name, (estimator, needs_scaling, _) in model_defs.items():
        pipeline = wrap_pipeline(clone(estimator), needs_scaling)
        try:
            r = evaluate_model(name, pipeline, X_train, y_train, X_test, y_test)
        except Exception as e:
            print(f"  {name:<{col_w}}  FAILED: {e}")
            continue

        results.append(r)

        cv  = r["cv"]
        tst = r["test"]
        line = (
            f"{name:<{col_w}}  "
            f"{r['avg_cv_r2']:>9.4f}  "
            f"{cv['temp_C']['cv_r2_mean']:>8.4f}  "
            f"{cv['rh_pct']['cv_r2_mean']:>7.4f}  "
            f"{cv['cost_usd']['cv_r2_mean']:>8.4f}  "
            f"{tst['temp_C']['mae']:>9.4f}  "
            f"{tst['rh_pct']['mae']:>7.3f}  "
            f"{tst['cost_usd']['mae']:>9.4f}  "
            f"{r['infer_us']:>8.2f}  "
            f"{r['fit_s']:>6.2f}"
        )
        print(line)

        # Feature importances
        imp = extract_importances(name, pipeline, feature_names)
        if imp:
            importances[name] = imp

    # ── Sort leaderboard by avg CV R² ─────────────────────────────────────────
    results.sort(key=lambda r: r["avg_cv_r2"], reverse=True)

    print("\n── Leaderboard (by avg CV R²) ───────────────────────────────────────")
    for rank, r in enumerate(results, 1):
        marker = " ◄ current" if r["name"] == "MLP_128x64" else ""
        print(f"  #{rank:>2}  {r['name']:<{col_w}}  avg CV R²={r['avg_cv_r2']:.4f}  "
              f"speed={r['infer_us']:.2f}µs/pred{marker}")

    # ── Feature importance summary ────────────────────────────────────────────
    if importances:
        print("\n── Feature importances (avg across trees or |coef|) ─────────────────")
        imp_df = pd.DataFrame(importances).T.fillna(0)
        imp_df = imp_df.sort_values(imp_df.columns[0], ascending=False)
        for feat in imp_df.columns:
            vals = {m: f"{v:.3f}" for m, v in imp_df[feat].items()}
            print(f"  {feat:<20}  " +
                  "  ".join(f"{m[:12]}: {v}" for m, v in vals.items()))
        imp_df.to_csv(os.path.join(OUTPUT_DIR, "feature_importance.csv"))

    # ── Save outputs ──────────────────────────────────────────────────────────
    leaderboard_rows = []
    for r in results:
        row = {
            "rank":        results.index(r) + 1,
            "model":       r["name"],
            "avg_cv_r2":   r["avg_cv_r2"],
            "temp_cv_r2":  r["cv"]["temp_C"]["cv_r2_mean"],
            "rh_cv_r2":    r["cv"]["rh_pct"]["cv_r2_mean"],
            "cost_cv_r2":  r["cv"]["cost_usd"]["cv_r2_mean"],
            "temp_mae":    r["test"]["temp_C"]["mae"],
            "rh_mae":      r["test"]["rh_pct"]["mae"],
            "cost_mae":    r["test"]["cost_usd"]["mae"],
            "infer_us":    r["infer_us"],
            "fit_s":       r["fit_s"],
        }
        leaderboard_rows.append(row)

    lb_df = pd.DataFrame(leaderboard_rows)
    lb_path = os.path.join(OUTPUT_DIR, "model_comparison.csv")
    lb_df.to_csv(lb_path, index=False)

    report = {
        "dataset": {"n_train": len(X_train), "n_test": len(X_test),
                    "features": feature_names, "targets": TARGETS,
                    "cyclic_month": True},
        "leaderboard": leaderboard_rows,
        "feature_importances": importances,
    }
    rpt_path = os.path.join(OUTPUT_DIR, "model_comparison.json")
    with open(rpt_path, "w") as f:
        json.dump(report, f, indent=2)

    make_plots(results)

    print(f"\n  Saved: {lb_path}")
    print(f"  Saved: {rpt_path}")
    print("── Done ─────────────────────────────────────────────────────────────")


if __name__ == "__main__":
    main()
