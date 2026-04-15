"""
GreenLight/scripts/train_nn.py

Train the greenhouse surrogate model on the parameter sweep dataset.

Architecture: XGBoost (MultiOutputRegressor, one tree per target) wrapped in a
StandardScaler pipeline.  Replaces the earlier MLP 128×64 surrogate after model
comparison showed XGBoost +0.06 avg CV R² over the MLP, especially on cost
prediction (0.88 vs 0.78 CV R²) — the primary Iris plan-scoring target.

Feature engineering:
  start_month (1-12) → month_sin, month_cos  (cyclical encoding)
  This makes Dec → Jan a smooth transition rather than a 11-step jump.

Usage:
    python scripts/train_nn.py

Outputs (all in models/):
    nn_surrogate.pkl         — trained sklearn Pipeline (scaler + XGBoost MOR)
    nn_surrogate_meta.json   — training metadata (features, targets, metrics)

Inputs:
    data/training_data.csv   — 500-row LHS sweep produced by parameter_sweep.py

Features (7 after encoding):
    in_tSpDay, in_tSpNight, in_thetaLampMax, in_heatDeadZone, in_rhMax,
    month_sin, month_cos

Targets (3):
    out_mean_tAir_C     — mean air temperature (°C)
    out_mean_rh_pct     — mean relative humidity (%)
    out_cost_total_usd_m2 — monthly operating cost ($/m²)
"""

import json
import os
import pickle
import sys
import warnings

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import KFold, cross_val_predict, train_test_split
from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import xgboost as xgb

warnings.filterwarnings("ignore")

# ── Paths ─────────────────────────────────────────────────────────────────────
project_dir = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
DATA_CSV    = os.path.join(project_dir, "data", "training_data.csv")
MODELS_DIR  = os.path.join(project_dir, "models")
MODEL_PKL   = os.path.join(MODELS_DIR, "nn_surrogate.pkl")
META_JSON   = os.path.join(MODELS_DIR, "nn_surrogate_meta.json")

os.makedirs(MODELS_DIR, exist_ok=True)

# ── Feature / target config ───────────────────────────────────────────────────
RAW_FEATURES = [
    "in_tSpDay",        # daytime heating setpoint (°C)
    "in_tSpNight",      # nighttime heating setpoint (°C)
    "in_thetaLampMax",  # max supplemental lighting (W/m²)
    "in_heatDeadZone",  # dead zone before fan kicks in (°C)
    "in_rhMax",         # humidity cap (%)
    "start_month",      # 1–12, encoded as sin/cos
]

TARGETS = [
    "out_mean_tAir_C",        # thermal comfort
    "out_mean_rh_pct",        # humidity control
    "out_cost_total_usd_m2",  # operating cost
]

TARGET_LABELS = {
    "out_mean_tAir_C":        "Mean air temp (°C)",
    "out_mean_rh_pct":        "Mean RH (%)",
    "out_cost_total_usd_m2":  "Total cost ($/m²)",
}

N_CV_FOLDS  = 5
RANDOM_SEED = 42


# ── Feature engineering ───────────────────────────────────────────────────────

def engineer_features(X: pd.DataFrame) -> pd.DataFrame:
    """
    Replace raw start_month (1-12) with cyclical (sin, cos) encoding.
    Months 12 and 1 are thermally adjacent — linear encoding treats them as 11 apart.
    """
    X = X.copy()
    m = X["start_month"]
    X["month_sin"] = np.sin(2 * np.pi * m / 12)
    X["month_cos"] = np.cos(2 * np.pi * m / 12)
    return X.drop(columns=["start_month"])


# ── Data loading ──────────────────────────────────────────────────────────────

def load_data():
    if not os.path.exists(DATA_CSV):
        sys.exit(f"ERROR: training data not found at {DATA_CSV}\n"
                 "       Run scripts/parameter_sweep.py first.")

    df = pd.read_csv(DATA_CSV)

    missing = [c for c in RAW_FEATURES + TARGETS if c not in df.columns]
    if missing:
        sys.exit(f"ERROR: missing columns: {missing}")

    mask = df[TARGETS].replace([np.inf, -np.inf], np.nan).notna().all(axis=1)
    dropped = (~mask).sum()
    if dropped:
        print(f"  Dropped {dropped} rows with invalid target values")

    df = df[mask].reset_index(drop=True)
    X = engineer_features(df[RAW_FEATURES])
    y = df[TARGETS]
    print(f"  Loaded {len(df)} rows × {len(X.columns)} features → {len(TARGETS)} targets")
    return X, y


# ── Model ─────────────────────────────────────────────────────────────────────

def build_pipeline() -> Pipeline:
    """
    XGBoost wrapped in MultiOutputRegressor (one XGB tree per target) + StandardScaler.

    n_estimators=300, lr=0.05, max_depth=4 matched the settings used in compare_models.py
    where XGBoost reached avg CV R²=0.9009 vs MLP's 0.8448.
    """
    xgb_base = xgb.XGBRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=4,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=RANDOM_SEED,
        verbosity=0,
        n_jobs=-1,
    )
    return Pipeline([
        ("scaler", StandardScaler()),
        ("model",  MultiOutputRegressor(xgb_base, n_jobs=1)),
    ])


# ── Evaluation ────────────────────────────────────────────────────────────────

def evaluate(pipeline, X_test, y_test) -> dict:
    y_pred = pipeline.predict(X_test)
    metrics = {}
    for i, col in enumerate(TARGETS):
        mae = mean_absolute_error(y_test.iloc[:, i], y_pred[:, i])
        r2  = r2_score(y_test.iloc[:, i], y_pred[:, i])
        metrics[col] = {"mae": round(float(mae), 4), "r2": round(float(r2), 4)}
    return metrics


def cross_validate(pipeline, X, y) -> dict:
    """
    5-fold CV using cross_val_predict on full multi-output y.
    Avoids the 1D/2D mismatch that breaks cross_val_score with MultiOutputRegressor.
    """
    kf = KFold(n_splits=N_CV_FOLDS, shuffle=True, random_state=RANDOM_SEED)

    fold_r2s = {col: [] for col in TARGETS}
    for train_idx, val_idx in kf.split(X):
        pipe_fold = clone(pipeline)
        pipe_fold.fit(X.iloc[train_idx], y.iloc[train_idx])
        y_pred = pipe_fold.predict(X.iloc[val_idx])
        for i, col in enumerate(TARGETS):
            fold_r2s[col].append(
                r2_score(y.iloc[val_idx, i], y_pred[:, i])
            )

    cv_scores = {}
    for col in TARGETS:
        scores = np.array(fold_r2s[col])
        cv_scores[col] = {
            "cv_r2_mean": round(float(scores.mean()), 4),
            "cv_r2_std":  round(float(scores.std()), 4),
        }
    return cv_scores


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("── GreenByte surrogate trainer (XGBoost) ───────────────────────────")

    print("\n[1/4] Loading data...")
    X, y = load_data()
    features = list(X.columns)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.15, random_state=RANDOM_SEED
    )
    print(f"  Train: {len(X_train)}  Test: {len(X_test)}")

    print("\n[2/4] Training XGBoost (300 estimators × 3 targets)...")
    pipeline = build_pipeline()
    pipeline.fit(X_train, y_train)
    n_estimators = pipeline.named_steps["model"].estimators_[0].n_estimators
    print(f"  Done — {n_estimators} trees per target")

    print(f"\n[3/4] Evaluating on test set (n={len(X_test)})...")
    test_metrics = evaluate(pipeline, X_test, y_test)
    for col, m in test_metrics.items():
        print(f"  {TARGET_LABELS[col]:<25}  MAE={m['mae']:.4f}   R²={m['r2']:.4f}")

    print(f"\n[4/4] {N_CV_FOLDS}-fold cross-validation...")
    cv_metrics = cross_validate(pipeline, X, y)
    for col, m in cv_metrics.items():
        print(f"  {TARGET_LABELS[col]:<25}  CV R²={m['cv_r2_mean']:.4f} ± {m['cv_r2_std']:.4f}")

    # Save model
    with open(MODEL_PKL, "wb") as f:
        pickle.dump(pipeline, f)
    print(f"\n  Saved model → {MODEL_PKL}")

    # Feature importances (avg across the 3 per-target trees)
    estimators = pipeline.named_steps["model"].estimators_
    importances = np.mean([e.feature_importances_ for e in estimators], axis=0)
    imp_dict = dict(zip(features, importances.tolist()))

    # Save metadata
    meta = {
        "model_type":    "XGBoost (MultiOutputRegressor)",
        "features":      features,
        "raw_features":  RAW_FEATURES,
        "targets":       TARGETS,
        "n_estimators":  n_estimators,
        "n_train":       int(len(X_train)),
        "n_test":        int(len(X_test)),
        "n_total":       int(len(X)),
        "test_metrics":  test_metrics,
        "cv_metrics":    cv_metrics,
        "feature_importances": imp_dict,
        "training_data": DATA_CSV,
    }
    with open(META_JSON, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"  Saved metadata → {META_JSON}")

    # Feature importance summary
    print("\n── Feature importances (avg across targets) ─────────────────────────")
    for feat, imp in sorted(imp_dict.items(), key=lambda x: -x[1]):
        bar = "█" * int(imp * 40)
        print(f"  {feat:<22}  {imp:.4f}  {bar}")

    # Sanity check
    print("\n── Sanity check: spring daytime scenario (April, 66.5°F setpoint) ──")
    m = 4
    sample = pd.DataFrame([{
        "in_tSpDay":        19.17,
        "in_tSpNight":      17.17,
        "in_thetaLampMax":   0.0,
        "in_heatDeadZone":   5.0,
        "in_rhMax":         85.0,
        "month_sin":        np.sin(2 * np.pi * m / 12),
        "month_cos":        np.cos(2 * np.pi * m / 12),
    }])
    pred = pipeline.predict(sample)[0]
    for col, val in zip(TARGETS, pred):
        print(f"  {TARGET_LABELS[col]:<25}  {val:.3f}")

    print("\n── Done ─────────────────────────────────────────────────────────────")


if __name__ == "__main__":
    main()
