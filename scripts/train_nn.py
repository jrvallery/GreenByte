"""
GreenLight/scripts/train_nn.py

Train a neural network surrogate model on the parameter sweep dataset.
The NN replaces GreenLight inference at query time: given control settings
and start month, it predicts steady-state mean temperature, humidity, and
monthly cost — fast enough to drive a real-time optimizer or Iris planner.

Usage:
    python scripts/train_nn.py

Outputs (all in models/):
    nn_surrogate.pkl         — trained sklearn Pipeline (scaler + MLPRegressor)
    nn_surrogate_meta.json   — training metadata (features, targets, metrics)

Inputs:
    data/training_data.csv   — 500-row LHS sweep produced by parameter_sweep.py

Features (6):
    in_tSpDay, in_tSpNight, in_thetaLampMax, in_heatDeadZone, in_rhMax, start_month

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
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.multioutput import MultiOutputRegressor

warnings.filterwarnings("ignore", category=UserWarning)

# ── Paths ─────────────────────────────────────────────────────────────────────
project_dir = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
DATA_CSV    = os.path.join(project_dir, "data", "training_data.csv")
MODELS_DIR  = os.path.join(project_dir, "models")
MODEL_PKL   = os.path.join(MODELS_DIR, "nn_surrogate.pkl")
META_JSON   = os.path.join(MODELS_DIR, "nn_surrogate_meta.json")

os.makedirs(MODELS_DIR, exist_ok=True)

# ── Feature / target config ───────────────────────────────────────────────────
FEATURES = [
    "in_tSpDay",        # daytime heating setpoint (°C)
    "in_tSpNight",      # nighttime heating setpoint (°C)
    "in_thetaLampMax",  # max supplemental lighting (W/m²)
    "in_heatDeadZone",  # dead zone before fan kicks in (°C)
    "in_rhMax",         # humidity cap (%)
    "start_month",      # 1–12 (seasonal forcing)
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

# ── Network architecture ───────────────────────────────────────────────────────
# Two hidden layers — wide enough to capture seasonal nonlinearity
# without overfitting on 500 samples.
HIDDEN_LAYER_SIZES = (128, 64)
MAX_ITER = 5000
RANDOM_SEED = 42


def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    if not os.path.exists(DATA_CSV):
        sys.exit(f"ERROR: training data not found at {DATA_CSV}\n"
                 "       Run scripts/parameter_sweep.py first.")

    df = pd.read_csv(DATA_CSV)

    missing_cols = [c for c in FEATURES + TARGETS if c not in df.columns]
    if missing_cols:
        sys.exit(f"ERROR: missing columns in training data: {missing_cols}")

    # Drop any rows where targets are NaN or inf (failed sims)
    mask = df[TARGETS].replace([np.inf, -np.inf], np.nan).notna().all(axis=1)
    dropped = (~mask).sum()
    if dropped:
        print(f"  Dropped {dropped} rows with invalid target values")

    df = df[mask].reset_index(drop=True)
    print(f"  Loaded {len(df)} rows × {len(FEATURES)} features → {len(TARGETS)} targets")

    X = df[FEATURES]
    y = df[TARGETS]
    return X, y


def build_pipeline() -> Pipeline:
    mlp = MLPRegressor(
        hidden_layer_sizes=HIDDEN_LAYER_SIZES,
        activation="relu",
        solver="adam",
        alpha=1e-4,          # L2 regularization
        batch_size="auto",
        learning_rate_init=5e-4,
        max_iter=MAX_ITER,
        random_state=RANDOM_SEED,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=50,
        tol=1e-6,
    )
    return Pipeline([
        ("scaler", StandardScaler()),
        ("mlp",    mlp),
    ])


def evaluate(pipeline: Pipeline, X_test: pd.DataFrame, y_test: pd.DataFrame) -> dict:
    y_pred = pipeline.predict(X_test)
    metrics = {}
    for i, col in enumerate(TARGETS):
        mae = mean_absolute_error(y_test.iloc[:, i], y_pred[:, i])
        r2  = r2_score(y_test.iloc[:, i], y_pred[:, i])
        metrics[col] = {"mae": round(float(mae), 4), "r2": round(float(r2), 4)}
    return metrics


def cross_validate(pipeline: Pipeline, X: pd.DataFrame, y: pd.DataFrame) -> dict:
    """5-fold CV R² per target (uses separate Pipeline clone per fold)."""
    cv_scores = {}
    for col in TARGETS:
        from sklearn.base import clone
        pipe_clone = clone(pipeline)
        scores = cross_val_score(pipe_clone, X, y[col], cv=5, scoring="r2")
        cv_scores[col] = {
            "cv_r2_mean": round(float(scores.mean()), 4),
            "cv_r2_std":  round(float(scores.std()), 4),
        }
    return cv_scores


def main():
    print("── GreenLight NN surrogate trainer ─────────────────────────────────")

    # 1. Load data
    print("\n[1/4] Loading data...")
    X, y = load_data()

    # 2. Train / test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.15, random_state=RANDOM_SEED
    )
    print(f"  Train: {len(X_train)}  Test: {len(X_test)}")

    # 3. Train
    print("\n[2/4] Training MLP (hidden={}, max_iter={})...".format(
        HIDDEN_LAYER_SIZES, MAX_ITER))
    pipeline = build_pipeline()
    pipeline.fit(X_train, y_train)

    mlp = pipeline.named_steps["mlp"]
    print(f"  Converged: {mlp.n_iter_ < MAX_ITER}  |  Iterations: {mlp.n_iter_}  "
          f"|  Train loss: {mlp.loss_:.6f}")

    # 4. Evaluate on held-out test set
    print("\n[3/4] Evaluating on test set (n={})...".format(len(X_test)))
    test_metrics = evaluate(pipeline, X_test, y_test)
    for col, m in test_metrics.items():
        label = TARGET_LABELS[col]
        print(f"  {label:<25}  MAE={m['mae']:.4f}   R²={m['r2']:.4f}")

    # 5. Cross-validation
    print("\n[4/4] 5-fold cross-validation...")
    cv_metrics = cross_validate(pipeline, X, y)
    for col, m in cv_metrics.items():
        label = TARGET_LABELS[col]
        print(f"  {label:<25}  CV R²={m['cv_r2_mean']:.4f} ± {m['cv_r2_std']:.4f}")

    # 6. Save model
    with open(MODEL_PKL, "wb") as f:
        pickle.dump(pipeline, f)
    print(f"\n  Saved model → {MODEL_PKL}")

    # 7. Save metadata
    meta = {
        "features":     FEATURES,
        "targets":      TARGETS,
        "hidden_layers": list(HIDDEN_LAYER_SIZES),
        "n_train":      int(len(X_train)),
        "n_test":       int(len(X_test)),
        "n_total":      int(len(X)),
        "n_iter":       int(mlp.n_iter_),
        "train_loss": float(mlp.loss_),
        "test_metrics": test_metrics,
        "cv_metrics":   cv_metrics,
        "training_data": DATA_CSV,
    }
    with open(META_JSON, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"  Saved metadata → {META_JSON}")

    # 8. Quick sanity-check prediction
    print("\n── Sanity check: predict spring daytime setpoint scenario ──────────")
    sample = pd.DataFrame([{
        "in_tSpDay":       19.17,   # 66.5°F — calibrated setpoint
        "in_tSpNight":     17.17,   # 63°F
        "in_thetaLampMax":  0.0,    # no grow lights
        "in_heatDeadZone":  5.0,    # spring/winter fans
        "in_rhMax":        85.0,    # standard humidity cap
        "start_month":      4,      # April
    }])
    pred = pipeline.predict(sample)[0]
    for col, val in zip(TARGETS, pred):
        print(f"  {TARGET_LABELS[col]:<25}  {val:.3f}")

    print("\n── Done ─────────────────────────────────────────────────────────────")


if __name__ == "__main__":
    main()
