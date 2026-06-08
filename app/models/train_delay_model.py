"""
RailMind AI — Delay Prediction Model: Training Script
======================================================
Run once to fit an XGBRegressor on the delay dataset and persist
the model + encoders + metadata to saved_models/.

Usage
-----
    # From the project root (railmind-backend/)
    python -m app.models.train_delay_model

Output
------
    saved_models/delay_xgb_model.joblib
    saved_models/delay_encoders.joblib
    saved_models/delay_model_info.json
"""

import json
import os
import sys
from datetime import datetime, timezone

import joblib
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor

# ---------------------------------------------------------------------------
# Path resolution — works whether run as `python -m app.models.train_delay_model`
# from the project root OR directly as a script.
# ---------------------------------------------------------------------------
_THIS_DIR   = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_THIS_DIR, "..", ".."))

# Add project root to sys.path so `data.process_delay_data` is importable
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

DATA_CSV          = os.path.join(_PROJECT_ROOT, "data", "train_delays.csv")
SAVED_MODELS_DIR  = os.path.join(_PROJECT_ROOT, "saved_models")
MODEL_PATH        = os.path.join(SAVED_MODELS_DIR, "delay_xgb_model.joblib")
ENCODERS_PATH     = os.path.join(SAVED_MODELS_DIR, "delay_encoders.joblib")
MODEL_INFO_PATH   = os.path.join(SAVED_MODELS_DIR, "delay_model_info.json")

# ---------------------------------------------------------------------------
# Imports from the data pipeline
# ---------------------------------------------------------------------------
from data.process_delay_data import (          # noqa: E402
    engineer_features,
    get_feature_columns,
    load_and_clean,
    prepare_training_data,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _banner(msg: str) -> None:
    width = 60
    print(f"\n{'─' * width}")
    print(f"  {msg}")
    print(f"{'─' * width}")


# ---------------------------------------------------------------------------
# Main training routine
# ---------------------------------------------------------------------------
def train() -> None:
    os.makedirs(SAVED_MODELS_DIR, exist_ok=True)

    # ── 1. Load & clean ───────────────────────────────────────────────────
    _banner("Step 1 / 4 — Loading and cleaning data")
    if not os.path.exists(DATA_CSV):
        print(f"ERROR: Data file not found: {DATA_CSV}")
        print("       Run  python data/generate_synthetic_delay_data.py  first.")
        sys.exit(1)

    df = load_and_clean(DATA_CSV)
    print(f"  Rows after cleaning : {len(df):,}")

    # ── 2. Feature engineering ────────────────────────────────────────────
    _banner("Step 2 / 4 — Engineering features")
    df, encoders = engineer_features(df)
    print(f"  Feature columns : {get_feature_columns()}")

    # ── 3. Train / test split → XGBoost ──────────────────────────────────
    _banner("Step 3 / 4 — Training XGBRegressor (80 / 20 split)")
    X, y = prepare_training_data(df, encoders)
    n_samples = len(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42
    )
    print(f"  Train rows : {len(X_train):,}")
    print(f"  Test  rows : {len(X_test):,}")

    model = XGBRegressor(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        eval_metric="mae",
        verbosity=0,
    )
    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=False,
    )

    # ── 4. Evaluate ───────────────────────────────────────────────────────
    _banner("Step 4 / 4 — Evaluating on held-out test set")
    y_pred = model.predict(X_test)

    mae  = float(mean_absolute_error(y_test, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
    r2   = float(r2_score(y_test, y_pred))

    print(f"\n  Test MAE  : {mae:.2f} minutes")
    print(f"  Test RMSE : {rmse:.2f} minutes")
    print(f"  R² Score  : {r2:.4f}")

    # Feature importance
    feat_cols  = get_feature_columns()
    importances = model.feature_importances_
    print("\n  Feature importances:")
    for feat, imp in sorted(zip(feat_cols, importances), key=lambda x: -x[1]):
        bar = "█" * int(imp * 50)
        print(f"    {feat:<22s}  {imp:.4f}  {bar}")

    # ── 5. Persist artefacts ──────────────────────────────────────────────
    joblib.dump(model,    MODEL_PATH)
    joblib.dump(encoders, ENCODERS_PATH)

    model_info = {
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "mae":        round(mae,  4),
        "rmse":       round(rmse, 4),
        "r2":         round(r2,   4),
        "n_samples":  n_samples,
        "n_train":    len(X_train),
        "n_test":     len(X_test),
        "features":   feat_cols,
        "model_params": {
            "n_estimators":    300,
            "max_depth":       6,
            "learning_rate":   0.05,
            "subsample":       0.8,
            "colsample_bytree": 0.8,
        },
    }
    with open(MODEL_INFO_PATH, "w") as f:
        json.dump(model_info, f, indent=2)

    print(f"\n✅  Model saved   → {MODEL_PATH}")
    print(f"✅  Encoders saved → {ENCODERS_PATH}")
    print(f"✅  Info saved     → {MODEL_INFO_PATH}")
    print()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    train()
