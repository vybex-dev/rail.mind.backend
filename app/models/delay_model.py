"""
RailMind AI — Delay Prediction Inference Module
================================================
Provides a singleton `delay_predictor` that the FastAPI routers import.

Instantiation tries to load saved_models/delay_xgb_model.joblib; if the
files are absent it silently falls back to a rule-based mock so the API
stays functional before the first training run.

Usage
-----
    from app.models.delay_model import delay_predictor

    result = delay_predictor.predict(
        train_number="12301",
        source="HWH",
        destination="NDLS",
        hour=8,
        day_of_week=1,
        month=7,
    )
"""

import json
import os
import random
import sys
from typing import Optional

import numpy as np

# ---------------------------------------------------------------------------
# Path setup — works whether imported from FastAPI or run standalone
# ---------------------------------------------------------------------------
_THIS_DIR     = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_THIS_DIR, "..", ".."))

if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

# Canonical paths
_MODELS_DIR       = os.path.join(_PROJECT_ROOT, "saved_models")
_MODEL_PATH       = os.path.join(_MODELS_DIR, "delay_xgb_model.joblib")
_ENCODERS_PATH    = os.path.join(_MODELS_DIR, "delay_encoders.joblib")
_MODEL_INFO_PATH  = os.path.join(_MODELS_DIR, "delay_model_info.json")
_TRAINS_JSON_PATH = os.path.join(_PROJECT_ROOT, "data", "sample_trains.json")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
PEAK_HOURS     = {7, 8, 9, 17, 18, 19, 20}
MONSOON_MONTHS = {6, 7, 8, 9}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _delay_confidence(delay: float) -> str:
    """Confidence tier based on predicted magnitude."""
    if delay < 15:
        return "high"
    if delay < 45:
        return "medium"
    return "low"


def _delay_category(delay: float) -> str:
    """Human-readable delay bucket."""
    if delay <= 5:
        return "on_time"
    if delay <= 20:
        return "slight"
    if delay <= 60:
        return "moderate"
    return "severe"


def _safe_encode(encoder, value: str) -> int:
    """Encode `value` with a fitted LabelEncoder; return 0 for unseen labels."""
    classes = list(encoder.classes_)
    if value in classes:
        return int(encoder.transform([value])[0])
    return 0


# ---------------------------------------------------------------------------
# DelayPredictor
# ---------------------------------------------------------------------------
class DelayPredictor:
    """
    Wraps the trained XGBoost model for online delay prediction.

    Falls back to `get_mock_prediction()` when the model files are absent,
    so the backend remains usable during development or before training.
    """

    def __init__(self) -> None:
        self.model     = None
        self.encoders  = None
        self.model_info: Optional[dict] = None
        self.is_loaded  = False

        self._try_load()

    # ------------------------------------------------------------------
    # Load
    # ------------------------------------------------------------------
    def _try_load(self) -> None:
        """Attempt to load persisted model artefacts."""
        if not (os.path.exists(_MODEL_PATH) and os.path.exists(_ENCODERS_PATH)):
            print("DelayPredictor: using mock mode  "
                  "(run `python -m app.models.train_delay_model` to train)")
            return

        try:
            import joblib  # only needed at inference time

            self.model    = joblib.load(_MODEL_PATH)
            self.encoders = joblib.load(_ENCODERS_PATH)

            if os.path.exists(_MODEL_INFO_PATH):
                with open(_MODEL_INFO_PATH) as f:
                    self.model_info = json.load(f)

            self.is_loaded = True
            mae_str = (f"  MAE={self.model_info['mae']:.2f} min"
                       if self.model_info else "")
            print(f"DelayPredictor: model loaded ✅{mae_str}")

        except Exception as exc:  # pragma: no cover
            print(f"DelayPredictor: load failed ({exc}); using mock mode")
            self.model    = None
            self.encoders = None
            self.is_loaded = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def predict(
        self,
        train_number: str,
        source: str,
        destination: str,
        hour: int,
        day_of_week: int,
        month: int,
    ) -> dict:
        """
        Predict arrival delay for a single journey.

        Parameters
        ----------
        train_number : str   e.g. "12301"
        source       : str   Station code, e.g. "HWH"
        destination  : str   Station code, e.g. "NDLS"
        hour         : int   0-23
        day_of_week  : int   0 (Monday) – 6 (Sunday)
        month        : int   1-12

        Returns
        -------
        dict  with keys: predicted_delay_minutes, confidence, delay_category,
                         train_number, route, is_mock
        """
        source      = source.strip().upper()
        destination = destination.strip().upper()
        route_str   = f"{source} → {destination}"

        if self.is_loaded:
            return self._real_predict(
                train_number, source, destination, route_str,
                hour, day_of_week, month
            )
        return self.get_mock_prediction(train_number, source, destination,
                                        route_str, hour, month)

    # ------------------------------------------------------------------
    # Real prediction (model loaded)
    # ------------------------------------------------------------------
    def _real_predict(
        self,
        train_number: str,
        source: str,
        destination: str,
        route_str: str,
        hour: int,
        day_of_week: int,
        month: int,
    ) -> dict:
        # --- encode categorical inputs --------------------------------
        route_key    = f"{source}_to_{destination}"
        route_enc    = _safe_encode(self.encoders["route"], route_key)
        train_enc    = _safe_encode(self.encoders["train"], str(train_number))

        # --- derived binary features ----------------------------------
        is_weekend   = int(day_of_week >= 5)
        is_peak_hour = int(hour in PEAK_HOURS)

        # --- build feature vector (must match get_feature_columns()) --
        # ['hour_of_day', 'day_of_week', 'month', 'is_weekend',
        #  'is_peak_hour', 'route_encoded', 'train_encoded']
        features = np.array([[
            hour, day_of_week, month, is_weekend,
            is_peak_hour, route_enc, train_enc,
        ]], dtype=np.float32)

        # --- inference ------------------------------------------------
        raw_pred = float(self.model.predict(features)[0])
        delay    = round(float(np.clip(raw_pred, 0.0, 300.0)), 1)

        return {
            "predicted_delay_minutes": delay,
            "confidence":   _delay_confidence(delay),
            "delay_category": _delay_category(delay),
            "train_number": train_number,
            "route":        route_str,
            "is_mock":      False,
        }

    # ------------------------------------------------------------------
    # Mock prediction (model not loaded)
    # ------------------------------------------------------------------
    def get_mock_prediction(
        self,
        train_number: str,
        source: str,
        destination: str,
        route_str: str,
        hour: int,
        month: int,
    ) -> dict:
        """
        Rule-based mock that returns plausible delays without a trained model.

        Rules
        -----
        - Peak hour (7-9, 17-20)  →  delay ∈ [25, 55]
        - Off-peak                →  delay ∈ [5,  20]
        - Monsoon months (6-9)    →  add  10-15 min
        """
        if hour in PEAK_HOURS:
            base_delay = random.uniform(25.0, 55.0)
        else:
            base_delay = random.uniform(5.0, 20.0)

        if month in MONSOON_MONTHS:
            base_delay += random.uniform(10.0, 15.0)

        delay = round(min(float(base_delay), 300.0), 1)

        return {
            "predicted_delay_minutes": delay,
            "confidence":   _delay_confidence(delay),
            "delay_category": _delay_category(delay),
            "train_number": train_number,
            "route":        route_str,
            "is_mock":      True,
        }

    # ------------------------------------------------------------------
    # Catalogue helpers
    # ------------------------------------------------------------------
    def get_all_trains(self) -> list:
        """Return the 15-train catalogue from data/sample_trains.json."""
        if not os.path.exists(_TRAINS_JSON_PATH):
            return []
        with open(_TRAINS_JSON_PATH) as f:
            return json.load(f)

    def get_model_info(self) -> Optional[dict]:
        """Return training metadata if the model is loaded, else None."""
        return self.model_info


# ---------------------------------------------------------------------------
# Module-level singleton — imported by routers
# ---------------------------------------------------------------------------
delay_predictor = DelayPredictor()
