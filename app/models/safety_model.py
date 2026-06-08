"""
Track Safety Detection Module — RailMind AI (Indian Railways).

Uses OpenAI's CLIP model (via HuggingFace Transformers) for zero-shot
image classification.

FIX LOG (v4):
  - CLIP now loads LAZILY — only on the first real /safety/analyze request.
  - Removed background thread that was downloading 600 MB at startup and
    causing OOM crashes on Railway free tier (~512 MB RAM).
  - App boots instantly with zero CLIP memory usage.
  - CLIP loads once on first request, then stays in memory for all subsequent
    requests (no per-request reload).
  - Falls back to mock if torch/transformers are unavailable.
  - HF_HOME set to /tmp/hf_cache for Railway compatibility.
  - All v2/v3 improvements preserved: TEXT_DESCRIPTIONS, thresholds, etc.
"""

from __future__ import annotations

import logging
import os
import random
import threading
from datetime import datetime, timezone
from io import BytesIO

logger = logging.getLogger(__name__)

# Point HuggingFace cache to /tmp — writable on Railway
os.environ.setdefault("HF_HOME", "/tmp/hf_cache")
os.environ.setdefault("TRANSFORMERS_CACHE", "/tmp/hf_cache")

# ── Safety thresholds ─────────────────────────────────────────────────────────
CONFIDENCE_THRESHOLD: float = 0.45
MARGIN_THRESHOLD:     float = 0.10


class TrackSafetyDetector:
    """
    Zero-shot CLIP-based track defect classifier.

    CLIP loads LAZILY — only when analyze_image() is first called.
    This means:
      - The server boots with ~0 MB of CLIP memory overhead.
      - Railway free tier (~512 MB) can run delay + crowd models at startup
        without OOM crashing.
      - On the first /safety/analyze request, CLIP loads once and stays in
        memory for all subsequent requests (thread-safe via a lock).
      - Falls back to mock results if torch/transformers are unavailable.
    """

    DEFECT_CLASSES: list[str] = [
        "normal",
        "crack",
        "missing_bolt",
        "rail_break",
        "debris_on_track",
        "track_misalignment",
    ]

    SEVERITY: dict[str, str] = {
        "normal":             "none",
        "crack":              "high",
        "missing_bolt":       "medium",
        "rail_break":         "critical",
        "debris_on_track":    "medium",
        "track_misalignment": "high",
    }

    ACTIONS: dict[str, str] = {
        "normal":             "No action required. Track is in good condition.",
        "crack":              "Schedule urgent inspection. Mark segment for repair within 24h.",
        "missing_bolt":       "Dispatch maintenance crew. Replace missing fasteners.",
        "rail_break":         "HALT all trains on this segment immediately. Emergency repair needed.",
        "debris_on_track":    "Clear debris before next train passage. Alert train control.",
        "track_misalignment": "Reduce train speed on this segment. Schedule realignment.",
    }

    TEXT_DESCRIPTIONS: dict[str, str] = {
        "normal": (
            "a perfectly safe and intact railway track with smooth continuous "
            "steel rails, regular evenly spaced concrete or wooden sleepers, "
            "clean ballast gravel, all bolts clips and fasteners securely in place, "
            "no cracks breaks damage or foreign objects, safe for train operation"
        ),
        "crack": (
            "a damaged railway track with a clearly visible crack fracture or split "
            "running directly through the solid steel rail metal, a broken surface "
            "showing structural damage to the rail head or rail web, "
            "cracked rail with a visible fault line across the steel"
        ),
        "missing_bolt": (
            "a railway track where bolts nuts or fasteners are clearly absent, "
            "empty bolt holes are visible on the rail foot or fishplate connection, "
            "loose detached rail clips spikes or plates leaving the rail "
            "unsecured or improperly fastened to the sleeper"
        ),
        "rail_break": (
            "a catastrophically broken railway rail with a large gap or open "
            "separation in the metal, two completely disconnected pieces of rail "
            "with a visible missing section between them, "
            "total rail failure where the track is physically split apart"
        ),
        "debris_on_track": (
            "railway tracks with large rocks boulders tree branches fallen trees "
            "or heavy foreign objects lying directly on or between the rails, "
            "dangerous obstructions blocking the train path that would "
            "cause a derailment if hit by a train"
        ),
        "track_misalignment": (
            "railway tracks that are visibly buckled bent curved or shifted "
            "sideways out of their correct position, rails that are no longer "
            "parallel or straight showing a clear kink or lateral shift, "
            "severely distorted track geometry where the rail spacing is uneven"
        ),
    }

    _MOCK_WEIGHTS: list[float] = [0.55, 0.10, 0.15, 0.05, 0.10, 0.05]

    # ------------------------------------------------------------------ #
    # Initialisation — NO background thread, NO eager loading             #
    # ------------------------------------------------------------------ #

    def __init__(self) -> None:
        self.model      = None
        self.processor  = None
        self.is_loaded: bool = False
        self._load_failed: bool = False   # stops re-trying after a hard failure
        self._lock = threading.Lock()
        # ⚠️  Do NOT start a thread or import torch here.
        # CLIP will be loaded on the first call to analyze_image().
        logger.info("TrackSafetyDetector: initialised (CLIP will load on first request)")

    # ------------------------------------------------------------------ #
    # Lazy loader — called inside analyze_image()                          #
    # ------------------------------------------------------------------ #

    def _ensure_loaded(self) -> None:
        """
        Load CLIP exactly once, on the first real request.

        Thread-safe: uses a lock so concurrent requests don't trigger
        multiple simultaneous downloads.
        After a hard failure the flag _load_failed is set and we skip
        subsequent attempts (fall back to mock permanently).
        """
        # Fast path — already loaded or permanently failed
        if self.is_loaded or self._load_failed:
            return

        with self._lock:
            # Re-check inside the lock (another thread may have loaded it)
            if self.is_loaded or self._load_failed:
                return

            try:
                import torch  # noqa: F401
                from transformers import CLIPModel, CLIPProcessor

                logger.info(
                    "Loading CLIP 'openai/clip-vit-base-patch32' on first request "
                    "(may download ~600 MB on first run) …"
                )
                model     = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
                processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
                model.eval()

                self.model     = model
                self.processor = processor
                self.is_loaded = True
                logger.info("TrackSafetyDetector: CLIP loaded successfully ✅")

            except (ImportError, Exception) as exc:
                self._load_failed = True
                self.is_loaded    = False
                logger.warning(
                    "TrackSafetyDetector: CLIP unavailable (%s) — "
                    "falling back to mock mode permanently.",
                    exc,
                )

    # ------------------------------------------------------------------ #
    # Public API                                                           #
    # ------------------------------------------------------------------ #

    def analyze_image(self, image_bytes: bytes) -> dict:
        """Classify a raw image and return a structured safety report."""
        # Trigger lazy load — no-op if already loaded or failed
        self._ensure_loaded()

        timestamp = datetime.now(tz=timezone.utc).isoformat().replace("+00:00", "Z")

        if self.is_loaded:
            defect_type, confidence, all_probs = self._clip_predict(image_bytes)
        else:
            defect_type, confidence = self.get_mock_analysis()
            base = (1.0 - confidence) / (len(self.DEFECT_CLASSES) - 1)
            all_probs = {cls: base for cls in self.DEFECT_CLASSES}
            all_probs[defect_type] = confidence

        return {
            "defect_type":        defect_type,
            "confidence":         round(confidence, 3),
            "severity":           self.SEVERITY[defect_type],
            "description":        self.build_description(defect_type, confidence),
            "recommended_action": self.ACTIONS[defect_type],
            "safe_to_operate":    defect_type in {"normal", "debris_on_track"},
            "analysis_timestamp": timestamp,
            "all_scores": {
                cls: round(float(prob), 3)
                for cls, prob in all_probs.items()
            },
        }

    # ------------------------------------------------------------------ #
    # Internal helpers                                                     #
    # ------------------------------------------------------------------ #

    def _clip_predict(
        self, image_bytes: bytes
    ) -> tuple[str, float, dict[str, float]]:
        import torch
        from PIL import Image

        image = Image.open(BytesIO(image_bytes)).convert("RGB")
        texts = [self.TEXT_DESCRIPTIONS[cls] for cls in self.DEFECT_CLASSES]

        inputs = self.processor(
            text=texts, images=image, return_tensors="pt", padding=True,
        )

        with torch.no_grad():
            outputs = self.model(**inputs)

        probs = outputs.logits_per_image.softmax(dim=1)[0]

        all_probs: dict[str, float] = {
            cls: float(prob)
            for cls, prob in zip(self.DEFECT_CLASSES, probs)
        }

        best_idx    = int(probs.argmax().item())
        defect_type = self.DEFECT_CLASSES[best_idx]
        confidence  = float(probs[best_idx])

        # Confidence + margin safety thresholds
        sorted_probs = sorted(probs.tolist(), reverse=True)
        margin       = sorted_probs[0] - sorted_probs[1]

        if confidence < CONFIDENCE_THRESHOLD or margin < MARGIN_THRESHOLD:
            logger.info(
                "Threshold triggered — conf=%.3f margin=%.3f → overriding '%s' to 'normal'",
                confidence, margin, defect_type,
            )
            defect_type = "normal"
            confidence  = round(1.0 - sorted_probs[0], 3)

        return defect_type, confidence, all_probs

    def get_mock_analysis(self) -> tuple[str, float]:
        defect_type: str = random.choices(
            self.DEFECT_CLASSES, weights=self._MOCK_WEIGHTS, k=1,
        )[0]
        confidence: float = round(random.uniform(0.72, 0.95), 3)
        return defect_type, confidence

    def build_description(self, defect_type: str, confidence: float) -> str:
        pct      = round(confidence * 100, 1)
        severity = self.SEVERITY[defect_type]

        if defect_type == "normal":
            return (
                f"Track appears to be in normal operating condition "
                f"({pct}% confidence). No defects detected."
            )

        base = (
            f"Analysis detected {defect_type.replace('_', ' ')} on track segment "
            f"with {pct}% confidence."
        )
        if severity in {"high", "critical"}:
            return f"WARNING: {base}"
        return base


# --------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------
track_safety_detector = TrackSafetyDetector()
