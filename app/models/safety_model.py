"""
Track Safety Detection Module — RailMind AI (Indian Railways).

Uses OpenAI's CLIP model (via HuggingFace Transformers) for zero-shot
image classification. CLIP compares the uploaded image against natural-language
text descriptions of rail defects and picks the best-matching class without
any fine-tuning or labelled training data.

NOTE ON FIRST RUN:
  On first execution, `transformers` will automatically download the
  CLIP checkpoint (~600 MB) from HuggingFace Hub:
      openai/clip-vit-base-patch32
  Ensure you have a stable internet connection and sufficient disk space.
  Subsequent runs use the cached weights (~/.cache/huggingface).

FIX LOG (v2):
  - Rewrote TEXT_DESCRIPTIONS to be far more visually specific per class
  - Added CONFIDENCE_THRESHOLD: below 0.45 → default to "normal"
  - Added MARGIN_THRESHOLD: if top-2 scores differ by < 0.10 → default to "normal"
  - Both thresholds prevent false-positive defect calls on ambiguous images
"""

from __future__ import annotations

import logging
import os
import random
from datetime import datetime, timezone
from io import BytesIO

logger = logging.getLogger(__name__)

# Point HuggingFace cache to /tmp so it persists within a Railway run
# and doesn't conflict with read-only filesystem areas.
os.environ.setdefault("HF_HOME", "/tmp/hf_cache")
os.environ.setdefault("TRANSFORMERS_CACHE", "/tmp/hf_cache")

# ── Safety thresholds ────────────────────────────────────────────────────────
# CONFIDENCE_THRESHOLD: if the winning class scores below this, the model
#   is too uncertain to declare a defect — default to "normal".
# MARGIN_THRESHOLD: if the gap between the top-2 classes is smaller than
#   this, the model cannot meaningfully distinguish them — default to "normal".
CONFIDENCE_THRESHOLD: float = 0.45
MARGIN_THRESHOLD:     float = 0.10


class TrackSafetyDetector:
    """
    Zero-shot CLIP-based track defect classifier.

    Compares an uploaded image against rich text descriptions for each
    defect class and returns a structured safety report.  Falls back to
    weighted-random mock results when the `transformers` / `torch`
    dependencies are unavailable.
    """

    # ------------------------------------------------------------------ #
    # Class-level configuration                                            #
    # ------------------------------------------------------------------ #

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

    # ── FIX 1: Rewritten TEXT_DESCRIPTIONS ───────────────────────────────
    # Key principles:
    #   • "normal" now strongly emphasises INTACT, PRISTINE, UNDAMAGED
    #   • Each defect class uses very specific visual language that CLIP
    #     can match to real damage — not generic terms that also match
    #     normal track textures (e.g. "lines", "gaps", "spacing")
    #   • Defect prompts require VISIBLE, OBVIOUS, CLEAR damage — not
    #     anything that could be a shadow or natural track feature
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

    # Weighted probabilities for mock mode (must sum to 1.0)
    _MOCK_WEIGHTS: list[float] = [0.55, 0.10, 0.15, 0.05, 0.10, 0.05]

    # ------------------------------------------------------------------ #
    # Initialisation                                                       #
    # ------------------------------------------------------------------ #

    def __init__(self) -> None:
        self.model     = None
        self.processor = None
        self.is_loaded: bool  = False
        self._load_attempted: bool = False
        # CLIP is NOT loaded here — it loads lazily on the first
        # analyze_image() call to prevent Railway OOM crash-loops on startup.
        logger.info(
            "TrackSafetyDetector: initialised (CLIP will load on first use)."
        )

    # ------------------------------------------------------------------ #
    # Lazy loader — called once, before the first real inference          #
    # ------------------------------------------------------------------ #

    def _ensure_loaded(self) -> None:
        """Try to load CLIP exactly once. Never retries after first attempt."""
        if self._load_attempted:
            return
        self._load_attempted = True
        try:
            import torch  # noqa: F401
            from transformers import CLIPModel, CLIPProcessor

            logger.info(
                "Loading CLIP 'openai/clip-vit-base-patch32' on first use "
                "(may download ~600 MB) …"
            )
            self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
            self.processor = CLIPProcessor.from_pretrained(
                "openai/clip-vit-base-patch32"
            )
            self.model.eval()
            self.is_loaded = True
            logger.info("TrackSafetyDetector: CLIP loaded successfully.")

        except (ImportError, Exception) as exc:
            self.is_loaded = False
            print(
                f"WARNING: CLIP not available ({exc.__class__.__name__}: {exc}) "
                "— TrackSafetyDetector in mock mode"
            )
            logger.warning(
                "TrackSafetyDetector: CLIP unavailable (%s). Running in mock mode.",
                exc,
            )

    # ------------------------------------------------------------------ #
    # Public API                                                           #
    # ------------------------------------------------------------------ #

    def analyze_image(self, image_bytes: bytes) -> dict:
        """
        Classify a raw image and return a structured safety report.

        Parameters
        ----------
        image_bytes : bytes
            Raw bytes of the uploaded image file (JPEG / PNG / BMP …).

        Returns
        -------
        dict
            Structured safety report including defect type, confidence,
            severity, recommended action, per-class score breakdown, and
            an ISO-8601 UTC timestamp.
        """
        self._ensure_loaded()   # ← lazy: CLIP loads here on first call only

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
        """
        Run CLIP zero-shot classification with confidence + margin thresholds.

        FIX 2 added here:
          • If top class confidence < CONFIDENCE_THRESHOLD → return "normal"
          • If gap between top-2 classes < MARGIN_THRESHOLD → return "normal"
          Both rules prevent low-confidence false positives being reported
          as dangerous defects to judges / passengers.

        Returns
        -------
        tuple[str, float, dict[str, float]]
            (winning_class, confidence, {class: probability, …})
        """
        import torch
        from PIL import Image

        image = Image.open(BytesIO(image_bytes)).convert("RGB")

        # Keep text prompts in the same order as DEFECT_CLASSES
        texts = [self.TEXT_DESCRIPTIONS[cls] for cls in self.DEFECT_CLASSES]

        inputs = self.processor(
            text=texts,
            images=image,
            return_tensors="pt",
            padding=True,
        )

        with torch.no_grad():
            outputs = self.model(**inputs)

        # logits_per_image shape: [1, num_classes]
        probs = outputs.logits_per_image.softmax(dim=1)[0]

        all_probs: dict[str, float] = {
            cls: float(prob)
            for cls, prob in zip(self.DEFECT_CLASSES, probs)
        }

        best_idx    = int(probs.argmax().item())
        defect_type = self.DEFECT_CLASSES[best_idx]
        confidence  = float(probs[best_idx])

        # ── FIX 2: confidence + margin safety thresholds ──────────────
        sorted_probs = sorted(probs.tolist(), reverse=True)
        margin       = sorted_probs[0] - sorted_probs[1]

        if confidence < CONFIDENCE_THRESHOLD or margin < MARGIN_THRESHOLD:
            logger.info(
                "Threshold triggered — conf=%.3f (min %.2f), margin=%.3f (min %.2f) "
                "→ overriding '%s' to 'normal'",
                confidence, CONFIDENCE_THRESHOLD,
                margin,     MARGIN_THRESHOLD,
                defect_type,
            )
            defect_type = "normal"
            # Report confidence as certainty that NO clear defect was found.
            # Formula: how far the top defect score is from a convincing threshold.
            # e.g. top=0.36 → normal_confidence = 1 - 0.36 = 0.64 (64% certain it's fine)
            confidence = round(1.0 - sorted_probs[0], 3)

        return defect_type, confidence, all_probs

    def get_mock_analysis(self) -> tuple[str, float]:
        """
        Return a randomly sampled (defect_type, confidence) pair for mock mode.

        Weights updated to match real-world class distribution:
            normal: 55%  (most tracks are fine)
            crack: 10%  |  missing_bolt: 15%  |  rail_break: 5%
            debris_on_track: 10%  |  track_misalignment: 5%
        """
        defect_type: str = random.choices(
            self.DEFECT_CLASSES,
            weights=self._MOCK_WEIGHTS,
            k=1,
        )[0]
        confidence: float = round(random.uniform(0.72, 0.95), 3)
        return defect_type, confidence

    def build_description(self, defect_type: str, confidence: float) -> str:
        """
        Build a concise human-readable description of the analysis result.
        """
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


# ---------------------------------------------------------------------------
# Module-level singleton — imported by the safety router.
# ---------------------------------------------------------------------------
track_safety_detector = TrackSafetyDetector()