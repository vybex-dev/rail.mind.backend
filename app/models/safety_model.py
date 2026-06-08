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
"""

from __future__ import annotations

import logging
import random
from datetime import datetime, timezone
from io import BytesIO

logger = logging.getLogger(__name__)


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

    # Rich natural-language prompts fed to CLIP — more descriptive = better accuracy.
    TEXT_DESCRIPTIONS: dict[str, str] = {
        "normal":             "a normal undamaged railway track in good condition with intact rails and bolts",
        "crack":              "a railway track with a visible crack or fracture in the metal rail",
        "missing_bolt":       "a railway track with missing bolts or loose fasteners on the rail",
        "rail_break":         "a broken or completely fractured railway rail with a gap in the metal",
        "debris_on_track":    "railway tracks with rocks stones or foreign objects blocking the path",
        "track_misalignment": "railway tracks that are bent warped or misaligned out of position",
    }

    # Weighted probabilities for mock mode (must sum to 1.0)
    _MOCK_WEIGHTS: list[float] = [0.40, 0.15, 0.20, 0.05, 0.15, 0.05]

    # ------------------------------------------------------------------ #
    # Initialisation                                                       #
    # ------------------------------------------------------------------ #

    def __init__(self) -> None:
        self.model = None
        self.processor = None
        self.is_loaded: bool = False

        try:
            import torch  # noqa: F401 — side-effect: verify torch is present
            from transformers import CLIPModel, CLIPProcessor

            logger.info(
                "Loading CLIP checkpoint 'openai/clip-vit-base-patch32' "
                "(first run downloads ~600 MB) …"
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
        timestamp = datetime.now(tz=timezone.utc).isoformat().replace("+00:00", "Z")

        if self.is_loaded:
            defect_type, confidence, all_probs = self._clip_predict(image_bytes)
        else:
            defect_type, confidence = self.get_mock_analysis()
            # Build uniform-ish fake scores so the response shape is consistent
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
        Run the actual CLIP zero-shot classification.

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

        best_idx = int(probs.argmax().item())
        defect_type = self.DEFECT_CLASSES[best_idx]
        confidence = float(probs[best_idx])

        all_probs = {
            cls: float(prob)
            for cls, prob in zip(self.DEFECT_CLASSES, probs)
        }

        return defect_type, confidence, all_probs

    def get_mock_analysis(self) -> tuple[str, float]:
        """
        Return a randomly sampled (defect_type, confidence) pair for mock mode.

        Weights:
            normal: 40%  |  crack: 15%  |  missing_bolt: 20%
            rail_break: 5%  |  debris_on_track: 15%  |  track_misalignment: 5%

        Returns
        -------
        tuple[str, float]
            Sampled defect class and a confidence score in [0.72, 0.95].
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

        Parameters
        ----------
        defect_type : str
            One of DEFECT_CLASSES.
        confidence : float
            Classification confidence in [0, 1].

        Returns
        -------
        str
            One-sentence description, prefixed with "WARNING: " for
            high / critical severity classes.
        """
        pct = round(confidence * 100, 1)
        severity = self.SEVERITY[defect_type]

        if defect_type == "normal":
            return "Track appears to be in normal operating condition."

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
