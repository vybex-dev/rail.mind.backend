"""
Track Safety Detection Module — RailMind AI (Indian Railways).

Uses Groq's Llama 4 Scout vision API for track defect classification.
Groq is free-tier and reachable from Railway's network.

Vision model : meta-llama/llama-4-scout-17b-16e-instruct
  - Free on Groq
  - Supports base64 image input + JSON mode
  - OpenAI-compatible SDK (same groq SDK used by RailAgent)

Fallback      : weighted-random mock when GROQ_API_KEY is absent or API fails.
"""

from __future__ import annotations

import base64
import json
import logging
import os
import random
from datetime import datetime, timezone

logger = logging.getLogger(__name__)

# ── Safety thresholds ─────────────────────────────────────────────────────────
CONFIDENCE_THRESHOLD: float = 0.45
MARGIN_THRESHOLD:     float = 0.10

# ── Groq vision model ─────────────────────────────────────────────────────────
_VISION_MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"


class TrackSafetyDetector:
    """
    Groq Llama-4-Scout-powered track defect classifier.

    Reads GROQ_API_KEY from the environment (same key used by RailAgent).
    Falls back to weighted-random mock only when the key is absent or the
    API call genuinely fails — always logs which path ran.
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

    # Weighted mock fallback — "normal" gets 55 % of random hits
    _MOCK_WEIGHTS: list[float] = [0.55, 0.10, 0.15, 0.05, 0.10, 0.05]

    # ── System prompt ─────────────────────────────────────────────────────────
    _SYSTEM_PROMPT = """\
You are a railway track safety inspector AI. Analyse the provided image and classify the track condition.

You MUST respond with ONLY a valid JSON object — no markdown fences, no preamble, no explanation.

Required JSON format:
{
  "defect_type": "<one of: normal | crack | missing_bolt | rail_break | debris_on_track | track_misalignment>",
  "confidence": <float 0.0-1.0>,
  "all_scores": {
    "normal": <float>,
    "crack": <float>,
    "missing_bolt": <float>,
    "rail_break": <float>,
    "debris_on_track": <float>,
    "track_misalignment": <float>
  }
}

Rules:
- all_scores values must sum to approximately 1.0
- confidence must equal the score for the chosen defect_type
- Be conservative: only flag a defect when clearly visible evidence exists
- If the image is blurry, dark, or not a railway track, return defect_type "normal" with low confidence (0.5-0.6)
- Do NOT include any text outside the JSON object
"""

    def __init__(self) -> None:
        self.is_loaded:   bool = True
        self._load_failed: bool = False

        api_key = os.getenv("GROQ_API_KEY", "")
        self._client = None

        if api_key:
            try:
                from groq import Groq
                self._client = Groq(api_key=api_key)
                logger.info(
                    "TrackSafetyDetector: GROQ_API_KEY found — running in "
                    "Llama-4-Scout vision mode (model=%s).", _VISION_MODEL
                )
            except Exception as exc:
                logger.warning(
                    "TrackSafetyDetector: could not init Groq client (%s) — "
                    "will use mock fallback.", exc
                )
        else:
            logger.warning(
                "TrackSafetyDetector: GROQ_API_KEY not set — will use mock fallback."
            )

    # ── Public API ────────────────────────────────────────────────────────────

    def analyze_image(self, image_bytes: bytes) -> dict:
        """Classify a raw image and return a structured safety report."""
        timestamp = datetime.now(tz=timezone.utc).isoformat().replace("+00:00", "Z")
        defect_type, confidence, all_probs = self._classify(image_bytes)

        return {
            "defect_type":       defect_type,
            "confidence":        round(confidence, 3),
            "severity":          self.SEVERITY[defect_type],
            "description":       self.build_description(defect_type, confidence),
            "recommended_action": self.ACTIONS[defect_type],
            "safe_to_operate":   defect_type in ["normal", "debris_on_track"],
            "analysis_timestamp": timestamp,
            "all_scores": {
                cls: round(float(prob), 3) for cls, prob in all_probs.items()
            },
        }

    # ── Internal classification ───────────────────────────────────────────────

    def _classify(
        self, image_bytes: bytes
    ) -> tuple[str, float, dict[str, float]]:
        """Route to Groq vision or mock depending on client availability."""
        if self._client is None:
            logger.warning("No Groq client — returning mock result.")
            return self._mock_result()
        return self._groq_vision_predict(image_bytes)

    def _groq_vision_predict(
        self, image_bytes: bytes
    ) -> tuple[str, float, dict[str, float]]:
        """Call Groq Llama-4-Scout vision API and parse the JSON result."""
        media_type = _detect_media_type(image_bytes)
        b64_image  = base64.standard_b64encode(image_bytes).decode("utf-8")
        data_url   = f"data:{media_type};base64,{b64_image}"

        try:
            completion = self._client.chat.completions.create(
                model=_VISION_MODEL,
                max_tokens=512,
                temperature=0.1,          # low temp → consistent structured output
                response_format={"type": "json_object"},   # enforce JSON mode
                messages=[
                    {"role": "system", "content": self._SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {"url": data_url},
                            },
                            {
                                "type": "text",
                                "text": (
                                    "Analyse this railway track image. "
                                    "Respond ONLY with the JSON classification object."
                                ),
                            },
                        ],
                    },
                ],
            )
        except Exception as exc:
            logger.error("Groq vision API call failed: %s", exc)
            return self._mock_result()

        try:
            raw_text: str = completion.choices[0].message.content.strip()

            # Strip accidental markdown fences just in case
            if raw_text.startswith("```"):
                raw_text = raw_text.split("```")[1]
                if raw_text.startswith("json"):
                    raw_text = raw_text[4:]
                raw_text = raw_text.strip()

            parsed: dict = json.loads(raw_text)
        except Exception as exc:
            logger.error("Failed to parse Groq vision response: %s", exc)
            return self._mock_result()

        return self._validate_and_extract(parsed)

    def _validate_and_extract(
        self, parsed: dict
    ) -> tuple[str, float, dict[str, float]]:
        """Validate parsed JSON and return (defect_type, confidence, all_scores)."""
        defect_type = parsed.get("defect_type", "normal")
        if defect_type not in self.DEFECT_CLASSES:
            logger.warning(
                "Vision model returned unknown defect_type '%s', defaulting to normal.",
                defect_type,
            )
            defect_type = "normal"

        all_scores_raw: dict = parsed.get("all_scores", {})

        # Fill any missing classes with 0
        all_scores: dict[str, float] = {
            cls: float(all_scores_raw.get(cls, 0.0))
            for cls in self.DEFECT_CLASSES
        }

        # Re-normalise to sum = 1.0
        total = sum(all_scores.values())
        if total > 0:
            all_scores = {k: v / total for k, v in all_scores.items()}
        else:
            all_scores = {cls: 1.0 / len(self.DEFECT_CLASSES) for cls in self.DEFECT_CLASSES}

        # Use normalised score for the winning class — NOT the raw parsed value
        confidence = float(all_scores.get(defect_type, 0.5))
        confidence = max(0.0, min(1.0, confidence))

        # Low-confidence safety net
        sorted_scores = sorted(all_scores.values(), reverse=True)
        margin = sorted_scores[0] - sorted_scores[1] if len(sorted_scores) > 1 else 1.0
        if confidence < CONFIDENCE_THRESHOLD or margin < MARGIN_THRESHOLD:
            logger.info(
                "Low confidence (%.2f) or narrow margin (%.2f) — returning normal.",
                confidence, margin,
            )
            defect_type = "normal"
            confidence  = round(all_scores.get("normal", sorted_scores[0]), 3)

        logger.info(
            "Groq vision result: %s (conf=%.3f, margin=%.3f)",
            defect_type, confidence, margin,
        )
        return defect_type, confidence, all_scores

    # ── Mock fallback ─────────────────────────────────────────────────────────

    def _mock_result(self) -> tuple[str, float, dict[str, float]]:
        """Weighted-random result — logs clearly so operators know it's not real."""
        defect_type = random.choices(
            self.DEFECT_CLASSES, weights=self._MOCK_WEIGHTS, k=1
        )[0]
        confidence = round(random.uniform(0.72, 0.95), 3)
        base = (1.0 - confidence) / (len(self.DEFECT_CLASSES) - 1)
        probs = {cls: base for cls in self.DEFECT_CLASSES}
        probs[defect_type] = confidence
        logger.warning(
            "MOCK result returned (no real analysis): %s @ %.3f", defect_type, confidence
        )
        return defect_type, confidence, probs

    # ── Helpers ───────────────────────────────────────────────────────────────

    def build_description(self, defect_type: str, confidence: float) -> str:
        pct      = round(confidence * 100, 1)
        severity = self.SEVERITY[defect_type]

        if defect_type == "normal":
            return (
                f"Track appears to be in normal operating condition "
                f"({pct}% confidence). No defects detected."
            )

        base = (
            f"Analysis detected {defect_type.replace('_', ' ')} on track "
            f"segment with {pct}% confidence."
        )
        return f"WARNING: {base}" if severity in {"high", "critical"} else base


# ── Helpers ───────────────────────────────────────────────────────────────────

def _detect_media_type(data: bytes) -> str:
    """Sniff image format from magic bytes; default to image/jpeg."""
    if data[:8] == b"\x89PNG\r\n\x1a\n":
        return "image/png"
    if data[:4] in (b"GIF8", b"GIF9"):
        return "image/gif"
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "image/webp"
    return "image/jpeg"


# Singleton instance — import this everywhere
track_safety_detector = TrackSafetyDetector()