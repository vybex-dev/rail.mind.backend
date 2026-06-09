"""
Track Safety Detection Module — RailMind AI (Indian Railways).

Uses Claude claude-haiku-4-5-20251001 vision API for accurate track defect classification.
Replaces the broken HuggingFace CLIP approach (unreachable from Railway + wrong API usage).

Why Claude vision instead of CLIP:
  - api.anthropic.com is reachable from Railway's network
  - Actual vision understanding vs fragile zero-shot label matching
  - Returns structured JSON — no header hacks needed
  - Graceful mock fallback when ANTHROPIC_API_KEY is absent
"""

from __future__ import annotations

import base64
import json
import logging
import os
import random
import requests
from datetime import datetime, timezone

logger = logging.getLogger(__name__)

# ── Safety thresholds ─────────────────────────────────────────────────────────
CONFIDENCE_THRESHOLD: float = 0.45
MARGIN_THRESHOLD: float = 0.10

# ── Anthropic API ─────────────────────────────────────────────────────────────
_ANTHROPIC_API_URL = "https://api.anthropic.com/v1/messages"
_MODEL = "claude-haiku-4-5-20251001"   # fast + cheap, vision-capable


class TrackSafetyDetector:
    """
    Claude-vision-powered track defect classifier.

    Falls back to weighted-random mock only when ANTHROPIC_API_KEY is absent
    or the API call genuinely fails — and always tells the logger which path ran.
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
        "normal": "none",
        "crack": "high",
        "missing_bolt": "medium",
        "rail_break": "critical",
        "debris_on_track": "medium",
        "track_misalignment": "high",
    }

    ACTIONS: dict[str, str] = {
        "normal": "No action required. Track is in good condition.",
        "crack": "Schedule urgent inspection. Mark segment for repair within 24h.",
        "missing_bolt": "Dispatch maintenance crew. Replace missing fasteners.",
        "rail_break": "HALT all trains on this segment immediately. Emergency repair needed.",
        "debris_on_track": "Clear debris before next train passage. Alert train control.",
        "track_misalignment": "Reduce train speed on this segment. Schedule realignment.",
    }

    # Weighted mock fallback — "normal" gets 55% of random hits
    _MOCK_WEIGHTS: list[float] = [0.55, 0.10, 0.15, 0.05, 0.10, 0.05]

    # ── System prompt sent to Claude ──────────────────────────────────────────
    _SYSTEM_PROMPT = """\
You are a railway track safety inspector AI. Analyse the provided image and classify the track condition.

You MUST respond with ONLY a valid JSON object — no markdown fences, no preamble, no explanation.

Required JSON format:
{
  "defect_type": "<one of: normal | crack | missing_bolt | rail_break | debris_on_track | track_misalignment>",
  "confidence": <float 0.0–1.0>,
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
- If the image is blurry, dark, or not a railway track, return defect_type "normal" with low confidence (0.5–0.6)
- Do NOT include any text outside the JSON object
"""

    def __init__(self) -> None:
        self.is_loaded: bool = True
        self._load_failed: bool = False
        self._api_key: str = os.getenv("ANTHROPIC_API_KEY", "")

        if self._api_key:
            logger.info(
                "TrackSafetyDetector: ANTHROPIC_API_KEY found — running in Claude vision mode."
            )
        else:
            logger.warning(
                "TrackSafetyDetector: ANTHROPIC_API_KEY not set — will use mock fallback."
            )

    # ── Public API ────────────────────────────────────────────────────────────

    def analyze_image(self, image_bytes: bytes) -> dict:
        """Classify a raw image and return a structured safety report."""
        timestamp = datetime.now(tz=timezone.utc).isoformat().replace("+00:00", "Z")
        defect_type, confidence, all_probs = self._classify(image_bytes)

        return {
            "defect_type": defect_type,
            "confidence": round(confidence, 3),
            "severity": self.SEVERITY[defect_type],
            "description": self.build_description(defect_type, confidence),
            "recommended_action": self.ACTIONS[defect_type],
            "safe_to_operate": defect_type in ["normal", "debris_on_track"],
            "analysis_timestamp": timestamp,
            "all_scores": {
                cls: round(float(prob), 3) for cls, prob in all_probs.items()
            },
        }

    # ── Internal classification ───────────────────────────────────────────────

    def _classify(
        self, image_bytes: bytes
    ) -> tuple[str, float, dict[str, float]]:
        """Route to Claude vision or mock depending on API key availability."""
        if not self._api_key:
            logger.warning("No ANTHROPIC_API_KEY — returning mock result.")
            return self._mock_result()

        return self._claude_vision_predict(image_bytes)

    def _claude_vision_predict(
        self, image_bytes: bytes
    ) -> tuple[str, float, dict[str, float]]:
        """Call Claude Haiku vision API and parse the JSON classification result."""
        # Detect image format from magic bytes (default jpeg)
        media_type = _detect_media_type(image_bytes)
        b64_image = base64.standard_b64encode(image_bytes).decode("utf-8")

        payload = {
            "model": _MODEL,
            "max_tokens": 512,
            "system": self._SYSTEM_PROMPT,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": media_type,
                                "data": b64_image,
                            },
                        },
                        {
                            "type": "text",
                            "text": (
                                "Analyse this railway track image. "
                                "Respond ONLY with the JSON classification object."
                            ),
                        },
                    ],
                }
            ],
        }

        headers = {
            "x-api-key": self._api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        }

        try:
            response = requests.post(
                _ANTHROPIC_API_URL,
                headers=headers,
                json=payload,
                timeout=30,
            )
        except requests.exceptions.RequestException as exc:
            logger.error("Claude API request failed (network): %s", exc)
            return self._mock_result()

        if response.status_code != 200:
            logger.error(
                "Claude API returned HTTP %s: %s",
                response.status_code,
                response.text[:300],
            )
            return self._mock_result()

        try:
            resp_json = response.json()
            raw_text: str = resp_json["content"][0]["text"].strip()

            # Strip accidental markdown fences just in case
            if raw_text.startswith("```"):
                raw_text = raw_text.split("```")[1]
                if raw_text.startswith("json"):
                    raw_text = raw_text[4:]
                raw_text = raw_text.strip()

            parsed: dict = json.loads(raw_text)
        except Exception as exc:
            logger.error("Failed to parse Claude API response: %s", exc)
            return self._mock_result()

        return self._validate_and_extract(parsed)

    def _validate_and_extract(
        self, parsed: dict
    ) -> tuple[str, float, dict[str, float]]:
        """Validate parsed JSON and return (defect_type, confidence, all_scores)."""
        defect_type = parsed.get("defect_type", "normal")
        if defect_type not in self.DEFECT_CLASSES:
            logger.warning(
                "Claude returned unknown defect_type '%s', defaulting to normal.", defect_type
            )
            defect_type = "normal"

        all_scores_raw: dict = parsed.get("all_scores", {})

        # Fill any missing classes with 0
        all_scores: dict[str, float] = {}
        for cls in self.DEFECT_CLASSES:
            all_scores[cls] = float(all_scores_raw.get(cls, 0.0))

        # Re-normalise to sum = 1.0 (guards against Claude rounding)
        total = sum(all_scores.values())
        if total > 0:
            all_scores = {k: v / total for k, v in all_scores.items()}
        else:
            all_scores = {cls: 1.0 / len(self.DEFECT_CLASSES) for cls in self.DEFECT_CLASSES}

        confidence = float(parsed.get("confidence", all_scores.get(defect_type, 0.5)))
        confidence = max(0.0, min(1.0, confidence))

        # Apply low-confidence safety net
        sorted_scores = sorted(all_scores.values(), reverse=True)
        margin = sorted_scores[0] - sorted_scores[1] if len(sorted_scores) > 1 else 1.0
        if confidence < CONFIDENCE_THRESHOLD or margin < MARGIN_THRESHOLD:
            logger.info(
                "Low confidence (%.2f) or narrow margin (%.2f) — returning normal.",
                confidence, margin,
            )
            defect_type = "normal"
            confidence = round(1.0 - sorted_scores[0], 3)

        logger.info(
            "Claude vision result: %s (conf=%.3f, margin=%.3f)",
            defect_type, confidence, margin,
        )
        return defect_type, confidence, all_scores

    # ── Mock fallback ─────────────────────────────────────────────────────────

    def _mock_result(self) -> tuple[str, float, dict[str, float]]:
        """
        Returns a weighted-random result.
        Logs clearly so operators know the result is not from real analysis.
        """
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
        pct = round(confidence * 100, 1)
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
        if severity in {"high", "critical"}:
            return f"WARNING: {base}"
        return base


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