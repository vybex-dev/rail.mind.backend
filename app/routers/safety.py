"""
Track Safety Detection Router — RailMind AI.

Prefix : /safety
Tags   : ["Track Safety"]

Routes
------
POST /safety/analyze — upload an image; returns defect classification + AI alert
GET  /safety/recent  — last ≤10 analyses (in-memory, resets on restart)
GET  /safety/status  — CLIP model load status and supported defect classes

Schema note
-----------
`SafetyResponse` (app/schemas/safety.py) must include:

    alert_message: str = ""

so that the AI-generated alert is serialised and returned to the client.

In-memory ring buffer
---------------------
`recent_analyses` stores the last 10 result dicts (plain dicts, not models)
so they survive Pydantic serialisation round-trips cleanly.
The list resets whenever the server process restarts.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, File, HTTPException, UploadFile

from app.agent.rail_agent import rail_agent
from app.models.safety_model import TrackSafetyDetector, track_safety_detector
from app.schemas.safety import (
    RecentAnalysesResponse,
    SafetyResponse,
    SafetyStatusResponse,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/safety", tags=["Track Safety"])

# ---------------------------------------------------------------------------
# In-memory ring buffer — max 10 entries, oldest dropped when full.
# ---------------------------------------------------------------------------
recent_analyses: list[dict] = []
_MAX_RECENT = 10


# ---------------------------------------------------------------------------
# POST /safety/analyze
# ---------------------------------------------------------------------------


@router.post(
    "/analyze",
    response_model=SafetyResponse,
    summary="Analyse a track image",
    description=(
        "Upload a railway track image (JPEG, PNG, …). "
        "The CLIP model classifies it into one of the defect categories "
        "and returns a structured safety report together with an AI-generated "
        "operations alert from RailMind Agent. "
        "Falls back to weighted-random mock results when CLIP is unavailable."
    ),
)
async def analyze_track(
    file: UploadFile = File(..., description="Railway track image to analyse."),
    location: str = "Unknown",
) -> SafetyResponse:
    """
    Classify an uploaded track image, then generate an AI safety alert.

    Flow
    ----
    1. Validate MIME type.
    2. Read bytes.
    3. Run CLIP safety classifier.
    4. Ask RailMind Agent (Claude) to produce a formal operations alert.
    5. Merge alert into result dict, update ring buffer, and return.

    The agent call is wrapped in its own try/except so a Claude API failure
    never blocks the core classification response.
    """
    # ── 1. Validate MIME type ──────────────────────────────────────────────
    content_type: str = file.content_type or ""
    if not content_type.startswith("image/"):
        raise HTTPException(
            status_code=400,
            detail=(
                f"Only image files accepted (received content-type '{content_type}'). "
                "Please upload a JPEG, PNG, or similar image."
            ),
        )

    # ── 2. Read bytes ──────────────────────────────────────────────────────
    try:
        contents: bytes = await file.read()
    except Exception as exc:
        logger.exception("Failed to read uploaded file: %s", exc)
        raise HTTPException(
            status_code=500,
            detail=f"Could not read uploaded file: {exc}",
        ) from exc

    if not contents:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    # ── 3. Run CLIP classifier ─────────────────────────────────────────────
    try:
        result: dict = track_safety_detector.analyze_image(contents)
    except Exception as exc:
        logger.exception("Safety analysis failed: %s", exc)
        raise HTTPException(
            status_code=500,
            detail=f"Safety analysis failed: {exc}",
        ) from exc

    logger.info(
        "[SAFETY] %s | severity: %s | confidence: %s | location: %s",
        result.get("defect_type", "unknown"),
        result.get("severity",    "unknown"),
        result.get("confidence",  "?"),
        location,
    )

    # ── 4. AI safety alert ─────────────────────────────────────────────────
    try:
        alert = rail_agent.generate_safety_alert(result, location=location)
    except Exception as exc:
        logger.warning("Agent safety alert failed (non-fatal): %s", exc)
        alert = rail_agent._fallback_safety_alert(result)   # always available

    result["alert_message"] = alert

    # ── 5. Update ring buffer ──────────────────────────────────────────────
    if len(recent_analyses) >= _MAX_RECENT:
        recent_analyses.pop(0)
    recent_analyses.append(result)

    # ── 6. Return ──────────────────────────────────────────────────────────
    return SafetyResponse(**result)


# ---------------------------------------------------------------------------
# GET /safety/recent
# ---------------------------------------------------------------------------


@router.get(
    "/recent",
    response_model=RecentAnalysesResponse,
    summary="Recent analyses",
    description=(
        "Returns the last ≤10 track safety analyses performed "
        "in the current server process. Resets on server restart."
    ),
)
async def get_recent_analyses() -> RecentAnalysesResponse:
    """Return the in-memory ring buffer of recent safety analyses."""
    try:
        analyses = [SafetyResponse(**entry) for entry in recent_analyses]
        return RecentAnalysesResponse(analyses=analyses, count=len(analyses))
    except Exception as exc:
        logger.exception("Error retrieving recent analyses: %s", exc)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to retrieve recent analyses: {exc}",
        ) from exc


# ---------------------------------------------------------------------------
# GET /safety/status
# ---------------------------------------------------------------------------


@router.get(
    "/status",
    response_model=SafetyStatusResponse,
    summary="Model status",
    description=(
        "Returns whether the CLIP model is loaded (AI mode) or "
        "running as a mock, plus the list of supported defect classes."
    ),
)
async def get_model_status() -> SafetyStatusResponse:
    """Report the current operational mode of the TrackSafetyDetector."""
    try:
        loaded: bool = track_safety_detector.is_loaded
        return SafetyStatusResponse(
            clip_model_loaded=loaded,
            mode="ai" if loaded else "mock",
            supported_defects=TrackSafetyDetector.DEFECT_CLASSES,
        )
    except Exception as exc:
        logger.exception("Error fetching safety model status: %s", exc)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to retrieve model status: {exc}",
        ) from exc