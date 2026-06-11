"""
Pydantic schemas for the Track Safety Detection module.

Response → SafetyResponse, RecentAnalysesResponse, SafetyStatusResponse
(No request schema needed — input is a multipart file upload.)
"""


from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Responses
# ---------------------------------------------------------------------------


class SafetyResponse(BaseModel):
    """Structured safety analysis result for a single uploaded image."""

    defect_type: str                        # one of TrackSafetyDetector.DEFECT_CLASSES
    confidence: float                       # 0.0 – 1.0
    severity: str                           # "none" | "medium" | "high" | "critical"
    description: str                        # human-readable sentence
    recommended_action: str
    safe_to_operate: bool
    analysis_timestamp: str                 # ISO-8601 UTC
    all_scores: dict[str, float]            # per-class softmax probabilities
    alert_message: str = ""                 # AI-generated operations alert from RailAgent


class RecentAnalysesResponse(BaseModel):
    """The last ≤10 safety analyses performed in this server process."""

    analyses: list[SafetyResponse]
    count: int = Field(ge=0, le=10)


class SafetyStatusResponse(BaseModel):
    """Current status of the TrackSafetyDetector model."""

    clip_model_loaded: bool
    mode: str                               # "ai" | "mock"
    supported_defects: list[str]