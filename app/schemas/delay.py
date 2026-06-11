"""
Pydantic schemas for the Delay Prediction module.

Request  → DelayRequest
Response → DelayResponse, TrainsListResponse, DelayStatsResponse
"""



from typing import Any

from pydantic import BaseModel, Field, field_validator


# ---------------------------------------------------------------------------
# Request
# ---------------------------------------------------------------------------


class DelayRequest(BaseModel):
    """Payload sent by the client to request a delay prediction."""

    train_number: str = Field(
        ...,
        examples=["12301"],
        description="Indian Railways train number (e.g. '12301').",
    )
    source: str = Field(
        ...,
        examples=["HWH"],
        description="Origin station code (e.g. 'HWH', 'NDLS').",
    )
    destination: str = Field(
        ...,
        examples=["NDLS"],
        description="Destination station code (e.g. 'NDLS', 'MAS').",
    )
    hour: int = Field(
        ...,
        ge=0,
        le=23,
        description="Hour of departure in 24-h format (0–23).",
    )
    day_of_week: int = Field(
        ...,
        ge=0,
        le=6,
        description="Day of week: 0 = Monday … 6 = Sunday.",
    )
    month: int = Field(
        ...,
        ge=1,
        le=12,
        description="Calendar month (1 = January … 12 = December).",
    )

    @field_validator("hour")
    @classmethod
    def _check_hour(cls, v: int) -> int:
        if not (0 <= v <= 23):
            raise ValueError("hour must be between 0 and 23")
        return v

    @field_validator("day_of_week")
    @classmethod
    def _check_dow(cls, v: int) -> int:
        if not (0 <= v <= 6):
            raise ValueError("day_of_week must be between 0 (Mon) and 6 (Sun)")
        return v

    @field_validator("month")
    @classmethod
    def _check_month(cls, v: int) -> int:
        if not (1 <= v <= 12):
            raise ValueError("month must be between 1 and 12")
        return v


# ---------------------------------------------------------------------------
# Nested schema for agent alternative train suggestions
# ---------------------------------------------------------------------------


class AlternativeTrain(BaseModel):
    """A single alternative train suggested by the RailMind AI agent."""

    train_name: str = ""
    train_number: str = ""
    note: str = ""


# ---------------------------------------------------------------------------
# Responses
# ---------------------------------------------------------------------------


class DelayResponse(BaseModel):
    """Structured delay prediction returned to the client.

    Field names match what DelayPredictor.predict() actually returns:
        predicted_delay_minutes, confidence (str), delay_category,
        train_number, route, is_mock
    plus the agent fields merged in by the router.
    """

    # Core ML prediction fields — match DelayPredictor.predict() output exactly
    train_number: str
    route: str
    predicted_delay_minutes: float
    confidence: str             # "high" | "medium" | "low"
    delay_category: str         # "on_time" | "slight" | "moderate" | "severe"
    is_mock: bool = False

    # AI agent fields — all optional with safe defaults
    agent_suggestion: str = ""
    agent_reasoning: str = ""
    urgency: str = "low"        # "low" | "medium" | "high"
    suggested_alternatives: list[AlternativeTrain] = Field(default_factory=list)
    action_needed: str = ""


class TrainsListResponse(BaseModel):
    """List of all trains known to the delay predictor."""

    trains: list[Any]
    count: int


class DelayStatsResponse(BaseModel):
    """Operational statistics for the delay prediction service."""

    predictions_today: int
    avg_delay_minutes: float
    worst_route: str
    worst_delay_minutes: int
    on_time_percentage: float
    last_updated: str           # ISO-8601 UTC