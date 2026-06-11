"""
Pydantic schemas for the Crowd Forecasting module.

Request  → CrowdRequest
Response → CrowdResponse, StationsListResponse, HeatmapResponse
"""



from typing import Any, Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Request
# ---------------------------------------------------------------------------


class CrowdRequest(BaseModel):
    """Payload for a crowd forecast request."""

    station_code: str = Field(
        ...,
        examples=["NDLS"],
        description="Station code (e.g. 'NDLS', 'BCT', 'MAS').",
    )
    hours_ahead: int = Field(
        default=2,
        ge=1,
        le=24,
        description="How many hours ahead to forecast (1–24).",
    )


# ---------------------------------------------------------------------------
# Sub-models
# ---------------------------------------------------------------------------


class StationInfo(BaseModel):
    """Compact station descriptor used in the stations list."""

    station_code: str
    station_name: str
    base_crowd: int
    platforms: int


class ForecastHour(BaseModel):
    """A single hour slot in the crowd forecast."""

    hour: int
    time_label: str
    crowd_count: int
    congestion_level: str


class PlatformAllocation(BaseModel):
    """Platform status and recommendation."""

    platform: int
    status: str                         # "available" | "occupied" | "recommended"
    current_train: Optional[str] = None
    recommendation: str


# ---------------------------------------------------------------------------
# Responses
# ---------------------------------------------------------------------------


class CrowdResponse(BaseModel):
    """Crowd forecast result — matches CrowdForecaster.predict_crowd() output exactly."""

    station: str                                    # e.g. "New Delhi"
    station_code: str
    current_estimated_crowd: int
    congestion_level: str                           # "low" | "medium" | "high"
    forecast: list[ForecastHour]
    platform_allocation: list[PlatformAllocation]
    alert: Optional[str] = None
    advisory: str = ""                              # AI crowd advisory from RailAgent


class StationsListResponse(BaseModel):
    """All stations supported by the crowd forecaster."""

    stations: list[StationInfo]
    count: int


class HeatmapResponse(BaseModel):
    """Hour-by-hour crowd heatmap for a station (24 h)."""

    station_code: str
    station_name: str
    heatmap: Any                # dict[str, list] from the model
    generated_at: str           # ISO-8601 UTC