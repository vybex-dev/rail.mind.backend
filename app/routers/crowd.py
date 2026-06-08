"""
Crowd Forecasting Router — RailMind AI.

Prefix : /crowd
Tags   : ["Crowd Forecasting"]

Routes
------
GET  /crowd/stations               — list all supported stations
POST /crowd/predict                — forecast crowd level for a station
GET  /crowd/heatmap/{station_code} — 24-hour crowd heatmap for a station
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone

from fastapi import APIRouter, HTTPException

from app.models.crowd_model import crowd_forecaster
from app.schemas.crowd import (
    CrowdRequest,
    CrowdResponse,
    HeatmapResponse,
    StationInfo,
    StationsListResponse,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/crowd", tags=["Crowd Forecasting"])


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _assert_station_exists(station_code: str) -> None:
    known: dict = getattr(crowd_forecaster, "STATIONS", {})
    if station_code.upper() not in known:
        raise HTTPException(
            status_code=404,
            detail=(
                f"Station '{station_code}' not found. "
                "Use GET /crowd/stations to retrieve valid codes."
            ),
        )


# ---------------------------------------------------------------------------
# GET /crowd/stations
# ---------------------------------------------------------------------------


@router.get(
    "/stations",
    response_model=StationsListResponse,
    summary="List all stations",
)
async def get_stations() -> StationsListResponse:
    try:
        raw: list[dict] = crowd_forecaster.get_stations()

        # get_stations() returns dicts with keys:
        #   station_code, station_name, base_crowd, platforms
        stations = [
            StationInfo(
                station_code=s.get("station_code", ""),
                station_name=s.get("station_name", ""),
                base_crowd=s.get("base_crowd", 0),
                platforms=s.get("platforms", 0),
            )
            for s in raw
        ]

        return StationsListResponse(stations=stations, count=len(stations))

    except Exception as exc:
        logger.exception("Error fetching station list: %s", exc)
        raise HTTPException(status_code=500, detail=f"Failed to retrieve station list: {exc}") from exc


# ---------------------------------------------------------------------------
# POST /crowd/predict
# ---------------------------------------------------------------------------


@router.post(
    "/predict",
    response_model=CrowdResponse,
    summary="Forecast crowd level",
)
async def predict_crowd(request: CrowdRequest) -> CrowdResponse:
    _assert_station_exists(request.station_code)

    try:
        result: dict = crowd_forecaster.predict_crowd(
            station_code=request.station_code.upper(),
            hours_ahead=request.hours_ahead,
        )
        return CrowdResponse(**result)

    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Crowd prediction failed for station %s: %s", request.station_code, exc)
        raise HTTPException(status_code=500, detail=f"Crowd prediction failed: {exc}") from exc


# ---------------------------------------------------------------------------
# GET /crowd/heatmap/{station_code}
# ---------------------------------------------------------------------------


@router.get(
    "/heatmap/{station_code}",
    response_model=HeatmapResponse,
    summary="Station crowd heatmap",
)
async def get_heatmap(station_code: str) -> HeatmapResponse:
    code = station_code.upper()
    _assert_station_exists(code)

    try:
        heatmap_data: dict = crowd_forecaster.get_heatmap_data(code)

        now_iso = datetime.now(tz=timezone.utc).isoformat().replace("+00:00", "Z")

        stations_dict: dict = getattr(crowd_forecaster, "STATIONS", {})
        station_entry = stations_dict.get(code, {})
        station_name: str = station_entry.get("name", code) if isinstance(station_entry, dict) else code

        return HeatmapResponse(
            station_code=code,
            station_name=station_name,
            heatmap=heatmap_data,
            generated_at=now_iso,
        )

    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Heatmap generation failed for station %s: %s", station_code, exc)
        raise HTTPException(status_code=500, detail=f"Heatmap generation failed: {exc}") from exc