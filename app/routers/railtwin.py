"""
RailMind AI — RailTwin Router
Digital Interlocking & Junction Intelligence

Endpoints:
  GET  /railtwin/layout/{station}       — topology JSON for a station
  GET  /railtwin/state/{station}        — live operational state (occupancy, signals, points)
  POST /railtwin/route/request          — request a route allocation
  GET  /railtwin/trains/{station}       — active trains with positions and ETAs
  GET  /railtwin/alerts/{station}       — operational alerts (conflicts, delays, crowd surges)
  GET  /railtwin/stations               — list all supported stations
"""

from __future__ import annotations

import json
import math
import os
import random
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

router = APIRouter(prefix="/railtwin", tags=["RailTwin"])

# ── Layout directory ────────────────────────────────────────────────────────
# Searches multiple candidate locations so it works regardless of how/where
# the project is launched (uvicorn from root, from app/, Railway, etc.)
def _find_layout_dir() -> Path:
    candidates = [
        Path(__file__).resolve().parent.parent.parent / "station_layouts",  # backend root
        Path(__file__).resolve().parent.parent / "station_layouts",          # one up from routers/
        Path(__file__).resolve().parent / "station_layouts",                 # same dir as router
        Path.cwd() / "station_layouts",                                      # cwd
        Path.cwd().parent / "station_layouts",                               # parent of cwd
    ]
    for c in candidates:
        if c.exists() and any(c.glob("*.json")):
            return c
    return Path.cwd() / "station_layouts"   # graceful 404 fallback

LAYOUT_DIR = _find_layout_dir()

SUPPORTED_STATIONS = ["NDLS", "BPL", "HWH", "MAS", "SBC"]

# ── Synthetic train fleet ───────────────────────────────────────────────────
TRAIN_FLEET: dict[str, list[dict]] = {
    "NDLS": [
        {"number": "12002", "name": "Bhopal Shatabdi", "from": "BPL", "to": "NDLS", "platform": 3, "eta_min": 4},
        {"number": "12301", "name": "Howrah Rajdhani", "from": "HWH", "to": "NDLS", "platform": 1, "eta_min": 12},
        {"number": "12626", "name": "Kerala Express", "from": "TVC", "to": "NDLS", "platform": 7, "eta_min": 28},
        {"number": "12952", "name": "Mumbai Rajdhani", "from": "MMCT", "to": "NDLS", "platform": 5, "eta_min": 45},
        {"number": "22453", "name": "Vande Bharat", "from": "KLK", "to": "NDLS", "platform": 2, "eta_min": 6},
    ],
    "BPL": [
        {"number": "12002", "name": "Bhopal Shatabdi", "from": "NDLS", "to": "BPL", "platform": 1, "eta_min": 8},
        {"number": "12154", "name": "Habibganj Express", "from": "NZM", "to": "BPL", "platform": 3, "eta_min": 20},
        {"number": "19665", "name": "Kurj Express", "from": "INDB", "to": "BPL", "platform": 2, "eta_min": 35},
    ],
    "HWH": [
        {"number": "12301", "name": "Howrah Rajdhani", "from": "NDLS", "to": "HWH", "platform": 8, "eta_min": 5},
        {"number": "12019", "name": "Shatabdi Express", "from": "RNC", "to": "HWH", "platform": 4, "eta_min": 15},
        {"number": "13005", "name": "Amritsar Mail", "from": "ASR", "to": "HWH", "platform": 6, "eta_min": 22},
        {"number": "12841", "name": "Coromandel Express", "from": "MAS", "to": "HWH", "platform": 2, "eta_min": 40},
    ],
    "MAS": [
        {"number": "12841", "name": "Coromandel Express", "from": "HWH", "to": "MAS", "platform": 5, "eta_min": 10},
        {"number": "12163", "name": "Chennai Express", "from": "LTT", "to": "MAS", "platform": 3, "eta_min": 18},
        {"number": "22159", "name": "Vande Bharat", "from": "MYS", "to": "MAS", "platform": 1, "eta_min": 30},
    ],
    "SBC": [
        {"number": "22159", "name": "Vande Bharat", "from": "MAS", "to": "SBC", "platform": 4, "eta_min": 7},
        {"number": "16505", "name": "Bangalore Express", "from": "GY", "to": "SBC", "platform": 2, "eta_min": 25},
        {"number": "12677", "name": "Ernakulam SF", "from": "ERS", "to": "SBC", "platform": 6, "eta_min": 38},
    ],
}


def _load_layout(station_id: str) -> dict:
    station_id = station_id.upper()
    layout_file = LAYOUT_DIR / f"{station_id.lower()}.json"
    if not layout_file.exists():
        raise HTTPException(status_code=404, detail=f"Layout for {station_id} not found")
    with open(layout_file) as f:
        return json.load(f)


def _seeded_rand(seed: int, lo: float = 0.0, hi: float = 1.0) -> float:
    """Deterministic pseudo-random float based on seed + current minute."""
    minute_bucket = int(time.time() // 60)
    r = random.Random(seed + minute_bucket)
    return lo + r.random() * (hi - lo)


def _generate_track_states(layout: dict) -> dict[str, str]:
    """Return occupancy state per track id: free | occupied | reserved."""
    states: dict[str, str] = {}
    for t in layout.get("tracks", []):
        tid = t["id"]
        r = _seeded_rand(hash(tid), 0, 1)
        if r < 0.18:
            states[tid] = "occupied"
        elif r < 0.30:
            states[tid] = "reserved"
        else:
            states[tid] = "free"
    return states


def _generate_signal_states(layout: dict, track_states: dict) -> dict[str, str]:
    """Derive signal aspects from track occupancy."""
    aspects: dict[str, str] = {}
    for s in layout.get("signals", []):
        sid = s["id"]
        base = s.get("aspect", "green")
        # Slightly randomise within realistic bounds
        r = _seeded_rand(hash(sid + "sig"), 0, 1)
        if r < 0.20:
            aspects[sid] = "red"
        elif r < 0.35:
            aspects[sid] = "yellow"
        else:
            aspects[sid] = "green"
    return aspects


def _generate_point_states(layout: dict) -> dict[str, str]:
    """Return current point positions."""
    states: dict[str, str] = {}
    for p in layout.get("points", []):
        pid = p["id"]
        r = _seeded_rand(hash(pid + "pt"), 0, 1)
        states[pid] = "reverse" if r < 0.35 else "normal"
    return states


def _generate_train_positions(station_id: str, layout: dict) -> list[dict]:
    """Return trains with animated SVG position coordinates."""
    trains = []
    fleet = TRAIN_FLEET.get(station_id, [])
    platform_bays = layout.get("platformBays", [])
    now = datetime.utcnow()

    for idx, t in enumerate(fleet):
        pf_idx = (t["platform"] - 1) % max(len(platform_bays), 1)
        pb = platform_bays[pf_idx] if platform_bays else None

        # Derive position: trains not yet at station approach from the right
        eta = t["eta_min"]
        if eta <= 0:
            # At platform
            px = (pb["x"] + 40) if pb else 200
            py = (pb["y"] + pb["height"] // 2) if pb else 250
            state = "at_platform"
        elif eta <= 5:
            # Entering — on main line near station entry
            px = max(200, 900 - eta * 130)
            py = layout["tracks"][0]["y1"] if layout.get("tracks") else 80
            state = "arriving"
        else:
            # Approaching — further along main line
            px = 1050
            py = layout["tracks"][0]["y1"] if layout.get("tracks") else 80
            state = "approaching"

        trains.append({
            "number": t["number"],
            "name": t["name"],
            "from": t["from"],
            "to": t["to"],
            "platform": t["platform"],
            "eta_min": t["eta_min"],
            "eta_label": f"+{t['eta_min']} min" if t["eta_min"] > 0 else "AT PLATFORM",
            "x": px,
            "y": py,
            "state": state,
            "section": _train_section(state, t["platform"]),
        })
    return trains


def _train_section(state: str, platform: int) -> str:
    if state == "at_platform":
        return f"PF{platform}"
    elif state == "arriving":
        return "HOME SIGNAL"
    else:
        return "OUTER"


# ── Routes ──────────────────────────────────────────────────────────────────

@router.get("/stations", summary="List supported stations")
async def list_stations() -> dict:
    return {
        "stations": [
            {"id": "NDLS", "name": "New Delhi", "zone": "NR", "platforms": 16},
            {"id": "BPL", "name": "Bhopal Jn", "zone": "WCR", "platforms": 6},
            {"id": "HWH", "name": "Howrah Jn", "zone": "ER", "platforms": 15},
            {"id": "MAS", "name": "Chennai Central", "zone": "SR", "platforms": 12},
            {"id": "SBC", "name": "KSR Bengaluru", "zone": "SWR", "platforms": 10},
        ]
    }


@router.get("/layout/{station}", summary="Station topology")
async def get_layout(station: str) -> dict:
    return _load_layout(station)


@router.get("/state/{station}", summary="Live operational state")
async def get_state(station: str) -> dict:
    layout = _load_layout(station)
    track_states = _generate_track_states(layout)
    signal_states = _generate_signal_states(layout, track_states)
    point_states = _generate_point_states(layout)

    occupied_count = sum(1 for v in track_states.values() if v == "occupied")
    reserved_count = sum(1 for v in track_states.values() if v == "reserved")

    return {
        "station": station.upper(),
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "tracks": track_states,
        "signals": signal_states,
        "points": point_states,
        "summary": {
            "total_tracks": len(track_states),
            "occupied": occupied_count,
            "reserved": reserved_count,
            "free": len(track_states) - occupied_count - reserved_count,
        }
    }


@router.get("/trains/{station}", summary="Active train movements")
async def get_trains(station: str) -> dict:
    layout = _load_layout(station)
    trains = _generate_train_positions(station.upper(), layout)
    return {
        "station": station.upper(),
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "trains": trains,
        "active_count": len(trains),
    }


@router.get("/alerts/{station}", summary="Operational alerts")
async def get_alerts(station: str) -> dict:
    station = station.upper()
    fleet = TRAIN_FLEET.get(station, [])
    alerts = []

    # Route conflict alerts
    r1 = _seeded_rand(hash(station + "conf"), 0, 1)
    if r1 < 0.4 and len(fleet) >= 2:
        t1, t2 = fleet[0], fleet[1]
        alerts.append({
            "id": "ALT001",
            "severity": "critical",
            "type": "route_conflict",
            "title": "ROUTE CONFLICT DETECTED",
            "message": f"Platform {t1['platform']} expected conflict in {t1['eta_min'] + 2} min. "
                       f"Train {t1['number']} and {t2['number']} on converging paths.",
            "recommendation": f"Recommend alternate route via Platform {t2['platform'] + 1}",
            "timestamp": datetime.utcnow().isoformat() + "Z",
        })

    # Crowd surge
    r2 = _seeded_rand(hash(station + "crowd"), 0, 1)
    if r2 < 0.5:
        pf = random.Random(hash(station)).randint(3, 8)
        pct = int(_seeded_rand(hash(station + "pct"), 28, 58))
        alerts.append({
            "id": "ALT002",
            "severity": "warning",
            "type": "congestion",
            "title": "CONGESTION ALERT",
            "message": f"Crowd surge predicted. Platform {pf} +{pct}% within 20 min.",
            "recommendation": "Divert passengers to adjacent platforms. Deploy RPF.",
            "timestamp": datetime.utcnow().isoformat() + "Z",
        })

    # Delay impact
    r3 = _seeded_rand(hash(station + "delay"), 0, 1)
    if r3 < 0.55 and fleet:
        delayed = fleet[-1]
        delay_mins = int(_seeded_rand(hash(station + "dm"), 8, 35))
        alerts.append({
            "id": "ALT003",
            "severity": "info",
            "type": "delay_impact",
            "title": "DELAY IMPACT",
            "message": f"Train {delayed['number']} delayed by {delay_mins} min. "
                       f"May affect Platform {delayed['platform']} allocation.",
            "recommendation": "Pre-position shunting loco. Notify downstream stations.",
            "timestamp": datetime.utcnow().isoformat() + "Z",
        })

    # Signal degradation
    r4 = _seeded_rand(hash(station + "sig"), 0, 1)
    if r4 < 0.25:
        alerts.append({
            "id": "ALT004",
            "severity": "warning",
            "type": "signal_fault",
            "title": "SIGNAL DEGRADATION",
            "message": "Calling-on signal C2 intermittent. Manual working may be required.",
            "recommendation": "SM to authorize caution order. Contact S&T department.",
            "timestamp": datetime.utcnow().isoformat() + "Z",
        })

    return {
        "station": station,
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "alerts": alerts,
        "critical_count": sum(1 for a in alerts if a["severity"] == "critical"),
        "warning_count": sum(1 for a in alerts if a["severity"] == "warning"),
    }


class RouteRequest(BaseModel):
    station: str
    route_id: str
    train_number: str


@router.post("/route/request", summary="Request route allocation")
async def request_route(req: RouteRequest) -> dict:
    layout = _load_layout(req.station)
    routes = {r["id"]: r for r in layout.get("routes", [])}
    route = routes.get(req.route_id)
    if not route:
        raise HTTPException(status_code=404, detail=f"Route {req.route_id} not found")

    # Simulate conflict check
    r = random.random()
    if r < 0.15:
        return {
            "success": False,
            "route_id": req.route_id,
            "train_number": req.train_number,
            "status": "CONFLICT",
            "reason": "Conflicting route already locked",
        }

    return {
        "success": True,
        "route_id": req.route_id,
        "train_number": req.train_number,
        "status": "LOCKED",
        "route_name": route["name"],
        "tracks_reserved": route["tracks"],
        "signals_cleared": route["signals"],
        "points_set": route["points"],
        "timestamp": datetime.utcnow().isoformat() + "Z",
    }