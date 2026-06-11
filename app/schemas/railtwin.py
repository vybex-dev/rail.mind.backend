"""
RailTwin Pydantic Schemas
"""
from __future__ import annotations
from typing import Optional, List, Dict, Any
from pydantic import BaseModel


class TrackSegment(BaseModel):
    id: str
    x1: float
    y1: float
    x2: float
    y2: float
    type: str   # main | platform | loop | yard
    label: str


class SignalDef(BaseModel):
    id: str
    x: float
    y: float
    type: str   # home | starter | calling | advanced | outer
    aspect: str = "green"
    label: str


class PointDef(BaseModel):
    id: str
    x: float
    y: float
    state: str = "normal"   # normal | reverse
    angle: float = 15.0
    label: str


class CrossoverDef(BaseModel):
    id: str
    x: float
    y: float
    toY: float


class PlatformBay(BaseModel):
    id: str
    platform: int
    x: float
    y: float
    width: float
    height: float


class RouteDef(BaseModel):
    id: str
    name: str
    tracks: List[str]
    signals: List[str]
    points: List[str]


class StationLayout(BaseModel):
    id: str
    name: str
    zone: str
    platforms: int
    width: int
    height: int
    tracks: List[Dict[str, Any]]
    crossovers: List[Dict[str, Any]]
    points: List[Dict[str, Any]]
    signals: List[Dict[str, Any]]
    platformBays: List[Dict[str, Any]]
    routes: List[Dict[str, Any]]


class OperationalState(BaseModel):
    station: str
    timestamp: str
    tracks: Dict[str, str]    # track_id → free | occupied | reserved
    signals: Dict[str, str]   # signal_id → red | yellow | green
    points: Dict[str, str]    # point_id → normal | reverse
    summary: Dict[str, int]


class TrainPosition(BaseModel):
    number: str
    name: str
    from_: str
    to: str
    platform: int
    eta_min: int
    eta_label: str
    x: float
    y: float
    state: str    # at_platform | arriving | approaching
    section: str


class OpsAlert(BaseModel):
    id: str
    severity: str   # critical | warning | info
    type: str
    title: str
    message: str
    recommendation: str
    timestamp: str


class RouteRequestSchema(BaseModel):
    station: str
    route_id: str
    train_number: str


class RouteResponse(BaseModel):
    success: bool
    route_id: str
    train_number: str
    status: str
    reason: Optional[str] = None
    route_name: Optional[str] = None
    tracks_reserved: Optional[List[str]] = None
    signals_cleared: Optional[List[str]] = None
    points_set: Optional[List[str]] = None
    timestamp: Optional[str] = None