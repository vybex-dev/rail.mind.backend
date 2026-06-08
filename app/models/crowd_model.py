"""
RailMind AI — Station Crowd Forecasting Module
===============================================
Provides a singleton `crowd_forecaster` that FastAPI routers import.

On startup it tries to load data/crowd_data.csv and pre-computes per-station,
per-hour average crowd counts.  If the CSV is absent it falls back to the
same rule-based multiplier table used by the data generator so every endpoint
always returns a plausible response.

Usage
-----
    from app.models.crowd_model import crowd_forecaster

    result = crowd_forecaster.predict_crowd("NDLS", hours_ahead=2)
"""

import os
import random
import sys
from datetime import datetime, timedelta
from typing import Optional

import numpy as np

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
_THIS_DIR     = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_THIS_DIR, "..", ".."))

if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

_CROWD_CSV = os.path.join(_PROJECT_ROOT, "data", "crowd_data.csv")

# ---------------------------------------------------------------------------
# Shared constants (mirror those in generate_crowd_data.py)
# ---------------------------------------------------------------------------
HOUR_MULTIPLIERS: dict[int, float] = {
    0:  0.15, 1:  0.15, 2:  0.15, 3:  0.15, 4:  0.15, 5:  0.15,
    6:  0.40,
    7:  0.85,
    8:  1.00, 9:  1.00,
    10: 0.65, 11: 0.65,
    12: 0.70, 13: 0.70,
    14: 0.55, 15: 0.55, 16: 0.55,
    17: 0.80,
    18: 1.00, 19: 1.00, 20: 1.00,
    21: 0.75,
    22: 0.50,
    23: 0.25,
}

MONSOON_MONTHS = {6, 7, 8, 9}
PEAK_HOURS     = {7, 8, 9, 17, 18, 19, 20}

# Trains used for mock platform allocation (train_number, short_name)
_SAMPLE_TRAINS = [
    ("12301", "Howrah Rajdhani Exp"),
    ("12951", "Mumbai Rajdhani Exp"),
    ("12002", "Bhopal Shatabdi Exp"),
    ("22439", "Vande Bharat Exp"),
    ("12433", "Chennai Rajdhani Exp"),
    ("12595", "Gorakhpur Humsafar Exp"),
    ("12071", "Dadar Jan Shatabdi Exp"),
    ("12213", "Delhi Duronto Exp"),
]

_PLATFORM_RECOMMENDATIONS = [
    "Use for incoming Shatabdi — low crowd expected",
    "Reserve for Rajdhani departure — premium service",
    "Available for suburban services",
    "Recommended for long-distance arrivals",
    "Keep clear — maintenance window scheduled",
    "Optimal for next departure — good crowd spread",
    "Low footfall predicted — good for diverted trains",
    "High footfall expected — deploy extra staff",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _congestion(crowd: int, base: int) -> str:
    if crowd < base * 0.40:
        return "low"
    if crowd < base * 0.75:
        return "medium"
    return "high"


def _time_label(hour: int) -> str:
    """Convert 24h hour int to readable label, e.g. 13 → '1:00 PM'."""
    dt = datetime.now().replace(hour=hour % 24, minute=0, second=0, microsecond=0)
    return dt.strftime("%-I:%M %p")   # e.g. "8:00 AM"


def _rule_based_crowd(base: int, hour: int, now: datetime) -> int:
    """Estimate crowd purely from multipliers (no CSV needed)."""
    mult = HOUR_MULTIPLIERS[hour % 24]
    if now.weekday() >= 5:
        mult *= 0.75
    if now.month in MONSOON_MONTHS:
        mult *= 1.15
    raw = base * mult + random.gauss(0, base * 0.05)
    return max(0, int(raw))


# ---------------------------------------------------------------------------
# CrowdForecaster
# ---------------------------------------------------------------------------
class CrowdForecaster:
    """
    Forecasts station crowd levels for the next 8 hours and provides
    platform allocation recommendations.
    """

    STATIONS: dict[str, dict] = {
        "NDLS": {"name": "New Delhi",       "base_crowd": 4500, "platforms": 16},
        "BCT":  {"name": "Mumbai Central",  "base_crowd": 3200, "platforms":  8},
        "MAS":  {"name": "Chennai Central", "base_crowd": 2800, "platforms": 12},
        "HWH":  {"name": "Howrah Junction", "base_crowd": 3800, "platforms": 15},
        "BPL":  {"name": "Bhopal Junction", "base_crowd": 1500, "platforms":  6},
    }

    def __init__(self) -> None:
        self.hourly_averages: dict[str, dict[int, float]] = {}
        self.is_loaded = False
        self._try_load()

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------
    def _try_load(self) -> None:
        """Load crowd CSV and pre-compute per-station hourly averages."""
        if not os.path.exists(_CROWD_CSV):
            print("CrowdForecaster: crowd_data.csv not found — using rule-based mode")
            return

        try:
            import pandas as pd

            df = pd.read_csv(_CROWD_CSV, parse_dates=["timestamp"])
            df["hour"] = df["timestamp"].dt.hour

            for code in self.STATIONS:
                sub = df[df["station_code"] == code]
                avg_by_hour = (
                    sub.groupby("hour")["crowd_count"]
                    .mean()
                    .to_dict()
                )
                self.hourly_averages[code] = avg_by_hour

            self.is_loaded = True
            rows = len(df)
            stations = df["station_code"].nunique()
            print(f"CrowdForecaster: loaded ✅  "
                  f"({rows:,} rows, {stations} stations)")

        except Exception as exc:          # pragma: no cover
            print(f"CrowdForecaster: load failed ({exc}); using rule-based mode")
            self.is_loaded = False

    # ------------------------------------------------------------------
    # Crowd estimate for a single station + hour
    # ------------------------------------------------------------------
    def _estimate(self, station_code: str, hour: int, now: datetime) -> int:
        """Return a crowd count with small live noise added."""
        base = self.STATIONS[station_code]["base_crowd"]

        if self.is_loaded and station_code in self.hourly_averages:
            avg = self.hourly_averages[station_code].get(hour % 24, base * 0.5)
            # Add ±5 % live noise so successive calls feel dynamic
            noise = random.gauss(0, base * 0.03)
            return max(0, int(avg + noise))

        return _rule_based_crowd(base, hour, now)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def predict_crowd(
        self,
        station_code: str,
        hours_ahead: int = 2,
    ) -> dict:
        """
        Predict crowd levels for the requested station for the next 8 hours.

        Parameters
        ----------
        station_code : str   One of NDLS, BCT, MAS, HWH, BPL
        hours_ahead  : int   Unused parameter kept for API compatibility;
                             the forecast always covers 8 hours ahead.

        Returns
        -------
        dict  Full forecast payload (see module docstring).
        """
        station_code = station_code.strip().upper()
        if station_code not in self.STATIONS:
            station_code = "NDLS"    # graceful fallback

        meta  = self.STATIONS[station_code]
        base  = meta["base_crowd"]
        now   = datetime.now()
        current_hour = now.hour

        # ── Current crowd ────────────────────────────────────────────────
        current_crowd = self._estimate(station_code, current_hour, now)
        current_congestion = _congestion(current_crowd, base)

        # ── 8-hour forecast ──────────────────────────────────────────────
        forecast: list[dict] = []
        for offset in range(8):
            fh    = (current_hour + offset) % 24
            count = self._estimate(station_code, fh, now)
            forecast.append({
                "hour":             fh,
                "time_label":       _time_label(fh),
                "crowd_count":      count,
                "congestion_level": _congestion(count, base),
            })

        # ── Alert logic ──────────────────────────────────────────────────
        alert: Optional[str] = None
        upcoming_congestion = [f["congestion_level"] for f in forecast[:2]]

        if current_congestion == "high":
            alert = "Station currently at peak capacity"
        elif "high" in upcoming_congestion:
            alert = "Peak hour approaching — expect heavy crowds"

        # ── Platform allocation ──────────────────────────────────────────
        platform_alloc = self.generate_platform_allocation(station_code, forecast)

        return {
            "station":                meta["name"],
            "station_code":           station_code,
            "current_estimated_crowd": current_crowd,
            "congestion_level":       current_congestion,
            "forecast":               forecast,
            "platform_allocation":    platform_alloc,
            "alert":                  alert,
        }

    # ------------------------------------------------------------------
    # Platform allocation
    # ------------------------------------------------------------------
    def generate_platform_allocation(
        self,
        station_code: str,
        forecast: list,
    ) -> list:
        """
        Generate a mock platform allocation recommendation.

        Returns up to 4 platform entries based on the station's platform count
        and the upcoming crowd forecast.
        """
        station_code = station_code.strip().upper()
        if station_code not in self.STATIONS:
            station_code = "NDLS"

        total_platforms = self.STATIONS[station_code]["platforms"]
        num_to_show     = min(4, total_platforms)

        # Decide statuses based on forecast congestion in next 2 hours
        upcoming_high = sum(
            1 for f in forecast[:2] if f["congestion_level"] == "high"
        )

        statuses = ["available", "occupied", "recommended", "available"]
        if upcoming_high >= 2:
            statuses = ["occupied", "occupied", "recommended", "available"]

        result: list[dict] = []
        used_trains: set[int] = set()

        for i in range(num_to_show):
            plat_num = i + 1
            status   = statuses[i % len(statuses)]

            # Pick a current train for occupied platforms
            current_train: Optional[str] = None
            if status == "occupied":
                idx = (hash(station_code + str(plat_num)) % len(_SAMPLE_TRAINS))
                while idx in used_trains and len(used_trains) < len(_SAMPLE_TRAINS):
                    idx = (idx + 1) % len(_SAMPLE_TRAINS)
                used_trains.add(idx)
                tn, tname = _SAMPLE_TRAINS[idx]
                current_train = f"{tn} {tname}"

            rec_idx = (hash(station_code + str(plat_num) + "rec")
                       % len(_PLATFORM_RECOMMENDATIONS))
            recommendation = _PLATFORM_RECOMMENDATIONS[rec_idx]

            result.append({
                "platform":       plat_num,
                "status":         status,
                "current_train":  current_train,
                "recommendation": recommendation,
            })

        return result

    # ------------------------------------------------------------------
    # Heatmap data
    # ------------------------------------------------------------------
    def get_heatmap_data(self, station_code: str) -> dict:
        """
        Return 24-hour crowd profile for chart rendering.

        Parameters
        ----------
        station_code : str

        Returns
        -------
        dict with keys: hours, time_labels, crowd_counts,
                        congestion_levels, peak_hours
        """
        station_code = station_code.strip().upper()
        if station_code not in self.STATIONS:
            station_code = "NDLS"

        base = self.STATIONS[station_code]["base_crowd"]
        now  = datetime.now()

        hours: list[int]   = list(range(24))
        counts: list[int]  = []
        levels: list[str]  = []

        for h in hours:
            if self.is_loaded and station_code in self.hourly_averages:
                avg = self.hourly_averages[station_code].get(h, base * 0.5)
                c   = max(0, int(avg))
            else:
                c = _rule_based_crowd(base, h, now)
            counts.append(c)
            levels.append(_congestion(c, base))

        time_labels = [
            datetime.now().replace(hour=h, minute=0).strftime("%-I%p").lower()
            for h in hours
        ]

        return {
            "hours":             hours,
            "time_labels":       time_labels,
            "crowd_counts":      counts,
            "congestion_levels": levels,
            "peak_hours":        sorted(PEAK_HOURS),
        }

    # ------------------------------------------------------------------
    # Station catalogue
    # ------------------------------------------------------------------
    def get_stations(self) -> list:
        """Return station metadata list for use in dropdowns / listings."""
        return [
            {
                "station_code": code,
                "station_name": meta["name"],
                "base_crowd":   meta["base_crowd"],
                "platforms":    meta["platforms"],
            }
            for code, meta in self.STATIONS.items()
        ]


# ---------------------------------------------------------------------------
# Module-level singleton — imported by FastAPI routers
# ---------------------------------------------------------------------------
crowd_forecaster = CrowdForecaster()
