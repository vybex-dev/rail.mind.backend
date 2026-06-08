"""
RailMind AI — Station Crowd Data Generator
==========================================
Generates 90 days of synthetic hourly crowd data for 5 major Indian
railway stations with realistic temporal patterns.

Usage
-----
    python data/generate_crowd_data.py

Output
------
    data/crowd_data.csv
"""

import os
import random
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Station catalogue
# ---------------------------------------------------------------------------
STATIONS: dict[str, dict] = {
    "NDLS": {"name": "New Delhi",       "base_crowd": 4500, "platforms": 16},
    "BCT":  {"name": "Mumbai Central",  "base_crowd": 3200, "platforms":  8},
    "MAS":  {"name": "Chennai Central", "base_crowd": 2800, "platforms": 12},
    "HWH":  {"name": "Howrah Junction", "base_crowd": 3800, "platforms": 15},
    "BPL":  {"name": "Bhopal Junction", "base_crowd": 1500, "platforms":  6},
}

# ---------------------------------------------------------------------------
# Hour multipliers  (index 0 = midnight, index 23 = 11 pm)
# ---------------------------------------------------------------------------
HOUR_MULTIPLIERS: dict[int, float] = {
    0:  0.15, 1:  0.15, 2:  0.15, 3:  0.15, 4:  0.15, 5:  0.15,
    6:  0.40,
    7:  0.85,
    8:  1.00, 9:  1.00,          # morning peak
    10: 0.65, 11: 0.65,
    12: 0.70, 13: 0.70,
    14: 0.55, 15: 0.55, 16: 0.55,
    17: 0.80,
    18: 1.00, 19: 1.00, 20: 1.00, # evening peak
    21: 0.75,
    22: 0.50,
    23: 0.25,
}

MONSOON_MONTHS   = {6, 7, 8, 9}
WEEKEND_MULT     = 0.75
MONSOON_MULT     = 1.15
FESTIVAL_MULT    = 1.80
FESTIVAL_DAYS    = 8      # number of festival days randomly scattered
NOISE_STD_RATIO  = 0.08   # std = base * this


# ---------------------------------------------------------------------------
# congestion label helper
# ---------------------------------------------------------------------------
def _congestion(crowd: int, base: int) -> str:
    if crowd < base * 0.40:
        return "low"
    if crowd < base * 0.75:
        return "medium"
    return "high"


# ---------------------------------------------------------------------------
# Core generator
# ---------------------------------------------------------------------------
def generate_all_stations_data(days: int = 90) -> pd.DataFrame:
    """
    Generate hourly crowd rows for all STATIONS over `days` calendar days.

    Parameters
    ----------
    days : int
        Number of days of history to generate (default 90).

    Returns
    -------
    pd.DataFrame
        Columns: timestamp, station_code, station_name, crowd_count,
                 congestion_level
    """
    rng = np.random.default_rng(seed=2024)

    # Determine festival dates (same set for all stations, reflects national events)
    start_dt = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
    all_dates = [start_dt + timedelta(days=d) for d in range(days)]
    festival_dates: set[str] = {
        d.strftime("%Y-%m-%d")
        for d in random.sample(all_dates, min(FESTIVAL_DAYS, days))
    }

    records: list[dict] = []

    for code, meta in STATIONS.items():
        base    = meta["base_crowd"]
        name    = meta["name"]
        noise_std = base * NOISE_STD_RATIO
        max_crowd = int(base * 1.9)

        for day_offset in range(days):
            day_dt  = start_dt + timedelta(days=day_offset)
            is_weekend  = day_dt.weekday() >= 5       # Sat / Sun
            is_monsoon  = day_dt.month in MONSOON_MONTHS
            is_festival = day_dt.strftime("%Y-%m-%d") in festival_dates

            for hour in range(24):
                ts = day_dt + timedelta(hours=hour)

                mult = HOUR_MULTIPLIERS[hour]
                if is_weekend:
                    mult *= WEEKEND_MULT
                if is_monsoon:
                    mult *= MONSOON_MULT
                if is_festival:
                    mult *= FESTIVAL_MULT

                crowd_raw = base * mult + rng.normal(0, noise_std)
                crowd     = int(np.clip(crowd_raw, 0, max_crowd))

                records.append({
                    "timestamp":        ts.strftime("%Y-%m-%d %H:%M:%S"),
                    "station_code":     code,
                    "station_name":     name,
                    "crowd_count":      crowd,
                    "congestion_level": _congestion(crowd, base),
                })

    df = pd.DataFrame(records)

    # ── Save ──────────────────────────────────────────────────────────────
    out_path = os.path.join(os.path.dirname(__file__), "crowd_data.csv")
    df.to_csv(out_path, index=False)

    # ── Summary ───────────────────────────────────────────────────────────
    print(f"✅  Crowd data written → {out_path}")
    print(f"   Total rows : {len(df):,}")
    print(f"   Date range : {df['timestamp'].min()[:10]}  →  "
          f"{df['timestamp'].max()[:10]}")
    print()
    print("── Rows per station ─────────────────────────────────────────────")
    per_station = df.groupby(["station_code", "station_name"]).size()
    for (code, name), n in per_station.items():
        print(f"   {code}  {name:<22s}  {n:,} rows")

    print()
    print("── Mean crowd per station ───────────────────────────────────────")
    mean_crowd = df.groupby("station_code")["crowd_count"].mean()
    for code, avg in mean_crowd.items():
        print(f"   {code}  {avg:,.0f} pax/hr (avg)")

    print()
    print("── Congestion distribution ──────────────────────────────────────")
    congestion_dist = df["congestion_level"].value_counts()
    total = len(df)
    for level, count in congestion_dist.items():
        pct = count / total * 100
        print(f"   {level:<8s}  {count:>8,}  ({pct:.1f}%)")

    return df


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    generate_all_stations_data(days=90)
