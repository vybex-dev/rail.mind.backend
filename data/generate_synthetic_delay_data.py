"""
RailMind AI — Synthetic Delay Data Generator
=============================================
Generates 50,000 realistic train-delay rows using the 15 trains defined in
sample_trains.json and domain-specific statistical patterns observed in
actual Indian Railways data.

Usage
-----
    python data/generate_synthetic_delay_data.py

Output
------
    data/train_delays.csv  — ready to feed into process_delay_data.py
"""

import json
import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
RNG_SEED     = 42
NUM_ROWS     = 50_000
OUTPUT_PATH  = os.path.join(os.path.dirname(__file__), "train_delays.csv")
TRAINS_PATH  = os.path.join(os.path.dirname(__file__), "sample_trains.json")

rng = np.random.default_rng(RNG_SEED)


# ---------------------------------------------------------------------------
# Delay-generation parameters
# ---------------------------------------------------------------------------
PEAK_HOURS       = {7, 8, 9, 17, 18, 19, 20}   # high-traffic windows
MONSOON_MONTHS   = {6, 7, 8, 9}                 # June–September
MONSOON_FACTOR   = 1.4

# Base mean delays by time-of-day category
DELAY_PROFILE = {
    "peak":     {"mean": 35.0, "std": 20.0},
    "night":    {"mean": 18.0, "std": 12.0},   # 21:00–05:59
    "off_peak": {"mean": 12.0, "std":  9.0},
    "weekend":  {"mean": 10.0, "std":  8.0},   # overrides all above on Sat/Sun
}


def _hour_category(hour: int, is_weekend: bool) -> str:
    if is_weekend:
        return "weekend"
    if hour in PEAK_HOURS:
        return "peak"
    if hour >= 21 or hour < 6:
        return "night"
    return "off_peak"


def _sample_delay(
    hour: int,
    is_weekend: bool,
    month: int,
    base_avg: float,
    n: int,
) -> np.ndarray:
    """
    Draw n delay samples given temporal context and train-specific base average.

    The final mean blends the time-of-day profile with the train's own
    historical average (60/40 split) to preserve train-level variation while
    honouring macro patterns.
    """
    cat = _hour_category(hour, is_weekend)
    profile = DELAY_PROFILE[cat]

    # blend profile mean with train's historical average
    blended_mean = 0.6 * profile["mean"] + 0.4 * base_avg
    blended_std  = profile["std"]

    if month in MONSOON_MONTHS:
        blended_mean *= MONSOON_FACTOR

    # truncated normal (≥0) drawn via clip of gaussian
    raw = rng.normal(loc=blended_mean, scale=blended_std, size=n)
    return np.clip(raw, 0, 300).round(1)


# ---------------------------------------------------------------------------
# Build the dataset
# ---------------------------------------------------------------------------
def generate(num_rows: int = NUM_ROWS) -> pd.DataFrame:
    # Load train catalogue
    with open(TRAINS_PATH, "r") as f:
        trains = json.load(f)

    num_trains = len(trains)
    rows_per_train = num_rows // num_trains
    remainder      = num_rows % num_trains

    records: list[dict] = []

    # Synthetic journey date pool spanning 3 years (2022-2024)
    start_date = datetime(2022, 1, 1)
    date_pool  = [start_date + timedelta(days=i) for i in range(3 * 365)]

    for idx, train in enumerate(trains):
        n = rows_per_train + (1 if idx < remainder else 0)

        # Randomly assign dates
        chosen_dates = [date_pool[i % len(date_pool)]
                        for i in rng.integers(0, len(date_pool), size=n)]

        # Hours distributed realistically:
        # 30 % peak, 25 % night, 45 % off-peak
        hour_weights = np.array([
            3.0 if h in PEAK_HOURS         # peak
            else 1.5 if (h >= 21 or h < 6) # night
            else 2.5                        # off-peak
            for h in range(24)
        ])
        hour_weights /= hour_weights.sum()
        hours = rng.choice(24, size=n, p=hour_weights)

        for i in range(n):
            dt         = chosen_dates[i].replace(hour=int(hours[i]),
                                                  minute=int(rng.integers(0, 60)))
            dow        = dt.weekday()           # 0=Mon … 6=Sun
            month      = dt.month
            is_weekend = dow >= 5

            delay = float(_sample_delay(
                hour=int(hours[i]),
                is_weekend=is_weekend,
                month=month,
                base_avg=train["avg_delay_minutes"],
                n=1,
            )[0])

            records.append({
                "train_number":          train["train_number"],
                "train_name":            train["train_name"],
                "source_station":        train["source_code"],
                "destination_station":   train["destination_code"],
                "scheduled_departure":   dt.strftime("%Y-%m-%d %H:%M:%S"),
                "day_of_week":           dow,
                "month":                 month,
                "delay_minutes":         delay,
            })

    df = pd.DataFrame(records)

    # Shuffle so rows are not grouped by train
    df = df.sample(frac=1, random_state=RNG_SEED).reset_index(drop=True)
    return df


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("🚂  RailMind AI — Generating synthetic delay data …")
    print(f"    Rows    : {NUM_ROWS:,}")
    print(f"    Output  : {OUTPUT_PATH}")
    print()

    df = generate(NUM_ROWS)
    df.to_csv(OUTPUT_PATH, index=False)

    # ── Summary statistics ────────────────────────────────────────────────
    print("✅  File written.\n")
    print("── Dataset summary ─────────────────────────────────────────────")
    print(f"   Shape          : {df.shape}")
    print(f"   Columns        : {list(df.columns)}")
    print()

    stats = df["delay_minutes"].describe()
    print("── delay_minutes statistics ─────────────────────────────────────")
    for stat, val in stats.items():
        print(f"   {stat:8s}: {val:.2f}")

    print()
    print("── Per-train mean delay (sorted) ────────────────────────────────")
    per_train = (
        df.groupby(["train_number", "train_name"])["delay_minutes"]
        .mean()
        .round(1)
        .sort_values()
    )
    for (tn, name), avg in per_train.items():
        print(f"   {tn}  {name:<42s}  {avg:5.1f} min")

    print()
    print("── Monthly mean delay (monsoon highlighted) ─────────────────────")
    MONTH_NAMES = ["Jan","Feb","Mar","Apr","May","Jun",
                   "Jul","Aug","Sep","Oct","Nov","Dec"]
    for m, avg in df.groupby("month")["delay_minutes"].mean().items():
        tag = " 🌧️  monsoon" if m in {6, 7, 8, 9} else ""
        print(f"   {MONTH_NAMES[m-1]:3s}: {avg:5.1f} min{tag}")

    print()
    print("── Peak vs off-peak mean delay ───────────────────────────────────")
    peak_mask = df["scheduled_departure"].apply(
        lambda s: int(s.split(" ")[1].split(":")[0]) in {7,8,9,17,18,19,20}
    )
    print(f"   Peak hours   : {df.loc[ peak_mask, 'delay_minutes'].mean():.1f} min")
    print(f"   Off-peak     : {df.loc[~peak_mask, 'delay_minutes'].mean():.1f} min")
    print()
