"""
RailMind AI — Delay Prediction Data Pipeline
=============================================
Handles loading, cleaning, and feature engineering for Indian Railways
delay datasets. Designed to be robust against column name variations
across different Kaggle datasets.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from typing import Optional
import logging

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Column alias map — keys are canonical names, values are accepted variants
# ---------------------------------------------------------------------------
COLUMN_ALIASES: dict[str, list[str]] = {
    "delay_minutes":          ["delay", "late_minutes", "delay_min", "arrival_delay",
                                "delay_in_minutes", "lateness", "delay_mins",
                                "arr_delay", "arrdelay"],
    "train_number":           ["train_no", "train_number", "trainno", "train_id",
                                "train_num", "trainnumber", "train_code"],
    "source_station":         ["source", "from_station", "origin", "src_station",
                                "from", "source_stn", "from_stn", "src"],
    "destination_station":    ["destination", "to_station", "dest", "dst_station",
                                "to", "destination_stn", "to_stn", "dst"],
    "scheduled_departure":    ["std", "scheduled_dep", "departure_time", "dep_time",
                                "sched_dep", "scheduled_departure_time",
                                "planned_departure", "dept_time"],
}


# ---------------------------------------------------------------------------
# 1. load_and_clean
# ---------------------------------------------------------------------------
def load_and_clean(filepath: str) -> pd.DataFrame:
    """
    Load a CSV file, normalise column names, remap known aliases to canonical
    names, remove bad rows and cap extreme outliers.

    Parameters
    ----------
    filepath : str
        Absolute or relative path to the CSV file.

    Returns
    -------
    pd.DataFrame
        Cleaned DataFrame with standardised column names.
    """
    logger.info(f"Loading data from: {filepath}")
    df = pd.read_csv(filepath, low_memory=False)
    logger.info(f"  Raw shape: {df.shape}")

    # --- normalise column names -----------------------------------------------
    df.columns = (
        df.columns
        .str.strip()
        .str.lower()
        .str.replace(r"\s+", "_", regex=True)
        .str.replace(r"[^\w]", "_", regex=True)   # replace non-word chars
        .str.strip("_")
    )

    # --- resolve alias → canonical name ---------------------------------------
    col_set = set(df.columns)
    rename_map: dict[str, str] = {}

    for canonical, aliases in COLUMN_ALIASES.items():
        if canonical in col_set:
            continue  # already present; nothing to do
        for alias in aliases:
            if alias in col_set:
                rename_map[alias] = canonical
                logger.info(f"  Remapping '{alias}' → '{canonical}'")
                break
        else:
            logger.warning(f"  Could not find a column for '{canonical}'. "
                           f"Checked aliases: {aliases}")

    df.rename(columns=rename_map, inplace=True)

    # --- validate required column exists --------------------------------------
    if "delay_minutes" not in df.columns:
        raise ValueError(
            "Could not locate a 'delay_minutes' column. "
            f"Available columns: {list(df.columns)}"
        )

    # --- coerce delay_minutes to numeric --------------------------------------
    df["delay_minutes"] = pd.to_numeric(df["delay_minutes"], errors="coerce")

    # --- drop null / negative delays ------------------------------------------
    before = len(df)
    df = df.dropna(subset=["delay_minutes"])
    df = df[df["delay_minutes"] >= 0]
    removed = before - len(df)
    if removed:
        logger.info(f"  Dropped {removed} rows with null or negative delay_minutes.")

    # --- cap at 300 minutes (5 hours) to remove extreme outliers --------------
    capped = (df["delay_minutes"] > 300).sum()
    if capped:
        logger.info(f"  Capping {capped} rows where delay_minutes > 300.")
    df["delay_minutes"] = df["delay_minutes"].clip(upper=300)

    # --- reset index ----------------------------------------------------------
    df.reset_index(drop=True, inplace=True)
    logger.info(f"  Cleaned shape: {df.shape}")
    return df


# ---------------------------------------------------------------------------
# 2. engineer_features
# ---------------------------------------------------------------------------
def engineer_features(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """
    Add temporal and categorical features required by the delay model.

    Temporal features are derived from 'scheduled_departure' when available;
    otherwise they are assigned randomly so the pipeline still runs end-to-end
    on synthetic or incomplete data.

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned DataFrame (output of load_and_clean).

    Returns
    -------
    (df_with_features, encoders_dict)
        df_with_features : pd.DataFrame  — original columns + new feature cols
        encoders_dict    : dict          — {"route": LabelEncoder,
                                            "train": LabelEncoder}
    """
    df = df.copy()
    n = len(df)

    # --- temporal features ----------------------------------------------------
    if "scheduled_departure" in df.columns:
        # Attempt to parse as datetime; fall back gracefully
        parsed = pd.to_datetime(df["scheduled_departure"], errors="coerce")
        valid_mask = parsed.notna()
        valid_ratio = valid_mask.mean()
        logger.info(f"  scheduled_departure parse success rate: {valid_ratio:.1%}")

        if valid_ratio >= 0.5:
            # Enough valid values — use parsed timestamps
            df["_dt"] = parsed
            df.loc[~valid_mask, "_dt"] = pd.NaT

            # Fill NaT with random values so every row gets a real number
            rand_hours   = np.random.randint(0, 24, size=n)
            rand_dow     = np.random.randint(0, 7,  size=n)
            rand_month   = np.random.randint(1, 13, size=n)

            hour_arr  = np.where(valid_mask, df["_dt"].dt.hour.fillna(0).astype(int), rand_hours)
            dow_arr   = np.where(valid_mask, df["_dt"].dt.dayofweek.fillna(0).astype(int), rand_dow)
            month_arr = np.where(valid_mask, df["_dt"].dt.month.fillna(1).astype(int), rand_month)

            df["hour_of_day"]  = hour_arr.astype(int)
            df["day_of_week"]  = dow_arr.astype(int)
            df["month"]        = month_arr.astype(int)
            df.drop(columns=["_dt"], inplace=True)
        else:
            logger.warning("  scheduled_departure parse rate < 50%; using random temporal values.")
            df["hour_of_day"] = np.random.randint(0, 24, size=n)
            df["day_of_week"] = np.random.randint(0, 7,  size=n)
            df["month"]       = np.random.randint(1, 13, size=n)
    else:
        logger.warning("  No 'scheduled_departure' column found; using random temporal values.")
        df["hour_of_day"] = np.random.randint(0, 24, size=n)
        df["day_of_week"] = np.random.randint(0, 7,  size=n)
        df["month"]       = np.random.randint(1, 13, size=n)

    # --- derived binary features ----------------------------------------------
    df["is_weekend"]   = (df["day_of_week"] >= 5).astype(int)
    peak_hours         = {7, 8, 9, 17, 18, 19, 20}
    df["is_peak_hour"] = df["hour_of_day"].isin(peak_hours).astype(int)

    # --- route feature --------------------------------------------------------
    if "source_station" in df.columns and "destination_station" in df.columns:
        df["source_station"]      = df["source_station"].astype(str).str.strip().str.upper()
        df["destination_station"] = df["destination_station"].astype(str).str.strip().str.upper()
        df["route"] = df["source_station"] + "_to_" + df["destination_station"]
    else:
        logger.warning("  source_station or destination_station missing; route set to 'UNKNOWN'.")
        df["route"] = "UNKNOWN"

    # --- label encoders -------------------------------------------------------
    route_encoder = LabelEncoder()
    df["route_encoded"] = route_encoder.fit_transform(df["route"].astype(str))

    if "train_number" in df.columns:
        train_encoder = LabelEncoder()
        df["train_encoded"] = train_encoder.fit_transform(
            df["train_number"].astype(str)
        )
    else:
        logger.warning("  'train_number' column missing; train_encoded set to 0.")
        train_encoder = LabelEncoder()
        train_encoder.fit(["UNKNOWN"])
        df["train_encoded"] = 0

    encoders_dict = {
        "route": route_encoder,
        "train": train_encoder,
    }

    logger.info(f"  Feature engineering complete. Shape: {df.shape}")
    return df, encoders_dict


# ---------------------------------------------------------------------------
# 3. get_feature_columns
# ---------------------------------------------------------------------------
def get_feature_columns() -> list[str]:
    """Return the ordered list of feature column names used for model training."""
    return [
        "hour_of_day",
        "day_of_week",
        "month",
        "is_weekend",
        "is_peak_hour",
        "route_encoded",
        "train_encoded",
    ]


# ---------------------------------------------------------------------------
# 4. prepare_training_data
# ---------------------------------------------------------------------------
def prepare_training_data(
    df: pd.DataFrame,
    encoders: dict,          # noqa: ARG001  (kept for API consistency)
) -> tuple[np.ndarray, np.ndarray]:
    """
    Extract the feature matrix X and target vector y from a fully-featured
    DataFrame.

    Parameters
    ----------
    df       : pd.DataFrame  — output of engineer_features
    encoders : dict          — encoders dict (unused here; accepted for API
                               consistency so callers can pass it through)

    Returns
    -------
    (X, y)
        X : np.ndarray, shape (n_samples, n_features)
        y : np.ndarray, shape (n_samples,)
    """
    feature_cols = get_feature_columns()
    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        raise ValueError(
            f"DataFrame is missing required feature columns: {missing}. "
            "Did you run engineer_features() first?"
        )

    X = df[feature_cols].values.astype(np.float32)
    y = df["delay_minutes"].values.astype(np.float32)
    logger.info(f"  X shape: {X.shape}  |  y shape: {y.shape}")
    return X, y


# ---------------------------------------------------------------------------
# Quick smoke-test (run directly: python process_delay_data.py)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import os

    sample_path = os.path.join(os.path.dirname(__file__), "train_delays.csv")
    if os.path.exists(sample_path):
        df_raw      = load_and_clean(sample_path)
        df_feat, enc = engineer_features(df_raw)
        X, y         = prepare_training_data(df_feat, enc)
        print(f"\n✅  Pipeline OK — X: {X.shape}, y: {y.shape}")
        print(f"   Feature columns : {get_feature_columns()}")
        print(f"   Delay stats (min/mean/max): "
              f"{y.min():.1f} / {y.mean():.1f} / {y.max():.1f}")
    else:
        print("⚠  No train_delays.csv found. Run generate_synthetic_delay_data.py first.")
