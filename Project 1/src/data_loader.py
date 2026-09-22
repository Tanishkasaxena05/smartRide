# ============================================================
# DATA LOADER
# ============================================================

import pandas as pd

from src.config import (
    DATA_FILE,
    LATITUDE_COLUMN,
    LONGITUDE_COLUMN,
    TIMESTAMP_COLUMN,
)

from src.geocoding import ensure_coordinates


def load_data():
    """
    Load the CSV dataset.
    """

    print("Loading dataset...")

    df = pd.read_csv(DATA_FILE)

    print(f"Dataset loaded successfully.")
    print(f"Rows: {len(df):,}")
    print(f"Columns: {len(df.columns)}")

    return df


def clean_data(df):
    """
    Basic validation and cleaning of geographical/timestamp data.
    """

    print("\nCleaning data...")

    original_size = len(df)

    # ----------------------------------------------------------
    # NEW: the raw CSV only has "Pickup Location" (a zone name),
    # not coordinates. Derive Pickup Latitude/Longitude from it
    # (geocoded once, then cached) before anything else runs.
    # ----------------------------------------------------------
    df = ensure_coordinates(df, LATITUDE_COLUMN, LONGITUDE_COLUMN)

    # ----------------------------------------------------------
    # NEW: the raw CSV has separate "Date" and "Time" columns,
    # not a single "Timestamp" column. Combine them if needed.
    # ----------------------------------------------------------
    if TIMESTAMP_COLUMN not in df.columns and "Date" in df.columns and "Time" in df.columns:
        date_parsed = pd.to_datetime(df["Date"], format="mixed", dayfirst=True, errors="coerce")
        time_parsed = pd.to_timedelta(df["Time"].astype(str), errors="coerce")
        df[TIMESTAMP_COLUMN] = date_parsed + time_parsed

    # Remove rows where latitude/longitude are missing
    df = df.dropna(
        subset=[
            LATITUDE_COLUMN,
            LONGITUDE_COLUMN
        ]
    ).copy()

    # Convert coordinates to numeric
    df[LATITUDE_COLUMN] = pd.to_numeric(
        df[LATITUDE_COLUMN],
        errors="coerce"
    )

    df[LONGITUDE_COLUMN] = pd.to_numeric(
        df[LONGITUDE_COLUMN],
        errors="coerce"
    )

    # Remove rows that became NaN
    df = df.dropna(
        subset=[
            LATITUDE_COLUMN,
            LONGITUDE_COLUMN
        ]
    )

    # Valid latitude
    df = df[
        df[LATITUDE_COLUMN].between(-90, 90)
    ]

    # Valid longitude
    df = df[
        df[LONGITUDE_COLUMN].between(-180, 180)
    ]

    # Timestamp
    if TIMESTAMP_COLUMN in df.columns:

        df[TIMESTAMP_COLUMN] = pd.to_datetime(
            df[TIMESTAMP_COLUMN],
            errors="coerce"
        )

        df = df.dropna(
            subset=[TIMESTAMP_COLUMN]
        )

        # Extract hour
        df["hour"] = df[TIMESTAMP_COLUMN].dt.hour

        # Extract date
        df["date"] = df[TIMESTAMP_COLUMN].dt.date

        # Day of week
        df["day_of_week"] = (
            df[TIMESTAMP_COLUMN]
            .dt.day_name()
        )

    # Remove duplicate rows
    df = df.drop_duplicates()

    removed = original_size - len(df)

    print(f"Original rows: {original_size:,}")
    print(f"Clean rows: {len(df):,}")
    print(f"Removed rows: {removed:,}")

    return df


def add_time_period(df):
    """
    Add Morning/Afternoon/Evening/Night category.
    """

    def get_period(hour):

        if 6 <= hour < 12:
            return "Morning"

        elif 12 <= hour < 17:
            return "Afternoon"

        elif 17 <= hour < 22:
            return "Evening"

        else:
            return "Night"

    if "hour" in df.columns:

        df["time_period"] = (
            df["hour"]
            .apply(get_period)
        )

    return df
