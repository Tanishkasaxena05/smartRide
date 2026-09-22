# ============================================================
# BUILD DATASET
# ============================================================
# Turns the raw ride-level CSV into the aggregated, model-ready
# dataset app.py expects: one row per (date, hour, cluster) with
# the resulting ride demand, a day-of-month field, and a rainfall
# feature.
#
# Steps:
#   1. Load + clean raw bookings
#   2. Geocode each Pickup Location zone name -> lat/long
#   3. DBSCAN those zones into a smaller number of geographic
#      clusters (same technique as Project 1)
#   4. Attach an approximate seasonal rainfall value per date
#   5. Aggregate to (date, hour, cluster, day, rainfall_mm) -> demand

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN

from src.config import (
    RAW_DATA_FILE,
    PICKUP_LOCATION_COLUMN,
    CLUSTER_EPS_METERS,
    CLUSTER_MIN_SAMPLES,
    MONTHLY_AVG_RAINFALL_MM,
    DATASET_FILE,
)
from src.geocoding import geocode_pickup_locations

EARTH_RADIUS_METERS = 6_371_000


def load_and_clean(path=RAW_DATA_FILE):
    df = pd.read_csv(path)
    df["Date"] = pd.to_datetime(df["Date"], format="mixed", dayfirst=True)
    df["Hour"] = pd.to_datetime(df["Time"], format="%H:%M:%S").dt.hour
    df["day"] = df["Date"].dt.day          # day of month (1-31), matches the dashboard's slider
    df["month"] = df["Date"].dt.month
    df["date"] = df["Date"].dt.date
    return df


def assign_clusters(df, eps_meters=CLUSTER_EPS_METERS, min_samples=CLUSTER_MIN_SAMPLES):
    """Groups pickup zones into a handful of geographic clusters (DBSCAN,
    weighted by each zone's total ride volume — see Project 1 for the
    full explanation of why weighting on unique points beats clustering
    every individual row)."""

    locs = geocode_pickup_locations(df)

    counts = (
        df[PICKUP_LOCATION_COLUMN]
        .value_counts()
        .rename_axis(PICKUP_LOCATION_COLUMN)
        .reset_index(name="_ride_count")
    )
    locs = locs.merge(counts, on=PICKUP_LOCATION_COLUMN, how="left")
    locs["_ride_count"] = locs["_ride_count"].fillna(1)

    coords_rad = np.radians(locs[["lat", "lon"]].values)
    eps_radians = eps_meters / EARTH_RADIUS_METERS

    model = DBSCAN(eps=eps_radians, min_samples=min_samples, metric="haversine")
    locs["cluster"] = model.fit_predict(coords_rad, sample_weight=locs["_ride_count"])

    n_clusters = locs["cluster"].nunique() - (1 if -1 in locs["cluster"].values else 0)
    n_noise_zones = (locs["cluster"] == -1).sum()
    print(f"Geographic clustering: {n_clusters} clusters "
          f"({n_noise_zones} zones classified as noise / too sparse to group)")

    df = df.merge(
        locs[[PICKUP_LOCATION_COLUMN, "cluster"]],
        on=PICKUP_LOCATION_COLUMN,
        how="left",
    )
    return df


def add_rainfall(df, seed=42):
    """Adds an approximate rainfall_mm column per date, using Delhi NCR's
    typical monsoon-heavy seasonal pattern (NOT real historical weather —
    no such data exists for this dataset's dates). Most days get 0mm;
    a probability-of-rain scaled to the month's average decides which
    days get a nonzero (randomly sampled) rainfall amount."""

    unique_dates = pd.Series(df["date"].unique(), name="date").to_frame()
    unique_dates["month"] = pd.to_datetime(unique_dates["date"]).dt.month

    rng = np.random.default_rng(seed)
    rainfall = []
    for month in unique_dates["month"]:
        avg = MONTHLY_AVG_RAINFALL_MM[month]
        rain_prob = min(0.6, avg / 200)
        if rng.random() < rain_prob:
            # rainy day: sample around the month's average with spread
            rainfall.append(round(float(rng.gamma(shape=2.0, scale=avg / 2.0)), 1))
        else:
            rainfall.append(0.0)
    unique_dates["rainfall_mm"] = rainfall

    df = df.merge(unique_dates[["date", "rainfall_mm"]], on="date", how="left")
    return df


def finalize_raw(df):
    """Keeps ride-level granularity (one row per booking) with the columns
    the dashboard needs: date, hour, day, day_name, cluster, rainfall_mm,
    booking_id. Aggregation to demand happens downstream (in app.py / via
    aggregate_to_demand below) so the dashboard can also build a per-date
    calendar view (holiday/event/rainfall) before collapsing to counts."""
    df = df.dropna(subset=["cluster"]).copy()
    df["cluster"] = df["cluster"].astype(int)
    df["day_name"] = pd.to_datetime(df["date"]).dt.day_name()
    df["booking_id"] = df["Booking ID"].astype(str)

    return df[["date", "Hour", "day", "day_name", "cluster", "rainfall_mm", "booking_id"]].rename(
        columns={"Hour": "hour"}
    )


def aggregate_to_demand(raw_df):
    """Collapses ride-level rows into (date, hour, cluster, day, rainfall_mm)
    -> demand (ride count). Shared by app.py and run_analysis.py so both
    use identical aggregation logic."""
    agg = (
        raw_df.groupby(["date", "hour", "cluster", "day", "rainfall_mm"])["booking_id"]
        .count()
        .reset_index()
        .rename(columns={"booking_id": "demand"})
    )
    return agg


def build(output_path=DATASET_FILE):
    print("Loading and cleaning raw bookings...")
    df = load_and_clean()
    print(f"Loaded {len(df):,} bookings.")

    print("\nAssigning geographic clusters...")
    df = assign_clusters(df)

    print("\nAdding seasonal rainfall proxy...")
    df = add_rainfall(df)

    print("\nFinalizing ride-level dataset...")
    raw = finalize_raw(df)
    print(f"Saving {len(raw):,} ride-level rows.")

    raw.to_csv(output_path, index=False)
    print(f"\nSaved -> {output_path}")
    return raw


if __name__ == "__main__":
    build()
