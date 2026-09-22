# ============================================================
# DBSCAN CLUSTERING
# ============================================================

import numpy as np
import pandas as pd

from sklearn.cluster import DBSCAN

from src.config import (
    LATITUDE_COLUMN,
    LONGITUDE_COLUMN,
    EPS_METERS,
    MIN_SAMPLES,
)


EARTH_RADIUS_METERS = 6_371_000


def run_dbscan(
    df,
    eps_meters=EPS_METERS,
    min_samples=MIN_SAMPLES
):
    """
    Run DBSCAN using Haversine distance.

    eps_meters:
        Maximum neighborhood distance in meters.

    min_samples:
        Minimum total ride volume required within that radius
        to form a dense region (a zone's rides all share one exact
        coordinate, so this behaves like "does this area have at
        least this many rides").

    PERFORMANCE NOTE: many rows share the exact same coordinate
    (all rides from the same named pickup zone). Running DBSCAN on
    every individual row wastes memory recomputing identical
    neighborhoods over and over, and at larger search radii — where
    several zones' points fall within eps of each other — the
    per-point neighbor lists can grow large enough to exhaust
    memory and crash. Instead we cluster only the unique
    coordinates, using each one's ride count as its DBSCAN
    `sample_weight` (a point with weight w counts as w points
    toward min_samples) — same result, far less computation, and
    stays fast/safe at any eps or min_samples value.
    """

    if len(df) == 0:
        return df.copy()

    result = df.copy()

    # One row per unique coordinate, with its total ride count
    unique_points = (
        result
        .groupby([LATITUDE_COLUMN, LONGITUDE_COLUMN])
        .size()
        .reset_index(name="_ride_count")
    )

    coordinates_rad = np.radians(
        unique_points[[LATITUDE_COLUMN, LONGITUDE_COLUMN]].values
    )

    # Convert meters to radians
    eps_radians = (
        eps_meters /
        EARTH_RADIUS_METERS
    )

    print(
        f"Running DBSCAN: "
        f"eps={eps_meters}m, "
        f"min_samples={min_samples} "
        f"({len(unique_points)} unique locations)"
    )

    model = DBSCAN(
        eps=eps_radians,
        min_samples=min_samples,
        metric="haversine",
        algorithm="ball_tree",
        n_jobs=-1
    )

    unique_points["cluster"] = model.fit_predict(
        coordinates_rad,
        sample_weight=unique_points["_ride_count"]
    )

    # Map each unique location's cluster label back onto every ride
    result = result.merge(
        unique_points[[LATITUDE_COLUMN, LONGITUDE_COLUMN, "cluster"]],
        on=[LATITUDE_COLUMN, LONGITUDE_COLUMN],
        how="left"
    )

    return result


def run_time_based_clustering(df):
    """
    Run DBSCAN separately for each time period.

    Cluster IDs are prefixed with the time period
    to prevent confusion between clusters from
    different periods.
    """

    all_results = []

    periods = [
        "Morning",
        "Afternoon",
        "Evening",
        "Night"
    ]

    for period in periods:

        period_data = df[
            df["time_period"] == period
        ].copy()

        print(
            f"\n{'=' * 50}"
        )

        print(
            f"Processing: {period}"
        )

        print(
            f"Number of rides: "
            f"{len(period_data):,}"
        )

        if len(period_data) == 0:
            continue

        period_data = run_dbscan(
            period_data
        )

        # Create readable cluster IDs
        def create_label(cluster):

            if cluster == -1:
                return "Noise"

            return f"{period}_{cluster}"

        period_data["time_cluster"] = (
            period_data["cluster"]
            .apply(create_label)
        )

        all_results.append(
            period_data
        )

    if not all_results:
        return df.copy()

    final_df = pd.concat(
        all_results,
        ignore_index=True
    )

    return final_df