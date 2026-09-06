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
        Minimum number of points required
        to form a dense region.
    """

    if len(df) == 0:
        return df.copy()

    result = df.copy()

    # Extract coordinates
    coordinates = result[
        [
            LATITUDE_COLUMN,
            LONGITUDE_COLUMN
        ]
    ].values

    # Convert degrees to radians
    coordinates_rad = np.radians(
        coordinates
    )

    # Convert meters to radians
    eps_radians = (
        eps_meters /
        EARTH_RADIUS_METERS
    )

    print(
        f"Running DBSCAN: "
        f"eps={eps_meters}m, "
        f"min_samples={min_samples}"
    )

    model = DBSCAN(
        eps=eps_radians,
        min_samples=min_samples,
        metric="haversine",
        algorithm="ball_tree",
        n_jobs=-1
    )

    labels = model.fit_predict(
        coordinates_rad
    )

    result["cluster"] = labels

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