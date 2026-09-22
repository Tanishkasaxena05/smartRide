# ============================================================
# CLUSTER ANALYSIS
# ============================================================

import pandas as pd

from src.config import (
    LATITUDE_COLUMN,
    LONGITUDE_COLUMN,
    FARE_COLUMN,
    DISTANCE_COLUMN,
)


def create_cluster_summary(df):
    """
    Create summary information for each geographical cluster.
    """

    # Remove noise
    clustered = df[
        df["cluster"] != -1
    ].copy()

    if clustered.empty:
        return pd.DataFrame()

    # Basic aggregation
    summary = (
        clustered
        .groupby("cluster")
        .agg(
            ride_count=("cluster", "size"),
            center_latitude=(
                LATITUDE_COLUMN,
                "mean"
            ),
            center_longitude=(
                LONGITUDE_COLUMN,
                "mean"
            )
        )
        .reset_index()
    )

    # Average fare
    if FARE_COLUMN in clustered.columns:

        fare = (
            clustered
            .groupby("cluster")[FARE_COLUMN]
            .mean()
            .reset_index(
                name="average_fare"
            )
        )

        summary = summary.merge(
            fare,
            on="cluster",
            how="left"
        )

    # Average distance
    if DISTANCE_COLUMN in clustered.columns:

        distance = (
            clustered
            .groupby("cluster")[DISTANCE_COLUMN]
            .mean()
            .reset_index(
                name="average_distance"
            )
        )

        summary = summary.merge(
            distance,
            on="cluster",
            how="left"
        )

    # Find peak period
    # NOTE: picking the period with the highest raw count per cluster
    # tends to just pick whichever period is busiest system-wide (here,
    # Evening, since it has the most total rides overall) — so almost
    # every cluster ends up with the same "peak_period", which isn't
    # useful. Instead we compute "lift": how over-represented a period
    # is for THIS cluster compared to the dataset's overall pattern.
    # lift > 1 means this cluster is busier than usual in that period;
    # the period with the highest lift is what actually makes this
    # cluster distinctive.
    if "time_period" in clustered.columns:

        # Dataset-wide share of rides falling in each period
        overall_share = df["time_period"].value_counts(normalize=True)

        period_counts = (
            clustered
            .groupby(
                ["cluster", "time_period"]
            )
            .size()
            .reset_index(
                name="period_rides"
            )
        )

        cluster_totals = (
            clustered
            .groupby("cluster")
            .size()
            .reset_index(name="cluster_total")
        )

        period_counts = period_counts.merge(
            cluster_totals, on="cluster"
        )

        # What fraction of THIS cluster's rides fall in this period
        period_counts["cluster_share"] = (
            period_counts["period_rides"] / period_counts["cluster_total"]
        )

        # How that compares to the period's normal (dataset-wide) share
        period_counts["overall_share"] = (
            period_counts["time_period"].map(overall_share)
        )

        period_counts["lift"] = (
            period_counts["cluster_share"] / period_counts["overall_share"]
        )

        peak_period = (
            period_counts
            .sort_values(
                "lift",
                ascending=False
            )
            .drop_duplicates(
                "cluster"
            )
            [
                [
                    "cluster",
                    "time_period",
                    "period_rides",
                    "lift"
                ]
            ]
            .rename(
                columns={
                    "time_period":
                    "peak_period",
                    "lift":
                    "peak_period_lift"
                }
            )
        )

        summary = summary.merge(
            peak_period,
            on="cluster",
            how="left"
        )

    # Sort busiest first
    summary = summary.sort_values(
        "ride_count",
        ascending=False
    )

    return summary


def create_time_summary(df):
    """
    Calculate demand by time period.
    """

    if "time_period" not in df.columns:
        return pd.DataFrame()

    summary = (
        df
        .groupby("time_period")
        .size()
        .reset_index(
            name="ride_count"
        )
    )

    order = [
        "Morning",
        "Afternoon",
        "Evening",
        "Night"
    ]

    summary["time_period"] = pd.Categorical(
        summary["time_period"],
        categories=order,
        ordered=True
    )

    summary = summary.sort_values(
        "time_period"
    )

    return summary


def create_day_summary(df):
    """
    Calculate demand by day of week.
    """

    if "day_of_week" not in df.columns:
        return pd.DataFrame()

    summary = (
        df
        .groupby("day_of_week")
        .size()
        .reset_index(
            name="ride_count"
        )
    )

    order = [
        "Monday",
        "Tuesday",
        "Wednesday",
        "Thursday",
        "Friday",
        "Saturday",
        "Sunday"
    ]

    summary["day_of_week"] = pd.Categorical(
        summary["day_of_week"],
        categories=order,
        ordered=True
    )

    summary = summary.sort_values(
        "day_of_week"
    )

    return summary