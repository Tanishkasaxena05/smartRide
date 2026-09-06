# ============================================================
# MAIN ANALYSIS PIPELINE
# ============================================================

import os

from src.config import (
    OUTPUT_DIR,
    EPS_METERS,
    MIN_SAMPLES,
)

from src.data_loader import (
    load_data,
    clean_data,
    add_time_period,
)

from src.dbscan_clustering import (
    run_dbscan,
)

from src.cluster_analysis import (
    create_cluster_summary,
    create_time_summary,
    create_day_summary,
)

from src.map_visualization import (
    create_hotspot_map,
)


def main():

    print("\n")
    print("=" * 60)
    print("🚕 RIDE-HAILING PICKUP HOTSPOT DETECTION")
    print("=" * 60)

    # Create output directory
    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True
    )

    # --------------------------------------------------------
    # 1. LOAD DATA
    # --------------------------------------------------------

    df = load_data()

    # --------------------------------------------------------
    # 2. CLEAN DATA
    # --------------------------------------------------------

    df = clean_data(df)

    # --------------------------------------------------------
    # 3. ADD TIME INFORMATION
    # --------------------------------------------------------

    df = add_time_period(df)

    # --------------------------------------------------------
    # 4. RUN DBSCAN
    # --------------------------------------------------------

    df = run_dbscan(
        df,
        eps_meters=EPS_METERS,
        min_samples=MIN_SAMPLES
    )

    # --------------------------------------------------------
    # 5. BASIC CLUSTER STATISTICS
    # --------------------------------------------------------

    number_of_clusters = (
        len(
            set(df["cluster"])
        )
        - (
            1
            if -1 in df["cluster"].values
            else 0
        )
    )

    noise_count = (
        df["cluster"] == -1
    ).sum()

    noise_percentage = (
        noise_count /
        len(df)
    ) * 100

    print("\n")
    print("=" * 60)
    print("DBSCAN RESULTS")
    print("=" * 60)

    print(
        f"Clusters discovered: "
        f"{number_of_clusters}"
    )

    print(
        f"Noise points: "
        f"{noise_count:,}"
    )

    print(
        f"Noise percentage: "
        f"{noise_percentage:.2f}%"
    )

    # --------------------------------------------------------
    # 6. CLUSTER SUMMARY
    # --------------------------------------------------------

    cluster_summary = (
        create_cluster_summary(df)
    )

    print("\n")
    print("TOP HOTSPOTS")
    print("=" * 60)

    print(
        cluster_summary.head(10)
        .to_string(index=False)
    )

    # --------------------------------------------------------
    # 7. TIME SUMMARY
    # --------------------------------------------------------

    time_summary = (
        create_time_summary(df)
    )

    print("\n")
    print("TIME PERIOD DEMAND")
    print("=" * 60)

    print(
        time_summary.to_string(
            index=False
        )
    )

    # --------------------------------------------------------
    # 8. DAY SUMMARY
    # --------------------------------------------------------

    day_summary = (
        create_day_summary(df)
    )

    # --------------------------------------------------------
    # 9. SAVE RESULTS
    # --------------------------------------------------------

    df.to_csv(
        f"{OUTPUT_DIR}/clustered_rides.csv",
        index=False
    )

    cluster_summary.to_csv(
        f"{OUTPUT_DIR}/cluster_summary.csv",
        index=False
    )

    time_summary.to_csv(
        f"{OUTPUT_DIR}/time_summary.csv",
        index=False
    )

    day_summary.to_csv(
        f"{OUTPUT_DIR}/day_summary.csv",
        index=False
    )

    # --------------------------------------------------------
    # 10. CREATE MAP
    # --------------------------------------------------------

    create_hotspot_map(
        df,
        cluster_summary
    )

    # --------------------------------------------------------
    # DONE
    # --------------------------------------------------------

    print("\n")
    print("=" * 60)
    print("✅ ANALYSIS COMPLETE")
    print("=" * 60)

    print("\nGenerated files:")

    print(
        "1. outputs/clustered_rides.csv"
    )

    print(
        "2. outputs/cluster_summary.csv"
    )

    print(
        "3. outputs/time_summary.csv"
    )

    print(
        "4. outputs/day_summary.csv"
    )

    print(
        "5. outputs/hotspot_map.html"
    )


if __name__ == "__main__":
    main()