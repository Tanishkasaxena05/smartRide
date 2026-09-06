# ============================================================
# INTERACTIVE MAP
# ============================================================

import folium
from folium.plugins import MarkerCluster

from src.config import (
    MAP_CENTER_LAT,
    MAP_CENTER_LON,
    MAP_ZOOM,
    OUTPUT_DIR,
)


def create_hotspot_map(
    df,
    cluster_summary
):
    """
    Create an interactive Folium map.
    """

    city_map = folium.Map(
        location=[
            MAP_CENTER_LAT,
            MAP_CENTER_LON
        ],
        zoom_start=MAP_ZOOM,
        tiles="CartoDB positron"
    )

    # --------------------------------------------------------
    # Noise layer
    # --------------------------------------------------------

    noise = df[
        df["cluster"] == -1
    ]

    noise_group = folium.FeatureGroup(
        name="Noise / Outliers"
    )

    # Don't put every noise point on the map
    # if the dataset is huge.
    noise_sample = noise.head(3000)

    for _, row in noise_sample.iterrows():

        folium.CircleMarker(
            location=[
                row["Pickup Latitude"],
                row["Pickup Longitude"]
            ],
            radius=2,
            color="gray",
            fill=True,
            fill_opacity=0.4,
            popup="Noise / Outlier"
        ).add_to(noise_group)

    noise_group.add_to(city_map)

    # --------------------------------------------------------
    # Hotspot layer
    # --------------------------------------------------------

    hotspot_group = folium.FeatureGroup(
        name="Ride Hotspots"
    )

    for _, row in cluster_summary.iterrows():

        lat = row["center_latitude"]
        lon = row["center_longitude"]

        rides = row["ride_count"]
        cluster = row["cluster"]

        peak = row.get(
            "peak_period",
            "N/A"
        )

        # Radius based on number of rides
        radius = min(
            25,
            max(
                7,
                rides ** 0.5
            )
        )

        popup_html = f"""
        <div style="width:220px">
            <h4>🚕 Cluster {cluster}</h4>
            <b>Rides:</b> {rides:,}<br>
            <b>Peak Period:</b> {peak}<br>
            <b>Latitude:</b> {lat:.5f}<br>
            <b>Longitude:</b> {lon:.5f}
        </div>
        """

        folium.CircleMarker(
            location=[
                lat,
                lon
            ],
            radius=radius,
            popup=folium.Popup(
                popup_html,
                max_width=300
            ),
            tooltip=(
                f"Cluster {cluster} | "
                f"{rides:,} rides"
            ),
            fill=True,
            fill_opacity=0.7
        ).add_to(hotspot_group)

    hotspot_group.add_to(city_map)

    # Layer control
    folium.LayerControl().add_to(
        city_map
    )

    output_file = (
        f"{OUTPUT_DIR}/hotspot_map.html"
    )

    city_map.save(
        output_file
    )

    print(
        f"\nMap saved to: "
        f"{output_file}"
    )

    return city_map