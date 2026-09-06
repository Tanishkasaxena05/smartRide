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

# Color per time-of-day peak period (requested: Morning-yellow,
# Afternoon-orange, Evening-purple, Night-navy)
PEAK_PERIOD_COLORS = {
    "Morning": "#FFD700",    # yellow
    "Afternoon": "#FF8C00",  # orange
    "Evening": "#800080",    # purple
    "Night": "#000080",      # navy
}
DEFAULT_COLOR = "#808080"  # gray, for any missing/unknown peak period


def _add_legend(city_map):
    """Injects a small color-key box onto the map so the colors are explained."""
    legend_html = """
    <div style="
        position: fixed;
        bottom: 30px; left: 30px; z-index: 9999;
        background-color: rgba(255,255,255,0.95); padding: 10px 14px;
        border: 2px solid #444; border-radius: 6px;
        font-size: 14px; line-height: 1.6; color: #111 !important;">
        <b style="color:#111 !important;">Peak Period</b><br>
        <span style="color:#111 !important;"><span style="color:#FFD700;">&#9679;</span> Morning</span><br>
        <span style="color:#111 !important;"><span style="color:#FF8C00;">&#9679;</span> Afternoon</span><br>
        <span style="color:#111 !important;"><span style="color:#800080;">&#9679;</span> Evening</span><br>
        <span style="color:#111 !important;"><span style="color:#000080;">&#9679;</span> Night</span>
    </div>
    """
    city_map.get_root().html.add_child(folium.Element(legend_html))


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
        tiles="OpenStreetMap"  # free, no API key needed
        # (CartoDB's hosted tiles now require an API key for new
        # accounts, which is why the map used to show "API KEY
        # REQUIRED" watermarks instead of a real basemap)
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

        color = PEAK_PERIOD_COLORS.get(peak, DEFAULT_COLOR)

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
            color=color,
            fill=True,
            fill_color=color,
            weight=1,
            opacity=0.6,
            popup=folium.Popup(
                popup_html,
                max_width=300
            ),
            tooltip=(
                f"Cluster {cluster} | "
                f"{rides:,} rides | {peak}"
            ),
            fill_opacity=0.5
        ).add_to(hotspot_group)

    hotspot_group.add_to(city_map)

    # Layer control
    folium.LayerControl().add_to(
        city_map
    )

    _add_legend(city_map)

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