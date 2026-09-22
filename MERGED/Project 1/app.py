# ============================================================
# STREAMLIT DASHBOARD
# ============================================================

import streamlit as st
import pandas as pd
import folium

from streamlit_folium import st_folium

from src.config import (
    DATA_FILE,
    EPS_METERS,
    MIN_SAMPLES,
    MAP_CENTER_LAT,
    MAP_CENTER_LON,
    MAP_ZOOM,
)

from src.data_loader import (
    clean_data,
    add_time_period,
)

from src.dbscan_clustering import (
    run_dbscan,
)

from src.cluster_analysis import (
    create_cluster_summary,
)


# ------------------------------------------------------------
# PAGE CONFIG
# ------------------------------------------------------------

st.set_page_config(
    page_title="Ride Hotspot Detection",
    page_icon="🚕",
    layout="wide"
)


# ------------------------------------------------------------
# TITLE
# ------------------------------------------------------------

st.title(
    "🚕 Ride-Hailing Pickup Hotspot Detection"
)

st.write(
    """
    This dashboard uses DBSCAN clustering to identify
    geographical areas with high concentrations of
    ride pickup requests.
    """
)


# ------------------------------------------------------------
# LOAD DATA
# ------------------------------------------------------------

@st.cache_data
def load_dataset():

    df = pd.read_csv(
        DATA_FILE
    )

    df = clean_data(df)
    df = add_time_period(df)

    return df


df = load_dataset()


# ------------------------------------------------------------
# SIDEBAR
# ------------------------------------------------------------

st.sidebar.header(
    "⚙️ DBSCAN Settings"
)

eps_meters = st.sidebar.slider(
    "Search radius (meters)",
    min_value=50,
    max_value=2000,
    value=EPS_METERS,
    step=50,
    help="Larger radius merges more nearby pickup zones into the "
         "same hotspot. Small changes (under a few hundred meters) "
         "may show little effect since named zones can sit several "
         "kilometers apart."
)

min_samples = st.sidebar.slider(
    "Minimum samples",
    min_value=10,
    max_value=1000,
    value=MIN_SAMPLES,
    step=10,
    help="Each pickup zone has roughly 800-950 rides overall (fewer "
         "if you filter by time period below — e.g. Night has only "
         "~80-130 per zone). Raise this above a zone's ride count to "
         "start seeing it marked as noise."
)


# ------------------------------------------------------------
# TIME FILTER
# ------------------------------------------------------------

st.sidebar.header(
    "🕐 Time Filter"
)

period_options = [
    "All",
    "Morning",
    "Afternoon",
    "Evening",
    "Night"
]

selected_period = st.sidebar.selectbox(
    "Select period",
    period_options
)


# ------------------------------------------------------------
# FILTER DATA
# ------------------------------------------------------------

if selected_period == "All":

    filtered_df = df.copy()

else:

    filtered_df = df[
        df["time_period"]
        == selected_period
    ].copy()


st.write(
    f"### Analyzing {len(filtered_df):,} rides"
)


# ------------------------------------------------------------
# RUN DBSCAN
# ------------------------------------------------------------

if st.button(
    "🔍 Detect Hotspots"
):

    with st.spinner(
        "Running DBSCAN..."
    ):

        clustered = run_dbscan(
            filtered_df,
            eps_meters=eps_meters,
            min_samples=min_samples
        )

        st.session_state[
            "clustered"
        ] = clustered

        # Remember which settings produced this result, so we can
        # warn if the sliders get moved afterward without re-running.
        st.session_state["clustered_settings"] = (
            eps_meters, min_samples, selected_period
        )


# ------------------------------------------------------------
# CHECK RESULTS
# ------------------------------------------------------------

if "clustered" not in st.session_state:

    st.info(
        "Choose the parameters and click "
        "'Detect Hotspots'."
    )

    st.stop()

# Moving a slider does NOT auto-recompute — you must click the
# button again. Warn clearly if what's shown below is now stale.
current_settings = (eps_meters, min_samples, selected_period)
if st.session_state.get("clustered_settings") != current_settings:
    st.warning(
        "⚠️ Search radius, minimum samples, or time period changed "
        "since the results below were generated. Click "
        "'🔍 Detect Hotspots' again to apply the new settings."
    )


clustered = st.session_state[
    "clustered"
]


# ------------------------------------------------------------
# METRICS
# ------------------------------------------------------------

valid_clusters = (
    clustered[
        clustered["cluster"] != -1
    ]
)

number_clusters = (
    valid_clusters["cluster"]
    .nunique()
)

noise_count = (
    clustered["cluster"] == -1
).sum()

noise_percentage = (
    noise_count /
    len(clustered)
) * 100


col1, col2, col3, col4 = st.columns(4)


col1.metric(
    "Total Rides",
    f"{len(clustered):,}"
)

col2.metric(
    "Hotspots",
    number_clusters
)

col3.metric(
    "Noise Points",
    f"{noise_count:,}"
)

col4.metric(
    "Noise %",
    f"{noise_percentage:.1f}%"
)


# ------------------------------------------------------------
# CLUSTER SUMMARY
# ------------------------------------------------------------

summary = create_cluster_summary(
    clustered
)


st.subheader(
    "📊 Hotspot Summary"
)

st.dataframe(
    summary,
    use_container_width=True
)


# ------------------------------------------------------------
# MAP
# ------------------------------------------------------------

st.subheader(
    "🗺️ Interactive Hotspot Map"
)

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

# Color per time-of-day peak period (Morning-yellow,
# Afternoon-orange, Evening-purple, Night-navy)
PEAK_PERIOD_COLORS = {
    "Morning": "#FFD700",
    "Afternoon": "#FF8C00",
    "Evening": "#800080",
    "Night": "#000080",
}


# Add hotspots

for _, row in summary.iterrows():

    cluster = row["cluster"]
    rides = row["ride_count"]

    lat = row["center_latitude"]
    lon = row["center_longitude"]

    peak = row.get(
        "peak_period",
        "N/A"
    )

    color = PEAK_PERIOD_COLORS.get(peak, "#808080")

    radius = min(
        30,
        max(
            7,
            rides ** 0.5
        )
    )

    popup = f"""
    <b>Cluster:</b> {cluster}<br>
    <b>Rides:</b> {rides:,}<br>
    <b>Peak Period:</b> {peak}<br>
    <b>Latitude:</b> {lat:.5f}<br>
    <b>Longitude:</b> {lon:.5f}
    """

    folium.CircleMarker(
        location=[
            lat,
            lon
        ],
        radius=radius,
        color=color,
        fill_color=color,
        weight=1,
        opacity=0.6,
        popup=popup,
        tooltip=(
            f"Cluster {cluster}: "
            f"{rides:,} rides | {peak}"
        ),
        fill=True,
        fill_opacity=0.5
    ).add_to(city_map)

# Color-key legend
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


st_folium(
    city_map,
    width=None,
    height=600
)


# ------------------------------------------------------------
# BAR CHART
# ------------------------------------------------------------

st.subheader(
    "📈 Rides per Hotspot"
)

if not summary.empty:

    chart_data = (
        summary
        .set_index("cluster")
        ["ride_count"]
    )

    st.bar_chart(
        chart_data
    )


# ------------------------------------------------------------
# RAW CLUSTERED DATA
# ------------------------------------------------------------

with st.expander(
    "View clustered ride data"
):

    st.dataframe(
        clustered,
        use_container_width=True
    )