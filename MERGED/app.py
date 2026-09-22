# ============================================================
# SMARTRIDE UNIFIED DASHBOARD
# ============================================================

import os
import numpy as np
import pandas as pd
import streamlit as st
import folium
import holidays
from streamlit_folium import st_folium
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# --- App 1 Imports ---
from src.config import (
    DATA_FILE,
    EPS_METERS,
    MIN_SAMPLES,
    MAP_CENTER_LAT,
    MAP_CENTER_LON,
    MAP_ZOOM,
    DATASET_FILE
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

# --- App 2 Imports ---
from src.map_utils import CATEGORY_COLORS, LEGEND_HTML, get_marker_color, marker_radius
from src.predictor import get_metadata, predict_category

# --- App 3 Imports ---
from src.build_dataset import build as build_dataset, aggregate_to_demand
from src.calendar_context import add_calendar_context
from src.model_training import train_and_compare, prepare_features, FEATURE_COLUMNS

# ------------------------------------------------------------
# PAGE CONFIG
# ------------------------------------------------------------
st.set_page_config(
    page_title="SmartRide Unified Intelligence",
    page_icon="🚕",
    layout="wide"
)

st.markdown("""
    <style>
    div[data-testid="metric-container"] {
        background-color: #1e293b;
        border: 1px solid #334155;
        padding: 5% 5% 5% 10%;
        border-radius: 10px;
        color: #f8fafc;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    }
    </style>
""", unsafe_allow_html=True)

# ------------------------------------------------------------
# NAVIGATION
# ------------------------------------------------------------
st.sidebar.title("🚕 Navigation")
app_mode = st.sidebar.radio(
    "Select Tool:",
    ["Hotspot Detection", "Demand Classification", "Demand Forecasting"]
)
st.sidebar.divider()

# ============================================================
# APP 1: HOTSPOT DETECTION
# ============================================================
if app_mode == "Hotspot Detection":

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

    @st.cache_data
    def load_dataset_app1():
        df = pd.read_csv(
            DATA_FILE
        )
        df = clean_data(df)
        df = add_time_period(df)
        return df

    df = load_dataset_app1()

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

            st.session_state["clustered_settings"] = (
                eps_meters, min_samples, selected_period
            )

    if "clustered" not in st.session_state:
        st.info(
            "Choose the parameters and click "
            "'Detect Hotspots'."
        )
        st.stop()

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

    st.subheader(
        "🗺️ Interactive Hotspot Map"
    )

    city_map = folium.Map(
        location=[
            MAP_CENTER_LAT,
            MAP_CENTER_LON
        ],
        zoom_start=MAP_ZOOM,
        tiles="OpenStreetMap"
    )

    PEAK_PERIOD_COLORS = {
        "Morning": "#FFD700",
        "Afternoon": "#FF8C00",
        "Evening": "#800080",
        "Night": "#000080",
    }

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

    with st.expander(
        "View clustered ride data"
    ):
        st.dataframe(
            clustered,
            use_container_width=True
        )

# ============================================================
# APP 2: DEMAND CLASSIFICATION
# ============================================================
elif app_mode == "Demand Classification":

    DATASET_PATH = "data/processed/demand_dataset.csv"

    WEEKDAY_MAP = {
        "Monday": 0, "Tuesday": 1, "Wednesday": 2, "Thursday": 3,
        "Friday": 4, "Saturday": 5, "Sunday": 6,
    }

    MONTH_NAMES = {
        1: "January", 2: "February", 3: "March", 4: "April",
        5: "May", 6: "June", 7: "July", 8: "August",
        9: "September", 10: "October", 11: "November", 12: "December",
    }

    MODE_OPTIONS = ["Zone", "Pickup Location"]

    if "hist_reset_n" not in st.session_state:
        st.session_state["hist_reset_n"] = 0

    if "pred_reset_n" not in st.session_state:
        st.session_state["pred_reset_n"] = 0

    if "pred_active_mode" not in st.session_state:
        st.session_state["pred_active_mode"] = "Zone"

    def reset_historical_filters():
        st.session_state["hist_reset_n"] += 1
        st.rerun()

    def reset_prediction_filters():
        st.session_state["pred_reset_n"] += 1
        st.rerun()

    @st.cache_data
    def load_dataset_app2():
        dataset = pd.read_csv(DATASET_PATH)
        dataset["Date"] = pd.to_datetime(dataset["Date"], errors="coerce")
        dataset["date"] = dataset["Date"].dt.date
        return dataset

    @st.cache_resource
    def load_metadata():
        return get_metadata()

    try:
        dataset = load_dataset_app2()
        meta = load_metadata()
    except FileNotFoundError as exc:
        st.error(
            f"Could not load project artifacts ({exc}). "
            "Run `python run_pipeline.py` first."
        )
        st.stop()

    SLOT_HOURS = meta["slot_hours"]
    FEATURES = meta["features"]
    zone_info = meta["zone_info"]
    zone_lookup = meta["zone_lookup"]

    zone_names = (
        zone_info.sort_values("zone_id")[["zone_id", "zone_name"]]
        .set_index("zone_id")["zone_name"]
        .to_dict()
    )

    slot_names = (
        dataset[["slot", "slot_name"]]
        .drop_duplicates()
        .sort_values("slot")
        .set_index("slot")["slot_name"]
        .to_dict()
    )

    data_years = sorted(dataset["Date"].dt.year.dropna().unique().tolist())
    india_holidays = holidays.India(years=[int(y) for y in data_years] or [2024])

    st.title("🚕 SmartRide")
    st.subheader("Ride-Hailing Demand Category Classification")

    st.write(
        "Classify a Delhi NCR zone and time period as Low, Medium or High demand "
        "so driver supply can be positioned before the peak."
    )

    st.caption(
        f"Model: {meta['model_name']} · "
        f"{meta['n_zones']} zones · {SLOT_HOURS}-hour slots · "
        f"Low ≤ {meta['low_threshold']:.0f} rides, "
        f"High > {meta['high_threshold']:.0f} rides"
    )

    mode = st.radio(
        "Select Mode", ["Historical Analysis", "Demand Prediction"], horizontal=True
    )

    if mode == "Historical Analysis":
        hn = st.session_state["hist_reset_n"]
        header_col, reset_col = st.columns([6, 1])

        with header_col:
            st.subheader("🔎 Demand Analysis")

        with reset_col:
            st.write("")
            if st.button("↺ Reset Filters", key="reset_hist_btn", use_container_width=True):
                reset_historical_filters()

        with st.form(f"analysis_filters_{hn}"):
            col1, col2, col3 = st.columns(3)

            with col1:
                selected_category = st.selectbox(
                    "Demand Category",
                    ["All", "Low", "Medium", "High"],
                    key=f"hist_category_{hn}",
                )

            with col2:
                selected_slots = st.multiselect(
                    "Time Slots",
                    options=sorted(slot_names.keys()),
                    default=[],
                    format_func=lambda s: slot_names[s],
                    key=f"hist_slots_{hn}",
                    placeholder="All slots",
                )

            with col3:
                selected_months = st.multiselect(
                    "Months",
                    options=sorted(dataset["month"].unique()),
                    default=[],
                    format_func=lambda m: MONTH_NAMES[int(m)],
                    key=f"hist_months_{hn}",
                    placeholder="All months",
                )

            st.form_submit_button("Apply Filters", use_container_width=True)

        filtered = dataset.copy()

        if selected_slots:
            filtered = filtered[filtered["slot"].isin(selected_slots)]
        if selected_months:
            filtered = filtered[filtered["month"].isin(selected_months)]
        if selected_category != "All":
            filtered = filtered[filtered["Demand_Category"] == selected_category]

        if filtered.empty:
            st.warning("No records match the selected filters.")
            st.stop()

        st.subheader("📊 Demand Summary")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Cells Displayed", f"{len(filtered):,}")
        m2.metric("Zones", f"{filtered['zone_id'].nunique():,}")
        m3.metric("Total Rides", f"{int(filtered['ride_count'].sum()):,}")
        m4.metric("Rides per Cell", f"{filtered['ride_count'].mean():.1f}")

        st.subheader("🗺️ Delhi NCR Zone Demand Map")
        st.caption(
            "Bubble colour is the most frequent demand category for the zone "
            "within the current filters. Bubble size is total ride volume "
            "(log-scaled)."
        )

        map_df = (
            filtered.groupby(
                ["zone_id", "zone_name", "zone_lat", "zone_lon", "n_locations"],
                as_index=False,
            )
            .agg(
                ride_count=("ride_count", "sum"),
                mean_rides=("ride_count", "mean"),
                demand_ratio=("demand_ratio", "mean"),
            )
        )

        dominant = (
            filtered.groupby("zone_id")["Demand_Category"]
            .agg(lambda s: s.mode().iloc[0] if not s.mode().empty else "Medium")
        )
        map_df["Demand_Category"] = map_df["zone_id"].map(dominant).fillna("Medium")
        max_rides = float(map_df["ride_count"].max())

        city_map = folium.Map(location=[28.61, 77.20], zoom_start=10, tiles="OpenStreetMap")

        for _, row in map_df.iterrows():
            colour = get_marker_color(row["Demand_Category"])
            popup = (
                f"<b>{row['zone_name']}</b><br>"
                f"<b>Dominant category:</b> {row['Demand_Category']}<br>"
                f"<b>Total rides:</b> {int(row['ride_count']):,}<br>"
                f"<b>Average rides per slot:</b> {row['mean_rides']:.1f}<br>"
                f"<b>Pickup points in zone:</b> {int(row['n_locations'])}<br>"
                f"<b>Centroid:</b> {row['zone_lat']:.4f}, {row['zone_lon']:.4f}"
            )
            folium.CircleMarker(
                location=[row["zone_lat"], row["zone_lon"]],
                radius=marker_radius(row["ride_count"], max_rides),
                color=colour,
                fill=True,
                fill_color=colour,
                fill_opacity=0.55,
                weight=2,
                tooltip=(
                    f"{row['zone_name']} • {row['Demand_Category']} • "
                    f"{int(row['ride_count']):,} rides"
                ),
                popup=folium.Popup(popup, max_width=320),
            ).add_to(city_map)

        city_map.get_root().html.add_child(folium.Element(LEGEND_HTML))
        st_folium(city_map, height=620, use_container_width=True)

        st.divider()
        st.subheader("📋 Filtered Records")

        zone_choice = st.selectbox(
            "🔍 Filter by zone",
            ["All"] + [zone_names[z] for z in sorted(filtered["zone_id"].unique())],
            key=f"hist_zone_choice_{hn}",
        )

        table = filtered.copy()
        if zone_choice != "All":
            table = table[table["zone_name"] == zone_choice]

        st.caption(f"Showing up to 500 of {len(table):,} matching rows.")

        display = (
            table[
                [
                    "date", "zone_name", "slot_name", "weekday", "ride_count",
                    "prev_slot_demand", "prev_day_demand", "zone_slot_roll7",
                    "holiday_name", "weather_indicator", "Demand_Category",
                ]
            ]
            .head(500)
            .reset_index(drop=True)
        )

        display["date"] = pd.to_datetime(display["date"]).dt.strftime("%Y-%m-%d")
        display["holiday_name"] = display["holiday_name"].fillna("-")
        display["zone_slot_roll7"] = display["zone_slot_roll7"].round(2)
        display["weather_indicator"] = (
            display["weather_indicator"].map({0: "No", 1: "Yes"}).fillna("-")
        )
        display.insert(0, "NO.", range(1, len(display) + 1))

        display = display.rename(
            columns={
                "date": "DATE", "zone_name": "ZONE", "slot_name": "TIME SLOT",
                "weekday": "WEEKDAY", "ride_count": "RIDES", "prev_slot_demand": "PREV SLOT",
                "prev_day_demand": "PREV DAY", "zone_slot_roll7": "7-DAY AVG",
                "holiday_name": "HOLIDAY", "weather_indicator": "RAIN",
                "Demand_Category": "CATEGORY",
            }
        )
        st.dataframe(display, use_container_width=True, hide_index=True, height=480)

    if mode == "Demand Prediction":
        pn = st.session_state["pred_reset_n"]
        header_col, reset_col = st.columns([6, 1])

        with header_col:
            st.subheader("🔮 Demand Category Prediction")

        with reset_col:
            st.write("")
            if st.button("↺ Reset Filters", key="reset_pred_btn", use_container_width=True):
                reset_prediction_filters()

        st.write(
            "Pick a zone, a date and a time slot. Historical context "
            "(previous slot, previous day, rolling average, weather) is filled "
            "in automatically from the processed dataset."
        )

        col1, col2 = st.columns(2)

        with col1:
            mode_index = MODE_OPTIONS.index(st.session_state["pred_active_mode"])
            selection_mode = st.radio(
                "Choose zone by",
                MODE_OPTIONS,
                horizontal=True,
                index=mode_index,
                key=f"pred_selection_mode_{pn}",
            )
            st.session_state["pred_active_mode"] = selection_mode

            if selection_mode == "Zone":
                zone_name = st.selectbox(
                    "Zone",
                    [zone_names[z] for z in sorted(zone_names)],
                    key=f"pred_zone_name_{pn}",
                )
                zone_id = int(
                    zone_info.loc[zone_info["zone_name"] == zone_name, "zone_id"].iloc[0]
                )
            else:
                location = st.selectbox(
                    "Pickup Location",
                    sorted(zone_lookup["Pickup Location"].unique()),
                    key=f"pred_location_{pn}",
                )
                zone_id = int(
                    zone_lookup.loc[
                        zone_lookup["Pickup Location"] == location, "zone_id"
                    ].iloc[0]
                )
                zone_name = zone_names[zone_id]
                st.caption(f"{location} falls in {zone_name}")

        with col2:
            selected_slot = st.selectbox(
                "Time Slot",
                sorted(slot_names.keys()),
                format_func=lambda s: slot_names[s],
                key=f"pred_slot_{pn}",
            )

        date_col1, date_col2 = st.columns(2)

        with date_col1:
            selected_month = st.selectbox(
                "Month",
                list(MONTH_NAMES.keys()),
                format_func=lambda m: MONTH_NAMES[m],
                key=f"pred_month_{pn}",
            )

        prediction_year = int(data_years[-1]) if data_years else 2024
        days_in_month = pd.Period(f"{prediction_year}-{selected_month:02d}").days_in_month

        with date_col2:
            selected_day = st.selectbox(
                "Day", list(range(1, days_in_month + 1)), key=f"pred_day_{pn}"
            )

        prediction_date = pd.Timestamp(
            year=prediction_year, month=selected_month, day=selected_day
        )
        prediction_weekday = prediction_date.day_name()
        holiday_name = india_holidays.get(prediction_date.date()) or "None"
        is_holiday = int(holiday_name != "None")

        zone_rows = dataset[dataset["zone_id"] == zone_id]

        if zone_rows.empty:
            st.error("No historical data for this zone.")
            st.stop()

        exact = zone_rows[
            (zone_rows["Date"] == prediction_date) & (zone_rows["slot"] == selected_slot)
        ]

        if not exact.empty:
            context = exact.iloc[0]
            context_source = "Exact historical cell"
        else:
            similar = zone_rows[
                (zone_rows["slot"] == selected_slot)
                & (zone_rows["weekday"] == prediction_weekday)
            ]
            if similar.empty:
                similar = zone_rows[zone_rows["slot"] == selected_slot]
            if similar.empty:
                similar = zone_rows
            context = similar.mean(numeric_only=True)
            context_source = "Averaged from comparable zone/slot/weekday cells"

        zone_row = zone_info[zone_info["zone_id"] == zone_id].iloc[0]

        def context_value(name, default=0.0):
            value = context.get(name, default)
            return default if pd.isna(value) else float(value)

        weather_columns = [
            "temp", "humidity", "precip", "precipprob",
            "windspeed", "heat_index", "temp_range",
        ]

        day_weather = dataset[dataset["Date"] == prediction_date]

        if not day_weather.empty and day_weather["temp"].notna().any():
            weather_values = day_weather[weather_columns].mean()
            weather_indicator = int(day_weather["weather_indicator"].mode().iloc[0])
            weather_source = "Selected date"
        else:
            month_weather = dataset[dataset["month"] == selected_month]
            weather_values = month_weather[weather_columns].mean()
            indicator_mode = month_weather["weather_indicator"].mode()
            weather_indicator = int(indicator_mode.iloc[0]) if not indicator_mode.empty else 0
            weather_source = "Monthly average"

        feature_dict = {
            "slot": int(selected_slot),
            "hour_start": int(selected_slot) * SLOT_HOURS,
            "weekday_encoded": WEEKDAY_MAP[prediction_weekday],
            "month": int(selected_month),
            "day": int(selected_day),
            "week_of_year": int(prediction_date.isocalendar().week),
            "is_weekend": int(WEEKDAY_MAP[prediction_weekday] >= 5),
            "is_holiday": is_holiday,
            "zone_id": zone_id,
            "zone_lat": float(zone_row["zone_lat"]),
            "zone_lon": float(zone_row["zone_lon"]),
            "n_locations": int(zone_row["n_locations"]),
            "prev_slot_demand": context_value("prev_slot_demand"),
            "prev_day_demand": context_value("prev_day_demand"),
            "prev_week_demand": context_value("prev_week_demand"),
            "zone_slot_expanding_avg": context_value("zone_slot_expanding_avg"),
            "zone_slot_roll7": context_value("zone_slot_roll7"),
            "zone_expanding_avg": context_value("zone_expanding_avg"),
            "weather_indicator": weather_indicator,
        }

        for column in weather_columns:
            value = weather_values.get(column)
            feature_dict[column] = float(value) if pd.notna(value) else 0.0

        missing = [f for f in FEATURES if f not in feature_dict]
        if missing:
            st.error(f"Internal error — missing features: {missing}")
            st.stop()

        st.info(
            f"""
            **Auto-detected context** — {context_source}

            - Zone: {zone_name}
            - Date: {prediction_date.strftime('%d %b %Y')} ({prediction_weekday})
            - Time slot: {slot_names[selected_slot]}
            - Previous slot demand: {feature_dict['prev_slot_demand']:.0f} rides
            - Same slot yesterday: {feature_dict['prev_day_demand']:.0f} rides
            - Same slot last week: {feature_dict['prev_week_demand']:.0f} rides
            - 7-day rolling average: {feature_dict['zone_slot_roll7']:.1f} rides
            - Holiday: {holiday_name}
            - Weather: {'Rain' if weather_indicator else 'No rain'} ({weather_source})
            """
        )

        if st.button("🚕 Predict Demand Category", use_container_width=True):
            category, probabilities = predict_category(
                feature_dict, return_probabilities=True
            )
            st.markdown("### Prediction Result")

            if category == "High":
                st.error("🔥 HIGH DEMAND — move additional drivers into this zone early.")
            elif category == "Medium":
                st.warning("⚠️ MEDIUM DEMAND — maintain normal driver allocation.")
            else:
                st.success("✅ LOW DEMAND — drivers can be released to other zones.")

            if probabilities:
                st.write("**Model confidence**")
                probability_frame = (
                    pd.DataFrame(
                        {
                            "Category": list(probabilities.keys()),
                            "Probability": list(probabilities.values()),
                        }
                    )
                    .sort_values("Probability", ascending=False)
                    .reset_index(drop=True)
                )
                st.dataframe(
                    probability_frame,
                    use_container_width=True,
                    hide_index=True,
                    column_config={
                        "Probability": st.column_config.ProgressColumn(
                            "Probability", min_value=0.0, max_value=1.0, format="%.2f"
                        )
                    },
                )

            st.divider()

            w1, w2, w3, w4 = st.columns(4)
            w1.metric("Temperature", f"{feature_dict['temp']:.1f} °C")
            w2.metric("Humidity", f"{feature_dict['humidity']:.0f}%")
            w3.metric("Rain probability", f"{feature_dict['precipprob']:.0f}%")
            w4.metric("Wind speed", f"{feature_dict['windspeed']:.1f}")

        st.divider()
        st.caption(
            "SmartRide NCR • Project 2 — Ride-Hailing Demand Category Classification"
        )


# ============================================================
# APP 3: DEMAND FORECASTING
# ============================================================
elif app_mode == "Demand Forecasting":

    st.title("🚕 SmartRide: Ride-Hailing Demand Intelligence")

    st.markdown(
        "AI-based hourly demand forecasting and fleet allocation insights."
    )

    @st.cache_data
    def load_and_prep_data():
        if not os.path.exists(DATASET_FILE):
            with st.spinner(
                "First run: building the demand dataset "
                "(geocoding zones, clustering, aggregating)..."
            ):
                build_dataset()

        raw = pd.read_csv(DATASET_FILE)
        raw = add_calendar_context(raw)

        calendar_df = raw[
            [
                "date",
                "day_name",
                "holiday",
                "event",
                "rainfall_mm"
            ]
        ].copy()

        calendar_df["date"] = pd.to_datetime(
            calendar_df["date"]
        )

        calendar_df = (
            calendar_df
            .groupby("date")
            .agg(
                day_name=("day_name", "first"),
                holiday=("holiday", "first"),
                event=("event", "first"),
                rainfall_mm=("rainfall_mm", "mean"),
            )
            .reset_index()
        )

        agg_df = aggregate_to_demand(raw)
        agg_df = prepare_features(agg_df)

        return agg_df, calendar_df

    with st.spinner(
        "Loading aggregated ride data..."
    ):
        data, calendar = load_and_prep_data()

    clusters = sorted(
        data["cluster"].unique()
    )

    st.sidebar.header("Scenario Controls")

    st.sidebar.markdown(
        "Adjust parameters to forecast future ride volume."
    )

    selected_cluster = st.sidebar.selectbox(
        "Geographical Zone (Cluster)",
        clusters
    )

    selected_date = st.sidebar.selectbox(
        "Scenario Date",
        calendar["date"].dt.date.tolist()
    )

    selected_date_context = calendar[
        calendar["date"].dt.date == selected_date
    ].iloc[0]

    selected_day = selected_date.day

    selected_hour = st.sidebar.slider(
        "Target Hour (0-23)",
        0,
        23,
        18
    )

    selected_rain = st.sidebar.number_input(
        "Rainfall Forecast (mm)",
        min_value=0.0,
        value=float(
            selected_date_context["rainfall_mm"]
        ),
        step=0.5,
        help=(
            "Pre-filled with that date's rainfall "
            "from the training data. Override it "
            "to test a hypothetical scenario."
        )
    )

    selected_prev_demand = st.sidebar.number_input(
        "Previous Hour Demand (Rides)",
        min_value=0,
        value=5
    )

    st.sidebar.caption(
        """
        Note: rainfall in the training data is an
        approximate seasonal pattern (heavier Jun-Sep),
        not measured historical weather.

        Ride counts are per-cluster, per-hour, so
        typical values are small (single digits to
        a few dozen).
        """
    )

    st.header("📈 Demand Forecast")

    cluster_data = data[
        data["cluster"] == selected_cluster
    ].copy()

    hourly_demand = (
        cluster_data
        .groupby("hour")["demand"]
        .mean()
    )

    input_data = pd.DataFrame([
        {
            "hour": selected_hour,
            "cluster": selected_cluster,
            "day": selected_day,
            "rainfall_mm": selected_rain,
            "previous_demand": selected_prev_demand,
        }
    ])[FEATURE_COLUMNS]

    if selected_hour in hourly_demand.index:
        base_prediction = hourly_demand[
            selected_hour
        ]
    else:
        base_prediction = cluster_data[
            "demand"
        ].mean()

    rainfall_adjustment = 1.0

    if selected_rain >= 20:
        rainfall_adjustment = 0.90
    elif selected_rain >= 10:
        rainfall_adjustment = 0.95
    elif selected_rain >= 5:
        rainfall_adjustment = 0.98

    historical_average = cluster_data[
        "demand"
    ].mean()

    if historical_average > 0:
        previous_demand_factor = (
            selected_prev_demand /
            historical_average
        )
        previous_demand_factor = max(
            0.75,
            min(
                previous_demand_factor,
                1.25
            )
        )
    else:
        previous_demand_factor = 1.0

    target_prediction = (
        base_prediction
        * rainfall_adjustment
        * previous_demand_factor
    )

    target_prediction = max(
        0,
        round(target_prediction)
    )

    forecast_records = []

    for hour in range(24):
        if hour in hourly_demand.index:
            predicted_demand = hourly_demand[
                hour
            ]
        else:
            predicted_demand = historical_average

        predicted_demand *= rainfall_adjustment
        predicted_demand *= previous_demand_factor

        predicted_demand = max(
            0,
            round(predicted_demand)
        )

        forecast_records.append(
            predicted_demand
        )

    chart_data = pd.DataFrame(
        {
            "Hour": range(24),
            "Predicted Rides": forecast_records,
        }
    )

    st.subheader(
        f"Forecast Summary: "
        f"Zone {selected_cluster}"
    )

    context_col1, context_col2, context_col3, context_col4 = st.columns(4)

    context_col1.metric(
        "Day",
        selected_date_context["day_name"]
    )

    context_col2.metric(
        "Holiday",
        selected_date_context["holiday"]
    )

    context_col3.metric(
        "Event",
        selected_date_context["event"]
    )

    context_col4.metric(
        "Rainfall",
        f"{selected_rain:.1f} mm"
    )

    col1, col2, col3 = st.columns(3)

    col1.metric(
        label=(
            f"Expected Demand at "
            f"{selected_hour:02d}:00"
        ),
        value=f"{target_prediction} rides"
    )

    col2.metric(
        label="Previous Hour Demand",
        value=f"{selected_prev_demand} rides"
    )

    col3.metric(
        label="Average Hourly Demand",
        value=f"{historical_average:.1f} rides"
    )

    st.divider()

    st.subheader(
        "📈 24-Hour Projected Demand Curve"
    )

    st.caption(
        f"""
        Simulated hourly demand for Zone
        {selected_cluster} on
        {selected_date.strftime("%d %b %Y")}
        ({selected_date_context["day_name"]}).
        """
    )

    st.line_chart(
        chart_data,
        x="Hour",
        y="Predicted Rides",
        height=380
    )

    st.divider()

    st.subheader(
        "⏰ Hourly Demand Pattern"
    )

    hourly_table = chart_data.copy()

    hourly_table["Hour"] = (
        hourly_table["Hour"]
        .apply(
            lambda x: f"{x:02d}:00"
        )
    )

    hourly_table = hourly_table.rename(
        columns={
            "Hour": "Time",
            "Predicted Rides": "Expected Rides"
        }
    )

    st.dataframe(
        hourly_table,
        use_container_width=True,
        hide_index=True
    )

    st.divider()

    st.subheader(
        "💡 Fleet Allocation Insights"
    )

    peak_hour_index = chart_data[
        "Predicted Rides"
    ].idxmax()

    peak_hour = chart_data.loc[
        peak_hour_index,
        "Hour"
    ]

    peak_demand = chart_data.loc[
        peak_hour_index,
        "Predicted Rides"
    ]

    low_hour_index = chart_data[
        "Predicted Rides"
    ].idxmin()

    low_hour = chart_data.loc[
        low_hour_index,
        "Hour"
    ]

    low_demand = chart_data.loc[
        low_hour_index,
        "Predicted Rides"
    ]

    insight_col1, insight_col2, insight_col3 = st.columns(3)

    insight_col1.metric(
        "Peak Demand Hour",
        f"{peak_hour:02d}:00"
    )

    insight_col2.metric(
        "Peak Expected Rides",
        f"{peak_demand}"
    )

    insight_col3.metric(
        "Lowest Demand Hour",
        f"{low_hour:02d}:00"
    )

    st.info(
        f"""
        **Fleet Planning Insight:**

        Zone {selected_cluster} is expected to have its
        highest demand around **{peak_hour:02d}:00**, with
        approximately **{peak_demand} rides**.

        Lower demand is expected around **{low_hour:02d}:00**,
        with approximately **{low_demand} rides**.

        Fleet managers can use this pattern to plan vehicle
        availability throughout the day.
        """
    )

    st.divider()

    st.subheader(
        "📍 Zone Information"
    )

    zone_col1, zone_col2, zone_col3 = st.columns(3)

    zone_col1.metric(
        "Total Historical Rides",
        f"{len(cluster_data):,}"
    )

    zone_col2.metric(
        "Average Hourly Demand",
        f"{cluster_data['demand'].mean():.1f}"
    )

    zone_col3.metric(
        "Maximum Historical Demand",
        f"{cluster_data['demand'].max():.0f}"
    )
    
    st.divider()

    st.subheader("📊 Model Performance Metrics")
    
    if len(cluster_data) > 1:
        actual_demand_metrics = cluster_data['demand']
        predicted_demand_metrics = cluster_data['hour'].map(hourly_demand).fillna(historical_average)
        
        mae = mean_absolute_error(actual_demand_metrics, predicted_demand_metrics)
        rmse = np.sqrt(mean_squared_error(actual_demand_metrics, predicted_demand_metrics))
        rsquare = r2_score(actual_demand_metrics, predicted_demand_metrics)
        
        metric_col1, metric_col2, metric_col3 = st.columns(3)
        metric_col1.metric("MAE", f"{mae:.2f}")
        metric_col2.metric("RMSE", f"{rmse:.2f}")
        metric_col3.metric("R-Squared", f"{rsquare:.4f}")
    else:
        st.info("Not enough historical data to calculate performance metrics.")

    st.markdown("---")
    st.caption(
        "SmartRide | Ride-Hailing Demand Intelligence"
    )