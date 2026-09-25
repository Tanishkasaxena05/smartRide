"""
SmartRide NCR dashboard.

Bugs fixed from the previous version:

1. CRASH. The weather block ran unconditionally after both prediction
   modes. In "Historical Pattern" mode prediction_date was None, so
   weather_row was an empty DataFrame with no columns and
   weather_row["weather_indicator"] raised KeyError on page render.

2. LATENT NameError. The seven `temp = float(weather_values["temp"])`
   assignments existed only in the fallback branch. Because weather was
   100% null, the fallback always ran and the bug never surfaced -- fixing
   the weather join alone would have exposed it.

3. HARDCODED DATE. Historical Pattern passed month=1, day=1 to the model,
   silently asking about New Year's Day every time.

4. RESET DIDN'T RESET. Popping a widget's session_state key and calling
   st.rerun() is the textbook way to reset a Streamlit widget -- but the
   Historical Analysis filters live inside st.form(...), and Streamlit's
   forms cache submitted widget values against their key in a way that a
   backend session_state.pop() does not reliably clear: the browser-side
   widget can still report its last submitted value back on the next run.
   Fixed with VERSIONED KEYS: every resettable widget's key embeds a
   counter (hist_reset_n / pred_reset_n) that only changes when its
   Reset button is clicked. Bumping the counter makes the key genuinely
   new on the next run, so there is no old state anywhere to restore --
   this sidesteps the form-caching behavior entirely rather than fighting
   it. Demand Prediction's fields are not inside a form, so plain
   session_state deletion already worked there, but they are versioned
   too for consistency and to guard against the same failure mode.

5. MODE-AWARE PREDICTION RESET. Versioning pred_selection_mode's key
   would reset it to its literal default ("Zone") on every Reset click,
   which would silently flip a user working in "Pickup Location" mode
   back to "Zone" just for clearing their date/slot picks. The currently
   active mode is tracked separately in pred_active_mode (a key that is
   NEVER versioned/reset) and fed back in as the new radio widget's
   starting index after each reset, so the active mode survives a reset
   while every other prediction field returns to its true default.

Feature assembly for prediction goes through one function that returns a
dict keyed by feature name, validated against the model bundle.
"""

import folium
import holidays
import pandas as pd
import streamlit as st
from streamlit_folium import st_folium

from src.map_utils import CATEGORY_COLORS, LEGEND_HTML, get_marker_color, marker_radius
from src.predictor import get_metadata, predict_category

st.set_page_config(page_title="SmartRide", page_icon="🚕", layout="wide")

DATASET_PATH = "data/processed/demand_dataset.csv"

WEEKDAY_MAP = {
    "Monday": 0,
    "Tuesday": 1,
    "Wednesday": 2,
    "Thursday": 3,
    "Friday": 4,
    "Saturday": 5,
    "Sunday": 6,
}

MONTH_NAMES = {
    1: "January", 2: "February", 3: "March", 4: "April",
    5: "May", 6: "June", 7: "July", 8: "August",
    9: "September", 10: "October", 11: "November", 12: "December",
}

MODE_OPTIONS = ["Zone", "Pickup Location"]

# Reset counters. Each widget that must reliably reset embeds one of
# these in its `key`. Bumping the counter is the entire reset mechanism --
# see the module docstring point 4 for why plain session_state.pop() was
# not reliable for the form-based Historical Analysis filters.
if "hist_reset_n" not in st.session_state:
    st.session_state["hist_reset_n"] = 0

if "pred_reset_n" not in st.session_state:
    st.session_state["pred_reset_n"] = 0

# Survives resets on purpose -- see module docstring point 5.
if "pred_active_mode" not in st.session_state:
    st.session_state["pred_active_mode"] = "Zone"


def reset_historical_filters():
    st.session_state["hist_reset_n"] += 1
    st.rerun()


def reset_prediction_filters():
    # pred_active_mode is already kept in sync on every run (see below),
    # so it already holds whatever mode was active when Reset was clicked.
    # Only the counter needs to move.
    st.session_state["pred_reset_n"] += 1
    st.rerun()


# ======================================================================
# LOADERS
# ======================================================================

@st.cache_data
def load_dataset():
    dataset = pd.read_csv(DATASET_PATH)
    dataset["Date"] = pd.to_datetime(dataset["Date"], errors="coerce")
    dataset["date"] = dataset["Date"].dt.date
    return dataset


@st.cache_resource
def load_metadata():
    return get_metadata()


try:
    dataset = load_dataset()
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


# ======================================================================
# HEADER
# ======================================================================

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


# ======================================================================
# HISTORICAL ANALYSIS
# ======================================================================

if mode == "Historical Analysis":

    hn = st.session_state["hist_reset_n"]

    header_col, reset_col = st.columns([6, 1])

    with header_col:
        st.subheader("🔎 Demand Analysis")

    with reset_col:
        st.write("")  # vertical alignment with the subheader
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

    # ------------------------------------------------------------------
    # KPIs
    # ------------------------------------------------------------------
    st.subheader("📊 Demand Summary")

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Cells Displayed", f"{len(filtered):,}")
    m2.metric("Zones", f"{filtered['zone_id'].nunique():,}")
    m3.metric("Total Rides", f"{int(filtered['ride_count'].sum()):,}")
    m4.metric("Rides per Cell", f"{filtered['ride_count'].mean():.1f}")

    # ------------------------------------------------------------------
    # MAP
    # ------------------------------------------------------------------
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

    # ------------------------------------------------------------------
    # TABLE
    # ------------------------------------------------------------------
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
            "date": "DATE",
            "zone_name": "ZONE",
            "slot_name": "TIME SLOT",
            "weekday": "WEEKDAY",
            "ride_count": "RIDES",
            "prev_slot_demand": "PREV SLOT",
            "prev_day_demand": "PREV DAY",
            "zone_slot_roll7": "7-DAY AVG",
            "holiday_name": "HOLIDAY",
            "weather_indicator": "RAIN",
            "Demand_Category": "CATEGORY",
        }
    )

    st.dataframe(display, use_container_width=True, hide_index=True, height=480)


# ======================================================================
# DEMAND PREDICTION
# ======================================================================

if mode == "Demand Prediction":

    pn = st.session_state["pred_reset_n"]

    header_col, reset_col = st.columns([6, 1])

    with header_col:
        st.subheader("🔮 Demand Category Prediction")

    with reset_col:
        st.write("")  # vertical alignment with the subheader
        if st.button("↺ Reset Filters", key="reset_pred_btn", use_container_width=True):
            reset_prediction_filters()

    st.write(
        "Pick a zone, a date and a time slot. Historical context "
        "(previous slot, previous day, rolling average, weather) is filled "
        "in automatically from the processed dataset."
    )

    col1, col2 = st.columns(2)

    with col1:
        # index comes from pred_active_mode, which is NEVER versioned/reset
        # -- this is what lets Reset clear every other field while leaving
        # the user in whichever mode (Zone / Pickup Location) they were
        # already using.
        mode_index = MODE_OPTIONS.index(st.session_state["pred_active_mode"])

        selection_mode = st.radio(
            "Choose zone by",
            MODE_OPTIONS,
            horizontal=True,
            index=mode_index,
            key=f"pred_selection_mode_{pn}",
        )

        # Keep the stable tracker in sync with any live change, so the
        # *next* reset (whenever it happens) preserves whatever mode the
        # user is in at that time, not just whatever it was at page load.
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

    # ------------------------------------------------------------------
    # CONTEXT ASSEMBLY
    # ------------------------------------------------------------------
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
        # Same zone, same slot, same weekday -- averaged.
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

    # Weather for the selected date, falling back to the monthly average.
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

    # Single assembly point -- both branches reach here, so no variable can
    # be left undefined the way `temp` was previously.
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

    # ------------------------------------------------------------------
    # CONTEXT PANEL
    # ------------------------------------------------------------------
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

    # ------------------------------------------------------------------
    # PREDICT
    # ------------------------------------------------------------------
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
