import streamlit as st
import pandas as pd
import folium

from streamlit_folium import st_folium

from src.predictor import predict_category


# ==================================================
# PAGE CONFIG
# ==================================================

st.set_page_config(
    page_title="SmartRide NCR",
    page_icon="🚕",
    layout="wide"
)


# ==================================================
# CONSTANTS
# ==================================================

WEEKDAYS = [
    "Monday",
    "Tuesday",
    "Wednesday",
    "Thursday",
    "Friday",
    "Saturday",
    "Sunday"
]

CATEGORY_COLORS = {
    "Low": "#2ECC71",
    "Medium": "#F39C12",
    "High": "#E74C3C"
}


# ==================================================
# LOAD DATA
# ==================================================

@st.cache_data
def load_data():

    geo = pd.read_csv(
        "data/raw/geocoded_locations.csv"
    )

    dataset = pd.read_csv(
        "data/processed/demand_dataset.csv"
    )

    return geo, dataset


geo, dataset = load_data()


# ==================================================
# PAGE TITLE
# ==================================================

st.title(
    "🚕 SmartRide NCR"
)

st.subheader(
    "Ride-Hailing Demand Category Classification"
)

st.write(
    """
    Analyze historical ride demand across Delhi NCR
    and classify expected demand as Low, Medium, or High.
    """
)

st.caption(
    "Project 2 — Supervised Classification | "
    "Demand Category: Low / Medium / High"
)


# ==================================================
# FILTER SECTION
# ==================================================

st.subheader(
    "🔎 Demand Analysis"
)

with st.form(
    "analysis_filter_form"
):

    filter_col1, filter_col2 = st.columns(2)

    with filter_col1:

        selected_category = st.selectbox(
            "Demand Category",
            [
                "All",
                "Low",
                "Medium",
                "High"
            ]
        )

    with filter_col2:

        selected_hour = st.slider(
            "Analysis Hour Range",
            min_value=0,
            max_value=23,
            value=(0, 23),
            format="%02d:00"
        )

    apply_filters = st.form_submit_button(
        "Apply Filters",
        width="stretch"
    )


# ==================================================
# FILTER DATA
# ==================================================

filtered = dataset[
    dataset["hour"].between(
        selected_hour[0],
        selected_hour[1]
    )
].copy()


if selected_category != "All":

    filtered = filtered[
        filtered["Demand_Category"]
        == selected_category
    ].copy()


# ==================================================
# EMPTY RESULT
# ==================================================

if filtered.empty:

    st.warning(
        "No records found for the selected filters."
    )

    st.stop()


# ==================================================
# KPI SUMMARY
# ==================================================

st.subheader(
    "📊 Demand Summary"
)

metric1, metric2, metric3, metric4 = st.columns(4)

metric1.metric(
    "Rows Displayed",
    f"{len(filtered):,}"
)

metric2.metric(
    "Locations",
    f"{filtered['Pickup Location'].nunique():,}"
)

metric3.metric(
    "Total Rides",
    f"{filtered['ride_count'].sum():,}"
)

metric4.metric(
    "Average Rides",
    f"{filtered['ride_count'].mean():.1f}"
)


# ==================================================
# MAP DATA
# ==================================================

@st.cache_data
def prepare_map_data(
    filtered_data,
    category
):

    map_df = (
        filtered_data
        .groupby(
            [
                "Pickup Location",
                "lat",
                "lon"
            ],
            as_index=False
        )["ride_count"]
        .sum()
    )

    # ----------------------------------------------
    # ALL CATEGORIES
    # ----------------------------------------------
    # One bubble per location.
    # Categories are assigned according to the
    # aggregate ride volume of the current filter.

    if category == "All":

        ranked_rides = (
            map_df["ride_count"]
            .rank(
                method="first"
            )
        )

        map_df["Demand_Category"] = pd.qcut(
            ranked_rides,
            q=3,
            labels=[
                "Low",
                "Medium",
                "High"
            ]
        )

    # ----------------------------------------------
    # SPECIFIC CATEGORY
    # ----------------------------------------------
    # The data has already been filtered to the
    # selected category, so every location bubble
    # receives that category.
    else:

        map_df["Demand_Category"] = category

    return map_df


map_df = prepare_map_data(
    filtered,
    selected_category
)


# ==================================================
# MAP
# ==================================================

st.subheader(
    "🗺️ Delhi NCR Demand Map"
)

st.caption(
    "Bubble color represents demand category. "
    "Bubble size represents ride volume within the "
    "selected filters. The map automatically expands "
    "to the available page width."
)


city_map = folium.Map(
    location=[
        28.61,
        77.20
    ],
    zoom_start=10,
    tiles="OpenStreetMap"
)


for _, row in map_df.iterrows():

    category = row["Demand_Category"]

    marker_color = CATEGORY_COLORS.get(
        category,
        "#2ECC71"
    )

    # Keep the original bubble-size behavior.
    radius = min(
        8 + (row["ride_count"] ** 0.40),
        35
    )

    popup = f"""
    <b>{row['Pickup Location']}</b><br>
    <b>Category:</b> {category}<br>
    <b>Total Rides:</b> {int(row['ride_count']):,}<br>
    <b>Latitude:</b> {row['lat']:.4f}<br>
    <b>Longitude:</b> {row['lon']:.4f}
    """

    folium.CircleMarker(
        location=[
            row["lat"],
            row["lon"]
        ],
        radius=radius,
        color=marker_color,
        fill=True,
        fill_color=marker_color,
        fill_opacity=0.6,
        weight=2,
        tooltip=(
            f"{row['Pickup Location']} • "
            f"{category} Demand • "
            f"{int(row['ride_count']):,} rides"
        ),
        popup=popup
    ).add_to(city_map)


# ==================================================
# MAP LEGEND
# ==================================================

legend_html = """
<div style="
    position: fixed;
    bottom: 30px;
    left: 30px;
    z-index: 9999;
    background-color: rgba(255,255,255,0.96);
    padding: 14px;
    border: 2px solid #444;
    border-radius: 8px;
    font-size: 14px;
    color: #111;
    line-height: 1.6;
">

<b>Demand Categories</b><br><br>

<span style="color:#2ECC71;">●</span>
Low Demand<br>

<span style="color:#F39C12;">●</span>
Medium Demand<br>

<span style="color:#E74C3C;">●</span>
High Demand<br><br>

<b>Bubble Size</b><br>
Larger Bubble = More Rides

</div>
"""


city_map.get_root().html.add_child(
    folium.Element(
        legend_html
    )
)


st_folium(
    city_map,
    height=650,
    width="stretch"
)



# ==================================================
# FILTERED DATASET
# ==================================================

st.divider()

st.subheader(
    "📋 Filtered Dataset"
)

st.caption(
    f"Showing up to 500 rows from the current "
    f"filter selection ({len(filtered):,} matching rows)."
)

display_columns = [
    "Pickup Location",
    "hour",
    "weekday",
    "ride_count",
    "previous_demand",
    "Demand_Category"
]

display_df = (
    filtered[
        display_columns
    ]
    .head(500)
    .reset_index(drop=True)
)


# ==================================================
# TABLE NUMBER COLUMN
# ==================================================

display_df.insert(
    0,
    "NO.",
    range(
        1,
        len(display_df) + 1
    )
)


# ==================================================
# TABLE HEADINGS
# ==================================================

display_df = display_df.rename(
    columns={
        "Pickup Location": "PICKUP LOCATION",
        "hour": "HOUR",
        "weekday": "WEEKDAY",
        "ride_count": "RIDE COUNT",
        "previous_demand": "PREVIOUS DEMAND",
        "Demand_Category": "DEMAND CATEGORY"
    }
)
st.dataframe(
    display_df,
    width="stretch",
    hide_index=True,
    height=520,
    column_config={
        "NO.": st.column_config.NumberColumn(
            "NO.",
            width=45
        ),
        "HOUR": st.column_config.NumberColumn(
            "HOUR",
            width=60
        ),
        "RIDE COUNT": st.column_config.NumberColumn(
            "RIDE COUNT",
            width=90
        )
    }
)

# ==================================================
# DEMAND PREDICTION
# ==================================================

st.divider()

st.subheader(
    "🔮 Demand Category Prediction"
)

st.write(
    """
    Select a specific scenario and classify its expected
    demand level. This is a classification prediction,
    not Project 3's numerical ride-demand forecasting.
    """
)


prediction_col1, prediction_col2 = st.columns(2)


with prediction_col1:

    prediction_location = st.selectbox(
        "Pickup Location",
        sorted(
            geo["Pickup Location"]
            .dropna()
            .unique()
        ),
        key="prediction_location"
    )

    prediction_weekday = st.selectbox(
        "Weekday",
        WEEKDAYS,
        key="prediction_weekday"
    )


with prediction_col2:

    prediction_hours = list(
        range(
            selected_hour[0],
            selected_hour[1] + 1
        )
    )

    prediction_hour = st.selectbox(
    "Prediction Hour",
    prediction_hours,
    index=len(prediction_hours) // 2,
    format_func=lambda hour: f"{hour:02d}:00",
    key="prediction_hour"
)

# ==================================================
# AUTOMATIC PREVIOUS DEMAND
# ==================================================

if prediction_hour == 0:

    previous_demand = 0

else:

    previous_hour = prediction_hour - 1

    previous_rows = dataset[
        (
            dataset["Pickup Location"]
            == prediction_location
        )
        &
        (
            dataset["weekday"]
            == prediction_weekday
        )
        &
        (
            dataset["hour"]
            == previous_hour
        )
    ]

    if previous_rows.empty:

        previous_demand = 0

    else:

        previous_demand = int(
            previous_rows[
                "ride_count"
            ].sum()
        )

    st.info(
        f"""
        Previous Hour Demand (Auto Detected):

        {previous_demand:,} rides
        """
    )
        

# ==================================================
# PREDICTION LOCATION
# ==================================================

prediction_geo = geo[
    geo["Pickup Location"]
    == prediction_location
].copy()


if prediction_geo.empty:

    st.error(
        "Coordinates for the selected pickup location "
        "could not be found."
    )

else:

    prediction_row = (
        prediction_geo
        .iloc[0]
    )


# ==================================================
# PREDICTION BUTTON
# ==================================================

predict_clicked = st.button(
    "🚕 Predict Demand Category",
    width="stretch"
)


if (
    predict_clicked
    and not prediction_geo.empty
):

    category = predict_category(
        prediction_hour,
        prediction_weekday,
        previous_demand,
        prediction_row["lat"],
        prediction_row["lon"]
    )

    st.markdown(
        f"### Prediction Result"
    )

    result_col1, result_col2, result_col3 = st.columns(3)

    result_col1.metric(
        "LOCATION",
        prediction_location
    )

    result_col2.metric(
        "TIME",
        f"{prediction_hour:02d}:00"
    )

    result_col3.metric(
        "PREVIOUS DEMAND",
        f"{previous_demand:,}"
    )

    if category == "High":

        st.error(
            "🔥 HIGH DEMAND"
        )

        st.write(
            """
            The model expects high ride demand for this
            scenario. Increasing driver availability before
            the expected peak may help reduce waiting time.
            """
        )

    elif category == "Medium":

        st.warning(
            "⚠️ MEDIUM DEMAND"
        )

        st.write(
            """
            The model expects moderate ride demand.
            Normal driver availability should generally
            be sufficient, with monitoring during peak periods.
            """
        )

    else:

        st.success(
            "✅ LOW DEMAND"
        )

        st.write(
            """
            The model expects relatively low ride demand.
            Standard driver allocation should generally be sufficient.
            """
        )

# ==================================================
# FOOTER
# ==================================================

st.divider()

st.caption(
    "SmartRide NCR • Project 2 — Ride-Hailing Demand "
    "Category Classification"
)