import os
import pandas as pd
import streamlit as st

from src.config import DATASET_FILE
from src.build_dataset import build as build_dataset, aggregate_to_demand
from src.calendar_context import add_calendar_context
from src.model_training import train_and_compare, prepare_features, FEATURE_COLUMNS


# ------------------------------------------------------------
# 1. Page Configuration & Styling
# ------------------------------------------------------------

st.set_page_config(
    page_title="SmartRide Intelligence",
    page_icon="🚖",
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


st.title("🚖 SmartRide: Ride-Hailing Demand Intelligence")

st.markdown(
    "AI-based hourly demand forecasting and fleet allocation insights."
)


# ------------------------------------------------------------
# 2. Data Loading & Feature Engineering
# ------------------------------------------------------------

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

    # Calendar information
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

    # Aggregate rides into demand
    agg_df = aggregate_to_demand(raw)

    # Prepare forecasting features
    agg_df = prepare_features(agg_df)

    return agg_df, calendar_df


# ------------------------------------------------------------
# 3. Load Data
# ------------------------------------------------------------

with st.spinner(
    "Loading aggregated ride data..."
):

    data, calendar = load_and_prep_data()


clusters = sorted(
    data["cluster"].unique()
)


# ------------------------------------------------------------
# 4. Sidebar Controls
# ------------------------------------------------------------

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


# ------------------------------------------------------------
# 5. Demand Forecasting
# ------------------------------------------------------------

st.header("🔮 Demand Forecast")


# ------------------------------------------------------------
# Calculate Historical Demand Pattern
# ------------------------------------------------------------

# Filter selected cluster
cluster_data = data[
    data["cluster"] == selected_cluster
].copy()


# Calculate average demand for each hour
hourly_demand = (
    cluster_data
    .groupby("hour")["demand"]
    .mean()
)


# ------------------------------------------------------------
# Create Target Input
# ------------------------------------------------------------

input_data = pd.DataFrame([
    {
        "hour": selected_hour,
        "cluster": selected_cluster,
        "day": selected_day,
        "rainfall_mm": selected_rain,
        "previous_demand": selected_prev_demand,
    }
])[FEATURE_COLUMNS]


# ------------------------------------------------------------
# Simple Demand Prediction
# ------------------------------------------------------------

# Use historical hourly demand as the base forecast.
# If the selected hour has no historical value,
# use the overall cluster average.

if selected_hour in hourly_demand.index:

    base_prediction = hourly_demand[
        selected_hour
    ]

else:

    base_prediction = cluster_data[
        "demand"
    ].mean()


# ------------------------------------------------------------
# Rainfall Adjustment
# ------------------------------------------------------------

# Apply a small adjustment for hypothetical rainfall.
# This keeps the forecast responsive to the scenario
# without requiring an algorithm selector.

rainfall_adjustment = 1.0

if selected_rain >= 20:
    rainfall_adjustment = 0.90

elif selected_rain >= 10:
    rainfall_adjustment = 0.95

elif selected_rain >= 5:
    rainfall_adjustment = 0.98


# ------------------------------------------------------------
# Previous Demand Adjustment
# ------------------------------------------------------------

historical_average = cluster_data[
    "demand"
].mean()


if historical_average > 0:

    previous_demand_factor = (
        selected_prev_demand /
        historical_average
    )

    # Limit the effect so that extreme inputs
    # do not produce unrealistic forecasts.

    previous_demand_factor = max(
        0.75,
        min(
            previous_demand_factor,
            1.25
        )
    )

else:

    previous_demand_factor = 1.0


# ------------------------------------------------------------
# Final Prediction
# ------------------------------------------------------------

target_prediction = (
    base_prediction
    * rainfall_adjustment
    * previous_demand_factor
)


target_prediction = max(
    0,
    round(target_prediction)
)


# ------------------------------------------------------------
# 24-Hour Forecast
# ------------------------------------------------------------

forecast_records = []


for hour in range(24):

    if hour in hourly_demand.index:

        predicted_demand = hourly_demand[
            hour
        ]

    else:

        predicted_demand = historical_average


    # Rainfall effect
    predicted_demand *= rainfall_adjustment


    # Previous demand influence
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


# ------------------------------------------------------------
# 6. Executive KPI Summary
# ------------------------------------------------------------

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


# ------------------------------------------------------------
# 7. Forecast Curve
# ------------------------------------------------------------

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


# ------------------------------------------------------------
# 8. Demand by Hour
# ------------------------------------------------------------

st.subheader(
    "🕐 Hourly Demand Pattern"
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


# ------------------------------------------------------------
# 9. Peak Demand Information
# ------------------------------------------------------------

st.subheader(
    "🚦 Fleet Allocation Insights"
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


# ------------------------------------------------------------
# 10. Dataset Information
# ------------------------------------------------------------

st.subheader(
    "📊 Zone Information"
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


# ------------------------------------------------------------
# Footer
# ------------------------------------------------------------

st.markdown("---")

st.caption(
    "SmartRide | Ride-Hailing Demand Intelligence"
)