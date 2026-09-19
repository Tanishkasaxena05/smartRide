import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

# 1. Page Configuration & Professional Slate Styling
st.set_page_config(page_title="SmartRide Intelligence", page_icon="🚖", layout="wide")

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
st.markdown("AI-based hourly demand forecasting and fleet allocation insights[cite: 1].")

# 2. Data Loading & Feature Engineering (Cached)
HOLIDAYS_2024 = {
    "2024-01-26": "Republic Day",
    "2024-03-25": "Holi",
    "2024-04-11": "Eid al-Fitr",
    "2024-08-15": "Independence Day",
    "2024-10-02": "Gandhi Jayanti",
    "2024-10-12": "Dussehra",
    "2024-10-31": "Diwali",
    "2024-11-01": "Diwali holiday",
}

EVENTS_2024 = {
    "2024-01-26": "Republic Day Parade",
    "2024-11-14": "India International Trade Fair",
    "2024-11-15": "India International Trade Fair",
    "2024-11-16": "India International Trade Fair",
    "2024-11-17": "India International Trade Fair",
    "2024-11-18": "India International Trade Fair",
    "2024-11-19": "India International Trade Fair",
    "2024-11-20": "India International Trade Fair",
    "2024-11-21": "India International Trade Fair",
    "2024-11-22": "India International Trade Fair",
    "2024-11-23": "India International Trade Fair",
    "2024-11-24": "India International Trade Fair",
    "2024-11-25": "India International Trade Fair",
    "2024-11-26": "India International Trade Fair",
    "2024-11-27": "India International Trade Fair",
}


def add_calendar_context(df):
    dates = pd.to_datetime(df["date"], dayfirst=True)
    date_keys = dates.dt.strftime("%Y-%m-%d")
    df["holiday"] = date_keys.map(HOLIDAYS_2024).fillna("No holiday")
    df["event"] = date_keys.map(EVENTS_2024).fillna("No scheduled event")
    return df


@st.cache_data
def load_and_prep_data():
    df = pd.read_csv("geocoded_clusters_hourly_weather.csv")
    df = add_calendar_context(df)
    calendar_df = df[['date', 'day_name', 'holiday', 'event', 'rainfall_mm']].copy()
    calendar_df['date'] = pd.to_datetime(calendar_df['date'], dayfirst=True)
    calendar_df = calendar_df.groupby('date').agg(
        day_name=('day_name', 'first'),
        holiday=('holiday', 'first'),
        event=('event', 'first'),
        rainfall_mm=('rainfall_mm', 'mean'),
    ).reset_index()

    agg_df = df.groupby(['date', 'hour', 'cluster', 'day', 'rainfall_mm'])['booking_id'].count().reset_index()
    agg_df.rename(columns={'booking_id': 'demand'}, inplace=True)
    agg_df = agg_df.sort_values(by=['cluster', 'date', 'hour'])
    agg_df['previous_demand'] = agg_df.groupby('cluster')['demand'].shift(1)
    agg_df = agg_df.dropna()
    return agg_df, calendar_df

# 3. Model Training & Diagnostics (Cached)
@st.cache_resource
def train_model(df):
    feature_cols = ['hour', 'cluster', 'day', 'rainfall_mm', 'previous_demand']
    X = df[feature_cols]
    y = df['demand']
    
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)
    
    y_pred = model.predict(X)
    mae = mean_absolute_error(y, y_pred)
    r2 = r2_score(y, y_pred)
    
    importance = pd.DataFrame({
        'Importance': model.feature_importances_
    }, index=['Hour', 'Cluster', 'Day', 'Rainfall', 'Prev Demand']).sort_values(by='Importance', ascending=True)
    
    return model, mae, r2, importance, sorted(df['cluster'].unique()), sorted(df['day'].unique())

with st.spinner("Loading aggregated ride data and training AI model..."):
    data, calendar = load_and_prep_data()
    model, mae, r2, importance_df, clusters, days = train_model(data)

# 4. Sidebar Scenario Controls
st.sidebar.header("Scenario Controls")
st.sidebar.markdown("Adjust parameters to forecast future ride volume[cite: 1].")

selected_cluster = st.sidebar.selectbox("Geographical Zone (Cluster)", clusters)
selected_date = st.sidebar.selectbox(
    "Scenario Date", calendar['date'].dt.date.tolist()
)
selected_date_context = calendar[calendar['date'].dt.date == selected_date].iloc[0]
selected_day = selected_date.day
selected_hour = st.sidebar.slider("Target Hour (0-23)", 0, 23, 18)
selected_rain = st.sidebar.number_input(
    "Rainfall Forecast (mm)",
    min_value=0.0,
    value=float(selected_date_context['rainfall_mm']),
    step=0.5,
)
selected_prev_demand = st.sidebar.number_input("Previous Hour Demand (Rides)", min_value=0, value=50)

# 5. Core Demand Prediction & Simulation Pipeline
input_data = pd.DataFrame([{
    'hour': selected_hour,
    'cluster': selected_cluster,
    'day': selected_day,
    'rainfall_mm': selected_rain,
    'previous_demand': selected_prev_demand
}])

target_prediction = int(model.predict(input_data)[0])

# Generate full 24-hour forecast curve
forecast_records = []
current_prev = selected_prev_demand

for h in range(24):
    sim_row = pd.DataFrame([{
        'hour': h,
        'cluster': selected_cluster,
        'day': selected_day,
        'rainfall_mm': selected_rain,
        'previous_demand': current_prev
    }])
    pred_val = int(model.predict(sim_row)[0])
    forecast_records.append(pred_val)
    current_prev = pred_val

chart_data = pd.DataFrame({
    'Hour': range(24),
    'Predicted Rides': forecast_records,
})

# 6. Executive KPI Summary
st.subheader(f"Forecast Summary: Zone {selected_cluster}")
context_col1, context_col2, context_col3, context_col4 = st.columns(4)
context_col1.metric("Day", selected_date_context['day_name'])
context_col2.metric("Holiday", selected_date_context['holiday'])
context_col3.metric("Event", selected_date_context['event'])
context_col4.metric("Rainfall", f"{selected_rain:.1f} mm")

col1, col2, col3 = st.columns(3)

col1.metric(
    label=f"Expected Demand at {selected_hour:02d}:00", 
    value=f"{target_prediction} rides",
    delta="Driver fleet positioning needed"
)
col2.metric(
    label="Mean Absolute Error (MAE)", 
    value=f"± {mae:.1f} rides", 
    delta="Model accuracy margin", 
    delta_color="off"
)
col3.metric(
    label="Confidence Score (R²)", 
    value=f"{r2 * 100:.1f} %", 
    delta="Variance explained"
)

st.divider()

# 7. Interactive Dashboards & Graphs
left_col, right_col = st.columns([2, 1])

with left_col:
    st.subheader("📈 24-Hour Projected Demand Curve")
    st.caption(f"Continuous hourly volume simulation for Zone {selected_cluster} on {selected_date:%d %b %Y}.")
    st.line_chart(
        chart_data,
        x="Hour",
        y="Predicted Rides",
        color="#3b82f6",
        height=380,
    )

with right_col:
    st.subheader("🧠 Feature Importance")
    st.caption("Weight of each operational factor in model decisions.")
    st.bar_chart(importance_df, color="#10b981")