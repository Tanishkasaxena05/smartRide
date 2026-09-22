# ============================================================
# PROJECT 3 CONFIGURATION — Ride Demand Prediction
# ============================================================

RAW_DATA_FILE = "data/ncr_ride_bookings.csv"
GEOCODE_CACHE = "data/geocoded_locations.csv"

# This is the file app.py reads. Built by run_build_dataset.py.
DATASET_FILE = "data/geocoded_clusters_hourly_weather.csv"

OUTPUT_DIR = "outputs"

PICKUP_LOCATION_COLUMN = "Pickup Location"

# Geographic clustering (reuses the Project 1 DBSCAN approach) —
# groups nearby pickup zones into a smaller number of "clusters" so
# the demand model predicts per-area, not per-individual-zone-name.
CLUSTER_EPS_METERS = 1500
CLUSTER_MIN_SAMPLES = 200

# Approximate monthly rainfall pattern for Delhi NCR (mm) — heavier
# in the Jun-Sep monsoon, drier otherwise. This is a CLIMATOLOGICAL
# APPROXIMATION, not measured historical weather (no real weather
# data is available for this dataset's dates), used only to give the
# model a realistic seasonal "rain" feature as the brief requests.
MONTHLY_AVG_RAINFALL_MM = {
    1: 15, 2: 18, 3: 13, 4: 7, 5: 15, 6: 60,
    7: 180, 8: 175, 9: 95, 10: 15, 11: 3, 12: 8,
}

RANDOM_STATE = 42
TEST_SIZE = 0.2
