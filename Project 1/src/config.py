# ============================================================
# PROJECT CONFIGURATION
# ============================================================

# Input dataset
DATA_FILE = "data/ncr_ride_bookings.csv"

# Output directory
OUTPUT_DIR = "outputs"

# ------------------------------------------------------------
# COLUMN NAMES
# ------------------------------------------------------------
# CHANGE THESE IF YOUR CSV USES DIFFERENT COLUMN NAMES
# ------------------------------------------------------------

LATITUDE_COLUMN = "Pickup Latitude"
LONGITUDE_COLUMN = "Pickup Longitude"

TIMESTAMP_COLUMN = "Timestamp"

# Optional columns
TRIP_ID_COLUMN = "Booking ID"
FARE_COLUMN = "Booking Value"
DISTANCE_COLUMN = "Ride Distance"

# NEW: the raw CSV only has a zone NAME, not coordinates.
# This is the column geocoding.py reads to derive lat/long from.
PICKUP_LOCATION_COLUMN = "Pickup Location"

# NEW: cached lat/long per pickup zone, so geocoding only runs once.
GEOCODE_CACHE = "data/geocoded_locations.csv"


# ------------------------------------------------------------
# DBSCAN PARAMETERS
# ------------------------------------------------------------

# Radius around each point in meters
EPS_METERS = 200

# Minimum number of points required to form a dense region
MIN_SAMPLES = 20


# ------------------------------------------------------------
# TIME WINDOWS
# ------------------------------------------------------------

TIME_PERIODS = {
    "Morning": (6, 12),
    "Afternoon": (12, 17),
    "Evening": (17, 22),
    "Night": (22, 6)
}


# ------------------------------------------------------------
# MAP SETTINGS
# ------------------------------------------------------------

# NCR approximate center
MAP_CENTER_LAT = 28.6139
MAP_CENTER_LON = 77.2090

MAP_ZOOM = 10
