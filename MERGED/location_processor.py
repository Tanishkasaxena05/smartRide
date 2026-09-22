import pandas as pd
import time
from geopy.geocoders import Nominatim


INPUT_FILE = "data/cleaned/ncr_ride_bookings_cleaned.csv"
OUTPUT_FILE = "data/cleaned/ncr_ride_bookings_with_coordinates.csv"
LOCATION_FILE = "data/location_coordinates.csv"


# --------------------------------------------------
# 1. Load cleaned data
# --------------------------------------------------

df = pd.read_csv(INPUT_FILE)

print("Rows:", len(df))
print("Unique pickup locations:", df["pickup_location"].nunique())

locations = (
    df["pickup_location"]
    .dropna()
    .astype(str)
    .str.strip()
    .unique()
)

print("\nPickup locations:")
for location in locations:
    print("-", location)


# --------------------------------------------------
# 2. Geocoder
# --------------------------------------------------

geolocator = Nominatim(
    user_agent="ncr_ride_hailing_hotspot_project"
)


# --------------------------------------------------
# 3. Geocode each unique location
# --------------------------------------------------

results = []

for location in locations:

    print(f"Searching: {location}")

    try:
        # NCR helps disambiguate locations
        query = f"{location}, Delhi NCR, India"

        result = geolocator.geocode(
            query,
            timeout=10
        )

        if result:

            results.append({
                "pickup_location": location,
                "latitude": result.latitude,
                "longitude": result.longitude
            })

            print(
                f"  Found: "
                f"{result.latitude}, {result.longitude}"
            )

        else:

            results.append({
                "pickup_location": location,
                "latitude": None,
                "longitude": None
            })

            print("  NOT FOUND")

    except Exception as e:

        print("  Error:", e)

        results.append({
            "pickup_location": location,
            "latitude": None,
            "longitude": None
        })

    # Be polite to the geocoding service
    time.sleep(1)


# --------------------------------------------------
# 4. Save coordinate mapping
# --------------------------------------------------

location_df = pd.DataFrame(results)

location_df.to_csv(
    LOCATION_FILE,
    index=False
)

print("\nSaved:", LOCATION_FILE)


# --------------------------------------------------
# 5. Merge coordinates with ride data
# --------------------------------------------------

df = df.merge(
    location_df,
    on="pickup_location",
    how="left"
)

df.to_csv(
    OUTPUT_FILE,
    index=False
)

print("Saved:", OUTPUT_FILE)


# --------------------------------------------------
# 6. Check missing coordinates
# --------------------------------------------------

missing = df["latitude"].isna().sum()

print("\nRows without coordinates:", missing)

if missing > 0:

    print("\nLocations that were not found:")

    print(
        df.loc[
            df["latitude"].isna(),
            "pickup_location"
        ].unique()
    )