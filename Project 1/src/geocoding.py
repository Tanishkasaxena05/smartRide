# ============================================================
# GEOCODING
# ============================================================
# Your dataset only has "Pickup Location" (a place NAME), not
# coordinates. DBSCAN needs numeric lat/long, so this module
# converts names -> coordinates:
#   1. Try real geocoding via OpenStreetMap (needs internet).
#   2. If that's blocked/unavailable, fall back to approximate
#      NCR sub-region coordinates (offline, no internet needed).
# Results are cached so this only runs once.

import os
import hashlib
import logging
import numpy as np
import pandas as pd

from src.config import (
    GEOCODE_CACHE,
    PICKUP_LOCATION_COLUMN,
)

NCR_BOUNDS = {"lat_min": 28.20, "lat_max": 28.95, "lon_min": 76.75, "lon_max": 77.55}

# ------------------------------------------------------------
# Offline fallback: keyword -> approximate NCR sub-region
# ------------------------------------------------------------
_GURGAON_KW = ["gurgaon", "gurugram", "dlf", "cyber hub", "civil lines gurgaon",
               "badshahpur", "basai dhankot", "ardee city", "ambience mall gurgaon"]
_GHAZIABAD_KW = ["ghaziabad", "vaishali", "indirapuram", "vasundhara", "sahibabad"]
_NOIDA_KW = ["noida", "botanical garden"]
_NORTH_DELHI_KW = ["adarsh nagar", "azadpur", "ashok vihar", "gtb nagar", "model town",
                   "civil lines", "shalimar bagh", "pitampura", "rohini"]
_EAST_DELHI_KW = ["akshardham", "anand vihar", "dilshad garden", "preet vihar", "mayur vihar"]
_WEST_DELHI_KW = ["ashok park", "dwarka mor", "janakpuri", "rajouri garden", "tilak nagar",
                  "uttam nagar", "vikaspuri", "paschim vihar"]
_SOUTH_DELHI_KW = ["chhatarpur", "chirag delhi", "saket", "vasant kunj", "ghitorni",
                   "bhikaji cama", "ashram", "aiims", "aya nagar", "arjangarh", "malviya nagar"]


def _match_region(name):
    n = str(name).lower()
    if "dwarka" in n:
        return (28.59, 77.04)
    if any(k in n for k in _GURGAON_KW):
        return (28.45, 77.03)
    if "faridabad" in n:
        return (28.40, 77.31)
    if any(k in n for k in _GHAZIABAD_KW):
        return (28.67, 77.45)
    if any(k in n for k in _NOIDA_KW):
        return (28.55, 77.35)
    if "bahadurgarh" in n:
        return (28.69, 76.93)
    if "bhiwadi" in n:
        return (28.21, 76.87)
    if any(k in n for k in _NORTH_DELHI_KW):
        return (28.70, 77.19)
    if any(k in n for k in _EAST_DELHI_KW):
        return (28.62, 77.30)
    if any(k in n for k in _WEST_DELHI_KW):
        return (28.62, 77.05)
    if any(k in n for k in _SOUTH_DELHI_KW):
        return (28.53, 77.20)
    return (28.63, 77.22)  # default: Central Delhi


def _offline_geocode_one(name, jitter_deg=0.04):
    base_lat, base_lon = _match_region(name)
    h = int(hashlib.md5(str(name).encode()).hexdigest(), 16)
    dlat = ((h % 1000) / 1000 - 0.5) * 2 * jitter_deg
    dlon = (((h // 1000) % 1000) / 1000 - 0.5) * 2 * jitter_deg
    return base_lat + dlat, base_lon + dlon


def _offline_geocode_all(zones):
    zones = zones.copy()
    coords = zones[PICKUP_LOCATION_COLUMN].apply(_offline_geocode_one)
    zones["lat"] = coords.apply(lambda t: t[0])
    zones["lon"] = coords.apply(lambda t: t[1])
    print("Using OFFLINE approximate coordinates (keyword-region + deterministic "
          "jitter) — no internet needed, but these are NOT real geocoded points.")
    return zones


# ------------------------------------------------------------
# Online geocoding (OpenStreetMap / Nominatim), with fast-fail
# ------------------------------------------------------------
def _online_geocode_all(zones):
    logging.getLogger("geopy").setLevel(logging.ERROR)  # silence retry noise

    from geopy.geocoders import Nominatim
    from geopy.extra.rate_limiter import RateLimiter

    geolocator = Nominatim(user_agent="smartride_project1", timeout=5)
    geocode = RateLimiter(geolocator.geocode, min_delay_seconds=1, max_retries=1, error_wait_seconds=1.0)

    lats, lons = [], []
    consecutive_failures = 0
    names = zones[PICKUP_LOCATION_COLUMN].tolist()
    print(f"Geocoding {len(names)} zone names (~1 request/sec)...")

    for i, loc in enumerate(names):
        try:
            result = geocode(f"{loc}, Delhi NCR, India")
            if result and (NCR_BOUNDS["lat_min"] <= result.latitude <= NCR_BOUNDS["lat_max"]) \
                    and (NCR_BOUNDS["lon_min"] <= result.longitude <= NCR_BOUNDS["lon_max"]):
                lats.append(result.latitude)
                lons.append(result.longitude)
                consecutive_failures = 0
            else:
                lats.append(np.nan)
                lons.append(np.nan)
                consecutive_failures += 1
        except Exception:
            lats.append(np.nan)
            lons.append(np.nan)
            consecutive_failures += 1

        if (i + 1) % 25 == 0:
            print(f"  ...{i + 1}/{len(names)} done")

        if consecutive_failures >= 8:
            print(f"  {consecutive_failures} attempts in a row failed — "
                  f"connection to Nominatim looks unavailable. Stopping early.")
            for _ in range(i + 1, len(names)):
                lats.append(np.nan)
                lons.append(np.nan)
            break

    zones = zones.copy()
    zones["lat"] = lats
    zones["lon"] = lons
    return zones


# ------------------------------------------------------------
# Public API
# ------------------------------------------------------------
def geocode_pickup_locations(df):
    """Returns [PICKUP_LOCATION_COLUMN, 'lat', 'lon'] for every unique
    pickup location in df. Uses cache if present, else tries online
    geocoding, falling back to offline approximation if that mostly fails."""

    if os.path.exists(GEOCODE_CACHE):
        cached = pd.read_csv(GEOCODE_CACHE)
        cached["lat"] = pd.to_numeric(cached.get("lat"), errors="coerce")
        cached["lon"] = pd.to_numeric(cached.get("lon"), errors="coerce")
        cached = cached.dropna(subset=["lat", "lon"])
        if len(cached) > 0:
            print(f"Using cached coordinates from {GEOCODE_CACHE} ({len(cached)} zones)")
            return cached
        print(f"{GEOCODE_CACHE} exists but has no usable rows — regeocoding.")

    zones = pd.Series(
        df[PICKUP_LOCATION_COLUMN].dropna().unique(), name=PICKUP_LOCATION_COLUMN
    ).to_frame()

    result = _online_geocode_all(zones)
    n_success = result["lat"].notna().sum()

    if n_success < 0.5 * len(zones):
        print(f"\nOnline geocoding only succeeded for {n_success}/{len(zones)} zones "
              f"— falling back to offline approximate coordinates.\n")
        result = _offline_geocode_all(zones)
    else:
        result = result.dropna(subset=["lat", "lon"])
        print(f"Geocoded {len(result)}/{len(zones)} zones successfully.")

    os.makedirs(os.path.dirname(GEOCODE_CACHE) or ".", exist_ok=True)
    result.to_csv(GEOCODE_CACHE, index=False)
    return result


def ensure_coordinates(df, lat_col, lon_col):
    """Adds lat_col/lon_col to df (derived from Pickup Location) if not already present."""
    if lat_col in df.columns and lon_col in df.columns:
        return df

    zones = geocode_pickup_locations(df)
    zones = zones.rename(columns={"lat": lat_col, "lon": lon_col})
    merged = df.merge(
        zones[[PICKUP_LOCATION_COLUMN, lat_col, lon_col]],
        on=PICKUP_LOCATION_COLUMN,
        how="left",
    )
    return merged
