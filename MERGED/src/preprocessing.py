
import pandas as pd
import holidays


RIDES_PATH = "data/raw/ncr_ride_bookings.csv"
GEO_PATH = "data/raw/geocoded_locations.csv"
WEATHER_PATH = "data/raw/kaggel_weather_2013_to_2024.csv"

WEATHER_COLUMNS = [
    "temp",
    "humidity",
    "precip",
    "precipprob",
    "windspeed",
    "heat_index",
    "temp_range",
]


def parse_mixed_dates(series, dayfirst=False):
    """Parse a date column that mixes '01-02-2024' and '1/13/2024' styles."""
    series = series.astype(str).str.strip()

    parsed = pd.to_datetime(
        series,
        format="mixed",
        dayfirst=dayfirst,
        errors="coerce",
    )

    # Rescue anything the mixed parser could not place.
    still_missing = parsed.isna() & series.ne("nan") & series.ne("")

    if still_missing.any():
        for fmt in ("%m-%d-%Y", "%m/%d/%Y", "%Y-%m-%d", "%d-%m-%Y", "%d/%m/%Y"):
            if not still_missing.any():
                break
            retry = pd.to_datetime(
                series[still_missing], format=fmt, errors="coerce"
            )
            parsed.loc[still_missing] = retry
            still_missing = parsed.isna() & series.ne("nan") & series.ne("")

    return parsed.dt.normalize()


def load_data():
    rides = pd.read_csv(RIDES_PATH, na_values=["null", "NULL", ""])
    geo = pd.read_csv(GEO_PATH)
    weather = pd.read_csv(WEATHER_PATH, na_values=["null", "NULL", ""])

    # ---------------- geocoded locations ----------------
    geo.columns = geo.columns.str.strip()
    geo["Pickup Location"] = geo["Pickup Location"].astype(str).str.strip()
    geo["lat"] = pd.to_numeric(geo["lat"], errors="coerce")
    geo["lon"] = pd.to_numeric(geo["lon"], errors="coerce")
    geo = geo.dropna(subset=["lat", "lon"]).drop_duplicates(
        subset=["Pickup Location"]
    )

    # ---------------- weather ----------------
    weather.columns = weather.columns.str.strip()

    if "DATE" not in weather.columns:
        raise KeyError(
            "Weather file has no DATE column. Columns found: "
            f"{list(weather.columns)[:15]}"
        )

    weather["DATE"] = parse_mixed_dates(weather["DATE"])

    unparsed = weather["DATE"].isna().sum()
    if unparsed:
        print(f"[weather] {unparsed:,} rows had an unparseable date and were dropped.")

    missing_cols = [
        c for c in WEATHER_COLUMNS + ["conditions"] if c not in weather.columns
    ]
    if missing_cols:
        raise KeyError(f"Weather file is missing expected columns: {missing_cols}")

    weather = weather[["DATE", "conditions"] + WEATHER_COLUMNS].copy()

    for col in WEATHER_COLUMNS:
        weather[col] = pd.to_numeric(weather[col], errors="coerce")

    weather = (
        weather.dropna(subset=["DATE"])
        .drop_duplicates(subset=["DATE"])
        .sort_values("DATE")
        .reset_index(drop=True)
    )

    print(
        f"[weather] {len(weather):,} daily rows parsed, "
        f"{weather['DATE'].min().date()} to {weather['DATE'].max().date()}"
    )

    return rides, geo, weather


def clean_data(rides):
    rides = rides.copy()
    rides.columns = rides.columns.str.strip()

    rides["Date"] = pd.to_datetime(rides["Date"], errors="coerce").dt.normalize()
    rides["Time"] = pd.to_datetime(
        rides["Time"], format="%H:%M:%S", errors="coerce"
    )

    rides = rides.drop_duplicates()
    rides = rides.dropna(subset=["Date", "Time", "Pickup Location"])

    rides["Pickup Location"] = (
        rides["Pickup Location"].astype(str).str.strip()
    )

    # Every booking request counts as demand, including cancellations and
    # "No Driver Found" -- those are exactly the moments where supply was
    # short, which is what this project is meant to anticipate.
    if "Booking Status" in rides.columns:
        print("\n[rides] Booking status mix (all are counted as demand):")
        print(rides["Booking Status"].value_counts())

    holiday_years = sorted(rides["Date"].dt.year.dropna().unique().tolist())
    india_holidays = holidays.India(years=holiday_years)

    rides["is_holiday"] = (
        rides["Date"].dt.date.map(lambda d: int(d in india_holidays))
    )
    rides["holiday_name"] = (
        rides["Date"].dt.date.map(lambda d: india_holidays.get(d, "None"))
    )

    print(
        f"\n[rides] {len(rides):,} usable bookings, "
        f"{rides['Date'].min().date()} to {rides['Date'].max().date()}, "
        f"{rides['Pickup Location'].nunique()} pickup locations"
    )

    return rides


def check_weather_overlap(rides, weather):
    """Fail loudly if the weather join would produce nothing."""
    ride_dates = set(rides["Date"].dt.normalize().unique())
    weather_dates = set(weather["DATE"].unique())
    overlap = ride_dates & weather_dates

    coverage = len(overlap) / max(len(ride_dates), 1)
    print(
        f"[weather] covers {len(overlap):,} of {len(ride_dates):,} ride dates "
        f"({coverage:.1%})"
    )

    if coverage < 0.5:
        raise ValueError(
            "Weather data covers less than half the ride dates. "
            "Check the DATE parsing before continuing -- this is the bug "
            "that silently nulled every weather feature."
        )

    return coverage