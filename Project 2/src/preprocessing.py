import pandas as pd


def load_data():

    rides = pd.read_csv(
        "data/raw/ncr_ride_bookings.csv",
        na_values=["null"]
    )

    geo = pd.read_csv(
        "data/raw/geocoded_locations.csv"
    )

    return rides, geo


def clean_data(rides):

    rides = rides.copy()

    rides.columns = rides.columns.str.strip()

    rides["Date"] = pd.to_datetime(
        rides["Date"],
        errors="coerce"
    )

    rides["Time"] = pd.to_datetime(
        rides["Time"],
        format="%H:%M:%S",
        errors="coerce"
    )

    rides = rides.drop_duplicates()

    rides = rides.dropna(
        subset=[
            "Date",
            "Time",
            "Pickup Location"
        ]
    )

    rides["Pickup Location"] = (
        rides["Pickup Location"]
        .astype(str)
        .str.strip()
    )

    return rides