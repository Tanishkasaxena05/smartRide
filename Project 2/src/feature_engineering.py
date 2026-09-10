import pandas as pd


def create_demand_dataset(rides, geo):

    rides = rides.copy()

    rides["hour"] = rides["Time"].dt.hour

    rides["weekday"] = (
        rides["Date"]
        .dt.day_name()
    )

    demand = (
        rides.groupby(
            [
                "Pickup Location",
                "hour",
                "weekday"
            ]
        )
        .size()
        .reset_index(
            name="ride_count"
        )
    )

    demand = demand.merge(
        geo,
        on="Pickup Location",
        how="left"
    )

    weekday_map = {
        "Monday": 0,
        "Tuesday": 1,
        "Wednesday": 2,
        "Thursday": 3,
        "Friday": 4,
        "Saturday": 5,
        "Sunday": 6
    }

    demand["weekday_encoded"] = (
        demand["weekday"]
        .map(weekday_map)
        )
    previous_demand = (
        demand[
            [
                "Pickup Location",
                "weekday",
                "hour",
                "ride_count"
            ]
        ]
        .copy()
    )

    previous_demand["hour"] = (
        previous_demand["hour"] + 1
    )

    previous_demand = previous_demand.rename(
        columns={
            "ride_count": "previous_demand"
        }
    )

    demand = demand.merge(
        previous_demand[
            [
                "Pickup Location",
                "weekday",
                "hour",
                "previous_demand"
            ]
        ],
        on=[
            "Pickup Location",
            "weekday",
            "hour"
        ],
        how="left"
    )

    demand["previous_demand"] = (
        demand["previous_demand"]
        .fillna(0)
        .astype(int)
    )

    demand["Demand_Category"] = pd.qcut(
        demand["ride_count"],
        q=3,
        labels=[
            "Low",
            "Medium",
            "High"
        ],
        duplicates="drop"
    )

    print("\nDemand Category Distribution:\n")
    print(
        demand["Demand_Category"]
        .value_counts()
    )

    return demand