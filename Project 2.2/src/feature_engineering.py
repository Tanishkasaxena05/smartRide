
import numpy as np
import pandas as pd
import holidays

from src.zoning import build_zones, assign_slots, slot_label, slot_name


RAIN_PATTERN = "rain|storm|thunder|drizzle|shower"

WEATHER_COLUMNS = [
    "temp",
    "humidity",
    "precip",
    "precipprob",
    "windspeed",
    "heat_index",
    "temp_range",
]

# Features the model is allowed to see.  Note what is absent:
# ride_count, demand_ratio and Demand_Category.
MODEL_FEATURES = [
    "slot",
    "hour_start",
    "weekday_encoded",
    "month",
    "day",
    "week_of_year",
    "is_weekend",
    "is_holiday",
    "zone_id",
    "zone_lat",
    "zone_lon",
    "n_locations",
    "prev_slot_demand",
    "prev_day_demand",
    "prev_week_demand",
    "zone_slot_expanding_avg",
    "zone_slot_roll7",
    "zone_expanding_avg",
    "weather_indicator",
] + WEATHER_COLUMNS


def create_demand_dataset(
    rides,
    geo,
    weather,
    n_zones=10,
    slot_hours=4,
    train_fraction=0.80,
):
    rides = rides.copy()
    weather = weather.copy()

    slots_per_day = 24 // slot_hours

    # ------------------------------------------------------------------
    # ZONES
    # ------------------------------------------------------------------
    zone_lookup, zone_info, kmeans = build_zones(geo, n_zones=n_zones)

    rides = rides.merge(zone_lookup, on="Pickup Location", how="left")

    unmapped = int(rides["zone_id"].isna().sum())
    if unmapped:
        missing = (
            rides.loc[rides["zone_id"].isna(), "Pickup Location"]
            .value_counts()
            .head(10)
        )
        print(
            f"\n[zoning] {unmapped:,} rides had no geocoded location and were "
            f"dropped. Top offenders:\n{missing.to_string()}"
        )
        rides = rides.dropna(subset=["zone_id"])

    rides["zone_id"] = rides["zone_id"].astype(int)
    rides["hour"] = rides["Time"].dt.hour
    rides["slot"] = assign_slots(rides["hour"], slot_hours)

    # ------------------------------------------------------------------
    # AGGREGATE TO ZONE / SLOT COUNTS
    # ------------------------------------------------------------------
    counts = (
        rides.groupby(["Date", "zone_id", "slot"])
        .size()
        .reset_index(name="ride_count")
    )

    # Complete grid. A zone/slot with no bookings is genuine zero demand,
    # not a missing row -- and a complete grid makes the lag shifts below
    # line up exactly with real time offsets.
    all_dates = pd.date_range(
        rides["Date"].min(), rides["Date"].max(), freq="D"
    )
    zone_ids = sorted(zone_info["zone_id"].unique().tolist())

    grid = pd.MultiIndex.from_product(
        [all_dates, zone_ids, range(slots_per_day)],
        names=["Date", "zone_id", "slot"],
    ).to_frame(index=False)

    demand = grid.merge(counts, on=["Date", "zone_id", "slot"], how="left")
    demand["ride_count"] = demand["ride_count"].fillna(0).astype(int)

    demand = demand.sort_values(["zone_id", "Date", "slot"]).reset_index(drop=True)

    print(
        f"\n[grain] {len(demand):,} cells "
        f"({len(all_dates)} days x {len(zone_ids)} zones x {slots_per_day} slots)"
    )
    print("[grain] ride_count distribution:")
    print(demand["ride_count"].describe().round(2).to_string())

    zero_share = (demand["ride_count"] == 0).mean()
    one_share = (demand["ride_count"] == 1).mean()
    print(
        f"[grain] zero-demand cells: {zero_share:.1%}   "
        f"single-ride cells: {one_share:.1%}"
    )

    # ------------------------------------------------------------------
    # LAG FEATURES  (all strictly backward-looking)
    # ------------------------------------------------------------------
    by_zone = demand.groupby("zone_id", sort=False)["ride_count"]

    demand["prev_slot_demand"] = by_zone.shift(1)
    demand["prev_day_demand"] = by_zone.shift(slots_per_day)
    demand["prev_week_demand"] = by_zone.shift(7 * slots_per_day)

    by_zone_slot = demand.groupby(["zone_id", "slot"], sort=False)["ride_count"]

    demand["zone_slot_expanding_avg"] = by_zone_slot.transform(
        lambda s: s.shift(1).expanding().mean()
    )
    demand["zone_slot_roll7"] = by_zone_slot.transform(
        lambda s: s.shift(1).rolling(7, min_periods=1).mean()
    )
    demand["zone_expanding_avg"] = demand.groupby("zone_id", sort=False)[
        "ride_count"
    ].transform(lambda s: s.shift(1).expanding().mean())

    lag_columns = [
        "prev_slot_demand",
        "prev_day_demand",
        "prev_week_demand",
        "zone_slot_expanding_avg",
        "zone_slot_roll7",
        "zone_expanding_avg",
    ]
    for col in lag_columns:
        demand[col] = demand[col].fillna(0.0).astype(float)

    for col in ["prev_slot_demand", "prev_day_demand", "prev_week_demand"]:
        demand[col] = demand[col].astype(int)

    # ------------------------------------------------------------------
    # CALENDAR FEATURES
    # ------------------------------------------------------------------
    demand["date"] = demand["Date"].dt.date
    demand["weekday"] = demand["Date"].dt.day_name()
    demand["weekday_encoded"] = demand["Date"].dt.weekday
    demand["month"] = demand["Date"].dt.month
    demand["day"] = demand["Date"].dt.day
    demand["week_of_year"] = (
        demand["Date"].dt.isocalendar().week.astype(int)
    )
    demand["is_weekend"] = (demand["weekday_encoded"] >= 5).astype(int)
    demand["hour_start"] = demand["slot"] * slot_hours
    demand["slot_label"] = demand["slot"].map(
        lambda s: slot_label(s, slot_hours)
    )
    demand["slot_name"] = demand["slot"].map(lambda s: slot_name(s, slot_hours))

    holiday_years = sorted(demand["Date"].dt.year.unique().tolist())
    india_holidays = holidays.India(years=holiday_years)

    demand["is_holiday"] = demand["date"].map(
        lambda d: int(d in india_holidays)
    )
    demand["holiday_name"] = demand["date"].map(
        lambda d: india_holidays.get(d, "None")
    )

    # ------------------------------------------------------------------
    # WEATHER
    # ------------------------------------------------------------------
    weather["weather_indicator"] = (
        weather["conditions"]
        .astype(str)
        .str.lower()
        .str.contains(RAIN_PATTERN, regex=True)
        .astype(int)
    )

    demand = demand.merge(
        weather[["DATE", "weather_indicator"] + WEATHER_COLUMNS],
        left_on="Date",
        right_on="DATE",
        how="left",
    ).drop(columns=["DATE"])

    weather_null = demand["temp"].isna().mean()
    print(f"\n[weather] {weather_null:.1%} of cells have no weather match")

    if weather_null > 0.5:
        raise ValueError(
            "More than half of the demand cells have no weather data. "
            "The DATE parsing in preprocessing.py needs attention."
        )

    # Fill the remaining gaps with that month's average rather than dropping
    # the columns, so SimpleImputer never silently removes a feature.
    for col in WEATHER_COLUMNS:
        demand[col] = demand.groupby("month")[col].transform(
            lambda s: s.fillna(s.mean())
        )
        demand[col] = demand[col].fillna(demand[col].mean())

    demand["weather_indicator"] = (
        demand["weather_indicator"].fillna(0).astype(int)
    )

    # ------------------------------------------------------------------
    # ZONE GEOMETRY
    # ------------------------------------------------------------------
    demand = demand.merge(zone_info, on="zone_id", how="left")

    # ------------------------------------------------------------------
    # DISPLAY-ONLY RATIO  (never a model feature -- it contains ride_count)
    # ------------------------------------------------------------------
    baseline = demand["zone_slot_expanding_avg"].replace(0, np.nan)
    demand["demand_ratio"] = (demand["ride_count"] / baseline).fillna(1.0)

    # ------------------------------------------------------------------
    # TARGET -- tertiles of raw ride_count, fitted on the training window
    # ------------------------------------------------------------------
    unique_dates = sorted(demand["Date"].unique())
    cutoff_index = max(1, int(len(unique_dates) * train_fraction))
    cutoff_index = min(cutoff_index, len(unique_dates) - 1)
    cutoff_date = unique_dates[cutoff_index]

    train_counts = demand.loc[demand["Date"] < cutoff_date, "ride_count"]

    low_threshold = float(train_counts.quantile(1 / 3))
    high_threshold = float(train_counts.quantile(2 / 3))

    if high_threshold <= low_threshold:
        high_threshold = low_threshold + 1.0

    print(
        f"\n[target] thresholds from training window "
        f"(< {pd.Timestamp(cutoff_date).date()}):"
    )
    print(f"[target] Low  : ride_count <= {low_threshold:.0f}")
    print(f"[target] High : ride_count >  {high_threshold:.0f}")

    def classify(count):
        if count <= low_threshold:
            return "Low"
        if count <= high_threshold:
            return "Medium"
        return "High"

    demand["Demand_Category"] = demand["ride_count"].map(classify)

    print("\n[target] class distribution:")
    print(demand["Demand_Category"].value_counts().to_string())
    print("\n[target] class share:")
    print(
        demand["Demand_Category"]
        .value_counts(normalize=True)
        .round(3)
        .to_string()
    )

    # ------------------------------------------------------------------
    # TIDY
    # ------------------------------------------------------------------
    demand = demand.sort_values(["Date", "zone_id", "slot"]).reset_index(
        drop=True
    )

    metadata = {
        "n_zones": int(len(zone_ids)),
        "slot_hours": int(slot_hours),
        "slots_per_day": int(slots_per_day),
        "low_threshold": low_threshold,
        "high_threshold": high_threshold,
        "zone_lookup": zone_lookup,
        "zone_info": zone_info,
        "kmeans": kmeans,
        "features": MODEL_FEATURES,
    }

    return demand, metadata
