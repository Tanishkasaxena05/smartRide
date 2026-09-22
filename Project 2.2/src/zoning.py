
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans


def build_zones(geo, n_zones=10, random_state=42):
    """
    Cluster pickup locations into n_zones by lat/lon.

    Returns
    -------
    zone_lookup : DataFrame  [Pickup Location, zone_id]
    zone_info   : DataFrame  [zone_id, zone_name, zone_lat, zone_lon,
                              n_locations]
    kmeans      : the fitted estimator (saved with the model bundle so the
                  dashboard maps locations to zones identically)
    """
    geo = geo.dropna(subset=["lat", "lon"]).copy()

    n_zones = int(min(n_zones, len(geo)))

    kmeans = KMeans(n_clusters=n_zones, n_init=10, random_state=random_state)
    geo["zone_id"] = kmeans.fit_predict(geo[["lat", "lon"]].to_numpy())

    centroids = pd.DataFrame(
        kmeans.cluster_centers_, columns=["zone_lat", "zone_lon"]
    )
    centroids["zone_id"] = centroids.index

    # Name each zone after the pickup location nearest its centroid, so the
    # dashboard shows "Zone 3 - Connaught Place" rather than a bare integer.
    names = []
    for zone_id, group in geo.groupby("zone_id"):
        centre = centroids.loc[centroids["zone_id"] == zone_id]
        clat = float(centre["zone_lat"].iloc[0])
        clon = float(centre["zone_lon"].iloc[0])

        distance = np.hypot(group["lat"] - clat, group["lon"] - clon)
        anchor = group.loc[distance.idxmin(), "Pickup Location"]

        names.append(
            {
                "zone_id": int(zone_id),
                "zone_name": f"Zone {int(zone_id)} - {anchor}",
                "n_locations": int(len(group)),
            }
        )

    zone_info = (
        centroids.merge(pd.DataFrame(names), on="zone_id")
        .sort_values("zone_id")
        .reset_index(drop=True)
    )

    zone_lookup = geo[["Pickup Location", "zone_id"]].copy()
    zone_lookup["zone_id"] = zone_lookup["zone_id"].astype(int)

    print(f"\n[zoning] {len(geo)} locations grouped into {n_zones} zones")
    print(zone_info[["zone_id", "zone_name", "n_locations"]].to_string(index=False))

    return zone_lookup, zone_info, kmeans


def assign_slots(hour_series, slot_hours=4):
    """Map an hour-of-day to a time slot index."""
    return (hour_series // slot_hours).astype(int)


def slot_label(slot, slot_hours=4):
    start = int(slot) * slot_hours
    end = start + slot_hours
    return f"{start:02d}:00-{end:02d}:00"


def slot_name(slot, slot_hours=4):
    """Human-friendly slot name for the dashboard."""
    start = int(slot) * slot_hours
    if start < 6:
        period = "Night"
    elif start < 12:
        period = "Morning"
    elif start < 16:
        period = "Afternoon"
    elif start < 20:
        period = "Evening Peak"
    else:
        period = "Late Evening"
    return f"{period} ({slot_label(slot, slot_hours)})"
