# 🚕 Ride-Hailing Pickup Hotspot Detection Using DBSCAN

## Project Overview

This project identifies geographical areas where ride-hailing
pickup requests are concentrated.

DBSCAN (Density-Based Spatial Clustering of Applications with Noise)
is used to discover natural geographical clusters without specifying
the number of clusters in advance.

## Note on this dataset

The bundled `data/ncr_ride_bookings.csv` only records pickup zones as
**names** (e.g. "Palam Vihar"), not coordinates. `src/geocoding.py`
handles this automatically:

1. It first tries real geocoding via OpenStreetMap/Nominatim (needs
   internet — takes a few minutes the first time, then caches to
   `data/geocoded_locations.csv`).
2. If that's unavailable (blocked network, firewall, etc.), it falls
   back automatically to approximate NCR sub-region coordinates, so
   the pipeline still runs end to end with no internet at all.

A working `data/geocoded_locations.csv` (offline-fallback coordinates)
is already included, so you can run everything immediately without
waiting on geocoding. Delete that file and rerun if you want it to
attempt real geocoding instead.

## Technologies

- Python
- Pandas
- NumPy
- Scikit-learn
- DBSCAN
- Geopy
- Folium
- Streamlit

## Project Structure

```
ABCD/
├── data/
│   ├── ncr_ride_bookings.csv
│   └── geocoded_locations.csv   (cached zone coordinates)
│
├── outputs/                      (created on first run)
│
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── geocoding.py              (derives coordinates from zone names)
│   ├── data_loader.py
│   ├── dbscan_clustering.py
│   ├── cluster_analysis.py
│   └── map_visualization.py
│
├── run_analysis.py
├── app.py
├── requirements.txt
└── README.md
```

`clean_data.py` and `location_processor.py` are standalone exploration
scripts from an earlier stage — they are **not** used by `run_analysis.py`
or `app.py` and can be ignored or deleted.

## Setup

Install dependencies:

```
pip install -r requirements.txt
```

## Run Analysis

```
python run_analysis.py
```

Generates:
- `outputs/clustered_rides.csv`
- `outputs/cluster_summary.csv`
- `outputs/time_summary.csv`
- `outputs/day_summary.csv`
- `outputs/hotspot_map.html`

## Run Dashboard

```
streamlit run app.py
```

Lets you interactively change:
- DBSCAN radius (`eps_meters`)
- Minimum samples
- Time period filter

## DBSCAN

Uses:
- Haversine distance (real-world distance on a sphere)
- `eps` in meters, converted internally to radians
- `min_samples`

Noise points receive cluster ID `-1`.

## Interpretation

Cluster 0, Cluster 1, etc. represent geographical pickup hotspots.
Cluster -1 represents points DBSCAN considers noise (isolated, not
part of any dense region).
