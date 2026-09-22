# 🚕 SmartRide — AI-Based Ride-Hailing Demand & Hotspot Intelligence

SmartRide turns raw ride-hailing booking data into three connected pieces of
operational intelligence: **where** demand clusters geographically, **how busy**
a zone will be at a given time, and **how many rides** to expect next hour.
Built as a student capstone project spanning unsupervised learning, supervised
classification, and regression — all sharing one real-world dataset.

**Live in one dashboard, three tabs:**

| Tab | Project | ML Type | What it answers |
|---|---|---|---|
| 🗺️ Hotspots | Project 1 | Unsupervised (DBSCAN) | Where do pickup requests cluster geographically? |
| 📊 Demand Category | Project 2 | Classification | Will this zone be Low / Medium / High demand at this hour? |
| 📈 Demand Prediction | Project 3 | Regression | How many rides should we expect next hour, in this zone? |

---

## 📋 Table of Contents

- [The Problem](#the-problem)
- [Dataset](#dataset)
- [How It Works](#how-it-works)
- [Results](#results)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Setup & Usage](#setup--usage)
- [Known Limitations](#known-limitations-read-before-your-viva)
- [Possible Extensions](#possible-extensions)
- [Team](#team)

---

## The Problem

Ride-hailing platforms (Uber, Ola, Lyft) generate huge volumes of
location- and time-stamped ride data. Turned into intelligence, that data
can tell an operator:

1. **Which neighborhoods** generate the most pickup requests, and how that
   shifts through the day (morning commute vs. evening rush vs. night)
2. **Whether a given zone, at a given hour**, is about to be quiet or slammed
3. **Roughly how many rides** to expect an hour from now, so drivers can be
   repositioned *before* demand spikes rather than after

This repo builds all three as one working system, not three disconnected
scripts.

## Dataset

[`ncr_ride_bookings.csv`](data/ncr_ride_bookings.csv) — 150,000 ride bookings
across the Delhi National Capital Region (Delhi, Gurgaon, Noida, Faridabad,
Ghaziabad, and surrounding areas), covering January–October 2024. Each row
is one booking: pickup/drop zone names, timestamp, vehicle type, fare,
distance, driver/customer ratings, and cancellation details.

**Important:** the raw data has pickup **zone names** ("Palam Vihar," "Cyber
Hub"), not GPS coordinates, and no weather data. Both are derived — see
[Known Limitations](#known-limitations-read-before-your-viva).

## How It Works

```
Raw bookings (zone names, timestamps)
        │
        ├─► Geocode zone names → coordinates (OpenStreetMap, with an
        │   offline NCR-region fallback if that's unreachable)
        │
        ├──────────────┬──────────────────────────────┐
        │              │                              │
   ┌────▼────┐   ┌──────▼───────┐              ┌───────▼──────┐
   │ Tab 1   │   │ Fixed cluster │              │  (shared with │
   │ Tunable │   │ assignment    │─────────────►│   Tab 2 & 3)  │
   │ DBSCAN  │   │ (45 zones)    │              └───────┬───────┘
   └─────────┘   └───────────────┘                      │
                                          ┌──────────────┴──────────────┐
                                     ┌────▼────┐                  ┌─────▼────┐
                                     │  Tab 2   │                  │  Tab 3   │
                                     │ Low/Med/ │                  │ Ride-    │
                                     │ High     │                  │ count    │
                                     │ classify │                  │ forecast │
                                     └──────────┘                  └──────────┘
```

**Why two different clusterings?** Tab 1 is exploratory — sliders let you
tune the search radius live and watch hotspots merge or split. Tabs 2-3
need *stable* zone IDs to train a model against, so they share one fixed
clustering computed once and cached, separate from Tab 1's live tuning.

### Project 1 — Pickup Hotspot Detection
**Algorithm:** DBSCAN (density-based clustering), haversine distance.
Groups pickup locations into hotspots without specifying the number of
zones upfront, and separately labels isolated points as noise (`-1`).
Optimized to cluster unique locations (weighted by ride volume) rather
than every individual row — the same result, but stays fast and memory-safe
at any search radius.

### Project 2 — Demand Category Classification
**Algorithm:** Random Forest Classifier. Buckets historical demand into
Low/Medium/High by tertile, then predicts the category for any
(zone, day-of-week, hour) combination — evaluated on a held-out test set.

### Project 3 — Ride Demand Prediction
**Algorithm:** Random Forest Regressor. Predicts expected ride count using
hour, zone, day, an approximate seasonal rainfall signal, and the previous
hour's demand (a lag feature) — also evaluated on held-out data, plus a
24-hour forecast simulation and 2024 holiday/event context for the
dashboard's date picker.

## Results

*(measured on held-out test data the models never trained on — not
training-set accuracy, which would overstate performance)*

| Project | Metric | Result |
|---|---|---|
| 1 — Hotspots | Clusters found (default settings) | 162, 0% noise |
| 2 — Classification | Test accuracy | 60.1% (balanced across Low/Med/High) |
| 3 — Prediction | MAE / R² | 0.65 rides / 88.9% |

Project 3's strong R² comes mostly from the previous-hour lag feature and
the hour/zone pattern — not from rainfall (see Limitations below). Check
the dashboard's Feature Importance panel to see this directly.

## Tech Stack

- **Python** · **Pandas** / **NumPy** — data wrangling
- **Scikit-learn** — DBSCAN, Random Forest (classification & regression)
- **Geopy** — geocoding (OpenStreetMap/Nominatim)
- **Folium** — interactive map visualization
- **Streamlit** — the dashboard itself

## Project Structure

```
SmartRideCombined/
├── data/
│   ├── ncr_ride_bookings.csv                 # raw dataset
│   ├── geocoded_locations.csv                # Project 1's zone coordinates (cached)
│   └── geocoded_clusters_hourly_weather.csv  # Projects 2-3's shared dataset (cached)
├── src/
│   ├── config.py               # all settings, in one place
│   ├── geocoding.py            # zone name → coordinates (shared)
│   ├── data_loader.py          # Project 1: row-level clean + coordinates
│   ├── dbscan_clustering.py    # Project 1: tunable DBSCAN
│   ├── cluster_analysis.py     # Project 1: cluster summaries, peak period
│   ├── map_visualization.py    # Project 1: Folium map
│   ├── classification.py       # Project 2: Low/Medium/High classifier
│   ├── build_dataset.py        # Project 3: fixed clustering + rainfall proxy
│   ├── calendar_context.py     # Project 3: 2024 holidays/events
│   └── model_training.py       # Project 3: demand regression
├── app.py                      # the combined 3-tab dashboard
├── run_analysis.py             # CLI: runs all 3 projects, no UI
└── requirements.txt
```

## Setup & Usage

```bash
git clone <this-repo-url>
cd SmartRideCombined
pip install -r requirements.txt
```

**Run the dashboard:**
```bash
streamlit run app.py
```
First launch builds the shared dataset automatically (a few seconds);
cached after that.

**Run the CLI pipeline** (all 3 projects, prints results, no browser):
```bash
python run_analysis.py
```

## Known Limitations (read before your viva)

Being upfront about these is more credible than pretending they don't
exist — each is a deliberate, explained trade-off, not an oversight:

- **Geocoding fallback:** the raw data has zone *names*, not coordinates.
  Real geocoding via OpenStreetMap is attempted first; if that's
  unreachable, an offline approximation (keyword-matched to NCR
  sub-regions) is used automatically so the pipeline never blocks on
  network access.
- **Rainfall is a climatological proxy, not measured weather** — no real
  historical weather data exists for this dataset's dates. It follows
  Delhi's known monsoon pattern (heavy Jun–Sep) so the model has a
  plausible signal to learn from, but it isn't ground truth.
- **Ride counts are naturally sparse** at the (zone, hour, date) level —
  median demand per zone-hour is 1 ride. This is a real property of
  spreading 150,000 rides across 45 zones × 24 hours × ~300 days, not a
  processing error.
- **Holidays/events are reference context only** — shown in the
  dashboard's date picker, not yet used as model input features.

## Possible Extensions

The original project brief describes a 4th component not yet built here:

- **Project 4 — Traffic / Travel-Time Prediction:** using `Avg CTAT`
  (actual trip time) and `Ride Distance` to predict travel time and flag
  congestion hotspots by zone. A natural next addition using the same
  clustering foundation.

Other extensions worth considering: LSTM/GRU for genuine time-series
forecasting, real-time weather API integration, and combining Projects 1
and 3 so predicted demand is shown directly on the hotspot map.

## Team

*(add your names/roles here)*

| Name | Contribution |
|---|---|
| | Project 1 — Hotspot Detection |
| | Project 2 — Demand Classification |
| | Project 3 — Demand Prediction |
| | Dashboard Integration |

## License

*(add a license here — MIT is a common default for student/academic projects)*
