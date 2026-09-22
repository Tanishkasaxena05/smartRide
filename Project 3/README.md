# 🚖 Project 3 — Ride-Hailing Demand Prediction

## Problem Statement

Predict the number of ride requests expected in a particular geographical
zone during a future hour, using supervised regression.

## What this dataset actually has (and doesn't)

`data/ncr_ride_bookings.csv` records individual bookings with a pickup
**zone name** (e.g. "Palam Vihar") and a timestamp — it has no
coordinates and no weather data. To build the feature set the brief
asks for (zone, hour, day, rain, previous-hour demand), this project:

1. **Geocodes** each zone name into approximate coordinates (falls back
   to an offline NCR-region approximation if live geocoding is
   unavailable — see Project 1's notes on this; same approach here).
2. **Clusters** the ~176 zones into ~45 broader geographic areas with
   DBSCAN, so the model predicts per-area demand rather than
   per-individual-street-name demand.
3. **Adds an approximate seasonal rainfall feature** — Delhi NCR's
   monsoon pattern (heavy Jun-Sep, dry otherwise). This is **not**
   measured historical weather; no such data exists for this dataset's
   dates. It's a climatological proxy so the model has a genuine
   "does rain suppress demand" signal to learn from.
4. **Adds 2024 holiday and event context** (Republic Day, Holi, Diwali,
   the India International Trade Fair, etc. — see
   `src/calendar_context.py`) so the dashboard's date picker shows what
   was happening on that day. This is reference data, not model input —
   holidays/events aren't yet used as training features, only shown for
   business context in the dashboard.

This pipeline (`src/build_dataset.py`) saves ride-level rows (one per
booking, tagged with cluster/rainfall/day-name) to
`data/geocoded_clusters_hourly_weather.csv`. Both `app.py` and
`run_analysis.py` then aggregate this to (date, hour, cluster) →
demand via the shared `aggregate_to_demand()` function, so the two
entry points can never drift out of sync with each other.

## Algorithms (all 5 from the brief, honestly evaluated)

`src/model_training.py` trains and compares:
- Linear Regression
- Decision Tree Regressor
- Random Forest Regressor
- Gradient Boosting Regressor
- XGBoost

Every model is evaluated on a **held-out test set** it never trained
on — not scored on its own training data, which would make every
model look artificially better than it really is.

**Actual results on this dataset:**

| Model | MAE | RMSE | R² |
|---|---|---|---|
| XGBoost | 0.60 | 1.47 | 90.4% |
| Gradient Boosting | 0.61 | 1.45 | 90.7% |
| Decision Tree | 0.62 | 1.55 | 89.4% |
| Random Forest | 0.65 | 1.59 | 88.9% |
| Linear Regression | 0.99 | 2.19 | 78.7% |

Tree-based ensemble methods clearly outperform plain Linear Regression
— expected, since ride demand depends on non-linear interactions
between hour, cluster, and rain, not a straight-line relationship.

## Setup

```
pip install -r requirements.txt
```

## Run the CLI pipeline

```
python run_analysis.py
```

Builds the dataset (first run only), trains all 5 models, prints and
saves the comparison table to `outputs/model_comparison.csv`.

## Run the dashboard

```
streamlit run app.py
```

- Pick an algorithm, zone (cluster), scenario date (auto-fills day,
  day name, holiday, event, and that date's rainfall), hour, and
  previous demand
- See the predicted ride count, calendar context (day/holiday/event),
  a 24-hour forecast curve, feature importance (for tree-based
  models), and the full algorithm comparison table

## Honest limitations (worth stating in your report)

- **Rainfall is a proxy, not real weather** — say so explicitly if
  asked in a viva.
- **Ride counts are small** (median demand = 1, max = 38 per
  cluster-hour) because 150,000 rides spread across 45 clusters × 24
  hours × ~300 days is naturally sparse. The model's strong R² (90%+)
  comes mostly from the `previous_demand` lag feature and the
  hour/cluster pattern, not from rainfall — check feature importance
  in the dashboard to see this directly.
- **Clusters, not raw zone names** — Project 3 predicts demand per
  geographic *cluster* (from Project 1's DBSCAN), not per individual
  named zone. This is deliberate: predicting at the named-zone level
  would be even sparser and less learnable.
