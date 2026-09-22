"""
SmartRide NCR -- end-to-end pipeline.

Raw CSVs  ->  cleaning  ->  feature engineering  ->  model training
           ->  evaluation artifacts  ->  models/demand_classifier.pkl

Run this once before `streamlit run app.py`. It writes:
  - data/processed/demand_dataset.csv   (what app.py loads for the dashboard)
  - models/demand_classifier.pkl        (what predictor.py loads for scoring)
  - outputs/*.png / *.csv               (evaluation artifacts, incl. the
                                          7-model comparison chart)
"""

import os

from src.preprocessing import load_data, clean_data, check_weather_overlap
from src.feature_engineering import create_demand_dataset
from src.model_training import train_model
from src.evaluation import run_evaluation

PROCESSED_PATH = "data/processed/demand_dataset.csv"


def main():
    print("=" * 70)
    print("STEP 1/5 -- Loading raw data")
    print("=" * 70)
    rides, geo, weather = load_data()

    print("\n" + "=" * 70)
    print("STEP 2/5 -- Cleaning rides")
    print("=" * 70)
    rides = clean_data(rides)
    check_weather_overlap(rides, weather)

    print("\n" + "=" * 70)
    print("STEP 3/5 -- Feature engineering")
    print("=" * 70)
    demand, metadata = create_demand_dataset(rides, geo, weather)

    os.makedirs("data/processed", exist_ok=True)
    demand.to_csv(PROCESSED_PATH, index=False)
    print(f"\n[pipeline] saved processed dataset -> {PROCESSED_PATH}")

    print("\n" + "=" * 70)
    print("STEP 4/5 -- Model training (comparing all candidates)")
    print("=" * 70)
    best_model, comparison, artifacts = train_model(demand, metadata)

    print("\n" + "=" * 70)
    print("STEP 5/5 -- Evaluation artifacts")
    print("=" * 70)
    run_evaluation(artifacts, demand, metadata["features"])

    print("\n" + "=" * 70)
    print("DONE. Run `streamlit run app.py` to launch the dashboard.")
    print("=" * 70)


if __name__ == "__main__":
    main()
