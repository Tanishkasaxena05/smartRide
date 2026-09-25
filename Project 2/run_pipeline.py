
import os

from src.preprocessing import load_data, clean_data, check_weather_overlap
from src.feature_engineering import create_demand_dataset, MODEL_FEATURES
from src.model_training import train_model
from src.evaluation import run_evaluation

# Grain configuration.
# 150k rides / (366 days x N_ZONES x slots) decides how much signal each
# cell carries. At 10 zones and 4-hour slots the average cell holds ~7
# rides. Raising N_ZONES or lowering SLOT_HOURS thins the cells out again.
N_ZONES = 10
SLOT_HOURS = 4
TRAIN_FRACTION = 0.80

PROCESSED_PATH = "data/processed/demand_dataset.csv"


def main():
    os.makedirs("data/processed", exist_ok=True)
    os.makedirs("models", exist_ok=True)
    os.makedirs("outputs", exist_ok=True)

    print("=" * 70)
    print("STEP 1 — LOAD")
    print("=" * 70)
    rides, geo, weather = load_data()

    print("\n" + "=" * 70)
    print("STEP 2 — CLEAN")
    print("=" * 70)
    rides = clean_data(rides)
    check_weather_overlap(rides, weather)

    print("\n" + "=" * 70)
    print("STEP 3 — FEATURE ENGINEERING")
    print("=" * 70)
    dataset, metadata = create_demand_dataset(
        rides,
        geo,
        weather,
        n_zones=N_ZONES,
        slot_hours=SLOT_HOURS,
        train_fraction=TRAIN_FRACTION,
    )

    dataset.to_csv(PROCESSED_PATH, index=False)
    print(f"\n[io] wrote {len(dataset):,} rows -> {PROCESSED_PATH}")

    print("\n" + "=" * 70)
    print("STEP 4 — TRAIN & COMPARE")
    print("=" * 70)
    model, comparison, artifacts = train_model(
        dataset, metadata, train_fraction=TRAIN_FRACTION
    )

    print("\n" + "=" * 70)
    print("STEP 5 — EVALUATE")
    print("=" * 70)
    run_evaluation(artifacts, dataset, MODEL_FEATURES)

    print("\n" + "=" * 70)
    print("PIPELINE COMPLETE")
    print("=" * 70)
    print("Next: streamlit run app.py")


if __name__ == "__main__":
    main()
