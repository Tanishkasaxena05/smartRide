# ============================================================
# PROJECT 3 — MAIN PIPELINE
# ============================================================

import os
import pandas as pd

from src.config import DATASET_FILE, OUTPUT_DIR
from src.build_dataset import build as build_dataset, aggregate_to_demand
from src.model_training import train_and_compare


def main():
    print("=" * 60)
    print("🚖 SMARTRIDE — RIDE DEMAND PREDICTION")
    print("=" * 60)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    if not os.path.exists(DATASET_FILE):
        print("\nBuilding dataset (geocode -> cluster -> rainfall)...")
        build_dataset()
    else:
        print(f"\nUsing existing dataset: {DATASET_FILE}")

    raw = pd.read_csv(DATASET_FILE)
    df = aggregate_to_demand(raw)

    print("\nTraining and comparing all algorithms...")
    fitted, comparison, splits = train_and_compare(df)

    print("\n" + "=" * 60)
    print("MODEL COMPARISON (held-out test set)")
    print("=" * 60)
    print(comparison.to_string(index=False))

    comparison.to_csv(f"{OUTPUT_DIR}/model_comparison.csv", index=False)
    print(f"\nSaved -> {OUTPUT_DIR}/model_comparison.csv")

    best = comparison.iloc[0]["Model"]
    print(f"\nBest model by MAE: {best}")

    # Example prediction matching the brief's demo
    example = pd.DataFrame([{
        "hour": 18, "cluster": int(df["cluster"].mode()[0]),
        "day": 15, "rainfall_mm": 50.0, "previous_demand": 5,
    }])
    pred = fitted[best].predict(example)[0]
    print(f"\nExample prediction ({best}): zone {example['cluster'].iloc[0]}, "
          f"6 PM, rain -> {max(0, round(pred))} rides")

    print("\n" + "=" * 60)
    print("✅ ANALYSIS COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
