from src.preprocessing import (
    load_data,
    clean_data
)

from src.feature_engineering import (
    create_demand_dataset
)

from src.model_training import (
    train_model
)


rides, geo = load_data()

rides = clean_data(rides)

dataset = create_demand_dataset(
    rides,
    geo
)
print(dataset["Demand_Category"].value_counts())
print(dataset["ride_count"].describe())
dataset.to_csv(
    "data/processed/demand_dataset.csv",
    index=False
)

train_model(dataset)

print("Pipeline completed")