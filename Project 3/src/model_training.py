# ============================================================
# MODEL TRAINING & COMPARISON
# ============================================================
# Trains every algorithm listed in the Project 3 brief, evaluated
# on a proper held-out test set (not the training data itself —
# scoring a model on the same rows it learned from overstates how
# good it actually is).

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.config import RANDOM_STATE, TEST_SIZE

try:
    from xgboost import XGBRegressor
    _HAS_XGBOOST = True
except ImportError:
    _HAS_XGBOOST = False

FEATURE_COLUMNS = ["hour", "cluster", "day", "rainfall_mm", "previous_demand"]
TARGET_COLUMN = "demand"


def prepare_features(df):
    """Adds the previous_demand lag feature and drops the resulting
    first row of each cluster's time series (which has no prior value)."""
    df = df.sort_values(["cluster", "date", "hour"]).copy()
    df["previous_demand"] = df.groupby("cluster")["demand"].shift(1)
    df = df.dropna(subset=["previous_demand"])
    return df


def get_model_registry():
    """The algorithms named in the Project 3 brief."""
    models = {
        "Linear Regression": LinearRegression(),
        "Decision Tree": DecisionTreeRegressor(max_depth=12, random_state=RANDOM_STATE),
        "Random Forest": RandomForestRegressor(n_estimators=200, random_state=RANDOM_STATE, n_jobs=-1),
        "Gradient Boosting": GradientBoostingRegressor(random_state=RANDOM_STATE),
    }
    if _HAS_XGBOOST:
        models["XGBoost"] = XGBRegressor(random_state=RANDOM_STATE, n_jobs=-1, verbosity=0)
    return models


def train_and_compare(df):
    """Trains every model on the same train/test split and returns
    (fitted_models, comparison_dataframe)."""
    df = prepare_features(df)

    X = df[FEATURE_COLUMNS]
    y = df[TARGET_COLUMN]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    models = get_model_registry()
    fitted = {}
    rows = []

    for name, model in models.items():
        model.fit(X_train, y_train)
        pred = model.predict(X_test)

        mae = mean_absolute_error(y_test, pred)
        rmse = mean_squared_error(y_test, pred) ** 0.5
        r2 = r2_score(y_test, pred)

        fitted[name] = model
        rows.append({"Model": name, "MAE": mae, "RMSE": rmse, "R2": r2})

    comparison = pd.DataFrame(rows).sort_values("MAE").reset_index(drop=True)
    return fitted, comparison, (X_train, X_test, y_train, y_test)


def best_model_name(comparison_df):
    return comparison_df.sort_values("MAE").iloc[0]["Model"]
