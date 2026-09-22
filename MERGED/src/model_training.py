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



import os
import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

from src.feature_engineering import MODEL_FEATURES

MODEL_PATH = "models/demand_classifier.pkl"


def build_candidates():
    candidates = {
        "Logistic Regression": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
                ("scaler", StandardScaler()),
                (
                    "model",
                    LogisticRegression(
                        max_iter=2000,
                        class_weight="balanced",
                    ),
                ),
            ]
        ),
        "Decision Tree": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
                (
                    "model",
                    DecisionTreeClassifier(
                        max_depth=12,
                        min_samples_leaf=20,
                        class_weight="balanced",
                        random_state=42,
                    ),
                ),
            ]
        ),
        "Random Forest": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
                (
                    "model",
                    RandomForestClassifier(
                        n_estimators=400,
                        max_depth=18,
                        min_samples_leaf=5,
                        class_weight="balanced",
                        random_state=42,
                        n_jobs=-1,
                    ),
                ),
            ]
        ),
        "Gradient Boosting": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
                (
                    "model",
                    GradientBoostingClassifier(
                        n_estimators=200,
                        max_depth=4,
                        learning_rate=0.1,
                        random_state=42,
                    ),
                ),
            ]
        ),
    }

    try:
        from xgboost import XGBClassifier

        candidates["XGBoost"] = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
                (
                    "model",
                    XGBClassifier(
                        n_estimators=300,
                        max_depth=6,
                        learning_rate=0.1,
                        subsample=0.9,
                        colsample_bytree=0.9,
                        objective="multi:softprob",
                        eval_metric="mlogloss",
                        random_state=42,
                        n_jobs=-1,
                    ),
                ),
            ]
        )
    except ImportError:
        print("[train] xgboost not installed -- skipping that candidate.")

    return candidates


def chronological_split(data, train_fraction=0.80):
    unique_dates = (
        data["Date"].dt.normalize().dropna().drop_duplicates().sort_values().tolist()
    )

    if len(unique_dates) < 2:
        raise ValueError("Not enough unique dates for a chronological split.")

    split_index = max(1, int(len(unique_dates) * train_fraction))
    split_index = min(split_index, len(unique_dates) - 1)
    split_date = unique_dates[split_index]

    train_mask = data["Date"].dt.normalize() < split_date

    print(
        f"\n[split] train < {pd.Timestamp(split_date).date()} "
        f"({train_mask.sum():,} rows) | test >= same date "
        f"({(~train_mask).sum():,} rows)"
    )

    return train_mask, split_date


def train_model(df, metadata, train_fraction=0.80):
    data = df.copy()
    data["Date"] = pd.to_datetime(data["Date"], errors="coerce")
    data = data.dropna(subset=["Date", "Demand_Category"])

    missing = [c for c in MODEL_FEATURES if c not in data.columns]
    if missing:
        raise KeyError(f"Dataset is missing model features: {missing}")

    X = data[MODEL_FEATURES].copy()
    y = data["Demand_Category"].copy()

    train_mask, split_date = chronological_split(data, train_fraction)

    X_train, X_test = X.loc[train_mask], X.loc[~train_mask]
    y_train, y_test = y.loc[train_mask], y.loc[~train_mask]

    if y_train.nunique() < 2:
        raise ValueError("Training window contains fewer than two classes.")
    if y_test.empty:
        raise ValueError("Test window is empty after the chronological split.")

    label_encoder = LabelEncoder().fit(y)
    y_train_enc = label_encoder.transform(y_train)
    y_test_enc = label_encoder.transform(y_test)

    # A column that is entirely NaN would be dropped by SimpleImputer without
    # warning -- that is how seven weather features silently vanished before.
    all_null = [c for c in MODEL_FEATURES if X_train[c].isna().all()]
    if all_null:
        print(f"[train] WARNING: these features are entirely null: {all_null}")

    results = []
    trained = {}

    for name, pipeline in build_candidates().items():
        print(f"\n[train] fitting {name} ...")
        pipeline.fit(X_train, y_train_enc)

        predictions_enc = pipeline.predict(X_test)
        predictions = label_encoder.inverse_transform(predictions_enc)

        accuracy = accuracy_score(y_test, predictions)
        macro_f1 = f1_score(y_test, predictions, average="macro", zero_division=0)
        per_class = f1_score(
            y_test,
            predictions,
            average=None,
            labels=label_encoder.classes_,
            zero_division=0,
        )

        row = {"Model": name, "Accuracy": accuracy, "Macro F1": macro_f1}
        for cls, score in zip(label_encoder.classes_, per_class):
            row[f"F1 ({cls})"] = score

        results.append(row)
        trained[name] = pipeline

        print(f"[train] {name}: accuracy {accuracy:.4f} | macro F1 {macro_f1:.4f}")

    comparison = (
        pd.DataFrame(results)
        .sort_values("Macro F1", ascending=False)
        .reset_index(drop=True)
    )

    print("\n[train] model comparison:")
    print(comparison.round(4).to_string(index=False))

    best_name = comparison.loc[0, "Model"]
    best_model = trained[best_name]

    print(f"\n[train] selected: {best_name}")

    best_predictions = label_encoder.inverse_transform(best_model.predict(X_test))

    print("\n[train] classification report:")
    print(classification_report(y_test, best_predictions, zero_division=0))

    print("[train] confusion matrix (rows = true, cols = predicted):")
    print(
        pd.DataFrame(
            confusion_matrix(
                y_test, best_predictions, labels=label_encoder.classes_
            ),
            index=[f"true {c}" for c in label_encoder.classes_],
            columns=[f"pred {c}" for c in label_encoder.classes_],
        ).to_string()
    )

    os.makedirs("models", exist_ok=True)

    bundle = {
        "model": best_model,
        "model_name": best_name,
        "label_encoder": label_encoder,
        "features": MODEL_FEATURES,
        "classes": list(label_encoder.classes_),
        "comparison": comparison,
        "split_date": pd.Timestamp(split_date),
        # Everything the dashboard needs to rebuild identical inputs.
        "n_zones": metadata["n_zones"],
        "slot_hours": metadata["slot_hours"],
        "slots_per_day": metadata["slots_per_day"],
        "low_threshold": metadata["low_threshold"],
        "high_threshold": metadata["high_threshold"],
        "zone_lookup": metadata["zone_lookup"],
        "zone_info": metadata["zone_info"],
        "kmeans": metadata["kmeans"],
    }

    joblib.dump(bundle, MODEL_PATH)
    print(f"\n[train] saved bundle -> {MODEL_PATH}")

    return best_model, comparison, {
        "X_test": X_test,
        "y_test": y_test,
        "predictions": best_predictions,
        "label_encoder": label_encoder,
        "model_name": best_name,
        "comparison": comparison,
    }
