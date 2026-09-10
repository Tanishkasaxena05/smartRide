import os
import joblib

from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import train_test_split

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score
)


def train_model(df):

    features = [
        "hour",
        "weekday_encoded",
        "previous_demand",
        "lat",
        "lon"
    ]

    X = df[features]

    y = df["Demand_Category"]

    X_train, X_test, y_train, y_test = (
        train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=42,
            stratify=y
        )
    )

    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=12,
        random_state=42
    )

    model.fit(
        X_train,
        y_train
    )

    preds = model.predict(X_test)

    print("\nAccuracy:\n")
    print(
        accuracy_score(
            y_test,
            preds
        )
    )

    print("\nMacro F1:\n")
    print(
        f1_score(
            y_test,
            preds,
            average="macro"
        )
    )

    print("\nClassification Report:\n")
    print(
        classification_report(
            y_test,
            preds,
            zero_division=0
        )
    )

    print("\nConfusion Matrix:\n")
    print(
        confusion_matrix(
            y_test,
            preds
        )
    )

    os.makedirs(
        "models",
        exist_ok=True
    )

    joblib.dump(
        model,
        "models/demand_classifier.pkl"
    )

    print(
        "\nModel saved successfully."
    )

    return model