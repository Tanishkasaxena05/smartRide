
import os

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.metrics import (
    ConfusionMatrixDisplay,
    classification_report,
    confusion_matrix,
)

OUTPUT_DIR = "outputs"

CLASS_ORDER = ["Low", "Medium", "High"]


def _ordered_classes(label_encoder):
    present = list(label_encoder.classes_)
    return [c for c in CLASS_ORDER if c in present] + [
        c for c in present if c not in CLASS_ORDER
    ]


def save_confusion_matrix(y_test, predictions, label_encoder):
    labels = _ordered_classes(label_encoder)
    matrix = confusion_matrix(y_test, predictions, labels=labels)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    ConfusionMatrixDisplay(matrix, display_labels=labels).plot(
        ax=axes[0], cmap="Blues", colorbar=False, values_format="d"
    )
    axes[0].set_title("Confusion matrix (counts)")

    normalised = matrix.astype(float) / np.maximum(
        matrix.sum(axis=1, keepdims=True), 1
    )
    ConfusionMatrixDisplay(normalised, display_labels=labels).plot(
        ax=axes[1], cmap="Blues", colorbar=False, values_format=".2f"
    )
    axes[1].set_title("Confusion matrix (row-normalised recall)")

    fig.tight_layout()
    path = os.path.join(OUTPUT_DIR, "confusion_matrix.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)

    pd.DataFrame(
        matrix,
        index=[f"true {c}" for c in labels],
        columns=[f"pred {c}" for c in labels],
    ).to_csv(os.path.join(OUTPUT_DIR, "confusion_matrix.csv"))

    print(f"[eval] {path}")


def save_classification_report(y_test, predictions, model_name):
    text = classification_report(y_test, predictions, zero_division=0)
    path = os.path.join(OUTPUT_DIR, "classification_report.txt")

    with open(path, "w") as handle:
        handle.write(f"Selected model: {model_name}\n\n")
        handle.write(text)

    report_df = pd.DataFrame(
        classification_report(
            y_test, predictions, zero_division=0, output_dict=True
        )
    ).transpose()
    report_df.to_csv(os.path.join(OUTPUT_DIR, "classification_report.csv"))

    print(f"[eval] {path}")


def save_feature_importance(model, features, top_n=20):
    estimator = model.named_steps.get("model", model)

    if hasattr(estimator, "feature_importances_"):
        importance = np.asarray(estimator.feature_importances_)
        label = "Feature importance"
    elif hasattr(estimator, "coef_"):
        importance = np.abs(np.asarray(estimator.coef_)).mean(axis=0)
        label = "Mean |coefficient|"
    else:
        print("[eval] model exposes no importances -- skipping.")
        return

    frame = (
        pd.DataFrame({"feature": features, "importance": importance})
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )

    frame.to_csv(os.path.join(OUTPUT_DIR, "feature_importance.csv"), index=False)

    top = frame.head(top_n).iloc[::-1]

    fig, ax = plt.subplots(figsize=(9, max(5, 0.32 * len(top))))
    ax.barh(top["feature"], top["importance"], color="#2E86C1")
    ax.set_xlabel(label)
    ax.set_title(f"Top {len(top)} features")
    fig.tight_layout()

    path = os.path.join(OUTPUT_DIR, "feature_importance.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)

    print(f"[eval] {path}")
    print("\n[eval] top 10 features:")
    print(frame.head(10).to_string(index=False))


def save_model_comparison(comparison):
    path = os.path.join(OUTPUT_DIR, "model_comparison.csv")
    comparison.to_csv(path, index=False)

    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(comparison))
    width = 0.38

    ax.bar(x - width / 2, comparison["Accuracy"], width, label="Accuracy")
    ax.bar(x + width / 2, comparison["Macro F1"], width, label="Macro F1")
    ax.set_xticks(x)
    ax.set_xticklabels(comparison["Model"], rotation=20, ha="right")
    ax.set_ylim(0, 1)
    ax.set_ylabel("Score")
    ax.set_title("Model comparison on the chronological hold-out")
    ax.legend()
    fig.tight_layout()

    fig.savefig(os.path.join(OUTPUT_DIR, "model_comparison.png"), dpi=150)
    plt.close(fig)

    print(f"[eval] {path}")


def save_class_balance(dataset):
    counts = dataset["Demand_Category"].value_counts()
    labels = [c for c in CLASS_ORDER if c in counts.index]

    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.bar(labels, [counts[c] for c in labels], color="#16A085")
    ax.set_ylabel("Cells")
    ax.set_title("Demand category distribution")
    fig.tight_layout()

    fig.savefig(os.path.join(OUTPUT_DIR, "class_balance.png"), dpi=150)
    plt.close(fig)


def run_evaluation(artifacts, dataset, features):
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("\n[eval] writing artifacts to outputs/ ...")

    save_confusion_matrix(
        artifacts["y_test"], artifacts["predictions"], artifacts["label_encoder"]
    )
    save_classification_report(
        artifacts["y_test"], artifacts["predictions"], artifacts["model_name"]
    )
    save_model_comparison(artifacts["comparison"])
    save_class_balance(dataset)

    import joblib

    bundle = joblib.load("models/demand_classifier.pkl")
    save_feature_importance(bundle["model"], features)
