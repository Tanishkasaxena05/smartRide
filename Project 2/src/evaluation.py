from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score
)


def evaluate_model(
    y_test,
    predictions
):

    print(
        "\nAccuracy:"
    )

    print(
        accuracy_score(
            y_test,
            predictions
        )
    )

    print(
        "\nMacro F1:"
    )

    print(
        f1_score(
            y_test,
            predictions,
            average="macro"
        )
    )

    print(
        "\nClassification Report:"
    )

    print(
        classification_report(
            y_test,
            predictions,
            zero_division=0
        )
    )

    print(
        "\nConfusion Matrix:"
    )

    print(
        confusion_matrix(
            y_test,
            predictions
        )
    )