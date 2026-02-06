import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


def evaluate_classifier(model, y_pred, y_test, labels=None, show_cm=True, digits=4):
    """
    Better evaluation wrapper for classifiers.
    """

    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("\nClassification Report:\n")
    print(classification_report(y_test, y_pred, labels=labels, digits=digits))

    if show_cm:
        cm = confusion_matrix(y_test, y_pred, labels=labels)

        cm_df = pd.DataFrame(
            cm,
            index=[f"True_{l}" for l in (labels if labels else sorted(set(y_test)))],
            columns=[f"Pred_{l}" for l in (labels if labels else sorted(set(y_test)))],
        )

        print("\nConfusion Matrix:\n")
        print(cm_df)

    return y_pred
