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

    return y_pred
