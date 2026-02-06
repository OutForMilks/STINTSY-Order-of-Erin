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


def check_feature_sparsity(X_train, X_test, bow, top_n=30):
    """
    Checks which BoW features/words appear in X_test but never in X_train.
    Returns a readable summary and a top table of unseen words in test.
    """

    # Print like this
    # for _, row in summary_df.iterrows():
    #     print(f"{row['Metric']}: {row['Value']}")

    # print(unseen_df)

    feature_names = bow.vectorizer.get_feature_names_out()

    # Columns present at least once
    train_present = X_train.sum(axis=0).A1 > 0
    test_present = X_test.sum(axis=0).A1 > 0

    # Features that exist in test but never in train
    unseen_in_train = test_present & (~train_present)
    unseen_indices = np.where(unseen_in_train)[0]

    # Count of those features in test
    unseen_counts = X_test[:, unseen_indices].sum(axis=0).A1

    # How many test rows contain at least one unseen feature
    test_rows_with_unseen = X_test[:, unseen_indices].sum(axis=1).A1 > 0

    # Table of top unseen words
    if len(unseen_indices) > 0:
        df_unseen = (
            pd.DataFrame(
                {
                    "Word": feature_names[unseen_indices],
                    "Count in Test": unseen_counts.astype(int),
                }
            )
            .sort_values("Count in Test", ascending=False)
            .head(top_n)
        )
    else:
        df_unseen = pd.DataFrame(columns=["Word", "Count in Test"])

    # Human-readable summary
    summary_df = pd.DataFrame(
        {
            "Metric": [
                "Total Features",
                "Unseen Features in Test",
                "Percentage of Unseen Features",
                "Total Test Docs",
                "Test Docs with Unseen Features",
                "Percentage of Test Docs with Unseen Features",
            ],
            "Value": [
                X_train.shape[1],
                len(unseen_indices),
                f"{len(unseen_indices)/X_train.shape[1]*100:.2f}%",
                X_test.shape[0],
                int(test_rows_with_unseen.sum()),
                f"{test_rows_with_unseen.mean()*100:.2f}%",
            ],
        }
    )

    return summary_df, df_unseen
