import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay


def report_classification_performance(
    model, y_pred, y_test, labels=None, show_cm=True, digits=4
):
    """
    Prints standard evaluation metrics for a classifier.

    Metrics printed:
    - Accuracy
    - Precision / Recall / F1-score per class
    - Macro / Weighted averages

    # Parameters
    * model: trained model object (kept for consistency, not used directly here).
    * y_pred: predicted labels.
    * y_test: true labels.
    * labels: optional list controlling class order in report.
    * show_cm: if True, also prints the confusion matrix.
    * digits: number of decimal places for the classification report.

    # Returns
        y_pred (for convenience in notebooks).
    """

    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("\nClassification Report:\n")
    print(classification_report(y_test, y_pred, labels=labels, digits=digits))

    return y_pred


def report_unseen_test_words(X_train, X_test, bow, top_n=30):
    """
    Checks which BoW features appear in X_test but never appear in X_train.

    Output tables:
    - summary_df: high-level statistics (counts and percentages)
    - df_unseen: top unseen words with frequency in the test set

    # Parameters
    * X_train: sparse BoW matrix for training data.
    * X_test: sparse BoW matrix for test data.
    * bow: fitted BoW object containing the vectorizer.
    * top_n: number of top unseen words to return.

    # Returns
        summary_df, df_unseen
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


def report_misleading_words_by_lift(
    y_pred,
    X_test,
    y_test,
    bow,
    label,
    lift_threshold=2.0,
    pct_wrong_threshold=0,
    top_n=30,
    min_docs=5,
):
    """
    Finds words that are unusually common in WRONG predictions for a given label.

    Wrong predictions here mean:
    - predicted == label
    - actual != label

    Computation:
    - pct_wrong = % of wrong docs containing the word
    - pct_other = % of other docs containing the word
    - lift = pct_wrong / pct_other

    # Parameters
    * y_pred: predicted labels.
    * X_test: sparse BoW test matrix.
    * y_test: true labels.
    * bow: fitted BoW object containing the vectorizer.
    * label: the predicted label we are analyzing.
    * lift_threshold: minimum lift value to keep.
    * pct_wrong_threshold: minimum pct_wrong value to keep.
    * top_n: number of top words to return.
    * min_docs: minimum number of wrong docs the word must appear in.

    # Returns
        DataFrame with columns: word, pct_wrong, lift
    """

    feature_names = bow.vectorizer.get_feature_names_out()

    # Wrongly predicted as the label
    mask_wrong = (y_pred == label) & (y_test != label).to_numpy()
    X_wrong = X_test[mask_wrong]
    n_wrong = X_wrong.shape[0]
    if n_wrong == 0:
        return pd.DataFrame(columns=["word", "pct_wrong", "lift"])

    # All other docs
    mask_other = ~mask_wrong
    X_other = X_test[mask_other]
    n_other = X_other.shape[0]

    # Document frequencies
    df_wrong = (X_wrong > 0).sum(axis=0).A1
    df_other = (X_other > 0).sum(axis=0).A1

    pct_wrong = df_wrong / n_wrong * 100
    pct_other = df_other / n_other * 100

    lift = (pct_wrong + 1e-8) / (
        pct_other + 1e-8
    )  # Add small value to avoid division by zero

    df = pd.DataFrame(
        {
            "word": feature_names,
            "pct_wrong": pct_wrong,
            "lift": lift,
            "wrong_docs": df_wrong.astype(int),
        }
    )

    # Filter for words with lift above threshold and minimum occurrence
    df = df[
        (df["lift"] >= lift_threshold)
        & (df["wrong_docs"] >= min_docs)
        & (df["pct_wrong"] >= pct_wrong_threshold)
    ]

    # Sort by lift descending
    df = df.sort_values("lift", ascending=False).head(top_n)

    df["lift"] = df["lift"].round(2)

    return df[["word", "pct_wrong", "lift"]]


def compare_word_usage_wrong_vs_right(y_pred, X_test, y_test, bow, label, top_n=20):
    """
    Returns the top words associated with wrong vs right predictions
    for a specific predicted label.

    This compares:
    - WRONG predictions: predicted == label but actual != label
    - RIGHT predictions: predicted == label and actual == label

    Percentages:
    - wrong_word_pct = % share of the word among all word occurrences in wrong docs
    - right_word_pct = % share of the word among all word occurrences in right docs

    # Parameters
    * y_pred: predicted labels.
    * X_test: sparse BoW test matrix.
    * y_test: true labels.
    * bow: fitted BoW object containing the vectorizer.
    * label: predicted label to analyze.
    * top_n: number of top words to return.

    # Returns
        DataFrame with columns:
        word | wrong_label_rank | wrong_word_pct | right_label_rank | right_word_pct
    """

    wrong_mask = (y_pred == label) & (y_test != label).to_numpy()
    right_mask = (y_pred == label) & (y_test == label).to_numpy()

    X_wrong_pred = X_test[wrong_mask]
    X_right_pred = X_test[right_mask]

    wrong_counts = X_wrong_pred.sum(axis=0).A1  # convert sparse to 1D array
    right_counts = X_right_pred.sum(axis=0).A1

    feature_names = bow.vectorizer.get_feature_names_out()

    # Totals (for %)
    wrong_total = wrong_counts.sum()
    right_total = right_counts.sum()

    # Avoid division by zero
    wrong_pct = (
        wrong_counts / wrong_total if wrong_total > 0 else np.zeros_like(wrong_counts)
    )
    right_pct = (
        right_counts / right_total if right_total > 0 else np.zeros_like(right_counts)
    )

    df = pd.DataFrame(
        {
            "word": feature_names,
            "wrong_count": wrong_counts,
            "wrong_word_pct": wrong_pct,
            "right_count": right_counts,
            "right_word_pct": right_pct,
        }
    )

    df["wrong_label_rank"] = (
        df["wrong_count"].rank(method="dense", ascending=False).astype(int)
    )
    df["right_label_rank"] = (
        df["right_count"].rank(method="dense", ascending=False).astype(int)
    )

    df = df.sort_values("wrong_count", ascending=False)

    # Final output columns (exactly what you asked)
    df_out = df[
        [
            "word",
            "wrong_label_rank",
            "wrong_word_pct",
            "right_label_rank",
            "right_word_pct",
        ]
    ].head(top_n)

    # Make % readable (optional)
    df_out["wrong_word_pct"] = (df_out["wrong_word_pct"] * 100).round(3)
    df_out["right_word_pct"] = (df_out["right_word_pct"] * 100).round(3)

    return df_out.head(top_n)

def comparative_confusion_matrix(cm1, cm2):
    """
    Returns a comparative figure of confusion matrix 1 and confusion matrix 2 
    with the differences between the 2

    # Parameters 
    cm1 - confusion matrix 1 or the initial/old confusion matrix
    cm2 - confusion matrix 2 or the new confusion matrix to be compared against cm1
    """
    cm_diff = cm2 - cm1 
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # confusion matrix - before
    disp1 = ConfusionMatrixDisplay(cm1, display_labels=[-1, 0, 1])
    disp1.plot(ax=axes[0], cmap='Blues', colorbar=False)
    axes[0].set_title('Before')
    
    # confusion matrix - after
    disp2 = ConfusionMatrixDisplay(cm2, display_labels=[-1, 0, 1])
    disp2.plot(ax=axes[1], cmap='Blues', colorbar=False)
    axes[1].set_title('After')
    
    # difference matrix
    im = axes[2].imshow(cm_diff, cmap='RdYlGn')
    
    for i in range(cm_diff.shape[0]):
        for j in range(cm_diff.shape[1]):
            value = cm_diff[i, j]
            label = f'+{value}' if value > 0 else str(value)
            color = 'black'
            axes[2].text(j, i, label, ha='center', va='center', color=color)
    
    axes[2].set_xticks([0, 1, 2])
    axes[2].set_yticks([0, 1, 2])
    axes[2].set_xticklabels([-1, 0, 1])
    axes[2].set_yticklabels([-1, 0, 1])
    axes[2].set_xlabel('Predicted Label')
    axes[2].set_ylabel('True Label')
    axes[2].set_title('Difference')
    
    display(fig)
    plt.close(fig)

    # summarizing changes
    print("Summary of Changes:")
    print(f"Before - Total True Positive: {np.trace(cm1)} / {cm1.sum()}")
    print(f"After  - Total True Positive: {np.trace(cm2)} / {cm2.sum()}")
    print(f"Improvement: {np.trace(cm2) - np.trace(cm1):+d} correct predictions")
    
    labels = [-1, 0, 1]
    print("\nPer-class diagonal change (correct predictions):")
    for i, label in enumerate(labels):
        change = cm_diff[i, i]
        print(f"  Class {label:2d}: {change:+d}")

def incorrect_confidence_score(model, X, y_true, y_pred):
    y_prob = model.predict_proba(X)
    confidence = y_prob.max(axis=1)
    
    incorrect_mask = y_pred != y_true.values
    incorrect_confidence = confidence[incorrect_mask]
    
    plt.figure(figsize=(10, 6))
    plt.hist(incorrect_confidence, bins=50, alpha=0.7, edgecolor='black')
    plt.xlabel('Confidence Score')
    plt.ylabel('Number of Samples')
    plt.title('Distribution of Incorrect Predictions and Level of confidence')
    plt.tight_layout()
    plt.show()
    
    print(f"Total incorrect predictions: {incorrect_mask.sum()}/{len(y_pred)}")
    print(f"Mean (incorrect) confidence score: {(incorrect_confidence).mean():.4f}")