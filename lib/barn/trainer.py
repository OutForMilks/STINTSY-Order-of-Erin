from __future__ import annotations

from typing import Any, Type
import torch
import torch.nn as nn
import pandas as pd
from pandas import Series
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix
import numpy as np
from numpy import ndarray
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from dataclasses import dataclass, field


@dataclass
class FoldMetrics:
    accuracy: float = 0.0
    f1_macro: float = 0.0
    f1_weighted: float = 0.0
    precision_macro: float = 0.0
    recall_macro: float = 0.0
    confusion_matrix: ndarray = field(default_factory=lambda: np.array([]))


@dataclass
class CVMetrics:
    train: list[FoldMetrics] = field(default_factory=list)
    val: list[FoldMetrics] = field(default_factory=list)

    def summary(self) -> pd.DataFrame:
        """Return a DataFrame summarising mean ± std across folds for val metrics."""
        rows = []
        for split_name, folds in [("train", self.train), ("val", self.val)]:
            for metric in ["accuracy", "f1_macro", "f1_weighted", "precision_macro", "recall_macro"]:
                values = [getattr(f, metric) for f in folds]
                rows.append({
                    "split": split_name,
                    "metric": metric,
                    "mean": np.mean(values),
                    "std": np.std(values),
                    "min": np.min(values),
                    "max": np.max(values),
                })
        return pd.DataFrame(rows).set_index(["split", "metric"])


def _compute_metrics(all_labels: list[int], all_preds: list[int]) -> FoldMetrics:
    """Compute all classification metrics from collected labels and predictions."""
    labels_arr = np.array(all_labels)
    preds_arr = np.array(all_preds)
    accuracy = (preds_arr == labels_arr).mean().item()
    return FoldMetrics(
        accuracy=accuracy,
        f1_macro=f1_score(labels_arr, preds_arr, average="macro", zero_division=0),
        f1_weighted=f1_score(labels_arr, preds_arr, average="weighted", zero_division=0),
        precision_macro=precision_score(labels_arr, preds_arr, average="macro", zero_division=0),
        recall_macro=recall_score(labels_arr, preds_arr, average="macro", zero_division=0),
        confusion_matrix=confusion_matrix(labels_arr, preds_arr),
    )


def train(
    model: nn.Module,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    train_loader: DataLoader[Any],
    epoch: int,
    device: torch.device,
) -> FoldMetrics:
    """
    Train a model for a specified number of epochs.

    # Parameters
    * model: PyTorch model to train.
    * criterion: Loss function.
    * optimizer: Optimizer instance.
    * train_loader: DataLoader for training data.
    * epoch: Number of epochs to train.
    * device: Device to run training on (e.g., 'cuda', 'cpu').

    # Returns
    FoldMetrics computed on the last epoch's predictions.
    """
    model.train()
    all_labels: list[int] = []
    all_preds: list[int] = []

    for _ in range(epoch):
        all_labels = []
        all_preds = []
        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            _, predicted = torch.max(outputs.data, 1)
            all_labels.extend(labels.cpu().tolist())
            all_preds.extend(predicted.cpu().tolist())

    return _compute_metrics(all_labels, all_preds)


def validate(
    model: nn.Module,
    val_loader: DataLoader[Any],
    device: torch.device,
) -> FoldMetrics:
    """
    Evaluate a model on validation data.

    # Parameters
    * model: PyTorch model to evaluate.
    * val_loader: DataLoader for validation data.
    * device: Device to run evaluation on (e.g., 'cuda', 'cpu').

    # Returns
    FoldMetrics for the validation set.
    """
    model.eval()
    all_labels: list[int] = []
    all_preds: list[int] = []

    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            all_labels.extend(labels.cpu().tolist())
            all_preds.extend(predicted.cpu().tolist())

    return _compute_metrics(all_labels, all_preds)
