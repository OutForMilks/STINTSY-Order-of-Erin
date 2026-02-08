from __future__ import annotations

from typing import Any, Type
import torch
import torch.nn as nn
import pandas as pd
from pandas import Series
from sklearn.model_selection import StratifiedKFold
import numpy as np
from numpy import ndarray
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader


def train(
    model: nn.Module,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    train_loader: DataLoader[Any],
    epoch: int,
    device: torch.device,
) -> float:
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
    Training accuracy as a float.
    """
    model.train()
    total_correct: int = 0
    total_samples: int = 0
    
    for _ in range(epoch):
        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            _, predicted = torch.max(outputs.data, 1)
            total_samples += labels.size(0)
            total_correct += (predicted == labels).sum().item()
    
    return total_correct / total_samples


def validate(
    model: nn.Module,
    val_loader: DataLoader[Any],
    device: torch.device,
) -> float:
    """
    Evaluate a model on validation data.

    # Parameters
    * model: PyTorch model to evaluate.
    * val_loader: DataLoader for validation data.
    * device: Device to run evaluation on (e.g., 'cuda', 'cpu').

    # Returns
    Validation accuracy as a float.
    """
    model.eval()
    total_correct: int = 0
    total_samples: int = 0
    
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total_samples += labels.size(0)
            total_correct += (predicted == labels).sum().item()
    
    return total_correct / total_samples


def crossvalid(
    model_class: Type[nn.Module],
    vocab_size: int,
    hidden_dim: int,
    n_hiddens: int,
    epochs: int,
    criterion: nn.Module,
    optimizer_class: Any,
    lr: float,
    dataset: Dataset[Any],
    k_fold: int,
    device: torch.device,
    batch_size: int,
    dropout: float,
    weight_decay: float,
) -> tuple[Series[float], Series[float]]:
    """
    Perform stratified k-fold cross-validation on a dataset.

    # Parameters
    We will split the parameters into three types:
        1. Constants: are parameters that are fixed
        2. Variant Hyperparameters: are hyperparameters that can be
            changed (in the context of a Grid Search).
        3. Constant Hyperparameters: are hyperparameters that are
            fixed (in the context of a Grid Search).

    ## Constants
    * model_class: Should ALWAYS be MyLittlePony.
    * k_fold: Number of folds.
    * device: Device to run on (e.g., 'cuda', 'cpu').
    * vocab_size: Vocabulary size for the model.
    * dataset: Dataset with `.y` attribute for labels.

    ## Variant Hyperparameters
    * hidden_dim: Hidden layer dimension.
    * n_hiddens: Number of hidden layers.
    * epochs: Number of training epochs per fold.
    * batch_size: Batch size for DataLoaders.
    * dropout: Dropout rate (passed to model_class).

    ## Constant Hyperparameters
    * optimizer_class: Optimizer class to instantiate.
    * criterion: Loss function.
    * lr: Learning rate.
    * weight_decay: L2 regularization strength for the optimizer.

    # Returns
    Tuple of (train_score, val_score) as pandas Series.

    # Reference
    https://stackoverflow.com/a/64386444
    Posted by Skipper, modified by community. See post 'Timeline' for change history
    Retrieved 2026-02-03, License - CC BY-SA 4.0
    """
    train_score: Series[float] = pd.Series(dtype=float)
    val_score: Series[float] = pd.Series(dtype=float)

    # stratified k fold
    labels: ndarray[Any, Any] = dataset.y  # type: ignore[attr-defined]
    stratified_folds = StratifiedKFold(n_splits=k_fold, shuffle=True, random_state=5)

    allocate: ndarray[Any, Any] = np.zeros(len(labels))
    fold_indices = list(stratified_folds.split(allocate, labels))

    for i in tqdm(range(k_fold), desc="K-Fold CV"):
        model = model_class(vocab_size, hidden_dim, n_hiddens, dropout).to(device)
        optimizer = optimizer_class(model.parameters(), lr=lr, weight_decay=weight_decay)

        train_indices, val_indices = fold_indices[i]

        train_set = torch.utils.data.Subset(dataset, train_indices)
        val_set = torch.utils.data.Subset(dataset, val_indices)

        train_loader = torch.utils.data.DataLoader(
            train_set, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True
        )
        val_loader = torch.utils.data.DataLoader(
            val_set, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True
        )
        
        train_acc = train(model, criterion, optimizer, train_loader, epochs, device)
        train_score.at[i] = train_acc
        val_acc = validate(model, val_loader, device)
        val_score.at[i] = val_acc

        fold_labels = labels[val_indices]
        distribution = np.bincount(fold_labels) / len(fold_labels)
    
    return train_score, val_score