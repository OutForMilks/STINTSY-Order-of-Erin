import torch
import torch.nn as nn
from .trainer import train, crossvalid


def grid_search(
    *,
    model_class,
    hyperparam_combinations,
    optimizer,
    criterion,
    learning_rate,
    weight_decay,
    dataset,
    vocab_size,
    device,
    k_fold=5,
    save_path="best_model.pt",
):
    """
    Run grid search over hyperparameter combinations using stratified k-fold cross-validation.

    Iterates through all hyperparameter combinations, evaluates each with k-fold CV,
    then retrains the best configuration on the full dataset and saves the model.

    # Parameters
    We will split the parameters into three types:
        1. Constants: are parameters that are fixed
        2. Variant Hyperparameters: are hyperparameters that can be
            changed (in the context of a Grid Search).
        3. Constant Hyperparameters: are hyperparameters that are
            fixed (in the context of a Grid Search).

    ## Constants
    * model_class: PyTorch model class to instantiate. Must accept (vocab_size, hidden_dim, n_hiddens, dropout).
    * vocab_size: Vocabulary size for the model.
    * device: Device to run on (e.g., 'cuda', 'cpu').
    * dataset: Dataset with .y attribute for labels.
    * k_fold: Number of cross-validation folds.
    * save_path: Path to save the best model checkpoint.

    ## Variant Hyperparameters
    * hyperparam_combinations: Iterable of (batch_size, epochs, n_hiddens, hidden_dim, dropout) tuples.

    ## Constant Hyperparameters
    * optimizer: Optimizer class (e.g., torch.optim.Adam).
    * criterion: Loss function module.
    * learning_rate: Learning rate.
    * weight_decay: Weight decay.

    # Returns
    Tuple of (results, best_cfg) where results is a list of dicts with scores per config,
    and best_cfg is (epochs, n_hiddens, hidden_dim, batch_size, dropout) of the best configuration.
    """
    results = []

    best_val = -float("inf")
    best_cfg = None

    print("=" * 100)
    print(
        f"{' Epochs':>6} | {'Hiddens':>7} | {'Neurons':>7} | {'Batch':>5} | "
        f"{'Dropout':>7} | {'Train':>7} | {'Val':>7}"
    )
    print("-" * 100)

    for batch_size, epochs, n_hiddens, hidden_dim, dropout in hyperparam_combinations:
        train_score, val_score = crossvalid(
            model_class     = model_class,
            k_fold          = k_fold,
            device          = device,
            vocab_size      = vocab_size,
            dataset         = dataset,

            hidden_dim      = hidden_dim,
            n_hiddens       = n_hiddens,
            epochs          = epochs,
            batch_size      = batch_size,
            dropout         = dropout,
            weight_decay    = weight_decay,

            optimizer_class = optimizer,
            criterion       = criterion,
            lr              = learning_rate,
        )

        mean_train = train_score.mean()
        mean_val = val_score.mean()

        results.append({
            "epochs": epochs,
            "n_hiddens": n_hiddens,
            "hidden_dim": hidden_dim,
            "batch_size": batch_size,
            "dropout": dropout,
            "weight_decay": weight_decay,
            "train_score": mean_train,
            "val_score": mean_val,
        })

        print(
            f"{epochs:6d} | {n_hiddens:7d} | {hidden_dim:7d} | {batch_size:5d} | "
            f"{dropout:7.2f} | {mean_train:7.4f} | {mean_val:7.4f}"
        )

        if mean_val > best_val:
            best_val = mean_val
            best_cfg = (epochs, n_hiddens, hidden_dim, batch_size, dropout)

    print("=" * 100)

    assert best_cfg is not None, "No configurations were evaluated"
    epochs, n_hiddens, hidden_dim, batch_size, dropout = best_cfg

    best_model = model_class(
        vocab_size, hidden_dim, n_hiddens, dropout
    ).to(device)

    best_optimizer = optimizer(
        best_model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )

    train_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
    )

    train(
        model        = best_model,
        criterion    = criterion,
        optimizer    = best_optimizer,
        train_loader = train_loader,
        epoch        = epochs,
        device       = device,
    )

    torch.save(
        {
            "model_state": best_model.state_dict(),
            "config": {
                "epochs": epochs,
                "n_hiddens": n_hiddens,
                "hidden_dim": hidden_dim,
                "batch_size": batch_size,
                "dropout": dropout,
                "weight_decay": weight_decay,
            },
            "best_val_score": best_val,
        },
        save_path,
    )

    return results, best_cfg
