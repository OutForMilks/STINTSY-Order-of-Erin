import optuna
from torch.utils.data import DataLoader
import torch.nn as nn
import torch

def objective(trial, model, train_set, val_set, device):
    # Variable
    n_hidden_layers = trial.suggest_int("n_hidden_layers", 2, 4)
    neurons_in_hidden = trial.suggest_int("neurons_in_hidden", 32, 64)
    dropout = trial.suggest_float("dropout", 0.2, 0.5)
    
    # Constants
    learning_rate = 1e-3
    weight_decay = 1e-5
    batch_size = 32
    epochs = 5

    model = model(
        vocab_size=train_set.vocab_size,
        neurons_in_hidden=neurons_in_hidden,
        n_hidden_layers=n_hidden_layers,
        dropout=dropout,
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size)

    for _ in range(epochs):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)

            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for xb, yb in val_loader:
            xb, yb = xb.to(device), yb.to(device)
            preds = torch.argmax(model(xb), dim=1)
            correct += (preds == yb).sum().item()
            total += yb.size(0)

    return correct / total