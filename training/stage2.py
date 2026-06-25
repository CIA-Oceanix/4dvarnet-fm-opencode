import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from models.solver import TweedieSolver
from training.losses import StateMSELoss


def train_stage2(
    model: TweedieSolver,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int = 400,
    lr: float = 1e-3,
    device: torch.device = torch.device("cpu"),
    verbose: bool = True,
):
    loss_fn = StateMSELoss(use_gradient_loss=True)

    for param in model.mean_estimator.parameters():
        param.requires_grad = False

    optimizer = optim.Adam(model.non_gaussian.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for batch in train_loader:
            batch = batch.to(device)
            B, T, D = batch.states.shape

            optimizer.zero_grad()
            pred = model(batch.obs)
            loss = loss_fn(pred, batch.states)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                pred = model(batch.obs)
                loss = loss_fn(pred, batch.states)
                val_loss += loss.item()

        if verbose and (epoch + 1) % 20 == 0:
            print(
                f"Stage2 Epoch {epoch + 1}/{epochs} | "
                f"Train Loss: {train_loss / len(train_loader):.6f} | "
                f"Val Loss: {val_loss / len(val_loader):.6f}"
            )

    return model
