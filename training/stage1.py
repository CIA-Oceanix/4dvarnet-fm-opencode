import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from models.solver import TweedieSolver
from training.losses import StateMSELoss


def train_stage1(
    model: TweedieSolver,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int = 200,
    lr: float = 1e-3,
    device: torch.device = torch.device("cpu"),
    verbose: bool = True,
):
    loss_fn = StateMSELoss(use_gradient_loss=True)
    optimizer = optim.Adam(model.mean_estimator.parameters(), lr=lr)

    best_val_loss = float('inf')

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for batch in train_loader:
            batch = batch.to(device)
            B, T, D = batch.states.shape

            optimizer.zero_grad()
            pred_mean = model.estimate_mean(batch.obs)
            loss = loss_fn(pred_mean, batch.states)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                pred_mean = model.estimate_mean(batch.obs)
                loss = loss_fn(pred_mean, batch.states)
                val_loss += loss.item()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.mean_estimator.state_dict(), 'checkpoint_stage1.pt')
            if verbose:
                print("✓ checkpoint saved")

        if verbose and (epoch + 1) % 20 == 0:
            print(
                f"Stage1 Epoch {epoch + 1}/{epochs} | "
                f"Train Loss: {train_loss / len(train_loader):.6f} | "
                f"Val Loss: {val_loss / len(val_loader):.6f}"
            )

    return model
