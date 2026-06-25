import torch
import torch.nn as nn


class StateMSELoss(nn.Module):
    def __init__(self, use_gradient_loss: bool = False):
        super().__init__()
        self.mse = nn.MSELoss()
        self.use_gradient_loss = use_gradient_loss

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        loss = self.mse(pred, target)
        if self.use_gradient_loss and pred.shape[1] > 1:
            pred_grad = pred[:, 1:] - pred[:, :-1]
            target_grad = target[:, 1:] - target[:, :-1]
            loss = loss + 0.1 * self.mse(pred_grad, target_grad)
        return loss
