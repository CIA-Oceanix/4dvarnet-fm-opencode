import torch
import torch.nn as nn
import pytorch_lightning as pl
from models.solver import TweedieSolver
from training.losses import StateMSELoss


class LitModel(pl.LightningModule):
    def __init__(
        self,
        model: nn.Module,
        model_type: str = "tweedie",
        stage: int = 1,
        lr: float = 1e-3,
        gradient_clip_val: float = 10.0,
        use_gradient_loss: bool = True,
        gradient_weight: float = 0.1,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["model"])
        self.model = model
        self.model_type = model_type
        self.stage = stage
        self.lr = lr
        self.gradient_clip_val = gradient_clip_val
        self.loss_fn = StateMSELoss(
            use_gradient_loss=use_gradient_loss,
            gradient_weight=gradient_weight,
        )
        self._frozen = False

    def configure_optimizers(self):
        if self.model_type == "tweedie":
            if self.stage == 1:
                params = self.model.mean_estimator.parameters()
            else:
                params = self.model.non_gaussian.parameters()
        else:
            params = self.model.parameters()
        return torch.optim.Adam(params, lr=self.lr)

    def on_train_start(self):
        if self._frozen:
            return
        if self.model_type == "tweedie":
            if self.stage == 1:
                for p in self.model.non_gaussian.parameters():
                    p.requires_grad = False
                for p in self.model.mean_estimator.parameters():
                    p.requires_grad = True
            else:
                for p in self.model.mean_estimator.parameters():
                    p.requires_grad = False
                for p in self.model.non_gaussian.parameters():
                    p.requires_grad = True
        self._frozen = True

    def _forward_and_loss(self, batch):
        if self.model_type == "tweedie":
            if self.stage == 1:
                pred = self.model.estimate_mean(batch.obs)
            else:
                pred = self.model(batch.obs)
            loss = self.loss_fn(pred, batch.states)
        elif self.model_type == "direct_unet":
            pred = self.model(batch.obs)
            loss = self.loss_fn(pred, batch.states)
        elif self.model_type == "vanilla_cfm":
            loss = self.model.compute_cfm_loss(batch)
        elif self.model_type == "joint_cfm":
            loss = self.model.compute_cfm_loss(batch)
        else:
            raise ValueError(f"Unknown model_type: {self.model_type}")
        return loss

    def training_step(self, batch, batch_idx):
        loss = self._forward_and_loss(batch)
        self.log("train_loss", loss, prog_bar=True, on_step=False, on_epoch=True, batch_size=batch.batch_size)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self._forward_and_loss(batch)
        self.log("val_loss", loss, prog_bar=True, on_epoch=True, batch_size=batch.batch_size)
        return loss

    def forward(self, obs, **kwargs):
        return self.model(obs, **kwargs)

    def load_legacy_checkpoint(self, ckpt_path: str):
        state = torch.load(ckpt_path, map_location="cpu")
        state = {k.replace("_orig_mod.", ""): v for k, v in state.items()}
        self.model.load_state_dict(state)
