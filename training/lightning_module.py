import torch
import torch.nn as nn
import pytorch_lightning as pl
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
        elif self.model_type == "tweedie_cfm":
            if self.stage == 1:
                params = self.model.mean_estimator.parameters()
            else:
                params = self.model.velocity_unet.parameters()
        elif self.model_type in ("joint_cfm", "joint_direct_unet") and self.stage == 2:
            params = self.model.param_flow.parameters() if self.model_type == "joint_cfm" \
                else self.model.param_head.parameters()
        elif self.model_type == "param_head":
            params = self.model.param_head.parameters()
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
        elif self.model_type == "tweedie_cfm":
            if self.stage == 1:
                for p in self.model.velocity_unet.parameters():
                    p.requires_grad = False
                for p in self.model.mean_estimator.parameters():
                    p.requires_grad = True
                self.model.set_stage(1)
            else:
                for p in self.model.mean_estimator.parameters():
                    p.requires_grad = False
                for p in self.model.velocity_unet.parameters():
                    p.requires_grad = True
                self.model.set_stage(2)
        elif self.model_type in ("joint_cfm", "joint_direct_unet"):
            if self.stage == 2:
                for p in self.model.unet.parameters():
                    p.requires_grad = False
                if self.model_type == "joint_cfm":
                    for p in self.model.param_flow.parameters():
                        p.requires_grad = True
                else:
                    for p in self.model.param_head.parameters():
                        p.requires_grad = True
                self.model.set_stage(2)
        elif self.model_type == "param_head":
            if getattr(self.model, "state_encoder", None) is not None:
                for p in self.model.state_encoder.parameters():
                    p.requires_grad = False
            for p in self.model.param_head.parameters():
                p.requires_grad = True
            self.model.set_stage(1)
        self._frozen = True

    def _forward_and_loss(self, batch):
        if self.model_type == "tweedie":
            if self.stage == 1:
                pred = self.model.estimate_mean(batch.obs)
            else:
                pred = self.model(batch.obs)
            loss = self.loss_fn(pred, batch.states)
        elif self.model_type == "direct_unet":
            pred = self.model(batch)
            loss = self.loss_fn(pred, batch.states)
        elif self.model_type == "vanilla_cfm":
            loss = self.model.compute_cfm_loss(batch)
        elif self.model_type == "joint_cfm":
            loss = self.model.compute_param_loss(batch) if self.stage == 2 \
                else self.model.compute_cfm_loss(batch)
        elif self.model_type == "joint_direct_unet":
            loss = self.model.compute_param_loss(batch) if self.stage == 2 \
                else self.model.compute_loss(batch)
        elif self.model_type == "predict_state_cfm":
            loss = self.model.compute_loss(batch)
        elif self.model_type == "tweedie_cfm":
            loss = self.model.compute_loss(batch)
        elif self.model_type == "param_head":
            loss = self.model.compute_loss(batch)
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

    def forward(self, batch, **kwargs):
        if self.model_type == "direct_unet":
            return self.model(batch)
        elif self.model_type == "predict_state_cfm":
            return self.model.sample(batch)
        elif self.model_type == "tweedie_cfm":
            if self.stage == 1:
                return self.model.estimate_mean(batch.obs)
            else:
                return self.model.sample(batch)
        return self.model(batch, **kwargs)

    def load_legacy_checkpoint(self, ckpt_path: str):
        state = torch.load(ckpt_path, map_location="cpu")
        state = {k.replace("_orig_mod.", ""): v for k, v in state.items()}
        self.model.load_state_dict(state)
