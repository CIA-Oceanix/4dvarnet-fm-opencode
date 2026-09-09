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
        use_cosine_scheduler: bool = True,
        max_epochs: int = None,
        obs_weight_lr_scale: float = 1.0,
        prior_unet_lr_scale: float = 1.0,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["model"])
        self.model = model
        self.model_type = model_type
        self.stage = stage
        self.lr = lr
        self.gradient_clip_val = gradient_clip_val
        self.use_cosine_scheduler = use_cosine_scheduler
        self.max_epochs = max_epochs
        self.obs_weight_lr_scale = obs_weight_lr_scale
        self.prior_unet_lr_scale = prior_unet_lr_scale
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
        elif self.model_type in ("joint_cfm", "joint_cfm_coupled", "joint_direct_unet") and self.stage == 2:
            params = self.model.param_flow.parameters() if self.model_type != "joint_direct_unet" \
                else self.model.param_head.parameters()
        elif self.model_type in ("param_head", "param_head_unet"):
            params = self.model.param_head.parameters()
        else:
            params = self.model.parameters()
        # obs_weight_lr_scale governs whichever trainable var-cost scalar the
        # model actually has: FourDVarNetPredictStateCFM's _obs_weight_raw, or
        # FourDVarNetSolver's _prior_weight_raw (the two are mutually
        # exclusive per model -- never both -- see models/fourdvarnet.py).
        # Explicit `is not None` checks throughout (never `or`/truthiness):
        # these are 0-dim tensors, and `bool(tensor(0.0))` is False, which
        # would silently misselect the wrong attribute if the value happens
        # to equal exactly zero.
        var_cost_weight_param = None
        prior_unet_params = []
        if self.model_type in ("fourdvarnet", "fourdvarnet_cfm"):
            obs_weight_param = getattr(self.model, "_obs_weight_raw", None)
            prior_weight_param = getattr(self.model, "_prior_weight_raw", None)
            var_cost_weight_param = obs_weight_param if obs_weight_param is not None else prior_weight_param
            prior_unet = getattr(self.model, "prior_unet", None)
            if prior_unet is not None and self.prior_unet_lr_scale != 1.0:
                prior_unet_params = list(prior_unet.parameters())
        use_var_cost_weight_group = var_cost_weight_param is not None and self.obs_weight_lr_scale != 1.0
        if use_var_cost_weight_group or prior_unet_params:
            # Only exclude var_cost_weight_param from other_params when it
            # actually gets its own group below (use_var_cost_weight_group) --
            # otherwise (e.g. prior_unet_lr_scale != 1.0 but obs_weight_lr_scale
            # left at its 1.0 default) it must stay in other_params so it's
            # still covered by SOME group, at the plain self.lr rate. Dropping
            # it unconditionally here silently excludes it from the optimizer
            # entirely whenever only prior_unet_params triggers this branch --
            # exactly the kind of silent-stall bug this PR exists to eliminate.
            excluded_ids = {id(p) for p in prior_unet_params}
            if use_var_cost_weight_group:
                excluded_ids.add(id(var_cost_weight_param))
            other_params = [p for p in params if id(p) not in excluded_ids]
            groups = [{"params": other_params, "lr": self.lr}]
            if use_var_cost_weight_group:
                groups.append({"params": [var_cost_weight_param], "lr": self.lr * self.obs_weight_lr_scale})
            if prior_unet_params:
                groups.append({"params": prior_unet_params, "lr": self.lr * self.prior_unet_lr_scale})
            optimizer = torch.optim.Adam(groups)
        else:
            optimizer = torch.optim.Adam(params, lr=self.lr)
        if self.use_cosine_scheduler:
            if not self.max_epochs:
                raise ValueError("use_cosine_scheduler=True requires max_epochs to be set")
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.max_epochs)
            return {"optimizer": optimizer, "lr_scheduler": scheduler}
        return optimizer

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
        elif self.model_type in ("joint_cfm", "joint_cfm_coupled", "joint_direct_unet"):
            if self.stage == 2:
                for p in self.model.unet.parameters():
                    p.requires_grad = False
                if self.model_type != "joint_direct_unet":
                    for p in self.model.param_flow.parameters():
                        p.requires_grad = True
                else:
                    for p in self.model.param_head.parameters():
                        p.requires_grad = True
                self.model.set_stage(2)
        elif self.model_type in ("param_head", "param_head_unet"):
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
        elif self.model_type in ("direct_unet", "monai_direct_unet"):
            pred = self.model(batch)
            loss = self.loss_fn(pred, batch.states)
        elif self.model_type in ("vanilla_cfm", "monai_vanilla_cfm"):
            loss = self.model.compute_cfm_loss(batch)
        elif self.model_type in ("joint_cfm", "joint_cfm_coupled"):
            loss = self.model.compute_param_loss(batch) if self.stage == 2 \
                else self.model.compute_cfm_loss(batch)
        elif self.model_type == "joint_direct_unet":
            loss = self.model.compute_param_loss(batch) if self.stage == 2 \
                else self.model.compute_loss(batch)
        elif self.model_type == "predict_state_cfm":
            loss = self.model.compute_loss(batch)
        elif self.model_type == "tweedie_cfm":
            loss = self.model.compute_loss(batch)
        elif self.model_type in ("param_head", "param_head_unet"):
            loss = self.model.compute_loss(batch)
        elif self.model_type in ("sda_prior", "sda_prior_cond", "monai_sda_prior", "monai_sda_prior_cond"):
            loss = self.model.compute_cfm_loss(batch)
        elif self.model_type == "fourdvarnet":
            loss = self.model.compute_loss(batch)
        elif self.model_type == "fourdvarnet_cfm":
            loss = self.model.compute_loss(batch)
        else:
            raise ValueError(f"Unknown model_type: {self.model_type}")
        return loss

    def training_step(self, batch, batch_idx):
        loss = self._forward_and_loss(batch)
        self.log("train_loss", loss, prog_bar=True, on_step=False, on_epoch=True, batch_size=batch.batch_size)
        if getattr(self.model, "_obs_weight_raw", None) is not None:
            self.log("obs_weight", self.model.obs_weight, on_step=False, on_epoch=True, batch_size=batch.batch_size)
        if getattr(self.model, "_prior_weight_raw", None) is not None:
            self.log("prior_weight", self.model.prior_weight, on_step=False, on_epoch=True, batch_size=batch.batch_size)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self._forward_and_loss(batch)
        self.log("val_loss", loss, prog_bar=True, on_epoch=True, batch_size=batch.batch_size)
        return loss

    def forward(self, batch, **kwargs):
        if self.model_type in ("direct_unet", "monai_direct_unet"):
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
