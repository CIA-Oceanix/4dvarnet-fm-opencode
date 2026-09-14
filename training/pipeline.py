import torch
import torch.nn as nn
from omegaconf import DictConfig
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import CSVLogger, TensorBoardLogger
from training.lightning_module import LitModel

# Val loss is logged every epoch and is noisy for the CFM models (stochastic
# sampling loss), so a plain patience-only EarlyStopping would reset on
# noise-driven "new lows" all the time. min_delta filters that out: patience
# only accumulates once genuine improvement stalls. Both are overridable per
# stage via `early_stopping_patience`/`early_stopping_min_delta` in the
# model config's training.stage1/stage2 block; set patience to 0/null to
# disable early stopping for a stage.
DEFAULT_EARLY_STOPPING_PATIENCE = 300
DEFAULT_EARLY_STOPPING_MIN_DELTA = 1e-3


def create_trainer(cfg: DictConfig, stage: int, max_epochs: int = None) -> pl.Trainer:
    stage_cfg = cfg.training.stage1 if stage == 1 else cfg.training.stage2
    if max_epochs is None:
        max_epochs = stage_cfg.max_epochs
    callbacks = [
        ModelCheckpoint(
            monitor="val_loss",
            mode="min",
            save_top_k=1,
            dirpath=cfg.paths.checkpoint_dir,
            filename=f"stage{stage}_best",
        )
    ]
    patience = stage_cfg.get("early_stopping_patience", DEFAULT_EARLY_STOPPING_PATIENCE)
    if patience:
        callbacks.append(EarlyStopping(
            monitor="val_loss",
            mode="min",
            patience=patience,
            min_delta=stage_cfg.get("early_stopping_min_delta", DEFAULT_EARLY_STOPPING_MIN_DELTA),
        ))
    csv_logger = CSVLogger(save_dir=cfg.paths.outputs_dir, name=f"stage{stage}")
    tb_logger = TensorBoardLogger(save_dir=cfg.paths.outputs_dir, name=f"stage{stage}")
    trainer = pl.Trainer(
        max_epochs=max_epochs,
        gradient_clip_val=stage_cfg.gradient_clip_val,
        callbacks=callbacks,
        logger=[csv_logger, tb_logger],
        accelerator=cfg.training.get("accelerator", "auto"),
        devices=1,
        log_every_n_steps=10,
    )
    return trainer


def load_best_checkpoint(lit_module: pl.LightningModule, trainer: pl.Trainer) -> None:
    """Restore the best (lowest val_loss) weights into `lit_module` in place.

    With early stopping, training ends `patience` epochs past the best
    epoch, not at it -- without this, the checkpoint saved to disk would be
    the (slightly overfit) final-epoch weights instead of the best ones.
    """
    ckpt_cb = trainer.checkpoint_callback
    best_path = getattr(ckpt_cb, "best_model_path", "") if ckpt_cb else ""
    if best_path:
        state = torch.load(best_path, map_location=lit_module.device)["state_dict"]
        lit_module.load_state_dict(state)


def train_stage(
    model: nn.Module,
    loaders: dict,
    cfg: DictConfig,
    stage: int,
    device: torch.device,
) -> nn.Module:
    stage_cfg = cfg.training.stage1 if stage == 1 else cfg.training.stage2
    lit_module = LitModel(
        model, model_type=cfg.model.model_type, stage=stage,
        lr=stage_cfg.lr,
        gradient_clip_val=stage_cfg.gradient_clip_val,
    )
    trainer = create_trainer(cfg, stage)
    trainer.fit(lit_module, loaders["train"], loaders["val"])
    load_best_checkpoint(lit_module, trainer)
    path = cfg.paths[f"checkpoint_stage{stage}"]
    torch.save(lit_module.model.state_dict(), path)
    return lit_module.model


def run_2stage_pipeline(
    model: nn.Module,
    loaders: dict,
    cfg: DictConfig,
    device: torch.device,
) -> nn.Module:
    model = train_stage(model, loaders, cfg, stage=1, device=device)
    if cfg.training.stage2.max_epochs > 0:
        model = train_stage(model, loaders, cfg, stage=2, device=device)
    return model
