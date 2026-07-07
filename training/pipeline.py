import torch
import torch.nn as nn
from omegaconf import DictConfig
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger, TensorBoardLogger
from training.lightning_module import LitModel


def create_trainer(cfg: DictConfig, stage: int) -> pl.Trainer:
    stage_cfg = cfg.training.stage1 if stage == 1 else cfg.training.stage2
    callbacks = [
        ModelCheckpoint(
            monitor="val_loss",
            mode="min",
            save_top_k=1,
            dirpath=cfg.paths.checkpoint_dir,
            filename=f"stage{stage}_best",
        )
    ]
    csv_logger = CSVLogger(save_dir=cfg.paths.outputs_dir, name=f"stage{stage}")
    tb_logger = TensorBoardLogger(save_dir=cfg.paths.outputs_dir, name=f"stage{stage}")
    trainer = pl.Trainer(
        max_epochs=stage_cfg.epochs,
        gradient_clip_val=stage_cfg.gradient_clip_val,
        callbacks=callbacks,
        logger=[csv_logger, tb_logger],
        accelerator=cfg.training.get("accelerator", "auto"),
        devices=1,
        log_every_n_steps=10,
    )
    return trainer


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
        use_gradient_loss=cfg.training.loss.use_gradient,
        gradient_weight=cfg.training.loss.gradient_weight,
    )
    trainer = create_trainer(cfg, stage)
    trainer.fit(lit_module, loaders["train"], loaders["val"])
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
    if cfg.training.stage2.epochs > 0:
        model = train_stage(model, loaders, cfg, stage=2, device=device)
    return model
