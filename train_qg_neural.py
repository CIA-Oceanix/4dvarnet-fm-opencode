#!/usr/bin/env python3
"""Train a DirectUNet / VanillaCFM (tau=0) neural estimator on QG S0.

Dedicated entry point for the QG neural baseline (mirrors the L96 L1/L2 setup
on the QG case study). Unlike `train.py` (which hard-codes the L63/L96 data and
eval machinery), this drives the self-contained QG neural data wrapper
(`data/qg_neural.py`) and shared Lightning training pieces.

Supervision target: daily mean of the full 2-layer streamfunction (`psi_daily`,
30 days per window), with an optional auxiliary PV-q loss (`training.
q_loss_weight` in `config/experiment/Q{1,2}_..._s0.yaml`, CLI-overridable via
`--q-loss-weight`). Observations: upper-layer psi grid-expanded, daily
aggregated, NaN-masked, padded to the full state width. Psi (+obs) is z-score
normalized per layer with a *global* (mean, std) computed once over the whole
training split (`precompute_qg_norm_stats.py` -> `data.normalize`/
`norm_stats_path` in the experiment YAML, default
`experiments/qg_psi_norm_stats.pt`) so eval maps back to physical units; PV
(q) is left in raw physical units (see `data/qg_neural.py`'s docstring for the
per-window-vs-global normalization trade-off).

For the q-loss, the model's normalized psi estimate is de-normalized to
physical psi, spectrally inverted to PV (per-window `rd`), and MSE-matched
directly (raw units) against the daily-mean PV target -- `q_loss_weight` is
derived by `precompute_qg_norm_stats.py` as `1/Var(q)` so this raw-unit term
contributes comparably to the (unit-variance) normalized psi loss.

Default `--train-seed`/`--val-seed`/`--test-seed` (42/10042/20042) and split
sizes (1000/100/100) match `reports/qg/generate_qg_window_chunk.py`'s
production convention, so pointing `--cache-dir` at a pre-generated 1000/100/100
truth cache (built by that array-job pipeline) hits it directly instead of
re-paying the ~2-year-spinup rollout. Default `--obs-geometry`/`--cols-per-day`/
`--obs-noise-std-frac`/`--init-lag-days` match the S0 DA-baseline reference case
(PLAN.md); train/val's on-the-fly obs redraws follow these same settings via
the shared `test_cfg` object passed to `QGNeuralDataset`.

Usage:
    python train_qg_neural.py --model-type direct_unet --exp-dir experiments/Q1_direct_unet_s0 \
        --cache-dir /path/to/qg_windows_1000_100_100/cache
    python train_qg_neural.py --model-type vanilla_cfm   --exp-dir experiments/Q2_vanilla_cfm_s0
    python train_qg_neural.py --model-type direct_unet --q-loss-weight 0.0 --eval-only <ckpt>

Q3 (oracle forcing+params) / Q4 (noisy forcing+params) -- see PLAN.md's
2026-09-10 QG Q3/Q4 section and `data/qg_neural.py`'s cond_mode docstring.
Both share `model_type=direct_unet` with Q1, so `--exp-id` picks the config:
    python train_qg_neural.py --model-type direct_unet --exp-id Q3_direct_unet_s0_oracle_cond \
        --exp-dir experiments/Q3_direct_unet_s0_oracle_cond --cache-dir ...
    python train_qg_neural.py --model-type direct_unet --exp-id Q4_direct_unet_s1_noisy_cond \
        --exp-dir experiments/Q4_direct_unet_s1_noisy_cond --cache-dir ...

Q1-obsdensity (obs-density-augmented Q1) -- same obs-only architecture as
Q1, but `--cols-per-day-min`/`--cols-per-day-max` resample `cols_per_day`
per training draw instead of a single fixed value (train split only; val/
test stay at the fixed `--cols-per-day` reference density), so one
checkpoint generalizes across observing-network densities. Mirrors L96's
obs-density-augmented training; see `data/qg_neural.py`'s
`cols_per_day_range` docstring and
`config/experiment/Q1_direct_unet_s0_obsdensity_aug.yaml`.
"""
import argparse
import json
import os
import time
from types import SimpleNamespace

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from data.normalization import load_norm_stats
from data.qg import QGConfig
from data.qg_neural import (
    QGNeuralDataset,
    denorm_psi,
    ensure_truth_cache,
    ensure_truth_cache_redrawn,
    layer_split,
    num_days,
    psi_daily,
    psi_to_q,
    q_daily,
    q_from_psi_norm,
    qg_collate,
)
from models.vanilla_cfm import VanillaCFM
from training.pipeline import create_trainer

BASE = os.path.dirname(os.path.abspath(__file__))
EXP_DIR = os.path.join(BASE, "experiments")


def build_cfg(**overrides) -> QGConfig:
    return QGConfig(**{k: v for k, v in overrides.items() if v is not None})


def build_model(model_type: str, cfg: QGConfig, param_dim: int = 0,
                cond_extra_dim: int = 0, ic_dim: int = 0,
                fdv_kwargs: dict | None = None) -> torch.nn.Module:
    if model_type == "direct_unet":
        # MONAI-backed, circular-padded 2D U-Net over the (ny, nx) grid --
        # QG's domain is doubly periodic (models.qg_dynamics.QGDynamics),
        # unlike L96's DirectUNet (models.direct_unet), which only convolves
        # along time and never exploits the field's 2D spatial/periodic
        # structure. Requires the `fdv-monai-proto` env (see
        # models.monai_unet_qg2d's docstring), not this project's default
        # `fdv` env.
        from models.monai_unet_qg2d import MonaiDirectUNetQG
        return MonaiDirectUNetQG(ny=cfg.ny, nx=cfg.nx, nlayers=2, param_dim=param_dim,
                                 cond_extra_dim=cond_extra_dim, ic_dim=ic_dim,
                                 hidden_channels=[64, 128, 256])
    if model_type == "vanilla_cfm":
        return VanillaCFM(state_dim=cfg.state_dim, param_dim=param_dim,
                          cond_extra_dim=cond_extra_dim,
                          hidden_channels=[64, 128, 256], time_emb_dim=64,
                          N_outer=10, sigma_prior=0.5, dropout=0.1,
                          train_tau_0_only=True)
    if model_type == "fourdvarnet":
        # No conditioning hooks at all (no param_dim/cond_extra_dim/ic_dim --
        # models.fourdvarnet.FourDVarNetSolver has none, unlike DirectUNet/
        # VanillaCFM): a QG fourdvarnet scheme is necessarily obs-only,
        # matching Q1 not Q3/Q4/Q5. `fdv_kwargs` mirrors L96's `cfg.model.fdv`
        # block (train.py's own "fourdvarnet" dispatch) -- read from the
        # experiment YAML's `model.fdv.*` in main(), not from CLI flags (too
        # many knobs to justify individual CLI args, same reasoning as
        # param_dim/cond_extra_dim already being YAML-only).
        from models.fourdvarnet import FourDVarNetSolver
        # qg_T/qg_ny/qg_nx are only actually used by unet_backbone="monai2d"
        # (models.monai_unet_qg2d.MonaiUNet2DQGSolver) -- harmless to always
        # pass them (ignored otherwise), avoids needing to special-case the
        # call based on which backbone the YAML picked.
        return FourDVarNetSolver(state_dim=cfg.state_dim, qg_T=num_days(cfg),
                                 qg_ny=cfg.ny, qg_nx=cfg.nx, **(fdv_kwargs or {}))
    raise ValueError(f"unknown model_type {model_type!r}")


# `unet_backbone="monai"` (MonaiUNet1D, DiffusionModelUNet) treats the T
# (days) axis as its own downsampled "spatial" dimension, requiring T
# divisible by 2**(len(hidden_channels)-1) (4 for the 3-level S-tier
# [32,64,128] config) -- QG's 30-day windows aren't (confirmed: crashed a
# real GPU job twice, "Sizes of tensors must match... Expected size 16 but
# got size 15", jobs 53462/53467 -- the main solve, then separately the
# prior_unet consistency term, both being fed unpadded T=30 tensors). 8
# covers up to a 4-level backbone (the S-tier config only needs 4). A
# convolutional U-Net's receptive field means the last real day or two's
# estimate can pick up a small edge effect from the adjacent zero-padding --
# an accepted, minor v1 limitation, not a correctness bug (same
# padding-boundary tradeoff any CNN makes).
_MONAI1D_PAD_TO = 8


def _pad_time_axis(x: torch.Tensor, pad_to: int = _MONAI1D_PAD_TO) -> tuple:
    """Pads a (B, T, ...) tensor's T axis (dim=1) up to the next multiple of
    `pad_to` with zeros (`new_zeros` preserves dtype, so a bool mask pads
    with False). Returns `(x, 0)` unchanged when already aligned."""
    T = x.shape[1]
    pad_T = -(-T // pad_to) * pad_to
    n_pad = pad_T - T
    if n_pad == 0:
        return x, 0
    pad_shape = (x.shape[0], n_pad) + tuple(x.shape[2:])
    return torch.cat([x, x.new_zeros(pad_shape)], dim=1), n_pad


def _pad_batch_for_monai1d(batch, T: int):
    """Pads `batch.obs`/`batch.obs_mask` (see `_pad_time_axis`) into a
    lightweight shim object exposing only what `FourDVarNetSolver.forward`/
    `.sample` read (zero obs / all-False obs_mask on the padded region
    contributes nothing to any obs-cost term, and "obs+state" -- Q6's own
    config -- doesn't reference obs_mask at all). Returns
    `(shim_or_batch, n_pad)` -- `n_pad=0` (batch returned unchanged) when
    already aligned. Caller must crop the model's output back to `[:T]`
    whenever `n_pad>0`."""
    padded_obs, n_pad = _pad_time_axis(batch.obs)
    if n_pad == 0:
        return batch, 0
    padded_mask, _ = _pad_time_axis(batch.obs_mask)
    return SimpleNamespace(obs=padded_obs, obs_mask=padded_mask), n_pad


def _padded_prior_cost(prior_unet, state: torch.Tensor) -> torch.Tensor:
    """`models.fourdvarnet._prior_cost`, but pads `state`'s T axis before
    feeding `prior_unet` (needed whenever `unet_backbone="monai"`, same
    constraint as `_pad_batch_for_monai1d`) and crops the *reconstruction*
    back to the real T before computing the sum-of-squares -- so the
    fabricated zero-padded region never contaminates the loss value itself
    (unlike a plain crop-the-input approach would, since `_prior_cost` is a
    scalar sum over all positions, not a per-position tensor you could crop
    after the fact)."""
    from models.fourdvarnet import _prior_ae
    T = state.shape[1]
    padded, n_pad = _pad_time_axis(state)
    recon = _prior_ae(prior_unet, padded)
    if n_pad:
        recon = recon[:, :T]
    return F.mse_loss(state, recon, reduction="sum")


def _plain_prior_cost(prior_unet, state: torch.Tensor) -> torch.Tensor:
    """`models.fourdvarnet._prior_cost`, unpadded -- for backbones with no
    T-axis divisibility constraint (`unet_backbone` "unet1d"/"monai2d")."""
    from models.fourdvarnet import _prior_cost
    return _prior_cost(prior_unet, state)


def epochs_for(model_type: str) -> int:
    return 200 if model_type == "direct_unet" else 400


class QGNeuralLightning(pl.LightningModule):
    """Combined psi (primary) + auxiliary PV-q loss trainer for QG neural models."""

    def __init__(self, model, model_type: str, norm: dict | None, qg_cfg: QGConfig,
                 q_loss_weight: float = 0.1, lr: float = 1e-3,
                 gradient_clip_val: float = 10.0,
                 use_cosine_scheduler: bool = True, max_epochs: int | None = None):
        super().__init__()
        self.model = model
        self.model_type = model_type
        self.norm = norm
        self.qg_cfg = qg_cfg
        self.q_loss_weight = q_loss_weight
        self.lr = lr
        self.gradient_clip_val = gradient_clip_val
        self.use_cosine_scheduler = use_cosine_scheduler
        self.max_epochs = max_epochs

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        if self.use_cosine_scheduler:
            if not self.max_epochs:
                raise ValueError("use_cosine_scheduler=True requires max_epochs to be set")
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=self.max_epochs)
            return {"optimizer": optimizer, "lr_scheduler": scheduler}
        return optimizer

    def _estimate_and_psi_loss(self, batch):
        if self.model_type == "direct_unet":
            est = self.model(batch)
            loss_psi = F.mse_loss(est, batch.states)
            return est, loss_psi
        if self.model_type == "vanilla_cfm":
            if not self.model.train_tau_0_only:
                raise ValueError("QG neural CFM baseline requires train_tau_0_only=True")
            b = batch.obs.shape[0]
            device = batch.obs.device
            tau = torch.zeros(b, device=device)
            x0 = torch.randn_like(batch.states) * self.model.sigma_prior
            v = self.model(x0, batch, tau)
            loss_psi = F.mse_loss(v, batch.states - x0)
            est = x0 + v
            return est, loss_psi
        if self.model_type == "fourdvarnet":
            # Calls forward() directly (matching direct_unet's convention)
            # rather than the model's own compute_loss() -- QG's psi loss
            # needs to feed into _q_loss()'s auxiliary PV-q term below, which
            # compute_loss() knows nothing about. With the default
            # tbptt_n_blocks=1 (unchanged by any QG config so far) this is
            # exactly equivalent to compute_loss()'s own main MSE term (see
            # FourDVarNetSolver.compute_loss's docstring) -- only its
            # optional prior-consistency term needs replicating by hand here.
            #
            # T-axis padding (_pad_batch_for_monai1d/_padded_prior_cost) is
            # only needed for unet_backbone="monai" (MonaiUNet1D treats T as
            # its own downsampled 1D-sequence axis). unet_backbone="monai2d"
            # (models.monai_unet_qg2d.MonaiUNet2DQGSolver, Q6's actual
            # backbone) merges T into the channel axis instead -- no
            # divisibility constraint on T at all, so padding here would
            # only waste compute and introduce a spurious edge effect for
            # no reason.
            T = batch.states.shape[1]
            needs_pad = self.model.unet_backbone == "monai"
            shim, n_pad = _pad_batch_for_monai1d(batch, T) if needs_pad else (batch, 0)
            est = self.model(shim)
            if n_pad:
                est = est[:, :T]
            loss_psi = F.mse_loss(est, batch.states)
            if self.model.prior_unet is not None and self.model.aux_var_cost_weight > 0:
                numel = est.numel()
                prior_cost_fn = _padded_prior_cost if needs_pad else _plain_prior_cost
                loss_psi = loss_psi + self.model.aux_var_cost_weight * (
                    prior_cost_fn(self.model.prior_unet, est) / numel
                    + prior_cost_fn(self.model.prior_unet, batch.states) / numel)
            return est, loss_psi
        raise ValueError(f"unsupported model_type {self.model_type!r}")

    def _q_loss(self, est, batch) -> torch.Tensor:
        if self.q_loss_weight <= 0 or batch.states_q is None:
            return torch.zeros((), device=est.device)
        device = est.device
        losses = []
        for b in range(est.shape[0]):
            q_pred = q_from_psi_norm(est[b], float(batch.rd[b]), self.qg_cfg,
                                     self.norm, device)
            losses.append(F.mse_loss(q_pred, batch.states_q[b]))
        return torch.stack(losses).mean()

    def _total_loss(self, batch):
        est, loss_psi = self._estimate_and_psi_loss(batch)
        loss_q = self._q_loss(est, batch)
        return loss_psi + self.q_loss_weight * loss_q, loss_psi, loss_q

    def training_step(self, batch, batch_idx):
        loss, loss_psi, loss_q = self._total_loss(batch)
        self.log("train_loss", loss, prog_bar=True, on_step=False, on_epoch=True,
                 batch_size=batch.batch_size)
        self.log("train_loss_psi", loss_psi, batch_size=batch.batch_size)
        if self.q_loss_weight > 0:
            self.log("train_loss_q", loss_q, batch_size=batch.batch_size)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, loss_psi, _loss_q = self._total_loss(batch)
        self.log("val_loss", loss, prog_bar=True, on_epoch=True,
                 batch_size=batch.batch_size)
        self.log("val_loss_psi", loss_psi, batch_size=batch.batch_size)
        return loss


def make_trainer_cfg(model_type: str, exp_dir: str, epochs: int, lr: float,
                     gradient_clip_val: float = 10.0):
    return OmegaConf.create({
        "training": {
            "stage1": {"epochs": epochs, "lr": lr, "gradient_clip_val": gradient_clip_val},
            "stage2": {"epochs": 0, "lr": lr, "gradient_clip_val": gradient_clip_val},
            "accelerator": "auto",
            "loss": {"use_gradient": False, "gradient_weight": 0.0},
        },
        "paths": {
            "checkpoint_dir": os.path.join(exp_dir, "checkpoints"),
            "outputs_dir": os.path.join(exp_dir, "logs"),
        },
    })


def estimate_windows(model, windows, cfg, model_type, device, norm=None, n_members=1,
                     cond_mode="none", param_norm_stats=None, noisy_max=1.5,
                     forcing_norm_stats=None, include_ic=False):
    """Return per-window physical psi estimates (W, days, 2*ny*nx) + per-window rd list."""
    dataset = QGNeuralDataset(windows, cfg, norm, cond_mode=cond_mode,
                              param_norm_stats=param_norm_stats, noisy_max=noisy_max,
                              forcing_norm_stats=forcing_norm_stats, include_ic=include_ic)
    loader = DataLoader(dataset, batch_size=8, shuffle=False, collate_fn=qg_collate)
    rds = [float(dataset.rd(i)) for i in range(len(windows))]
    model = model.to(device)
    model.eval()
    estimates = []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            if model_type == "direct_unet":
                pred = model(batch)
            elif model_type == "fourdvarnet":
                T = batch.states.shape[1]
                needs_pad = model.unet_backbone == "monai"
                shim, n_pad = _pad_batch_for_monai1d(batch, T) if needs_pad else (batch, 0)
                pred = model.sample(shim, N_outer=10)
                if n_members > 1:
                    members = [model.sample(shim, N_outer=10) for _ in range(n_members)]
                    pred = torch.stack(members).mean(dim=0)
                if n_pad:
                    pred = pred[:, :T]
            else:
                pred = model.sample(batch, N_outer=10)
                if n_members > 1:
                    members = [model.sample(batch, N_outer=10) for _ in range(n_members)]
                    pred = torch.stack(members).mean(dim=0)
            pred = pred.detach().cpu()
            for b in range(pred.shape[0]):
                estimates.append(denorm_psi(pred[b], cfg, norm))
    return np.stack([e.numpy() for e in estimates]), np.asarray(rds)


def pooled_metrics(est, truth):
    """est, truth: (W, days, D). Returns (per_dim_rmse, per_dim_ev)."""
    mse = np.mean((est - truth) ** 2, axis=(0, 1))
    var = np.var(truth, axis=(0, 1))
    rmse_dim = np.sqrt(mse)
    ev_dim = 1.0 - mse / np.where(var > 0.0, var, np.nan)
    return rmse_dim, ev_dim


def layer_summary(rmse_dim, ev_dim, cfg):
    split = layer_split(cfg)
    layer1 = {"rmse": float(np.mean(rmse_dim[:split])), "ev": float(np.mean(ev_dim[:split]))}
    layer2 = {"rmse": float(np.mean(rmse_dim[split:])), "ev": float(np.mean(ev_dim[split:]))}
    return {"layer1": layer1, "layer2": layer2,
            "pooled_rmse": float(np.mean(rmse_dim)), "pooled_ev": float(np.mean(ev_dim))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-type", choices=["direct_unet", "vanilla_cfm", "fourdvarnet"],
                    default="direct_unet")
    ap.add_argument("--exp-dir", default=None)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--q-loss-weight", type=float, default=None,
                    help="Overrides config/experiment/Q{1,2}_..._s0.yaml's "
                         "training.q_loss_weight if given.")
    ap.add_argument("--normalize", action=argparse.BooleanOptionalAction, default=None,
                    help="Overrides the experiment YAML's data.normalize if given.")
    ap.add_argument("--cosine-scheduler", action=argparse.BooleanOptionalAction, default=True,
                    help="Cosine-anneal the LR over training (default: on).")
    ap.add_argument("--gradient-clip-val", type=float, default=None,
                    help="Global gradient-norm clip applied by PyTorch Lightning's "
                         "Trainer. default=None (not 10.0) so the experiment YAML's "
                         "training.gradient_clip_val (if any) isn't silently overridden "
                         "-- falls back to 10.0 if neither is given. Note: "
                         "QGNeuralLightning's own gradient_clip_val attribute is dead "
                         "code (never read) -- this Trainer-level value is the only "
                         "one that actually does anything.")
    ap.add_argument("--seed", type=int, default=None,
                    help="If given, calls pl.seed_everything(seed) before building the "
                         "model/dataloaders, for reproducible model init + dataloader "
                         "shuffling. Default None: no explicit seeding (the project's "
                         "prior behavior for every run so far -- Q1/Q3/Q4/Q5 all ran "
                         "with whatever randomness torch/numpy picked up at process "
                         "start, undocumented and unreproducible).")
    ap.add_argument("--norm-stats-path", default=None,
                    help="Overrides the experiment YAML's data.norm_stats_path if given "
                         "(psi mean/std produced by precompute_qg_norm_stats.py).")
    ap.add_argument("--exp-id", default=None,
                    help="Experiment config name under config/experiment/<id>.yaml "
                         "(default: derived as Q1/Q2 from --model-type, for backward "
                         "compat; Q3/Q4 forcing+param-conditioned configs need this set "
                         "explicitly since they share model_type=direct_unet with Q1).")
    ap.add_argument("--cond-mode", choices=["none", "true", "noisy", "scenario"], default=None,
                    help="Overrides the experiment YAML's data.cond_mode if given -- "
                         "'none' (Q1/Q2, obs-only), 'true' (Q3, oracle forcing+params), "
                         "'noisy' (Q4, resampled-severity corrupted forcing+params), "
                         "'scenario' (eval-only: deterministic, reads whatever the "
                         "window's own S0/S1 scenario wrapper designates as believed -- "
                         "see data/qg_neural.py's _scenario_forcing_and_params).")
    ap.add_argument("--param-norm-stats-path", default=None,
                    help="Overrides the experiment YAML's data.param_norm_stats_path if "
                         "given ([U1,rd,rek] mean/std produced by "
                         "precompute_qg_norm_stats.py --output-params). Required "
                         "whenever cond_mode != 'none'.")
    ap.add_argument("--forcing-norm-stats-path", default=None,
                    help="Overrides the experiment YAML's data.forcing_norm_stats_path "
                         "if given (global scalar wind_curl mean/std produced by "
                         "precompute_qg_norm_stats.py --output-forcing). Required "
                         "whenever cond_mode != 'none' -- see data/qg_neural.py's "
                         "module docstring for why (a training run collapsed without it).")
    ap.add_argument("--noisy-max", type=float, default=None,
                    help="Overrides the experiment YAML's data.noisy_max if given -- "
                         "Q4's per-draw corruption-severity fraction is resampled in "
                         "[0, noisy_max] each __getitem__ call (default 1.5, matching "
                         "the L96 SDA3 noisy_da_max convention).")
    ap.add_argument("--s1-param-bias", type=float, default=None,
                    help="Overrides the experiment YAML's data.s1_param_bias if given "
                         "-- the rd/rek bias magnitude cond_mode='noisy' scales by its "
                         "resampled frac (default: QGConfig's own 0.15 class default; "
                         "the DA S1 reference campaign uses 0.1, see qg_da_s1_scratch.py "
                         "-- Q5 uses 0.1 with noisy_max=2.0 for an effective [0,0.2] range).")
    ap.add_argument("--s1-amp-bias", type=float, default=None,
                    help="Overrides the experiment YAML's data.s1_amp_bias if given -- "
                         "same as --s1-param-bias but for the wind-amplitude bias.")
    ap.add_argument("--include-ic", action=argparse.BooleanOptionalAction, default=False,
                    help="Overrides the experiment YAML's data.include_ic if given -- "
                         "Q5: condition on the raw initial-condition snapshot (always "
                         "the true window['init_state'], like obs -- never scenario-"
                         "corrupted), inverted to psi and z-scored with the psi norm "
                         "stats. A third conditioning class distinct from forcing/params "
                         "(one static field per window, not per-day/scalar) -- see "
                         "data/qg_neural.py's include_ic docstring.")
    ap.add_argument("--num-train", type=int, default=1000)
    ap.add_argument("--num-val", type=int, default=100)
    ap.add_argument("--num-test", type=int, default=100)
    ap.add_argument("--fixed-split-obs", action="store_true",
                    help="Use one fixed obs/init-state draw for train/val "
                         "(legacy behavior) instead of regenerating them on "
                         "the fly from the cached truth each epoch.")
    ap.add_argument("--cols-per-day-min", type=int, default=None,
                    help="Obs-density-augmented training: TRAIN split only "
                         "resamples cols_per_day ~ Uniform{min,...,max} on every "
                         "draw (requires on-the-fly obs, i.e. NOT --fixed-split-obs) "
                         "instead of a single fixed value, so the checkpoint "
                         "generalizes across observing-network densities. Val/test "
                         "stay at the fixed --cols-per-day reference value. Falls "
                         "back to the experiment YAML's data.cols_per_day_min if not "
                         "given; default None (no augmentation, original behavior).")
    ap.add_argument("--cols-per-day-max", type=int, default=None,
                    help="See --cols-per-day-min. Both must be given (CLI or YAML) "
                         "to enable augmentation.")
    ap.add_argument("--cache-dir", default="reports/qg_cache")
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--num-workers", type=int, default=4,
                    help="DataLoader worker processes for train/val (default 4). "
                         "num_workers=1 serializes each __getitem__'s CPU work "
                         "(obs/forcing prep) with GPU training -- a measured ~2x "
                         "epoch-time bottleneck for Q4's noisy-forcing conditioning "
                         "(see PLAN.md's 2026-09-10 QG Q3/Q4 section). Match sbatch's "
                         "--cpus-per-task to this + a couple cores for the main process.")
    ap.add_argument("--nx", type=int, default=None)
    ap.add_argument("--n-members", type=int, default=1)
    # Split seeds match the production 1000/100/100 truth-generation convention
    # (reports/qg/generate_qg_window_chunk.py's SPLIT_SEED_BASE) so a `--cache-dir`
    # pointed at that pre-generated cache hits it directly instead of re-paying the
    # ~2-year-spinup rollout (train/val truth is reused as-is even though its baked-in
    # obs used QGConfig defaults -- `on_the_fly_obs` overwrites it every draw anyway).
    ap.add_argument("--train-seed", type=int, default=42)
    ap.add_argument("--val-seed", type=int, default=10_042)
    ap.add_argument("--test-seed", type=int, default=20_042)
    # S0 reference-case obs/IC protocol (matches reports/qg/fix_qg_test_obs_ic.py's
    # ref_cfg and the DA-baseline reference case in PLAN.md), applied to `test_cfg` --
    # which also doubles as the shared cfg QGNeuralDataset uses to (re)draw obs for
    # every split, so train/val's on-the-fly obs follow the same protocol.
    ap.add_argument("--obs-geometry", default="random_columns")
    ap.add_argument("--cols-per-day", type=int, default=4)
    # default=None (not 0.01/1.0) so the experiment YAML's data.* values (if
    # any) aren't silently overridden -- Q5 needs obs_noise_std_frac=0.05/
    # init_lag_days=5.0 to actually take effect from its config alone,
    # without requiring the launcher to also pass these on the CLI.
    ap.add_argument("--obs-noise-std-frac", type=float, default=None)
    ap.add_argument("--init-lag-days", type=float, default=None)
    ap.add_argument("--eval-only", nargs="?", const="stage1_best.pt", default=None,
                    help="Path to a checkpoint; skip training and just evaluate.")
    args = ap.parse_args()
    if args.seed is not None:
        pl.seed_everything(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_type = args.model_type
    epochs = args.epochs if args.epochs is not None else epochs_for(model_type)
    config_name = args.exp_id or f"Q{1 if model_type == 'direct_unet' else 2}_{model_type}_s0"
    exp_dir = args.exp_dir or os.path.join(EXP_DIR, config_name)
    os.makedirs(exp_dir, exist_ok=True)

    # `config/experiment/Q{1,2,3,4}_..._s0.yaml` is the source of truth for
    # `training.q_loss_weight`/`data.normalize`/`data.norm_stats_path`/
    # `model.param_dim`/`model.cond_extra_dim`/`data.cond_mode` -- CLI flags
    # below only override it when explicitly given.
    exp_cfg = OmegaConf.load(os.path.join(BASE, "config", "experiment", f"{config_name}.yaml"))
    obs_noise_std_frac = (args.obs_noise_std_frac if args.obs_noise_std_frac is not None
                          else float(exp_cfg.data.get("obs_noise_std_frac", 0.01)))
    init_lag_days = (args.init_lag_days if args.init_lag_days is not None
                     else float(exp_cfg.data.get("init_lag_days", 1.0)))
    gradient_clip_val = (args.gradient_clip_val if args.gradient_clip_val is not None
                        else float(exp_cfg.training.get("gradient_clip_val", 10.0)))
    q_loss_weight = (args.q_loss_weight if args.q_loss_weight is not None
                     else float(exp_cfg.training.q_loss_weight))
    do_normalize = (args.normalize if args.normalize is not None
                    else bool(exp_cfg.data.get("normalize", True)))
    norm_stats_path = (args.norm_stats_path or
                       exp_cfg.data.get("norm_stats_path", "experiments/qg_psi_norm_stats.pt"))
    norm = load_norm_stats(norm_stats_path) if do_normalize else None
    param_dim = int(exp_cfg.model.get("param_dim", 0))
    cond_extra_dim = int(exp_cfg.model.get("cond_extra_dim", 0))
    # model.fdv.* mirrors L96's train.py "fourdvarnet" dispatch (cfg.model.fdv)
    # -- OmegaConf DictConfig -> plain dict, since FourDVarNetSolver.__init__
    # takes plain Python kwargs, not OmegaConf nodes.
    fdv_kwargs = (OmegaConf.to_container(exp_cfg.model.fdv, resolve=True)
                 if model_type == "fourdvarnet" else None)
    include_ic = bool(args.include_ic or exp_cfg.data.get("include_ic", False))
    ic_dim = 2 if include_ic else 0
    cond_mode = args.cond_mode or exp_cfg.data.get("cond_mode", "none")
    noisy_max = (args.noisy_max if args.noisy_max is not None
                else float(exp_cfg.data.get("noisy_max", 1.5)))
    param_norm_stats_path = (args.param_norm_stats_path or
                             exp_cfg.data.get("param_norm_stats_path", None))
    if cond_mode != "none" and not param_norm_stats_path:
        raise ValueError(f"cond_mode={cond_mode!r} requires data.param_norm_stats_path "
                         "(see precompute_qg_norm_stats.py --output-params)")
    param_norm = load_norm_stats(param_norm_stats_path) if cond_mode != "none" else None
    forcing_norm_stats_path = (args.forcing_norm_stats_path or
                               exp_cfg.data.get("forcing_norm_stats_path", None))
    if cond_mode != "none" and not forcing_norm_stats_path:
        raise ValueError(f"cond_mode={cond_mode!r} requires data.forcing_norm_stats_path "
                         "(see precompute_qg_norm_stats.py --output-forcing -- leaving "
                         "the forcing field unnormalized collapsed a real training run, "
                         "see data/qg_neural.py's module docstring)")
    forcing_norm = load_norm_stats(forcing_norm_stats_path) if cond_mode != "none" else None
    s1_param_bias = (args.s1_param_bias if args.s1_param_bias is not None
                     else exp_cfg.data.get("s1_param_bias", None))
    s1_amp_bias = (args.s1_amp_bias if args.s1_amp_bias is not None
                  else exp_cfg.data.get("s1_amp_bias", None))
    cols_per_day_min = (args.cols_per_day_min if args.cols_per_day_min is not None
                       else exp_cfg.data.get("cols_per_day_min", None))
    cols_per_day_max = (args.cols_per_day_max if args.cols_per_day_max is not None
                       else exp_cfg.data.get("cols_per_day_max", None))
    if (cols_per_day_min is None) != (cols_per_day_max is None):
        raise ValueError("--cols-per-day-min/--cols-per-day-max must both be given "
                         "(or both omitted) to enable obs-density-augmented training")
    cols_per_day_range = ((int(cols_per_day_min), int(cols_per_day_max))
                         if cols_per_day_min is not None else None)
    results_path = os.path.join(exp_dir, "results.json")
    est_path = os.path.join(exp_dir, "estimates_s0.npz")

    # `num_windows` must match the split's window count: `_truth_cache_path`
    # hashes the *whole* QGConfig (asdict), and `num_windows` is one of its
    # fields, so a mismatched default there silently misses a cache keyed
    # with the matching value (`generate_qg_window_chunk.py` always sets it
    # equal to the split size) and falls back to a full from-scratch rollout.
    test_cfg = build_cfg(nx=args.nx, seed=args.test_seed, num_windows=args.num_test,
                        obs_geometry=args.obs_geometry,
                        cols_per_day=args.cols_per_day,
                        obs_noise_std_frac=obs_noise_std_frac,
                        init_lag_days=init_lag_days,
                        s1_param_bias=s1_param_bias, s1_amp_bias=s1_amp_bias)
    state_dim = test_cfg.state_dim

    if os.path.exists(results_path) and args.eval_only is None:
        print(f"Results exist at {results_path}, skipping.")
        return

    # `test_cache_cfg` matches the production TEST cache's key exactly.
    # Unlike train_cfg/val_cfg below (whose cache was built with plain
    # QGConfig defaults, irrelevant since on_the_fly_obs overwrites them
    # regardless), the test cache was specifically corrected
    # (`fix_qg_test_obs_ic.py`) to have the real S0 reference-case obs/IC
    # protocol baked in -- obs_geometry="random_columns"/cols_per_day=4/
    # obs_noise_std_frac=0.01/init_lag_days=1.0, NOT QGConfig's raw class
    # defaults ("grid"/3/0.05/0.5) and NOT `test_cfg`'s own (possibly Q5-
    # overridden) values. Using either of those instead of the actual
    # baked-in key silently misses the cache and triggers a full
    # from-scratch truth rollout -- caught the hard way twice: first with
    # `test_cfg` directly (obvious once diagnosed), then again with a
    # nx/seed/num_windows-only `test_cache_cfg` that still didn't match
    # because QGConfig's plain defaults aren't the S0 reference values
    # either (see PLAN.md's 2026-09-12 note for both).
    test_cache_cfg = build_cfg(nx=args.nx, seed=args.test_seed, num_windows=args.num_test,
                              obs_geometry="random_columns", cols_per_day=4,
                              obs_noise_std_frac=0.01, init_lag_days=1.0)
    test_windows = ensure_truth_cache_redrawn(test_cache_cfg, test_cfg, args.num_test,
                                              args.cache_dir)
    if args.eval_only is None:
        # No obs-config overrides here: `_truth_cache_path` hashes the whole
        # QGConfig, and the pre-generated production truth was built with plain
        # QGConfig defaults for obs fields (only nx/dt/seed/num_windows set) --
        # overriding obs fields here would miss that cache and trigger a full
        # from-scratch rollout. `on_the_fly_obs` (below) discards whatever obs
        # this cache carries anyway, so its obs config is irrelevant.
        train_cfg = build_cfg(nx=args.nx, seed=args.train_seed, num_windows=args.num_train)
        val_cfg = build_cfg(nx=args.nx, seed=args.val_seed, num_windows=args.num_val)
        on_the_fly = not args.fixed_split_obs
        train_windows = ensure_truth_cache(train_cfg, args.num_train, args.cache_dir)
        val_windows = ensure_truth_cache(val_cfg, args.num_val, args.cache_dir)

        train_ds = QGNeuralDataset(train_windows, test_cfg, norm, on_the_fly_obs=on_the_fly,
                                   cond_mode=cond_mode, param_norm_stats=param_norm,
                                   noisy_max=noisy_max, forcing_norm_stats=forcing_norm,
                                   include_ic=include_ic,
                                   cols_per_day_range=cols_per_day_range)
        val_ds = QGNeuralDataset(val_windows, test_cfg, norm, on_the_fly_obs=on_the_fly,
                                 cond_mode=cond_mode, param_norm_stats=param_norm,
                                 noisy_max=noisy_max, forcing_norm_stats=forcing_norm,
                                 include_ic=include_ic)
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                                  collate_fn=qg_collate, num_workers=args.num_workers,
                                  persistent_workers=args.num_workers > 0)
        val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                                collate_fn=qg_collate, num_workers=args.num_workers,
                                persistent_workers=args.num_workers > 0)

    print(f"Device: {device}  model={model_type}  epochs={epochs}  state_dim={state_dim}"
          f"  q_loss_weight={q_loss_weight:.4e}")
    print(f"Test: {len(test_windows)}  eval_only={bool(args.eval_only)}")
    if norm is not None:
        print(f"psi norm stats ({norm_stats_path}): "
              f"psi1 mean={norm['mean'][0]:.4e} std={norm['std'][0]:.4e}  "
              f"psi2 mean={norm['mean'][1]:.4e} std={norm['std'][1]:.4e}")
    else:
        print("normalization disabled (--no-normalize)")

    model = build_model(model_type, test_cfg, param_dim=param_dim,
                       cond_extra_dim=cond_extra_dim, ic_dim=ic_dim,
                       fdv_kwargs=fdv_kwargs).to(device)

    total_train = 0.0
    if args.eval_only is None:
        tcfg = make_trainer_cfg(model_type, exp_dir, epochs, args.lr,
                               gradient_clip_val=gradient_clip_val)
        lit = QGNeuralLightning(model, model_type, norm, test_cfg,
                                q_loss_weight=q_loss_weight, lr=args.lr,
                                gradient_clip_val=gradient_clip_val,
                                use_cosine_scheduler=args.cosine_scheduler,
                                max_epochs=epochs)
        trainer = create_trainer(tcfg, 1)
        t0 = time.time()
        trainer.fit(lit, train_loader, val_loader)
        total_train = time.time() - t0
        ckpt = os.path.join(exp_dir, "stage1_best.pt")
        torch.save(lit.model.state_dict(), ckpt)
        print(f"Stage 1 done in {total_train:.1f}s, saved {ckpt}")
    else:
        loaded = torch.load(args.eval_only, map_location="cpu")
        state_dict = loaded["state_dict"] if isinstance(loaded, dict) and "state_dict" in loaded else loaded
        state_dict = {(k[6:] if k.startswith("model.") else k): v for k, v in state_dict.items()}
        model.load_state_dict(state_dict)
        print(f"Loaded checkpoint {args.eval_only}")

    model.eval()
    est_psi, est_rd = estimate_windows(model, test_windows, test_cfg, model_type,
                                       device, norm=norm, n_members=args.n_members,
                                       cond_mode=cond_mode, param_norm_stats=param_norm,
                                       noisy_max=noisy_max, forcing_norm_stats=forcing_norm,
                                       include_ic=include_ic)

    truth_psi = np.stack([psi_daily(w, test_cfg).numpy() for w in test_windows])
    truth_q = np.stack([q_daily(w, test_cfg).numpy() for w in test_windows])

    est_q = np.stack([
        psi_to_q(torch.tensor(est_psi[i], dtype=torch.float32), float(est_rd[i]),
                 test_cfg, device=device).cpu().numpy()
        for i in range(len(test_windows))
    ])

    rmse_psi, ev_psi = pooled_metrics(est_psi, truth_psi)
    rmse_q, ev_q = pooled_metrics(est_q, truth_q)

    np.savez_compressed(est_path,
                        estimates_psi=est_psi, truth_psi=truth_psi,
                        estimates_q=est_q, truth_q=truth_q, rd=est_rd)

    summ_psi = layer_summary(rmse_psi, ev_psi, test_cfg)
    summ_q = layer_summary(rmse_q, ev_q, test_cfg)

    result = {
        "model_type": model_type,
        "config": {"nx": test_cfg.nx, "state_dim": state_dim, "epochs": epochs,
                   "num_train_windows": args.num_train, "num_val_windows": args.num_val,
                   "num_test_windows": args.num_test,
                   "on_the_fly_split_obs": not args.fixed_split_obs,
                   "train_seed": args.train_seed, "val_seed": args.val_seed,
                   "test_seed": args.test_seed, "obs_geometry": test_cfg.obs_geometry,
                   "cols_per_day": test_cfg.cols_per_day,
                   "obs_noise_std_frac": test_cfg.obs_noise_std_frac,
                   "init_lag_days": test_cfg.init_lag_days,
                   "n_members": args.n_members, "q_loss_weight": q_loss_weight,
                   "normalize": do_normalize, "norm_stats_path": norm_stats_path,
                   "cond_mode": cond_mode, "param_dim": param_dim,
                   "cond_extra_dim": cond_extra_dim, "noisy_max": noisy_max,
                   "param_norm_stats_path": param_norm_stats_path,
                   "forcing_norm_stats_path": forcing_norm_stats_path,
                   "s1_param_bias": test_cfg.s1_param_bias,
                   "s1_amp_bias": test_cfg.s1_amp_bias,
                   "include_ic": include_ic, "ic_dim": ic_dim,
                   "gradient_clip_val": gradient_clip_val, "seed": args.seed,
                   "cols_per_day_min": cols_per_day_min,
                   "cols_per_day_max": cols_per_day_max},
        "norm": ({"psi1_mean": norm["mean"][0].item(), "psi1_std": norm["std"][0].item(),
                  "psi2_mean": norm["mean"][1].item(), "psi2_std": norm["std"][1].item()}
                 if norm is not None else None),
        "train_time_seconds": total_train,
        "s0": {"psi": summ_psi, "q": summ_q},
    }
    with open(results_path, "w") as f:
        json.dump(result, f, indent=2)

    print(f"\n=== S0 results ({model_type}) ===")
    p = summ_psi
    print(f"  PSI   pooled RMSE {p['pooled_rmse']:.6e}  EV {p['pooled_ev']:.4f}"
          f"   layer1 EV {p['layer1']['ev']:.4f}   layer2 EV {p['layer2']['ev']:.4f}")
    q = summ_q
    print(f"  PV-q  pooled RMSE {q['pooled_rmse']:.6e}  EV {q['pooled_ev']:.4f}"
          f"   layer1 EV {q['layer1']['ev']:.4f}   layer2 EV {q['layer2']['ev']:.4f}")
    print(f"  wrote {est_path} and {results_path}")


if __name__ == "__main__":
    main()
