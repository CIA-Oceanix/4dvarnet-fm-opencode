#!/usr/bin/env python3
"""Train a DirectUNet / VanillaCFM (tau=0) neural estimator on QG S0.

Dedicated entry point for the QG neural baseline (mirrors the L96 L1/L2 setup
on the QG case study). Unlike `train.py` (which hard-codes the L63/L96 data and
eval machinery), this drives the self-contained QG neural data wrapper
(`data/qg_neural.py`) and shared Lightning training pieces.

Supervision target: daily mean of the full 2-layer streamfunction (`psi_daily`,
30 days per window), with an optional auxiliary PV-q loss (`--q-loss-weight`).
Observations: upper-layer psi grid-expanded, daily aggregated, NaN-masked,
padded to the full state width. Per-layer normalization is computed on the
training split and recorded so eval maps back to physical units.

For the q-loss, the model's normalized psi estimate is de-normalized to physical
psi, spectrally inverted to PV (per-window `rd`), re-normalized, and MSE-matched
against the normalized daily-mean PV target. Because PV<->psi is a linear
spectral inversion this is a consistent auxiliary objective.

Usage:
    python train_qg_neural.py --model-type direct_unet --exp-dir experiments/Q1_direct_unet_s0
    python train_qg_neural.py --model-type vanilla_cfm   --exp-dir experiments/Q2_vanilla_cfm_s0
    python train_qg_neural.py --model-type direct_unet --q-loss-weight 0.0 --eval-only <ckpt>
"""
import argparse
import json
import os
import time

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from data.qg import QGConfig
from data.qg_neural import (
    QGNeuralDataset,
    QGNorm,
    compute_norm,
    denorm_state,
    ensure_truth_cache,
    layer_split,
    norm_q_from_psi,
    psi_daily,
    psi_to_q,
    q_daily,
    qg_collate,
)
from models.direct_unet import DirectUNet
from models.vanilla_cfm import VanillaCFM
from training.pipeline import create_trainer

BASE = os.path.dirname(os.path.abspath(__file__))
EXP_DIR = os.path.join(BASE, "experiments")


def build_cfg(**overrides) -> QGConfig:
    return QGConfig(**{k: v for k, v in overrides.items() if v is not None})


def build_model(model_type: str, state_dim: int) -> torch.nn.Module:
    if model_type == "direct_unet":
        return DirectUNet(state_dim=state_dim, param_dim=0, cond_extra_dim=0,
                          hidden_channels=[64, 128, 256])
    if model_type == "vanilla_cfm":
        return VanillaCFM(state_dim=state_dim, param_dim=0, cond_extra_dim=0,
                          hidden_channels=[64, 128, 256], time_emb_dim=64,
                          N_outer=10, sigma_prior=0.5, dropout=0.1,
                          train_tau_0_only=True)
    raise ValueError(f"unknown model_type {model_type!r}")


def epochs_for(model_type: str) -> int:
    return 200 if model_type == "direct_unet" else 400


class QGNeuralLightning(pl.LightningModule):
    """Combined psi (primary) + auxiliary PV-q loss trainer for QG neural models."""

    def __init__(self, model, model_type: str, norm: QGNorm, qg_cfg: QGConfig,
                 q_loss_weight: float = 0.1, lr: float = 1e-3,
                 gradient_clip_val: float = 10.0):
        super().__init__()
        self.model = model
        self.model_type = model_type
        self.norm = norm
        self.qg_cfg = qg_cfg
        self.q_loss_weight = q_loss_weight
        self.lr = lr
        self.gradient_clip_val = gradient_clip_val

    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), lr=self.lr)

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
        raise ValueError(f"unsupported model_type {self.model_type!r}")

    def _q_loss(self, est, batch) -> torch.Tensor:
        if self.q_loss_weight <= 0 or batch.states_q is None:
            return torch.zeros((), device=est.device)
        device = est.device
        losses = []
        for b in range(est.shape[0]):
            q_pred = norm_q_from_psi(est[b], float(batch.rd[b]), self.qg_cfg,
                                     batch.scale[b], device)
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


def make_trainer_cfg(model_type: str, exp_dir: str, epochs: int, lr: float):
    return OmegaConf.create({
        "training": {
            "stage1": {"epochs": epochs, "lr": lr, "gradient_clip_val": 10.0},
            "stage2": {"epochs": 0, "lr": lr, "gradient_clip_val": 10.0},
            "accelerator": "auto",
            "loss": {"use_gradient": False, "gradient_weight": 0.0},
        },
        "paths": {
            "checkpoint_dir": os.path.join(exp_dir, "checkpoints"),
            "outputs_dir": os.path.join(exp_dir, "logs"),
        },
    })


def estimate_windows(model, windows, cfg, model_type, device, n_members=1):
    """Return per-window physical psi estimates (W, days, 2*ny*nx) + per-window rd list."""
    dataset = QGNeuralDataset(windows, cfg)
    loader = DataLoader(dataset, batch_size=8, shuffle=False, collate_fn=qg_collate)
    rds = [float(dataset.rd(i)) for i in range(len(windows))]
    scales = [dataset.scale(i) for i in range(len(windows))]
    model = model.to(device)
    model.eval()
    estimates = []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            if model_type == "direct_unet":
                pred = model(batch)
            else:
                pred = model.sample(batch, N_outer=10)
                if n_members > 1:
                    members = [model.sample(batch, N_outer=10) for _ in range(n_members)]
                    pred = torch.stack(members).mean(dim=0)
            pred = pred.detach().cpu()
            s = torch.tensor([[sc.psi1, sc.psi2] for sc in scales], dtype=torch.float32)
            for b in range(pred.shape[0]):
                estimates.append(denorm_state(pred[b], cfg, s[b]))
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
    ap.add_argument("--model-type", choices=["direct_unet", "vanilla_cfm"], default="direct_unet")
    ap.add_argument("--exp-dir", default=None)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--q-loss-weight", type=float, default=0.1)
    ap.add_argument("--num-train", type=int, default=1000)
    ap.add_argument("--num-val", type=int, default=100)
    ap.add_argument("--num-test", type=int, default=200)
    ap.add_argument("--cache-dir", default="reports/qg_cache")
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--nx", type=int, default=None)
    ap.add_argument("--n-members", type=int, default=1)
    ap.add_argument("--eval-only", nargs="?", const="stage1_best.pt", default=None,
                    help="Path to a checkpoint; skip training and just evaluate.")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_type = args.model_type
    epochs = args.epochs if args.epochs is not None else epochs_for(model_type)
    exp_dir = args.exp_dir or os.path.join(EXP_DIR, f"Q{1 if model_type == 'direct_unet' else 2}_{model_type}_s0")
    os.makedirs(exp_dir, exist_ok=True)
    results_path = os.path.join(exp_dir, "results.json")
    est_path = os.path.join(exp_dir, "estimates_s0.npz")

    test_cfg = build_cfg(nx=args.nx)
    state_dim = test_cfg.state_dim

    if os.path.exists(results_path) and args.eval_only is None:
        print(f"Results exist at {results_path}, skipping.")
        return

    test_windows = ensure_truth_cache(test_cfg, args.num_test, args.cache_dir)
    if args.eval_only is None:
        train_cfg = build_cfg(nx=args.nx, seed=7)
        val_cfg = build_cfg(nx=args.nx, seed=99)
        train_windows = ensure_truth_cache(train_cfg, args.num_train, args.cache_dir)
        val_windows = ensure_truth_cache(val_cfg, args.num_val, args.cache_dir)
        norm = compute_norm(train_windows, test_cfg)

        train_ds = QGNeuralDataset(train_windows, test_cfg, norm)
        val_ds = QGNeuralDataset(val_windows, test_cfg, norm)
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                                  collate_fn=qg_collate, num_workers=1)
        val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                                collate_fn=qg_collate, num_workers=1)
    else:
        norm = QGNorm()

    print(f"Device: {device}  model={model_type}  epochs={epochs}  state_dim={state_dim}"
          f"  q_loss_weight={args.q_loss_weight}")
    print(f"Test: {len(test_windows)}  eval_only={bool(args.eval_only)}")
    print(f"norm: psi1={norm.psi1:.4e} psi2={norm.psi2:.4e} q1={norm.q1:.4e} "
          f"q2={norm.q2:.4e} obs={norm.obs:.4e}")

    model = build_model(model_type, state_dim).to(device)

    total_train = 0.0
    if args.eval_only is None:
        tcfg = make_trainer_cfg(model_type, exp_dir, epochs, args.lr)
        lit = QGNeuralLightning(model, model_type, norm, test_cfg,
                                q_loss_weight=args.q_loss_weight, lr=args.lr,
                                gradient_clip_val=10.0)
        trainer = create_trainer(tcfg, 1)
        t0 = time.time()
        trainer.fit(lit, train_loader, val_loader)
        total_train = time.time() - t0
        ckpt = os.path.join(exp_dir, "stage1_best.pt")
        torch.save(lit.model.state_dict(), ckpt)
        print(f"Stage 1 done in {total_train:.1f}s, saved {ckpt}")
    else:
        model.load_state_dict(torch.load(args.eval_only, map_location="cpu"))
        print(f"Loaded checkpoint {args.eval_only}")

    model.eval()
    est_psi, est_rd = estimate_windows(model, test_windows, test_cfg, model_type,
                                       device, n_members=args.n_members)

    truth_psi = np.stack([psi_daily(w, test_cfg).numpy() for w in test_windows])
    truth_q = np.stack([q_daily(w, test_cfg).numpy() for w in test_windows])

    est_q = np.stack([
        psi_to_q(torch.tensor(est_psi[i], dtype=torch.float32), float(est_rd[i]),
                 test_cfg, device=device).cpu().numpy()
        for i in range(len(test_windows))
    ])

    rmse_psi, ev_psi = pooled_metrics(est_psi, truth_psi)
    rmse_q, ev_q = pooled_metrics(est_q, truth_q)

    test_scales = np.array([[s.psi1, s.psi2] for s in
                            [QGNeuralDataset(test_windows, test_cfg).scale(i)
                             for i in range(len(test_windows))]])

    np.savez_compressed(est_path,
                        estimates_psi=est_psi, truth_psi=truth_psi,
                        estimates_q=est_q, truth_q=truth_q, rd=est_rd,
                        psi_scale=test_scales)

    summ_psi = layer_summary(rmse_psi, ev_psi, test_cfg)
    summ_q = layer_summary(rmse_q, ev_q, test_cfg)

    result = {
        "model_type": model_type,
        "config": {"nx": test_cfg.nx, "state_dim": state_dim, "epochs": epochs,
                   "num_train_windows": args.num_train, "num_test_windows": args.num_test,
                   "n_members": args.n_members, "q_loss_weight": args.q_loss_weight},
        "norm": {"psi1": norm.psi1, "psi2": norm.psi2,
                 "q1": norm.q1, "q2": norm.q2, "obs": norm.obs},
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
