#!/usr/bin/env python3
"""Gamma sweep for variance-weighted (PiGDM/SDA-style) guidance, with per-channel
observation noise and step-size instrumentation.

Fixes three things vs the first attempt:
  * per-channel sigma_y^2 (R_var/std_d^2) folded into the cost as weights
        w_d = 1 / (2 (sigma_{y,d}^2 + gamma r_tau^2))
    rather than a single scalar channel-mean;
  * the step-norm clamp is instrumented (and off by default) so we can see
    whether it was silently distorting the trajectory;
  * an UNGUIDED reference (guidance_weight=0) so we can tell "guidance too weak"
    from "guidance wrong".
"""
import argparse
import json
import numpy as np
import torch

from data.normalization import load_norm_stats, normalize
from evaluation.neural_inference import BatchDict, load_model, prepare_dataset
from evaluation.sda_sampler import sda_guided_sample     
from evaluation.sda_samplers_experimental import r_tau_sq


def pigdm_sample(model, batch, r_var_vec, N_outer, prior_var, gamma,
                 n_members, max_step_norm=None, stats=None):
    """Returns (members, diag) with diag = {step_rms, clamp_frac}."""
    B, T, _ = batch.obs.shape
    device = batch.obs.device
    dt = 1.0 / N_outer
    s0 = model.sigma_prior
    y = torch.nan_to_num(batch.obs, nan=0.0)
    tmask = batch.obs_mask.to(y.dtype).unsqueeze(-1)
    step_norms, clamp_hits, n_steps = [], 0, 0

    def _run_one():
        nonlocal clamp_hits, n_steps
        x = torch.randn(B, T, model.state_dim, device=device) * s0
        for step in range(N_outer):
            tv = step / N_outer
            tau = torch.full((B,), tv, device=device)
            beta = max(tv, dt / 2.0)
            a2 = ((1.0 - tv) * s0) ** 2
            r2 = r_tau_sq(tv, prior_var, s0)
            w_d = 1.0 / (2.0 * (r_var_vec + gamma * r2))          # (D,)
            coef = dt * a2 / (beta * max(1.0 - tv, 1e-6))
            with torch.enable_grad():
                x = x.detach().requires_grad_(True)
                v = model.forward(x, batch, tau)
                xh = x + (1.0 - tv) * v
                cost = ((((xh - y) ** 2) * tmask) * w_d).sum()
                grad = torch.autograd.grad(cost, x)[0]
            sv = -coef * grad
            nrm = sv.flatten(1).norm(dim=1)
            step_norms.append(float(nrm.mean()))
            n_steps += 1
            if max_step_norm is not None:
                over = (nrm > max_step_norm)
                clamp_hits += int(over.sum())
                sc = (max_step_norm / nrm.clamp_min(1e-12)).clamp(max=1.0)
                sv = sv * sc.view(B, *([1] * (sv.dim() - 1)))
            x = (x + dt * v.detach() + sv).detach()
        return x

    mem = torch.stack([_run_one() for _ in range(n_members)], dim=-1)
    diag = dict(step_rms=float(np.mean(step_norms)),
                clamp_frac=(clamp_hits / max(n_steps * B, 1)) if max_step_norm else 0.0)
    return mem, diag


def score(est, tru):
    return float(torch.sqrt(((est - tru) ** 2).mean(dim=(0, 1))).mean())


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint",
                   default="experiments/l96/SDA1_monai_prior_l96_norm/checkpoints/stage1_best.ckpt")
    p.add_argument("--dataset", default="experiments/l96_datasets_obsj2_int100_nwin200.pt")
    p.add_argument("--norm-stats", default="experiments/l96_norm_stats_obsj2.pt")
    p.add_argument("--windows", type=int, default=8)
    p.add_argument("--chunk", type=int, default=4)
    p.add_argument("--members", type=int, default=8)
    p.add_argument("--n-outer", type=int, default=10)
    p.add_argument("--r-var", type=float, default=0.5)
    p.add_argument("--gammas", type=float, nargs="+", default=[0.03, 0.1, 0.3, 1.0, 3.0])
    p.add_argument("--clamp", type=float, default=None)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output", default="pigdm_sweep.json")
    args = p.parse_args()

    device = torch.device(args.device)
    model, cfg = load_model(args.checkpoint, None, device=device)
    model.eval()
    stats = {k: v.to(device) for k, v in load_norm_stats(args.norm_stats).items()}
    std = stats["std"].view(1, 1, -1)
    r_var_vec = (args.r_var / (std.squeeze() ** 2))           # (D,) normalized-space sigma_y^2
    print(f"sigma_prior={model.sigma_prior}  R_var/std^2: min {float(r_var_vec.min()):.4f} "
          f"max {float(r_var_vec.max()):.4f} mean {float(r_var_vec.mean()):.4f}", flush=True)

    dataset, dls, obs_idx = prepare_dataset(cfg, args.dataset, batch_size=args.chunk)
    idx = torch.as_tensor(obs_idx, dtype=torch.long).to(device)

    configs = [("unguided(w=0)", None), ("baseline(w=40)", None)] + \
              [(f"pigdm g={g}", g) for g in args.gammas]
    agg = {name: dict(est=[], tru=[], sp=0.0, n=0, diag=[]) for name, _ in configs}

    done = 0
    for bd in dls["s0"]:
        if done >= args.windows:
            break
        d = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in bd.items()}
        tru = d["true_state"][..., idx].contiguous()
        d["obs"] = normalize(d["obs"], stats)
        batch = BatchDict(d)
        for name, g in configs:
            torch.manual_seed(args.seed)
            if g is None:
                w = 0.0 if "w=0" in name else 40.0
                mem, _ = sda_guided_sample(model, batch, R_var=float(r_var_vec.mean()),
                                           N_outer=args.n_outer, guidance_weight=w,
                                           n_members=args.members)
                diag = dict(step_rms=float("nan"), clamp_frac=0.0)
            else:
                mem, diag = pigdm_sample(model, batch, r_var_vec, args.n_outer,
                                         1.0, g, args.members, args.clamp, stats)
            m = mem.mean(-1) * std + stats["mean"].view(1, 1, -1)
            s = mem.std(-1) * std
            agg[name]["est"].append(m.cpu())
            agg[name]["tru"].append(tru.cpu())
            agg[name]["sp"] += float((s ** 2).sum())
            agg[name]["n"] += m.numel()
            agg[name]["diag"].append(diag)
        done += tru.shape[0]
        print(f"  {done}/{args.windows}", flush=True)

    print(f"\n{'config':<16} {'RMSE':>8} {'spread':>8} {'sp/RMSE':>8} {'step_rms':>10} {'clamp%':>7}")
    out = {}
    for name, _ in configs:
        a = agg[name]
        rmse = score(torch.cat(a["est"]), torch.cat(a["tru"]))
        sp = float(np.sqrt(a["sp"] / a["n"]))
        sr = float(np.nanmean([d["step_rms"] for d in a["diag"]]))
        cf = 100 * float(np.mean([d["clamp_frac"] for d in a["diag"]]))
        print(f"{name:<16} {rmse:8.4f} {sp:8.4f} {sp/rmse:8.3f} {sr:10.3f} {cf:7.1f}")
        out[name] = dict(rmse=rmse, spread=sp, ratio=sp / rmse, step_rms=sr, clamp_pct=cf)
    with open(args.output, "w") as f:
        json.dump(dict(results=out, config=vars(args)), f, indent=2)
    print(f"\nwrote {args.output}")


if __name__ == "__main__":
    main()
