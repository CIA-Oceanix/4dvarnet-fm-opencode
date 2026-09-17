#!/usr/bin/env python3
"""Blended-flow conditional sampler: mix an FM prior's flow with a deterministic one.

A deterministic estimator m_hat(y) IS a flow. Its operator is the constant map
Psi_det(x_tau) = m_hat, so by the Tweedie identity its velocity is

    v_det(x_tau, tau) = (m_hat - x_tau) / (1 - tau),

which integrates from x_0 ~ N(0, sigma0^2 I) to x_tau = (1-tau) x_0 + tau m_hat --
exactly the SDEdit warm-start state. Warm starting is therefore not a separate
trick but the DISCONTINUOUS member lam(tau) = 1{tau < tau0} of the one-parameter
family of blended velocities

    v_lam = (1 - lam(tau)) v_FM + lam(tau) v_det ,     lam(tau) = lam0 (1 - tau)^p .

lam0 = 0 recovers the cold (pure FM) sampler. p >= 1 is required: v_det carries a
1/(1-tau) factor, so lam/(1-tau) = lam0 (1-tau)^(p-1) stays bounded only for p >= 1.

Deliberately no post-hoc recentering: the point is what the blended flow itself
achieves, both in RMSE and in spread.
"""
import argparse
import json

import torch

from data.normalization import load_norm_stats, normalize
from evaluation.neural_inference import BatchDict, load_model, prepare_dataset
from reports.l96.fm_sampler_common import EnsembleScores, reliability_target


def sample(model, batch, r_vec, n_outer, gamma, n_members, sigma0,
           m_hat=None, lam0=0.0, p=1.0, tau0=None):
    """lam-blended flow with SDA-style variance-weighted guidance.

    ``tau0`` not None selects the hard-switch (warm-start) member instead.
    """
    B, T, _ = batch.obs.shape
    device = batch.obs.device
    dt = 1.0 / n_outer
    y = torch.nan_to_num(batch.obs, nan=0.0)
    tmask = batch.obs_mask.to(y.dtype).unsqueeze(-1)
    step0 = int(round(tau0 * n_outer)) if tau0 is not None else 0

    def _one():
        noise = torch.randn(B, T, model.state_dim, device=device) * sigma0
        if step0 > 0:
            t0 = step0 / n_outer
            x = (1.0 - t0) * noise + t0 * m_hat
        else:
            x = noise
        for step in range(step0, n_outer):
            tv = step / n_outer
            tau = torch.full((B,), tv, device=device)
            beta = max(tv, dt / 2.0)
            alpha2 = ((1.0 - tv) * sigma0) ** 2
            weight = 1.0 / (2.0 * (r_vec + gamma * alpha2 / beta ** 2))
            coef = dt * alpha2 / (beta * max(1.0 - tv, 1e-6))
            lam = 0.0 if (tau0 is not None or m_hat is None) else lam0 * (1.0 - tv) ** p
            with torch.enable_grad():
                x = x.detach().requires_grad_(True)
                v = model.forward(x, batch, tau)
                if lam > 0.0:
                    v_det = (m_hat - x) / max(1.0 - tv, 1e-6)
                    v = (1.0 - lam) * v + lam * v_det
                x_hat = x + (1.0 - tv) * v
                cost = ((((x_hat - y) ** 2) * tmask) * weight).sum()
                grad = torch.autograd.grad(cost, x)[0]
            x = (x + dt * v.detach() - coef * grad).detach()
        return x

    return torch.stack([_one() for _ in range(n_members)], dim=-1)


def main():
    q = argparse.ArgumentParser(description=__doc__)
    q.add_argument("--prior-ckpt", required=True)
    q.add_argument("--cond", action="store_true",
                   help="prior is an observation-conditional CFM (SDA2/SDA3)")
    q.add_argument("--mean-ckpt",
                   default="experiments/l96/L1b_monai_unet_s0s1_norm_obsdensity/checkpoints/stage1_best.ckpt")
    q.add_argument("--dataset", default="experiments/l96_datasets_obsj2_int100_nwin200.pt")
    q.add_argument("--norm-stats", default="experiments/l96_norm_stats_obsj2.pt")
    q.add_argument("--n-outer", type=int, default=50)
    q.add_argument("--gamma", type=float, default=0.01,
                   help="SDA Gamma; 1e-2 follows Rozet & Louppe")
    q.add_argument("--members", type=int, default=10)
    q.add_argument("--windows", type=int, default=200)
    q.add_argument("--chunk", type=int, default=10)
    q.add_argument("--r-var", type=float, default=0.5)
    q.add_argument("--tau0", type=float, default=0.3)
    q.add_argument("--lam0s", type=float, nargs="+", default=[0.5])
    q.add_argument("--ps", type=float, nargs="+", default=[1.0, 2.0])
    q.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    q.add_argument("--seed", type=int, default=0)
    q.add_argument("--output", required=True)
    args = q.parse_args()

    device = torch.device(args.device)
    prior, cfg = load_model(args.prior_ckpt, None, device=device)
    prior.eval()
    mean_model, _ = load_model(args.mean_ckpt, None, device=device)
    mean_model.eval()
    stats = {k: v.to(device) for k, v in load_norm_stats(args.norm_stats).items()}
    std = stats["std"].view(1, 1, -1)
    mean = stats["mean"].view(1, 1, -1)
    r_vec = args.r_var / (std.squeeze() ** 2)
    sigma0 = prior.sigma_prior
    n_members = args.members
    print(f"prior={type(prior).__name__} N_outer={args.n_outer} N={n_members} "
          f"windows={args.windows}", flush=True)

    _, loaders, obs_idx = prepare_dataset(cfg, args.dataset, batch_size=args.chunk,
                                          is_joint=args.cond)
    idx = torch.as_tensor(obs_idx, dtype=torch.long).to(device)

    specs = [("mean-only", None), ("cold(lam=0)", ("cold",)),
             (f"warm(tau0={args.tau0})", ("warm",))]
    specs += [(f"lam0={L:g},p={p:g}", ("lam", L, p))
              for L in args.lam0s for p in args.ps]
    scores = {name: EnsembleScores() for name, _ in specs}
    done = 0
    for batch_dict in loaders["s0"]:
        if done >= args.windows:
            break
        d = {k: (v.to(device) if torch.is_tensor(v) else v)
             for k, v in batch_dict.items()}
        truth = d["true_state"][..., idx].contiguous()
        d["obs"] = normalize(d["obs"], stats)
        batch = BatchDict(d)
        with torch.no_grad():
            m_hat = mean_model(batch)
        for name, spec in specs:
            if spec is None:
                members = m_hat.unsqueeze(-1)
            else:
                torch.manual_seed(args.seed)
                if spec[0] == "cold":
                    members = sample(prior, batch, r_vec, args.n_outer, args.gamma,
                                     n_members, sigma0)
                elif spec[0] == "warm":
                    members = sample(prior, batch, r_vec, args.n_outer, args.gamma,
                                     n_members, sigma0, m_hat=m_hat, tau0=args.tau0)
                else:
                    members = sample(prior, batch, r_vec, args.n_outer, args.gamma,
                                     n_members, sigma0, m_hat=m_hat,
                                     lam0=spec[1], p=spec[2])
            scores[name].add(members, truth, std, mean)
        done += truth.shape[0]
        print(f"  {done}/{args.windows}", flush=True)

    target = reliability_target(n_members)
    print(f"\ntarget spread/RMSE for N={n_members}: {target:.3f}")
    print(f"{'config':<20} {'rmse_repo':>10} {'rmse_pool':>10} {'spread':>8} {'ratio':>7}")
    out = {}
    for name, _ in specs:
        s = scores[name].summary()
        print(f"{name:<20} {s['rmse_repo']:10.4f} {s['rmse_pooled']:10.4f} "
              f"{s['spread']:8.4f} {s['ratio']:7.3f}")
        out[name] = s
    with open(args.output, "w") as fh:
        json.dump(dict(results=out, target=target, config=vars(args)), fh, indent=2)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
