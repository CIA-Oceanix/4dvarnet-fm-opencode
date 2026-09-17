#!/usr/bin/env python3
"""Decoupled mean/gain/non-Gaussian control of the FM operator -- NEGATIVE RESULT.

The lam-blend of ``run_blended_flow_sampler.py`` moves three things at once. Writing
the FM operator in the preprint's form Psi = mu_p + K_p (x_tau - beta mu_p) + Psi_NG,

    Psi_lam = (1-lam) Psi_FM + lam m_hat
            = [(1-lam) mu_p + lam m_hat] + (1-lam) K_p (x - beta mu_p) + (1-lam) Psi_NG

so blending toward a better mean ALSO shrinks the gain and the non-Gaussian term by
(1-lam) -- which is where the ensemble's dispersion goes. This probe gives each
component its own knob,

    a(tau)  mean blend       mu~  = (1-a) mu_p + a m_hat
    K~      gain             from a covariance model (K_hat here)
    c       non-Gaussian amplitude

realised with a single network call, Psi_NG recovered by subtraction:

    v = c v_FM + [ mu~ - c mu_p + (K~ - c K_p)(x - beta mu_p) + (c-1) x ] / (1 - tau)

Exact special cases, used as regression tests:
    cold      a=0,   K~=K_p,        c=1        -> v = v_FM
    lam-blend a=lam, K~=(1-lam)K_p, c=(1-lam)  -> v = (1-lam)v_FM + lam(m_hat-x)/(1-tau)

Measured outcome (10 windows, SDA3 prior + obsdensity mean): decoupling is WORSE on
both axes than the coupled blend -- ``decoup a0=.5`` 0.610/0.312 and ``blendK`` 0.752/
0.174 against the blend's 0.390/0.222. Psi_NG is defined relative to the pair
(mu_p, K_p) and is not invariant under a gain change, so re-using the network's
Psi_NG under a different K~ is inconsistent. Kept as the documented negative result
and as the regression harness for the general velocity form.
"""
import argparse
import json

import torch

from data.normalization import load_norm_stats, normalize
from evaluation.neural_inference import BatchDict, load_model, prepare_dataset
from reports.l96.fm_sampler_common import (
    EnsembleScores,
    estimate_prior_gain_variance,
    prior_gain,
    prior_mean,
    reliability_target,
)


def sample(model, batch, r_vec, n_outer, gamma, n_members, sigma0, mu_p, p_prior,
           m_hat=None, mode="cold", a0=0.0, p=2.0, k_target=None, tau0=None):
    B, T, _ = batch.obs.shape
    device = batch.obs.device
    dt = 1.0 / n_outer
    y = torch.nan_to_num(batch.obs, nan=0.0)
    tmask = batch.obs_mask.to(y.dtype).unsqueeze(-1)
    step0 = int(round(tau0 * n_outer)) if tau0 is not None else 0

    def _one():
        noise = torch.randn(B, T, model.state_dim, device=device) * sigma0
        x = ((1.0 - step0 / n_outer) * noise + (step0 / n_outer) * m_hat
             if step0 > 0 else noise)
        for step in range(step0, n_outer):
            tv = step / n_outer
            tau = torch.full((B,), tv, device=device)
            beta = max(tv, dt / 2.0)
            alpha2 = ((1.0 - tv) * sigma0) ** 2
            weight = 1.0 / (2.0 * (r_vec + gamma * alpha2 / beta ** 2))
            coef = dt * alpha2 / (beta * max(1.0 - tv, 1e-6))
            k_p = prior_gain(tv, p_prior, sigma0)
            lam = a0 * (1.0 - tv) ** p
            if mode == "cold":
                a, k_t, c = 0.0, k_p, 1.0
            elif mode == "blend":
                a, k_t, c = lam, (1.0 - lam) * k_p, (1.0 - lam)
            elif mode == "decoupled":
                a, k_t, c = lam, prior_gain(tv, k_target, sigma0), 1.0
            elif mode == "blendK":
                k_hat = prior_gain(tv, k_target, sigma0)
                a, k_t, c = lam, (1.0 - lam) * k_hat, (1.0 - lam)
            else:
                raise ValueError(f"unknown mode {mode!r}")
            with torch.enable_grad():
                x = x.detach().requires_grad_(True)
                v = model.forward(x, batch, tau)
                if mode != "cold":
                    mu_t = (1.0 - a) * mu_p + a * m_hat
                    extra = (mu_t - c * mu_p + (k_t - c * k_p) * (x - tv * mu_p)
                             + (c - 1.0) * x)
                    v = c * v + extra / max(1.0 - tv, 1e-6)
                x_hat = x + (1.0 - tv) * v
                cost = ((((x_hat - y) ** 2) * tmask) * weight).sum()
                grad = torch.autograd.grad(cost, x)[0]
            x = (x + dt * v.detach() - coef * grad).detach()
        return x

    return torch.stack([_one() for _ in range(n_members)], dim=-1)


def main():
    q = argparse.ArgumentParser(description=__doc__)
    q.add_argument("--prior-ckpt", required=True)
    q.add_argument("--cond", action="store_true")
    q.add_argument("--mean-ckpt",
                   default="experiments/l96/L1b_monai_unet_s0s1_norm_obsdensity/checkpoints/stage1_best.ckpt")
    q.add_argument("--dataset", default="experiments/l96_datasets_obsj2_int100_nwin200.pt")
    q.add_argument("--norm-stats", default="experiments/l96_norm_stats_obsj2.pt")
    q.add_argument("--n-outer", type=int, default=50)
    q.add_argument("--gamma", type=float, default=0.01)
    q.add_argument("--members", type=int, default=6)
    q.add_argument("--windows", type=int, default=10)
    q.add_argument("--chunk", type=int, default=5)
    q.add_argument("--r-var", type=float, default=0.5)
    q.add_argument("--tau0", type=float, default=0.3)
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

    _, loaders, obs_idx = prepare_dataset(cfg, args.dataset, batch_size=args.chunk,
                                          is_joint=args.cond)
    idx = torch.as_tensor(obs_idx, dtype=torch.long).to(device)

    specs = [("mean-only", None),
             ("cold", dict(mode="cold")),
             (f"warm(tau0={args.tau0})", dict(mode="cold", tau0=args.tau0)),
             ("blend a0=.5,p=2", dict(mode="blend", a0=0.5, p=2.0)),
             ("decoup a0=.5,p=2", dict(mode="decoupled", a0=0.5, p=2.0)),
             ("decoup a0=.9,p=2", dict(mode="decoupled", a0=0.9, p=2.0)),
             ("blendK a0=.5,p=2", dict(mode="blendK", a0=0.5, p=2.0))]
    scores = {name: EnsembleScores() for name, _ in specs}
    k_target = None
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
        mu_p = prior_mean(prior, batch, sigma0)
        p_prior = estimate_prior_gain_variance(prior, batch, args.n_outer,
                                               n_members, sigma0, seed=args.seed)
        if k_target is None:
            k_target = float(((m_hat - (truth - mean) / std) ** 2).mean())
            print(f"  P_prior={p_prior:.3f}  p_hat(oracle)={k_target:.4f}", flush=True)

        for name, spec in specs:
            if spec is None:
                members = m_hat.unsqueeze(-1)
            else:
                torch.manual_seed(args.seed)
                kw = dict(spec)
                mode = kw.pop("mode")
                members = sample(prior, batch, r_vec, args.n_outer, args.gamma,
                                 n_members, sigma0, mu_p, p_prior, m_hat=m_hat,
                                 mode=mode, k_target=k_target, **kw)
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
