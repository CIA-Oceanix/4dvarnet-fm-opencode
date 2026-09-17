#!/usr/bin/env python3
"""Run FM-prior (SDA-family) conditional-sampler configurations on L96.

One driver for every scheme in ``evaluation/fm_sampler.py``. Produces the JSON that
``generate_l96_fm_sampler_report.py`` renders into
``reports/l96/outputs/l96_fm_sampler_benchmark.md``.

Suites
------
``blend``      the (lam0, p) family: mean-only, cold, warm, and the blended flows.
               This is the suite behind the headline result.
``decoupled``  the decoupled mean/gain/non-Gaussian control -- a negative result,
               kept reproducible. Needs the unguided ``P_prior`` and ``mu_p``, so it
               costs one extra unguided ensemble per batch.
``all``        both.

Environment: requires the monai-enabled env for monai checkpoints (see the report's
reproduction section). Example::

    python reports/l96/run_sda_sampler_experiments.py --suite blend \\
      --prior-ckpt experiments/l96/SDA3_monai_cond_noisy_l96_norm/checkpoints/stage1_best.ckpt \\
      --cond --windows 200 --members 10 --output experiments/l96_fm_sampler_blend.json
"""
import argparse
import json
import time
from pathlib import Path

import torch

from data.normalization import load_norm_stats, normalize
from evaluation.fm_sampler import (
    Blend,
    Cold,
    Decoupled,
    EnsembleScores,
    SamplerConfig,
    SchemeContext,
    Warm,
    estimate_prior_gain_variance,
    prior_mean,
    reliability_target,
    sample,
)
from evaluation.neural_inference import BatchDict, load_model, prepare_dataset

MEAN_ONLY = "mean-only"


def build_suite(name: str, tau0: float) -> list:
    """(label, scheme) pairs; ``None`` marks the deterministic mean-only row."""
    blend = [
        (MEAN_ONLY, None),
        ("cold(lam0=0)", Cold()),
        (f"warm(tau0={tau0:g})", Warm(tau0=tau0)),
        ("lam0=0.5,p=1", Blend(lam0=0.5, p=1.0)),
        ("lam0=0.5,p=2", Blend(lam0=0.5, p=2.0)),
        ("lam0=0.9,p=1", Blend(lam0=0.9, p=1.0)),
        ("lam0=0.9,p=2", Blend(lam0=0.9, p=2.0)),
    ]
    decoupled = [
        ("decoup a0=.5,p=2", Decoupled(lam0=0.5, p=2.0, gain="target")),
        ("decoup a0=.9,p=2", Decoupled(lam0=0.9, p=2.0, gain="target")),
        ("blendK a0=.5,p=2", Decoupled(lam0=0.5, p=2.0, gain="target", scale_ng=True)),
    ]
    if name == "blend":
        return blend
    if name == "decoupled":
        return blend[:3] + [("lam0=0.5,p=2", Blend(lam0=0.5, p=2.0))] + decoupled
    return blend + decoupled


def main():
    q = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    q.add_argument("--prior-ckpt", required=True)
    q.add_argument("--cond", action="store_true",
                   help="prior is observation-conditional (SDA2/SDA3)")
    q.add_argument("--mean-ckpt",
                   default="experiments/l96/L1b_monai_unet_s0s1_norm_obsdensity/checkpoints/stage1_best.ckpt")
    q.add_argument("--dataset", default="experiments/l96_datasets_obsj2_int100_nwin200.pt")
    q.add_argument("--norm-stats", default="experiments/l96_norm_stats_obsj2.pt")
    q.add_argument("--suite", default="blend", choices=["blend", "decoupled", "all"])
    q.add_argument("--n-outer", type=int, default=50)
    q.add_argument("--gamma", type=float, default=1e-2,
                   help="SDA Gamma; 1e-2 follows Rozet & Louppe")
    q.add_argument("--members", type=int, default=10)
    q.add_argument("--windows", type=int, default=200)
    q.add_argument("--chunk", type=int, default=10)
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

    config = SamplerConfig(n_outer=args.n_outer, gamma=args.gamma,
                           members=args.members, seed=args.seed)
    suite = build_suite(args.suite, args.tau0)
    needs_gaussian = any(isinstance(s, Decoupled) for _, s in suite)
    print(f"prior={type(prior).__name__} suite={args.suite} N={args.members} "
          f"N_outer={args.n_outer} windows={args.windows} "
          f"gaussian_terms={needs_gaussian}", flush=True)

    _, loaders, obs_idx = prepare_dataset(cfg, args.dataset, batch_size=args.chunk,
                                          is_joint=args.cond)
    idx = torch.as_tensor(obs_idx, dtype=torch.long).to(device)

    scores = {label: EnsembleScores() for label, _ in suite}
    p_target = None
    done = 0
    t_start = time.time()
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

        ctx = SchemeContext(sigma0=sigma0, m_hat=m_hat)
        if needs_gaussian:
            ctx.mu_p = prior_mean(prior, batch, sigma0, seed=args.seed)
            ctx.p_prior = estimate_prior_gain_variance(
                prior, batch, args.n_outer, args.members, sigma0, seed=args.seed)
            if p_target is None:
                # oracle covariance target for the Decoupled probe
                p_target = float(((m_hat - (truth - mean) / std) ** 2).mean())
                print(f"  P_prior={ctx.p_prior:.3f} p_target={p_target:.4f}", flush=True)
            ctx.p_target = p_target

        for label, scheme in suite:
            if scheme is None:
                members = m_hat.unsqueeze(-1)
            else:
                torch.manual_seed(args.seed)
                members = sample(prior, batch, scheme, config, ctx, r_vec)
            scores[label].add(members, truth, std, mean)
        done += truth.shape[0]
        print(f"  {done}/{args.windows}  ({time.time() - t_start:.0f}s)", flush=True)

    target = reliability_target(args.members)
    print(f"\ntarget spread/RMSE for N={args.members}: {target:.3f}")
    print(f"{'config':<20} {'rmse_repo':>10} {'rmse_pool':>10} {'spread':>8} {'ratio':>7}")
    results = {}
    for label, scheme in suite:
        s = scores[label].summary()
        s["scheme"] = None if scheme is None else scheme.name
        print(f"{label:<20} {s['rmse_repo']:10.4f} {s['rmse_pooled']:10.4f} "
              f"{s['spread']:8.4f} {s['ratio']:7.3f}")
        results[label] = s

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(
        dict(results=results, target=target, windows_done=done,
             elapsed_s=round(time.time() - t_start, 1), config=vars(args)), indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
