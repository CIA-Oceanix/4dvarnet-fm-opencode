#!/usr/bin/env python3
"""Psi_mean / Psi_G / Psi_NG decomposition as a function of observation density.

Hypothesis under test: the canonical L96 observation configuration conditions the
posterior so strongly that the non-Gaussian component Psi_NG has little room to
matter. Thinning the fast-Y observations should widen the posterior and raise
Psi_NG's share.

Decomposition (Tweedie / flow-matching operator partition), with the linear
interpolant x_tau = (1-tau)*x0 + tau*x1:

    Psi(x_tau,y,tau) = E[x1|x_tau,y] = Psi_mean + Psi_G + Psi_NG
      Psi_mean = m(y) = E[x1|y]
      Psi_G    = K_tau * (x_tau - tau*m(y))
      Psi_NG   = residual (orthogonal to span{m, x_tau} by construction)

K_tau is fitted per tau by least squares over all (window, time, channel)
entries; the constrained form A = 1 - tau*K is used for the components, having
been verified to hold to <0.015 (see docs/results/l96_cfm_affine_velocity_decomposition.md).

Key reported quantity: ||Psi_NG|| / ||Psi_G||, the non-Gaussian component
relative to the Gaussian one, versus observation density.
"""
import argparse
import json

import numpy as np
import torch

from data.obs_density import apply_density_mask_to_obs, fast_channel_keep_mask
from evaluation.neural_inference import BatchDict, load_model, prepare_dataset
from models.vanilla_cfm import TweedieCFM


def solve2(S_mm, S_mx, S_xx, S_mmu, S_xmu):
    A = np.array([[S_mm, S_mx], [S_mx, S_xx]], dtype=np.float64)
    r = np.array([S_mmu, S_xmu], dtype=np.float64)
    a, b = np.linalg.solve(A, r)
    return float(a), float(b)



def psi_of(model, x_tau, batch, tau_scalar, tau_vec, mean_cache=None):
    """Assemble Psi(x_tau, y, tau) = E[x1|x_tau,y] for either interface.

    PredictStateCFM / FourDVarNetPredictStateCFM: the network returns mu directly.
    TweedieCFM: two-stage. The residual interpolant re-centers the standard one,
    r_tau = x_tau - tau*mean, since
        r_tau = (1-tau)x0 + tau*(x1-mean)  <=>  r_tau + tau*mean = (1-tau)x0 + tau*x1.
    Stage 2 predicts v = E[(x1-mean) - x0 | r_tau, obs, mean], so by the x1_hat
    identity in residual space E[x1-mean | .] = r_tau + (1-tau)v, giving
        Psi = mean + r_tau + (1-tau)*v.
    """
    if isinstance(model, TweedieCFM):
        mean = mean_cache if mean_cache is not None else model.estimate_mean(batch.obs)
        r_tau = x_tau - tau_scalar * mean
        v = model.forward(r_tau, batch.obs, mean, tau_vec)
        return mean + r_tau + (1.0 - tau_scalar) * v
    return model.forward(x_tau, batch, tau_vec)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint",
                   default="experiments/V3_predict_state_cfm_l96/checkpoints/stage1_best.ckpt")
    p.add_argument("--config", default=None,
                   help="config path/name; required for two-stage TweedieCFM")
    p.add_argument("--dataset", default="experiments/l96_datasets_obsj2_int100_nwin200.pt")
    p.add_argument("--case", default="s0", choices=["s0", "s1"])
    p.add_argument("--keep-k", type=int, nargs="+", default=[16, 8, 4, 0])
    p.add_argument("--batch-size", type=int, default=25)
    p.add_argument("--m-draws", type=int, default=4)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output", default="psi_decomposition_vs_density.json")
    args = p.parse_args()

    device = torch.device(args.device)
    model, cfg = load_model(args.checkpoint, args.config, device=device)
    model.eval()
    sigma = float(model.sigma_prior)
    print(f"model={type(model).__name__} state_dim={model.state_dim} sigma_prior={sigma}")

    dataset, dataloaders, obs_var_indices = prepare_dataset(
        cfg, args.dataset, batch_size=args.batch_size)
    idx = torch.as_tensor(obs_var_indices, dtype=torch.long).to(device)
    loader = dataloaders[args.case]

    taus = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]
    results = {}

    for keep_k in args.keep_k:
        torch.manual_seed(args.seed)
        acc = {t: dict(mm=0.0, mx=0.0, xx=0.0, mmu=0.0, xmu=0.0, mumu=0.0,
                       vv=0.0, n=0) for t in taus}
        m_err_sq = 0.0
        x1_sq = 0.0
        n_tot = 0

        with torch.no_grad():
            for bd in loader:
                batch_d = {k: (v.to(device) if torch.is_tensor(v) else v)
                           for k, v in bd.items()}
                x1 = batch_d["true_state"][..., idx].contiguous()
                B, T, D = x1.shape

                if keep_k < 16:
                    mask = fast_channel_keep_mask(B, T, keep_k, device=device)
                    batch_d["obs"] = apply_density_mask_to_obs(batch_d["obs"], mask)
                batch = BatchDict(batch_d)

                tau0 = torch.zeros(B, device=device)
                if isinstance(model, TweedieCFM):
                    # native, explicit Psi_mean -- no tau=0 proxy needed
                    m = model.estimate_mean(batch.obs)
                    mean_cache = m
                else:
                    mean_cache = None
                    ms = [psi_of(model, torch.randn_like(x1) * sigma, batch, 0.0, tau0)
                          for _ in range(args.m_draws)]
                    m = torch.stack(ms, 0).mean(0)
                m_err_sq += float(((m - x1) ** 2).sum())
                x1_sq += float((x1 ** 2).sum())
                n_tot += B * T * D

                for t in taus:
                    tt = torch.full((B,), float(t), device=device)
                    x0 = torch.randn_like(x1) * sigma
                    x_tau = (1.0 - t) * x0 + t * x1
                    mu = psi_of(model, x_tau, batch, t, tt, mean_cache)
                    a = acc[t]
                    a["mm"] += float((m * m).sum())
                    a["mx"] += float((m * x_tau).sum())
                    a["xx"] += float((x_tau * x_tau).sum())
                    a["mmu"] += float((m * mu).sum())
                    a["xmu"] += float((x_tau * mu).sum())
                    a["mumu"] += float((mu * mu).sum())
                    a["vv"] += float(((mu - x_tau) ** 2).sum())
                    a["n"] += B * T * D

        p_hat = m_err_sq / n_tot
        rows = []
        print(f"\n### keep_k={keep_k}   m(y) RMSE={np.sqrt(p_hat):.4f}  p_hat={p_hat:.4f}")
        print(f"{'tau':>5} {'K_fit':>8} {'|Psi_m|':>9} {'|Psi_G|':>9} {'|Psi_NG|':>9} "
              f"{'NG/G':>8} {'NG/|Psi|':>9} {'NG/|v|':>8}")
        for t in taus:
            a = acc[t]
            n = a["n"]
            A_fit, K = solve2(a["mm"], a["mx"], a["xx"], a["mmu"], a["xmu"])
            rss = max(a["mumu"] - A_fit * a["mmu"] - K * a["xmu"], 0.0)

            # canonical components under the constrained form A = 1 - tau*K
            # ||Psi_mean||^2 = S_mm ; ||Psi_G||^2 = K^2 * ||x_tau - tau*m||^2
            z_sq = a["xx"] - 2 * t * a["mx"] + (t ** 2) * a["mm"]
            psi_m = np.sqrt(a["mm"] / n)
            psi_G = abs(K) * np.sqrt(max(z_sq, 0.0) / n)
            psi_NG = np.sqrt(rss / n)
            psi_tot = np.sqrt(a["mumu"] / n)
            v_norm = np.sqrt(a["vv"] / n)

            ng_over_g = psi_NG / psi_G if psi_G > 0 else float("nan")
            print(f"{t:5.2f} {K:8.4f} {psi_m:9.4f} {psi_G:9.4f} {psi_NG:9.4f} "
                  f"{ng_over_g:8.4f} {psi_NG / psi_tot:9.4f} {psi_NG / v_norm:8.4f}")
            rows.append(dict(tau=t, K=K, A=A_fit, psi_mean=psi_m, psi_G=psi_G,
                             psi_NG=psi_NG, psi_total=psi_tot, v_norm=v_norm,
                             ng_over_g=ng_over_g,
                             ng_over_psi=float(psi_NG / psi_tot),
                             ng_over_v=float(psi_NG / v_norm)))

        results[str(keep_k)] = dict(keep_k=keep_k, m_rmse=float(np.sqrt(p_hat)),
                                    p_hat=float(p_hat), rows=rows)

    with open(args.output, "w") as f:
        json.dump(dict(checkpoint=args.checkpoint, case=args.case,
                       sigma_prior=sigma, by_keep_k=results), f, indent=2)
    print(f"\nwrote {args.output}")

    print("\n=== summary: max over tau of ||Psi_NG||/||Psi_G|| ===")
    print(f"{'keep_k':>7} {'m_RMSE':>8} {'p_hat':>8} {'max NG/G':>10} {'@tau':>6} {'mean NG/G':>10}")
    for k, r in results.items():
        vals = [(row["ng_over_g"], row["tau"]) for row in r["rows"]]
        best = max(vals)
        mean_ng_g = float(np.mean([v for v, _ in vals]))
        print(f"{k:>7} {r['m_rmse']:8.4f} {r['p_hat']:8.4f} {best[0]:10.4f} "
              f"{best[1]:6.2f} {mean_ng_g:10.4f}")


if __name__ == "__main__":
    main()
