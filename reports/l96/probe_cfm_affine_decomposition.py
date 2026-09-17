#!/usr/bin/env python3
"""How much of the CFM posterior mean mu = E[x1|x_tau,y] lives in span{m(y), x_tau}?

Under the linear interpolant x_tau = (1-tau)*x0 + tau*x1 (models/interpolant.py),
the exact velocity is v = (mu - x_tau)/(1-tau). Projecting mu onto the affine
span of the amortized mean m(y) := E[x1|y] and the current state x_tau gives

    mu(x_tau,tau,y) = a(tau)*m(y) + b(tau)*x_tau + rho(x_tau,tau,y)

with rho == 0 iff p(x1|y) is Gaussian with y-independent covariance. This probe
measures ||rho|| on a trained PredictStateCFM (V3) checkpoint, which predicts mu
directly, so no retraining is involved.

m(y) is taken as mu(x0, tau=0, y) averaged over several x0 draws -- at tau=0,
x_tau = x0 is independent of x1, so E[x1|x_tau,y] = E[x1|y] exactly.
"""
import argparse
import json

import numpy as np
import torch

from evaluation.neural_inference import BatchDict, load_model, prepare_dataset


def solve2(S_mm, S_mx, S_xx, S_mmu, S_xmu):
    """Least-squares (a,b) minimizing ||mu - a*m - b*x||^2, plus residual SS."""
    A = np.array([[S_mm, S_mx], [S_mx, S_xx]], dtype=np.float64)
    r = np.array([S_mmu, S_xmu], dtype=np.float64)
    a, b = np.linalg.solve(A, r)
    return float(a), float(b)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint",
                   default="experiments/V3_predict_state_cfm_l96/checkpoints/stage1_best.ckpt")
    p.add_argument("--dataset", default="experiments/l96_datasets_obsj2_int100_nwin200.pt")
    p.add_argument("--case", default="s0", choices=["s0", "s1"])
    p.add_argument("--batch-size", type=int, default=25)
    p.add_argument("--m-draws", type=int, default=4)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output", default="cfm_affine_decomposition.json")
    args = p.parse_args()

    device = torch.device(args.device)
    torch.manual_seed(args.seed)

    model, cfg = load_model(args.checkpoint, None, device=device)
    model.eval()
    sigma = float(model.sigma_prior)
    print(f"model={type(model).__name__} state_dim={model.state_dim} sigma_prior={sigma}")

    dataset, dataloaders, obs_var_indices = prepare_dataset(
        cfg, args.dataset, batch_size=args.batch_size)
    idx = torch.as_tensor(obs_var_indices, dtype=torch.long)
    loader = dataloaders[args.case]

    taus = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99]
    D = model.state_dim

    # scalar accumulators, per tau
    acc = {t: dict(mm=0.0, mx=0.0, xx=0.0, mmu=0.0, xmu=0.0, mumu=0.0, n=0) for t in taus}
    # per-channel accumulators, per tau
    accD = {t: {k: np.zeros(D) for k in ("mm", "mx", "xx", "mmu", "xmu", "mumu")} for t in taus}
    # m(y) quality
    m_err_sq = 0.0
    m_spread_sq = 0.0
    x1_sq = 0.0
    n_tot = 0

    with torch.no_grad():
        for bi, bd in enumerate(loader):
            batch = BatchDict({k: (v.to(device) if torch.is_tensor(v) else v)
                               for k, v in bd.items()})
            x1 = batch.true_state[..., idx.to(device)].contiguous()   # (B,T,D)
            B, T, _ = x1.shape
            tau0 = torch.zeros(B, device=device)

            # m(y) = E[x1|y] = mu at tau=0, averaged over x0 draws
            ms = []
            for _ in range(args.m_draws):
                x0 = torch.randn_like(x1) * sigma
                ms.append(model.forward(x0, batch, tau0))
            mstack = torch.stack(ms, 0)
            m = mstack.mean(0)
            m_spread_sq += float(mstack.var(0, unbiased=False).sum())
            m_err_sq += float(((m - x1) ** 2).sum())
            x1_sq += float((x1 ** 2).sum())
            n_tot += B * T * D

            for t in taus:
                tt = torch.full((B,), float(t), device=device)
                x0 = torch.randn_like(x1) * sigma
                x_tau = (1.0 - t) * x0 + t * x1
                mu = model.forward(x_tau, batch, tt)

                a = acc[t]
                a["mm"] += float((m * m).sum())
                a["mx"] += float((m * x_tau).sum())
                a["xx"] += float((x_tau * x_tau).sum())
                a["mmu"] += float((m * mu).sum())
                a["xmu"] += float((x_tau * mu).sum())
                a["mumu"] += float((mu * mu).sum())
                a["n"] += B * T * D

                d = accD[t]
                d["mm"] += (m * m).sum((0, 1)).double().cpu().numpy()
                d["mx"] += (m * x_tau).sum((0, 1)).double().cpu().numpy()
                d["xx"] += (x_tau * x_tau).sum((0, 1)).double().cpu().numpy()
                d["mmu"] += (m * mu).sum((0, 1)).double().cpu().numpy()
                d["xmu"] += (x_tau * mu).sum((0, 1)).double().cpu().numpy()
                d["mumu"] += (mu * mu).sum((0, 1)).double().cpu().numpy()

            print(f"  batch {bi}: B={B} done", flush=True)

    # posterior variance proxy p = E[(x1 - m)^2], noise variance s = sigma^2
    p_hat = m_err_sq / n_tot
    s = sigma ** 2
    m_rmse = float(np.sqrt(p_hat))
    print(f"\nm(y) RMSE vs truth = {m_rmse:.4f}  (p_hat={p_hat:.4f}, s=sigma^2={s:.4f})")
    print(f"m(y) x0-dispersion (std across {args.m_draws} draws) = "
          f"{np.sqrt(m_spread_sq / n_tot):.4f}")

    rows = []
    print(f"\n{'tau':>5} {'a_fit':>8} {'b_fit':>8} {'gamma_th':>9} {'1-tau*b':>8} "
          f"{'|rho|/|mu|':>11} {'|rho|/|mu-x|':>13} {'|rho|/|x1|':>11} {'perchan':>9}")
    for t in taus:
        a = acc[t]
        A, Bc = solve2(a["mm"], a["mx"], a["xx"], a["mmu"], a["xmu"])
        rss = a["mumu"] - A * a["mmu"] - Bc * a["xmu"]
        rss = max(rss, 0.0)
        S_vv = a["mumu"] - 2 * a["xmu"] + a["xx"]          # ||mu - x_tau||^2
        rel_mu = np.sqrt(rss / a["mumu"])
        rel_v = np.sqrt(rss / S_vv) if S_vv > 0 else float("nan")
        rel_x1 = np.sqrt(rss / x1_sq)

        # per-channel fit (a_d, b_d) -> residual
        d = accD[t]
        rss_d = 0.0
        for j in range(D):
            Ad, Bd = solve2(d["mm"][j], d["mx"][j], d["xx"][j], d["mmu"][j], d["xmu"][j])
            rss_d += max(d["mumu"][j] - Ad * d["mmu"][j] - Bd * d["xmu"][j], 0.0)
        rel_mu_d = np.sqrt(rss_d / a["mumu"])

        gam = t * p_hat / ((1 - t) ** 2 * s + t ** 2 * p_hat) if (t > 0 or s > 0) else 0.0
        print(f"{t:5.2f} {A:8.4f} {Bc:8.4f} {gam:9.4f} {1 - t * Bc:8.4f} "
              f"{rel_mu:11.4f} {rel_v:13.4f} {rel_x1:11.4f} {rel_mu_d:9.4f}")
        rows.append(dict(tau=t, a=A, b=Bc, gamma_theory=gam,
                         rel_resid_mu=float(rel_mu), rel_resid_v=float(rel_v),
                         rel_resid_x1=float(rel_x1), rel_resid_mu_perchannel=float(rel_mu_d)))

    out = dict(checkpoint=args.checkpoint, case=args.case, sigma_prior=sigma,
               m_rmse=m_rmse, p_hat=float(p_hat),
               m_x0_dispersion=float(np.sqrt(m_spread_sq / n_tot)),
               n_elements=n_tot, rows=rows)
    with open(args.output, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nwrote {args.output}")


if __name__ == "__main__":
    main()
