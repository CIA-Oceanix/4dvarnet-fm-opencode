#!/usr/bin/env python3
"""Sample-based Psi decomposition: estimate a method's implied operator from its
stored ensemble members, for schemes that do not expose Psi(x_tau, y, tau).

For a method whose output ensemble {x1^(n)}_{n=1..N} represents its implied
posterior q(x1|y), the tau-marginal of x_tau = beta*x1 + alpha*x0 is an exact
Gaussian mixture

    p_tau(z) = (1/N) sum_n N(z; beta*x1^(n), alpha^2)

so Tweedie gives the implied operator in closed form, with no KDE bandwidth:

    Psi~(z) = (z + alpha^2 * grad log p_tau(z)) / beta = sum_n w_n x1^(n),
    w_n = softmax_n( -(z - beta*x1^(n))^2 / (2 alpha^2) )

i.e. a softmax-weighted average of the members. The drawn member is excluded
(leave-one-out) so Psi~ is not evaluated on the sample that generated z.

IMPORTANT -- this is a **marginal (per-coordinate)** decomposition: each
(window, time, channel) entry is conditioned on its own z only, whereas a network
sees the whole x_tau field. Numbers are therefore comparable **across methods
measured this way**, but NOT directly against the query-based probe.
"""
import argparse
import json

import numpy as np
import torch


def solve2(S_mm, S_mx, S_xx, S_mmu, S_xmu):
    A = np.array([[S_mm, S_mx], [S_mx, S_xx]], dtype=np.float64)
    r = np.array([S_mmu, S_xmu], dtype=np.float64)
    a, b = np.linalg.solve(A, r)
    return float(a), float(b)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--members", required=True, help="path to members_s0.npz")
    p.add_argument("--label", default=None)
    p.add_argument("--sigma-prior", type=float, default=0.5)
    p.add_argument("--chunk", type=int, default=20, help="windows per chunk")
    p.add_argument("--max-windows", type=int, default=200)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output", required=True)
    args = p.parse_args()

    label = args.label or args.members
    device = torch.device(args.device)
    s0 = args.sigma_prior

    z = np.load(args.members, mmap_mode="r")
    members = z["members"]          # (W, T, D, N)
    truth = z["truth"]              # (W, T, D)
    W = min(args.max_windows, members.shape[0])
    N = members.shape[-1]
    print(f"{label}: members {members.shape} -> using {W} windows, N={N}")

    taus = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]
    acc = {t: dict(mm=0.0, mx=0.0, xx=0.0, mmu=0.0, xmu=0.0, mumu=0.0, vv=0.0, n=0)
           for t in taus}
    spread_sq = 0.0
    err_sq = 0.0
    n_tot = 0

    g = torch.Generator(device="cpu").manual_seed(args.seed)

    for start in range(0, W, args.chunk):
        stop = min(start + args.chunk, W)
        X = torch.from_numpy(np.ascontiguousarray(members[start:stop])).to(device)  # (w,T,D,N)
        Y = torch.from_numpy(np.ascontiguousarray(truth[start:stop])).to(device)    # (w,T,D)
        w, T, D, _ = X.shape

        m_tilde = X.mean(-1)                       # Psi~_mean = ensemble mean
        spread_sq += float(X.var(-1, unbiased=False).sum())
        err_sq += float(((m_tilde - Y) ** 2).sum())
        n_tot += w * T * D

        for t in taus:
            alpha = (1.0 - t) * s0
            pick = torch.randint(0, N, (w, T, D), generator=g).to(device)
            x1_draw = torch.gather(X, -1, pick.unsqueeze(-1)).squeeze(-1)
            x0 = torch.randn(x1_draw.shape, generator=g).to(device) * s0
            zz = t * x1_draw + (1.0 - t) * x0                      # (w,T,D)

            logw = -((zz.unsqueeze(-1) - t * X) ** 2) / (2.0 * alpha ** 2)
            logw.scatter_(-1, pick.unsqueeze(-1), float("-inf"))   # leave-one-out
            wgt = torch.softmax(logw, dim=-1)
            psi = (wgt * X).sum(-1)                                # Psi~(z)

            a = acc[t]
            a["mm"] += float((m_tilde * m_tilde).sum())
            a["mx"] += float((m_tilde * zz).sum())
            a["xx"] += float((zz * zz).sum())
            a["mmu"] += float((m_tilde * psi).sum())
            a["xmu"] += float((zz * psi).sum())
            a["mumu"] += float((psi * psi).sum())
            a["vv"] += float(((psi - zz) ** 2).sum())
            a["n"] += w * T * D

        del X, Y
        print(f"  windows {start}:{stop} done", flush=True)

    rows = []
    print(f"\n{'tau':>5} {'K_fit':>8} {'|Psi_m|':>9} {'|Psi_G|':>9} {'|Psi_NG|':>9} "
          f"{'NG/G':>8} {'NG/|Psi|':>9} {'NG/|v|':>8}")
    for t in taus:
        a = acc[t]
        n = a["n"]
        A_fit, K = solve2(a["mm"], a["mx"], a["xx"], a["mmu"], a["xmu"])
        rss = max(a["mumu"] - A_fit * a["mmu"] - K * a["xmu"], 0.0)
        z_sq = a["xx"] - 2 * t * a["mx"] + (t ** 2) * a["mm"]
        psi_m = np.sqrt(a["mm"] / n)
        psi_G = abs(K) * np.sqrt(max(z_sq, 0.0) / n)
        psi_NG = np.sqrt(rss / n)
        psi_tot = np.sqrt(a["mumu"] / n)
        v_norm = np.sqrt(a["vv"] / n)
        ng_g = psi_NG / psi_G if psi_G > 0 else float("nan")
        print(f"{t:5.2f} {K:8.4f} {psi_m:9.4f} {psi_G:9.4f} {psi_NG:9.4f} "
              f"{ng_g:8.4f} {psi_NG / psi_tot:9.4f} {psi_NG / v_norm:8.4f}")
        rows.append(dict(tau=t, K=K, A=A_fit, psi_mean=float(psi_m), psi_G=float(psi_G),
                         psi_NG=float(psi_NG), ng_over_g=float(ng_g),
                         ng_over_psi=float(psi_NG / psi_tot),
                         ng_over_v=float(psi_NG / v_norm)))

    out = dict(label=label, members=args.members, n_windows=W, n_members=N,
               sigma_prior=s0,
               ens_mean_rmse=float(np.sqrt(err_sq / n_tot)),
               ens_spread=float(np.sqrt(spread_sq / n_tot)), rows=rows)
    with open(args.output, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nens-mean RMSE {out['ens_mean_rmse']:.4f}  spread {out['ens_spread']:.4f}")
    print(f"mean NG/|Psi| {np.mean([r['ng_over_psi'] for r in rows]):.4f}  "
          f"mean NG/G {np.mean([r['ng_over_g'] for r in rows]):.4f}")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
