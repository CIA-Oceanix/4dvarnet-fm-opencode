"""Per-step posterior-mean predictions along the CFM Euler path.

Records x1_hat_k = D(x_tau_k, tau_k, y) = E[x1 | x_tau, y] at every Euler step for
M members and scores, per step: the member-averaged prediction (the implied
E[x1|y] at that tau), single members, their spread, the cross-tau error
correlation, uniform / cross-validated optimal averages over tau, and the
endpoint RMSE vs step count. Pooled all_obs RMSE in physical units.
See docs/results/l96_cfm_tau_consistency.md.

Opt-in extras (docs/results/l96_cfm_tau_consistency_ns1.md), off by default so the
default output is unchanged: --x0-zero (NS0a, one deterministic path from x0=0)
and --steps-grid/--members-grid (NS0c, endpoint ensemble-mean RMSE per N steps x
M members, cost N*M calls). --skip-main runs only the extras.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.normalization import denormalize, load_norm_stats  # noqa: E402
from evaluation.archive import resolve_norm_stats  # noqa: E402
from evaluation.neural_inference import BatchDict, load_model, prepare_dataset  # noqa: E402
from models.vanilla_cfm import PredictStateCFM  # noqa: E402


def mu_hat(model: torch.nn.Module, x: torch.Tensor, batch, tau_val: float, is_psc: bool) -> torch.Tensor:
    B = x.shape[0]
    tau = torch.full((B,), tau_val, device=x.device)
    out = model.forward(x, batch, tau)
    return out if is_psc else x + (1.0 - tau_val) * out


def run_path(model: torch.nn.Module, batch, x0: torch.Tensor, N: int, is_psc: bool,
             record: bool) -> tuple:
    x = x0
    mus = []
    for k in range(N):
        tau_k = k / N
        mu = mu_hat(model, x, batch, tau_k, is_psc)
        if record:
            mus.append(mu)
        x = x + (1.0 / N) * (mu - x) / max(1.0 - tau_k, 1e-3)
    return x, mus


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--dataset", default="experiments/l96_datasets_obsj2_int100_nwin200.pt")
    p.add_argument("--case", default="s0")
    p.add_argument("--members", type=int, default=30)
    p.add_argument("--n-outer", type=int, default=10)
    p.add_argument("--n-sweep", default="1,2,3,5,10,20")
    p.add_argument("--batch-size", type=int, default=25)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--x0-zero", action="store_true")
    p.add_argument("--steps-grid", default="")
    p.add_argument("--members-grid", default="")
    p.add_argument("--skip-main", action="store_true")
    p.add_argument("--out", required=True)
    a = p.parse_args()
    dev = torch.device("cuda")
    model, cfg = load_model(a.checkpoint, None, device=dev)
    model.eval()
    is_psc = isinstance(model, PredictStateCFM)
    sigma = float(model.sigma_prior)
    ns = load_norm_stats(str(resolve_norm_stats(cfg, required=True))) if cfg.data.get("normalize", False) else None
    _, dls, obs_idx = prepare_dataset(cfg, a.dataset, batch_size=a.batch_size, norm_stats=ns)
    idx = torch.as_tensor(obs_idx, dtype=torch.long, device=dev)
    N = a.n_outer
    sweep = [int(s) for s in a.n_sweep.split(",")]
    dn = (lambda t: denormalize(t.float().cpu().numpy(), ns)) if ns is not None else (lambda t: t.float().cpu().numpy())

    torch.manual_seed(a.seed)
    per_window_err = []
    member_sq = np.zeros(N)
    member_spread_sq = np.zeros(N)
    sweep_sq = {n: 0.0 for n in sweep}
    n_pts = 0
    steps_grid = [int(v) for v in a.steps_grid.split(",")] if a.steps_grid else []
    members_grid = [int(v) for v in a.members_grid.split(",")] if a.members_grid else []
    zero_mu_sq = np.zeros(N)
    zero_end_sq = 0.0
    grid_sq = {(n, m): 0.0 for n in steps_grid for m in members_grid}
    with torch.no_grad():
        for bd in dls[a.case]:
            batch = BatchDict({k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in bd.items()})
            truth = batch.true_state[..., idx].float().cpu().numpy()
            if a.x0_zero:
                xN, mus = run_path(model, batch, torch.zeros_like(batch.obs), N, is_psc, record=True)
                zero_mu_sq += np.array([((dn(m) - truth) ** 2).sum() for m in mus])
                zero_end_sq += float(((dn(xN) - truth) ** 2).sum())
            for n in steps_grid:
                acc = 0.0
                for i in range(max(members_grid)):
                    x0 = torch.randn_like(batch.obs) * sigma
                    xN, _ = run_path(model, batch, x0, n, is_psc, record=False)
                    acc = acc + dn(xN)
                    if i + 1 in members_grid:
                        grid_sq[(n, i + 1)] += float((((acc / (i + 1)) - truth) ** 2).sum())
            if a.skip_main:
                n_pts += truth.size
                print("batch done", flush=True)
                continue
            mus_all = []
            for _ in range(a.members):
                x0 = torch.randn_like(batch.obs) * sigma
                _, mus = run_path(model, batch, x0, N, is_psc, record=True)
                mus_all.append(np.stack([dn(m) for m in mus], 0))
            mus_all = np.stack(mus_all, 0)
            ens_mean = mus_all.mean(0)
            per_window_err.append(ens_mean - truth[None])
            member_sq += ((mus_all - truth[None, None]) ** 2).mean(axis=(0, 2, 3, 4)) * truth.size
            member_spread_sq += mus_all.var(axis=0, ddof=1).mean(axis=(1, 2, 3)) * truth.size
            for n in sweep:
                acc = 0.0
                for _ in range(a.members):
                    x0 = torch.randn_like(batch.obs) * sigma
                    xN, _ = run_path(model, batch, x0, n, is_psc, record=False)
                    acc = acc + dn(xN)
                sweep_sq[n] += float((((acc / a.members) - truth) ** 2).sum())
            n_pts += truth.size
            print("batch done", flush=True)

    extras = {}
    if a.x0_zero:
        extras["x0_zero"] = dict(rmse_mu_k=np.sqrt(zero_mu_sq / n_pts).tolist(),
                                 rmse_endpoint=float(np.sqrt(zero_end_sq / n_pts)))
    if grid_sq:
        extras["steps_members_grid"] = [dict(n_steps=n, members=m, calls=n * m,
                                             rmse_ensmean=float(np.sqrt(v / n_pts)))
                                        for (n, m), v in sorted(grid_sq.items())]
    if a.skip_main:
        res = dict(checkpoint=a.checkpoint, model=type(model).__name__, case=a.case,
                   n_outer=N, **extras)
        json.dump(res, open(a.out, "w"), indent=1)
        print(json.dumps(res, indent=1))
        return

    E = np.concatenate(per_window_err, 1)
    K, W = E.shape[0], E.shape[1]
    Ef = E.reshape(K, W, -1)
    rmse_k = np.sqrt((Ef ** 2).mean(axis=(1, 2)))
    G = np.einsum("kwp,jwp->kj", Ef, Ef) / (W * Ef.shape[2])
    corr = G / np.sqrt(np.outer(np.diag(G), np.diag(G)))
    tower_gap = np.sqrt(((Ef - Ef[-1:]) ** 2).mean(axis=(1, 2)))
    uniform = np.sqrt((Ef.mean(0) ** 2).mean())

    def best_weights(idx_w):
        Gw = np.einsum("kwp,jwp->kj", Ef[:, idx_w], Ef[:, idx_w])
        w = np.linalg.solve(Gw + 1e-9 * np.eye(K), np.ones(K))
        return w / w.sum()

    rng = np.random.default_rng(0)
    perm = rng.permutation(W)
    fa, fb = perm[: W // 2], perm[W // 2:]
    cv = []
    for tr, te in ((fa, fb), (fb, fa)):
        w = best_weights(tr)
        cv.append(((np.tensordot(w, Ef[:, te], 1)) ** 2).mean())
    w_full = best_weights(np.arange(W))
    res = dict(
        checkpoint=a.checkpoint, model=type(model).__name__, case=a.case, members=a.members, n_outer=N,
        taus=[k / N for k in range(N)],
        rmse_ensmean_mu_k=rmse_k.tolist(),
        rmse_single_member_mu_k=np.sqrt(member_sq / n_pts).tolist(),
        member_spread_mu_k=np.sqrt(member_spread_sq / n_pts).tolist(),
        tower_gap_vs_final=tower_gap.tolist(),
        err_corr=corr.round(4).tolist(),
        rmse_uniform_avg_over_k=float(uniform),
        rmse_optimal_affine_comb_cv=float(np.sqrt(np.mean(cv))),
        optimal_weights_full=w_full.round(3).tolist(),
        rmse_final_vs_nsteps={str(n): float(np.sqrt(sweep_sq[n] / n_pts)) for n in sweep},
        **extras,
    )
    json.dump(res, open(a.out, "w"), indent=1)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
