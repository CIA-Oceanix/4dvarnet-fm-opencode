"""tau-consistency probe for CFM posterior-mean operators D(x, tau, y) = E[x1 | x_tau, y].

Linear path x_tau = tau x1 + (1 - tau) x0, x0 ~ N(0, s0^2 I); b_tau = (1 - tau) s0.
Everything is computed in the model's normalized space (where s0 is defined).

  E1  orthogonality: slope of (x1 - D0) on (ODE mean - D0) -- 0 if D0 is MMSE.
  B2  state moments: member mean of x_tau vs tau * m, member var vs b^2 + tau^2 P.
  B4  total variance: Var_members D_tau + (b^2 / tau) diag(J_tau) constant in tau.
  A1  tau=0: ||J||_F / sqrt(n) should be 0.
  A3  tau->0+: (b^2 / tau) diag J -> P.
  C1  asymmetry ||(J - J^T) u|| / ||J u||.

J u by central finite differences (eps=1e-2 matches exact autograd to <0.1%),
J^T u by autograd. See docs/results/cfm_tau_consistency.md.
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

from data.normalization import load_norm_stats, normalize  # noqa: E402
from evaluation.archive import resolve_norm_stats  # noqa: E402
from evaluation.neural_inference import BatchDict, load_model, prepare_dataset  # noqa: E402
from models.vanilla_cfm import PredictStateCFM  # noqa: E402

NO = 8


def make_D(model: torch.nn.Module, batch, is_psc: bool):
    def D(x, tau_val):
        tau = torch.full((x.shape[0],), tau_val, device=x.device)
        out = model.forward(x, batch, tau)
        return out if is_psc else x + (1.0 - tau_val) * out
    return D


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--dataset", default="experiments/l96_datasets_obsj2_int100_nwin200.pt")
    p.add_argument("--case", default="s0")
    p.add_argument("--members", type=int, default=30)
    p.add_argument("--vjp-members", type=int, default=5)
    p.add_argument("--n-outer", type=int, default=10)
    p.add_argument("--fd-eps", type=float, default=1e-2)
    p.add_argument("--batch-size", type=int, default=50)
    p.add_argument("--max-batches", type=int, default=0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", required=True)
    a = p.parse_args()
    dev = torch.device("cuda")
    model, cfg = load_model(a.checkpoint, None, device=dev)
    model.eval()
    is_psc = isinstance(model, PredictStateCFM)
    s0 = float(model.sigma_prior)
    ns = load_norm_stats(str(resolve_norm_stats(cfg, required=True)))
    _, dls, obs_idx = prepare_dataset(cfg, a.dataset, batch_size=a.batch_size, norm_stats=ns)
    idx = torch.as_tensor(obs_idx, dtype=torch.long, device=dev)
    N, M, eps = a.n_outer, a.members, a.fd_eps
    taus = [k / N for k in range(N)]
    half = M // 2

    acc = {k: [] for k in ("var_x", "var_mu", "diagJ", "b2_gap", "b2_ref",
                           "rd_ab", "dd_ab", "rr_ab", "rd_full", "dd_full",
                           "mse_end", "mse_d0", "var_end", "dd_noise")}
    sym = {k: np.zeros(N) for k in ("asym", "Ju", "JTu")}
    Jfro = np.zeros(N)
    n_elem_sym = np.zeros(N)
    n_elem = 0

    torch.manual_seed(a.seed)
    for bi, bd in enumerate(dls[a.case]):
        if a.max_batches and bi >= a.max_batches:
            break
        batch = BatchDict({k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in bd.items()})
        D = make_D(model, batch, is_psc)
        truth = normalize(batch.true_state[..., idx].float(), {k: v.to(dev) for k, v in ns.items()})
        shp = (N,) + tuple(batch.obs.shape)
        S_x = torch.zeros(shp, device=dev)
        S_xx = torch.zeros(shp, device=dev)
        S_mu = torch.zeros(shp, device=dev)
        S_mumu = torch.zeros(shp, device=dev)
        S_uJu = torch.zeros(shp, device=dev)
        d0 = [torch.zeros_like(truth), torch.zeros_like(truth)]
        end = [torch.zeros_like(truth), torch.zeros_like(truth)]
        for i in range(M):
            h = 0 if i < half else 1
            x = torch.randn_like(batch.obs) * s0
            for k, tau in enumerate(taus):
                with torch.no_grad():
                    mu = D(x, tau)
                    u = torch.randint(0, 2, x.shape, device=dev).float() * 2 - 1
                    Ju = (D(x + eps * u, tau) - D(x - eps * u, tau)) / (2 * eps)
                S_x[k] += x
                S_xx[k] += x * x
                S_mu[k] += mu
                S_mumu[k] += mu * mu
                S_uJu[k] += u * Ju
                Jfro[k] += float((Ju ** 2).sum())
                if i < a.vjp_members:
                    with torch.enable_grad():
                        xg = x.detach().requires_grad_(True)
                        JTu = torch.autograd.grad(D(xg, tau), xg, u)[0]
                    sym["asym"][k] += float(((Ju - JTu) ** 2).sum())
                    sym["Ju"][k] += float((Ju ** 2).sum())
                    sym["JTu"][k] += float((JTu ** 2).sum())
                    n_elem_sym[k] += x.numel()
                if k == 0:
                    d0[h] += mu
                x = x + (1.0 / N) * (mu - x) / (1.0 - tau)
            end[h] += x
        cnt = [half, M - half]
        d0 = [d0[j] / cnt[j] for j in range(2)]
        end = [end[j] / cnt[j] for j in range(2)]
        d0_full = (d0[0] * cnt[0] + d0[1] * cnt[1]) / M
        end_full = (end[0] * cnt[0] + end[1] * cnt[1]) / M

        mean_x = S_x / M
        var_x = (S_xx / M - mean_x ** 2) * M / (M - 1)
        mean_mu = S_mu / M
        var_mu = (S_mumu / M - mean_mu ** 2) * M / (M - 1)
        diagJ = S_uJu / M
        tv = torch.tensor(taus, device=dev).view(N, 1, 1, 1)
        b2_gap = ((mean_x - tv * end_full.unsqueeze(0)) ** 2
                  - var_x / M - tv ** 2 * var_mu[-1].unsqueeze(0) / M)
        b2_ref = (tv * end_full.unsqueeze(0)) ** 2

        def tmean(z):
            return z.mean(dim=-2).cpu().numpy()

        acc["var_x"].append(tmean(var_x).transpose(1, 0, 2))
        acc["var_mu"].append(tmean(var_mu).transpose(1, 0, 2))
        acc["diagJ"].append(tmean(diagJ).transpose(1, 0, 2))
        acc["b2_gap"].append(tmean(b2_gap).transpose(1, 0, 2))
        acc["b2_ref"].append(tmean(b2_ref).transpose(1, 0, 2))
        rd = sum(((truth - d0[j]) * (end[1 - j] - d0[1 - j])) for j in range(2)) / 2
        dd = sum(((end[1 - j] - d0[1 - j]) ** 2) for j in range(2)) / 2
        rr = sum(((truth - d0[j]) ** 2) for j in range(2)) / 2
        acc["rd_ab"].append(tmean(rd))
        acc["dd_ab"].append(tmean(dd))
        acc["rr_ab"].append(tmean(rr))
        acc["rd_full"].append(tmean((truth - d0_full) * (end_full - d0_full)))
        acc["dd_full"].append(tmean((end_full - d0_full) ** 2))
        acc["mse_end"].append(tmean((truth - end_full) ** 2))
        acc["mse_d0"].append(tmean((truth - d0_full) ** 2))
        acc["var_end"].append(tmean(var_mu[-1]))
        acc["dd_noise"].append(tmean((var_mu[-1] + var_mu[0]) / half))
        n_elem += truth.numel()
        print(f"batch {bi} done", flush=True)

    A = {k: np.concatenate(v, 0) for k, v in acc.items()}
    W = A["mse_end"].shape[0]
    groups = {"all_obs": slice(None), "slow": slice(0, NO), "obs_fast": slice(NO, None)}

    def gm(arr, g):
        return float(arr[..., groups[g]].mean())

    res = dict(checkpoint=a.checkpoint, model=type(model).__name__, case=a.case, members=M,
               n_outer=N, sigma0=s0, fd_eps=eps, n_windows=W, taus=taus, space="normalized")
    for g in groups:
        r = {}
        P_emp = gm(A["mse_end"], g) / (1 + 1 / M)
        r["rmse_end"] = gm(A["mse_end"], g) ** 0.5
        r["rmse_d0"] = gm(A["mse_d0"], g) ** 0.5
        r["var_end_members"] = gm(A["var_end"], g)
        r["P_from_truth"] = P_emp
        r["E1_slope_split"] = gm(A["rd_ab"], g) / gm(A["dd_ab"], g)
        r["E1_slope_split_noisecorr"] = gm(A["rd_ab"], g) / (gm(A["dd_ab"], g) - gm(A["dd_noise"], g))
        r["E1_slope_full"] = gm(A["rd_full"], g) / gm(A["dd_full"], g)
        r["E1_corr_split"] = gm(A["rd_ab"], g) / (gm(A["rr_ab"], g) * gm(A["dd_ab"], g)) ** 0.5
        slopes = []
        rng = np.random.default_rng(0)
        for _ in range(200):
            bs = rng.integers(0, W, W)
            slopes.append(A["rd_ab"][bs][..., groups[g]].mean() / A["dd_ab"][bs][..., groups[g]].mean())
        r["E1_slope_split_ci95"] = [float(np.percentile(slopes, 2.5)), float(np.percentile(slopes, 97.5))]
        vx = [gm(A["var_x"][:, k], g) for k in range(N)]
        vmu = [gm(A["var_mu"][:, k], g) for k in range(N)]
        dj = [gm(A["diagJ"][:, k], g) for k in range(N)]
        b2 = [((1 - t) * s0) ** 2 for t in taus]
        r["B2_mean_rel_bias_sq"] = [None] + [gm(A["b2_gap"][:, k], g) / gm(A["b2_ref"][:, k], g)
                                             for k in range(1, N)]
        r["B2_var_x"] = vx
        r["B2_var_x_pred_selfP"] = [b2[k] + taus[k] ** 2 * r["var_end_members"] for k in range(N)]
        r["B2_var_x_pred_truthP"] = [b2[k] + taus[k] ** 2 * P_emp for k in range(N)]
        r["B4_var_mu"] = vmu
        r["B4_jac_term"] = [None] + [b2[k] / taus[k] * dj[k] for k in range(1, N)]
        r["B4_total"] = [None] + [vmu[k] + b2[k] / taus[k] * dj[k] for k in range(1, N)]
        r["diagJ_mean"] = dj
        r["A3_ratio_selfP_tau0.1"] = r["B4_jac_term"][1] / r["var_end_members"]
        r["A3_ratio_truthP_tau0.1"] = r["B4_jac_term"][1] / P_emp
        res[g] = r
    res["A1_C1"] = dict(
        J_fro_rms=[float((Jfro[k] / (n_elem * M)) ** 0.5) for k in range(N)],
        asym_rel=[float((sym["asym"][k] / sym["Ju"][k]) ** 0.5) for k in range(N)],
        JTu_over_Ju=[float((sym["JTu"][k] / sym["Ju"][k]) ** 0.5) for k in range(N)],
    )
    json.dump(res, open(a.out, "w"), indent=1)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
