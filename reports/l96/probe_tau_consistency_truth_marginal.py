"""NS1: tau-consistency of the CFM operator on TRUE marginals (no ODE states).

D(x, tau, y) = E[x1 | x_tau = x, y]; path x_tau = tau x1 + (1 - tau) x0,
x0 ~ N(0, s0^2 I), b_t = (1 - t) s0. x_tau is built from the test-set truth x1,
i.e. exactly the distribution the network was trained on.

  NS1a  per-tau orthogonality: slope of the residual x1 - D(x_tau^true, tau, y) on
        g(y) = ODE ensemble mean - D0(y). 0 at every tau for an exact operator
        (g is (x_tau, y)-measurable); at tau=0 it is the E1 slope.
  NS1b  pairwise martingale: x_s is a further-noised copy of x_tau^true
        (pair_from_tau); E[D(x_tau, tau) - D(x_s, s) | x_s, y] = 0, tested as the
        slope of that difference on features of x_s: D(x_s, s) - D0 and x_s.
  NS1c  second order (MMSE + second-order Tweedie) on true marginals:
        E|x1 - D(x_tau)|^2 = (b_tau^2 / tau) E[tr grad_x D] / n, ratio 1 when exact;
        tr by Hutchinson with central finite-difference JVPs (eps = 1e-2).

Normalized space. See docs/results/cfm_tau_consistency_ns1.md and
docs/scoping/cfm_tau_consistency_next_steps.md (v3).
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
GROUPS = {"all_obs": slice(None), "slow": slice(0, NO), "obs_fast": slice(NO, None)}


def pair_from_tau(x_tau: torch.Tensor, tau: float, s: float, s0: float) -> torch.Tensor:
    """Further-noised copy x_s of x_tau (s < tau) with law N(s x1, b_s^2 I) given x1."""
    b_s, b_tau = (1.0 - s) * s0, (1.0 - tau) * s0
    ratio = s / tau
    extra = max(b_s ** 2 - (ratio * b_tau) ** 2, 0.0) ** 0.5
    return ratio * x_tau + extra * torch.randn_like(x_tau)


def make_D(model: torch.nn.Module, batch, is_psc: bool):
    def D(x: torch.Tensor, tau_val: float) -> torch.Tensor:
        tau = torch.full((x.shape[0],), tau_val, device=x.device)
        out = model.forward(x, batch, tau)
        return out if is_psc else x + (1.0 - tau_val) * out
    return D


def slope_stats(num: np.ndarray, den: np.ndarray, g: str, n_boot: int = 200) -> dict:
    """Pooled slope sum(num)/sum(den) over (W, D) arrays, with a window bootstrap CI."""
    nu, de = num[:, GROUPS[g]].mean(axis=1), den[:, GROUPS[g]].mean(axis=1)
    rng = np.random.default_rng(0)
    W = nu.shape[0]
    boots = []
    for _ in range(n_boot):
        i = rng.integers(0, W, W)
        boots.append(nu[i].mean() / de[i].mean())
    return dict(slope=float(nu.mean() / de.mean()),
                ci95=[float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))])


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--dataset", default="experiments/l96_datasets_obsj2_int100_nwin200.pt")
    p.add_argument("--case", default="s0")
    p.add_argument("--members", type=int, default=30)
    p.add_argument("--n-outer", type=int, default=10)
    p.add_argument("--draws", type=int, default=4)
    p.add_argument("--taus", default="0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9")
    p.add_argument("--s-grid", default="0,0.1,0.2,0.3")
    p.add_argument("--batch-size", type=int, default=50)
    p.add_argument("--max-batches", type=int, default=0)
    p.add_argument("--fd-eps", type=float, default=1e-2)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", required=True)
    a = p.parse_args()
    dev = torch.device("cuda")
    model, cfg = load_model(a.checkpoint, None, device=dev)
    model.eval()
    is_psc = isinstance(model, PredictStateCFM)
    s0 = float(model.sigma_prior)
    ns = load_norm_stats(str(resolve_norm_stats(cfg, required=True)))
    ns_dev = None
    _, dls, obs_idx = prepare_dataset(cfg, a.dataset, batch_size=a.batch_size, norm_stats=ns)
    idx = torch.as_tensor(obs_idx, dtype=torch.long, device=dev)
    N, M, K = a.n_outer, a.members, a.draws
    taus = [float(t) for t in a.taus.split(",")]
    pairs = []
    for s in (float(v) for v in a.s_grid.split(",")):
        for t in sorted({round(s + 0.1, 3), round(s + 0.2, 3), round(s + 0.3, 3), 0.9}):
            if s < t <= 0.9:
                pairs.append((s, t))

    acc = {k: [] for k in ("rg", "gg", "g_noise", "rr", "uju", "d_f1", "f1f1", "d_f2", "f2f2", "dd", "dmean")}

    def tmean(z: torch.Tensor) -> np.ndarray:
        return z.mean(dim=-2).cpu().numpy()

    torch.manual_seed(a.seed)
    with torch.no_grad():
        for bi, bd in enumerate(dls[a.case]):
            if a.max_batches and bi >= a.max_batches:
                break
            batch = BatchDict({k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in bd.items()})
            if ns_dev is None:
                ns_dev = {k: v.to(dev) for k, v in ns.items()}
            D = make_D(model, batch, is_psc)
            x1 = normalize(batch.true_state[..., idx].float(), ns_dev)

            diff_sum = torch.zeros_like(x1)
            diff_sq = torch.zeros_like(x1)
            d0_sum = torch.zeros_like(x1)
            for _ in range(M):
                x = torch.randn_like(x1) * s0
                d0 = None
                for k in range(N):
                    tau = k / N
                    mu = D(x, tau)
                    if k == 0:
                        d0 = mu
                    x = x + (1.0 / N) * (mu - x) / (1.0 - tau)
                diff = x - d0
                diff_sum += diff
                diff_sq += diff * diff
                d0_sum += d0
            g = diff_sum / M
            g_noise = (diff_sq / M - g * g) / (M - 1)
            D0 = d0_sum / M

            rg, rr, uju = [], [], []
            for tau in taus:
                d_sum = torch.zeros_like(x1)
                sq_sum = torch.zeros_like(x1)
                uju_sum = torch.zeros_like(x1)
                for _ in range(K):
                    x_tau = tau * x1 + (1.0 - tau) * torch.randn_like(x1) * s0
                    d = D(x_tau, tau)
                    d_sum += d
                    sq_sum += (x1 - d) ** 2
                    u = torch.randint(0, 2, x1.shape, device=dev).float() * 2 - 1
                    ju = (D(x_tau + a.fd_eps * u, tau) - D(x_tau - a.fd_eps * u, tau)) / (2 * a.fd_eps)
                    uju_sum += u * ju
                r = x1 - d_sum / K
                rg.append(tmean(r * g))
                rr.append(tmean(sq_sum / K))
                uju.append(tmean(uju_sum / K))
            acc["rg"].append(np.stack(rg, 1))
            acc["rr"].append(np.stack(rr, 1))
            acc["uju"].append(np.stack(uju, 1))
            acc["gg"].append(tmean(g * g))
            acc["g_noise"].append(tmean(g_noise))

            cols = {k: [] for k in ("d_f1", "f1f1", "d_f2", "f2f2", "dd", "dmean")}
            for s, tau in pairs:
                z = {k: torch.zeros_like(x1) for k in cols}
                for _ in range(K):
                    x_tau = tau * x1 + (1.0 - tau) * torch.randn_like(x1) * s0
                    x_s = pair_from_tau(x_tau, tau, s, s0)
                    d_s = D(x_s, s)
                    delta = D(x_tau, tau) - d_s
                    f1 = d_s - D0
                    z["d_f1"] += delta * f1
                    z["f1f1"] += f1 * f1
                    z["d_f2"] += delta * x_s
                    z["f2f2"] += x_s * x_s
                    z["dd"] += delta * delta
                    z["dmean"] += delta
                for k in cols:
                    cols[k].append(tmean(z[k] / K))
            for k in cols:
                acc[k].append(np.stack(cols[k], 1))
            print(f"batch {bi} done", flush=True)

    A = {k: np.concatenate(v, 0) for k, v in acc.items()}
    res = dict(checkpoint=a.checkpoint, model=type(model).__name__, case=a.case, members=M,
               n_outer=N, draws=K, sigma0=s0, n_windows=int(A["gg"].shape[0]), space="normalized",
               taus=taus, pairs=[list(pr) for pr in pairs])
    for g in GROUPS:
        sl = GROUPS[g]
        gg = A["gg"][:, sl].mean()
        noise = A["g_noise"][:, sl].mean()
        r = dict(g_rms=float(gg ** 0.5), g_noise_frac=float(noise / gg))
        ns1a = []
        for j, tau in enumerate(taus):
            st = slope_stats(A["rg"][:, j], A["gg"], g)
            st["slope_noisecorr"] = float(A["rg"][:, j][:, sl].mean() / (gg - noise))
            mse = float(A["rr"][:, j][:, sl].mean())
            st["rmse_D_vs_x1"] = mse ** 0.5
            st["tau"] = tau
            if tau > 0:
                jac = ((1.0 - tau) * s0) ** 2 / tau * float(A["uju"][:, j][:, sl].mean())
                st["NS1c_jac_term"] = jac
                st["NS1c_ratio_jac_over_mse"] = jac / mse
            ns1a.append(st)
        r["NS1a"] = ns1a
        ns1b = []
        for j, (s, tau) in enumerate(pairs):
            dd = A["dd"][:, j][:, sl].mean()
            f1 = slope_stats(A["d_f1"][:, j], A["f1f1"][:, j], g)
            f2 = slope_stats(A["d_f2"][:, j], A["f2f2"][:, j], g)
            f1["corr"] = float(A["d_f1"][:, j][:, sl].mean() / (dd * A["f1f1"][:, j][:, sl].mean()) ** 0.5)
            f2["corr"] = float(A["d_f2"][:, j][:, sl].mean() / (dd * A["f2f2"][:, j][:, sl].mean()) ** 0.5)
            ns1b.append(dict(s=s, tau=tau, delta_rms=float(dd ** 0.5),
                             delta_mean=float(A["dmean"][:, j][:, sl].mean()),
                             on_Ds_minus_D0=f1, on_x_s=f2))
        r["NS1b"] = ns1b
        res[g] = r
    json.dump(res, open(a.out, "w"), indent=1)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
