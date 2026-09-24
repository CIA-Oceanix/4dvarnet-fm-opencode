"""Stochastic (DDIM-eta) sampling of a trained PredictStateCFM / VanillaCFM, no retraining.

Linear path x_tau = tau x1 + b_tau eps, b_tau = (1 - tau) s0. One step s -> tau (s < tau),
with D = D(x_s, s, y) and the implied noise eps_hat = (x_s - s D) / b_s:

    x_tau = tau D + sqrt(b_tau^2 - sigma^2) eps_hat + sigma xi,   sigma = eta * sqrt(v),

where v is the variance of q(x_tau | x_s, x1) under the forward-noising kernel
x_s = (s/tau) x_tau + c xi', c^2 = b_s^2 - (s/tau)^2 b_tau^2. eta = 0 is exactly the Euler
ODE step of the CFM sampler; eta = 1 is the ancestral sampler (at s = 0: fresh noise).
Every eta keeps x_tau | x1 on the path marginal when D is exact, so eta trades the ODE's
accumulated contraction against injected noise. See docs/results/cfm_sampler_schedule.md.
"""
import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.normalization import denormalize, load_norm_stats  # noqa: E402
from evaluation.archive import resolve_norm_stats  # noqa: E402
from evaluation.neural_inference import BatchDict, load_model, prepare_dataset  # noqa: E402
from models.vanilla_cfm import PredictStateCFM  # noqa: E402

NO = 8


def make_D(model: torch.nn.Module, batch, is_psc: bool):
    def D(x: torch.Tensor, tau_val: float) -> torch.Tensor:
        tau = torch.full((x.shape[0],), tau_val, device=x.device)
        out = model.forward(x, batch, tau)
        return out if is_psc else x + (1.0 - tau_val) * out
    return D


def eta_step(x: torch.Tensor, d: torch.Tensor, s: float, tau: float, s0: float,
             eta: float) -> torch.Tensor:
    b_s, b_t = (1.0 - s) * s0, (1.0 - tau) * s0
    eps_hat = (x - s * d) / b_s
    if b_t <= 0.0:
        return d
    r = s / tau
    c2 = b_s ** 2 - (r * b_t) ** 2
    v = b_t ** 2 if r == 0.0 else 1.0 / (1.0 / b_t ** 2 + r ** 2 / max(c2, 1e-12))
    sigma = eta * v ** 0.5
    return tau * d + max(b_t ** 2 - sigma ** 2, 0.0) ** 0.5 * eps_hat + sigma * torch.randn_like(x)


def tau_grid(n_steps: int, power: float = 1.0) -> list:
    """tau_k = 1 - (1 - k/N)^power: power = 1 is uniform, power > 1 concentrates steps near tau = 1."""
    return [1.0 - (1.0 - k / n_steps) ** power for k in range(n_steps + 1)]


def sample(D, x0: torch.Tensor, n_steps: int, s0: float, eta: float,
           power: float = 1.0) -> torch.Tensor:
    x = x0
    taus = tau_grid(n_steps, power)
    for s, tau in zip(taus[:-1], taus[1:]):
        x = eta_step(x, D(x, s), s, tau, s0, eta)
    return x


def fair_crps_sum(members: torch.Tensor, truth: torch.Tensor) -> torch.Tensor:
    M = members.shape[0]
    mae = (members - truth.unsqueeze(0)).abs().mean(0)
    srt, _ = members.sort(dim=0)
    w = (2 * torch.arange(1, M + 1, device=members.device, dtype=members.dtype) - M - 1)
    pair = 2.0 * (w.view(-1, *([1] * (members.dim() - 1))) * srt).sum(0) / (M * (M - 1))
    return mae - 0.5 * pair


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--dataset", default="experiments/l96_datasets_obsj2_int100_nwin200.pt")
    p.add_argument("--case", default="s0")
    p.add_argument("--members", type=int, default=30)
    p.add_argument("--grid", default="10:0,10:0.25,10:0.5,10:0.75,10:1,20:0,20:0.5,20:1",
                   help="comma-separated n_steps:eta[:power] triples (power: step schedule, 1 = uniform)")
    p.add_argument("--batch-size", type=int, default=50)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", required=True)
    a = p.parse_args()
    dev = torch.device("cuda")
    model, cfg = load_model(a.checkpoint, None, device=dev)
    model.eval()
    is_psc = isinstance(model, PredictStateCFM)
    s0 = float(model.sigma_prior)
    ns = load_norm_stats(str(resolve_norm_stats(cfg, required=True)))
    nsd = {k: v.to(dev) for k, v in ns.items()}
    _, dls, obs_idx = prepare_dataset(cfg, a.dataset, batch_size=a.batch_size, norm_stats=ns)
    idx = torch.as_tensor(obs_idx, dtype=torch.long, device=dev)
    grid = [(int(g.split(":")[0]), float(g.split(":")[1]),
             float(g.split(":")[2]) if g.count(":") == 2 else 1.0) for g in a.grid.split(",")]
    groups = {"all_obs": slice(None), "slow": slice(0, NO), "obs_fast": slice(NO, None)}
    acc = {g: {k: {"se": 0.0, "var": 0.0, "crps": 0.0, "se1": 0.0} for k in groups} for g in grid}
    n_pts = {k: 0 for k in groups}
    with torch.no_grad():
        for bi, bd in enumerate(dls[a.case]):
            batch = BatchDict({k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in bd.items()})
            D = make_D(model, batch, is_psc)
            truth = batch.true_state[..., idx].float()
            for k, sl in groups.items():
                n_pts[k] += truth[..., sl].numel()
            for g in grid:
                torch.manual_seed(a.seed * 1000 + bi)
                mem = torch.stack([denormalize(sample(D, torch.randn_like(batch.obs) * s0, g[0], s0, g[1], g[2]), nsd)
                                   for _ in range(a.members)], 0)
                mean = mem.mean(0)
                var = mem.var(0, unbiased=True)
                crps = fair_crps_sum(mem, truth)
                for k, sl in groups.items():
                    acc[g][k]["se"] += float(((mean - truth)[..., sl] ** 2).sum())
                    acc[g][k]["var"] += float(var[..., sl].sum())
                    acc[g][k]["crps"] += float(crps[..., sl].sum())
                    acc[g][k]["se1"] += float(((mem[0] - truth)[..., sl] ** 2).sum())
            print(f"batch {bi} done", flush=True)
    res = dict(checkpoint=a.checkpoint, model=type(model).__name__, case=a.case, members=a.members,
               units="physical, pooled over windows x time x channels", rows=[])
    for g in grid:
        row = {"n_steps": g[0], "eta": g[1], "power": g[2]}
        for k in groups:
            rmse = (acc[g][k]["se"] / n_pts[k]) ** 0.5
            spread = (acc[g][k]["var"] / n_pts[k]) ** 0.5
            row[k] = dict(rmse=rmse, spread=spread, spread_over_rmse=spread / rmse,
                          crps=acc[g][k]["crps"] / n_pts[k],
                          rmse_single_member=(acc[g][k]["se1"] / n_pts[k]) ** 0.5)
        res["rows"].append(row)
    json.dump(res, open(a.out, "w"), indent=1)
    for r in res["rows"]:
        x = r["all_obs"]
        print(f"N={r['n_steps']:3d} eta={r['eta']:.2f} p={r['power']:.1f}  rmse {x['rmse']:.4f}  spread {x['spread']:.4f}  "
              f"sp/rmse {x['spread_over_rmse']:.3f}  crps {x['crps']:.4f}  single {x['rmse_single_member']:.4f}")


if __name__ == "__main__":
    main()
