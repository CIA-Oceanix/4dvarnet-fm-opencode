import argparse
import itertools
import json
import os
import time

import numpy as np
import torch

from data.dataloader import FlowMatchingBatch
from data.normalization import denormalize, load_norm_stats, normalize
from eval_da_l63 import DEFAULT_OBS_INTERVAL, climatology, make_dataset, score
from evaluation.neural_inference import load_model
from evaluation.sda_sampler import sda_guided_sample

BASE = os.path.dirname(os.path.abspath(__file__))
EXP = os.path.join(BASE, "experiments")
OUT_DIR = os.path.join(EXP, "l63_da_table2")
def checkpoints(prefix):
    return {m: (os.path.join(EXP, f"{prefix}_{m}_monaiM", "checkpoints", "stage1_best.ckpt"),
                os.path.join(EXP, f"{prefix}_{m}_monaiM", "resolved_config.yaml"))
            for m in ("directunet", "vanillacfm", "sda2")}


NORM_STATS = os.path.join(EXP, "l63_norm_stats.pt")
METHODS = ("D-Unet", "CFM", "SDA2", "D-Unet+SDA2")
GW_GRID = [0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 40.0]
HYBRID_TAU0_GRID = [0.1, 0.3, 0.5]
HYBRID_GW_GRID = [0.25, 0.5, 1.0, 2.0, 5.0, 10.0]
R_VAR = 0.5
N_OUTER = 10


def make_batch(windows, device, stats):
    states = torch.stack([w["true_state"] for w in windows]).to(device)
    obs = torch.stack([w["obs"] for w in windows]).to(device)
    mask = torch.stack([w["obs_mask"] for w in windows]).to(device)
    forcing = torch.stack([w["forcing_corrupted"] for w in windows]).to(device)
    params = torch.tensor([[w["sigma"], w["rho"], w["beta"], w.get("c1", 1.0)] for w in windows],
                          dtype=torch.float32, device=device)
    s, o = normalize(states, stats), normalize(obs, stats)
    return FlowMatchingBatch(s, o, mask, forcing, params=params, true_params=params)


def sample(method, models, batch, n_members, gw=None, tau0=None):
    with torch.no_grad():
        if method == "D-Unet":
            return models["directunet"](batch).unsqueeze(-1)
        if method == "CFM":
            return torch.stack([models["vanillacfm"].sample(batch, N_outer=N_OUTER) for _ in range(n_members)], -1)
        mean_est = models["directunet"](batch) if method == "D-Unet+SDA2" else None
    x, _ = sda_guided_sample(models["sda2"], batch, R_VAR, N_outer=N_OUTER, guidance_weight=gw,
                             n_members=n_members,
                             mean_estimate=mean_est, tau0=tau0 if tau0 is not None else 0.0)
    return x.unsqueeze(-1) if x.dim() == 3 else x


def run(method, models, ds, device, stats, n_members, batch_size, gw=None, tau0=None, seed=0):
    torch.manual_seed(seed)
    trajs, members, truths = [], [], []
    for i in range(0, len(ds), batch_size):
        windows = [ds[j] for j in range(i, min(i + batch_size, len(ds)))]
        batch = make_batch(windows, device, stats)
        out = denormalize(sample(method, models, batch, n_members, gw, tau0).movedim(-1, 1).cpu(), stats)
        members.append(out.numpy())
        truths.append(torch.stack([w["true_state"] for w in windows]).numpy())
    members = np.concatenate(members).astype(np.float64)
    res = {"traj": members.mean(1), "truth": np.concatenate(truths).astype(np.float64)}
    if members.shape[1] > 1:
        res["members"], res["var"] = members, members.var(1, ddof=1)
    return res


def val_rmse(res):
    e = np.sqrt(((res["traj"] - res["truth"]) ** 2).mean(axis=(0, 1))).mean()
    return float(e) if np.isfinite(e) else float("inf")


def main():
    p = argparse.ArgumentParser(description="L63 neural schemes: EV / CRPSS / SS (Table 2 columns)")
    p.add_argument("--cases", nargs="+", default=["s0", "s1"], choices=["s0", "s1"])
    p.add_argument("--methods", nargs="+", default=list(METHODS), choices=list(METHODS))
    p.add_argument("--obs-interval", type=int, default=DEFAULT_OBS_INTERVAL)
    p.add_argument("--num-test", type=int, default=200)
    p.add_argument("--num-val", type=int, default=50)
    p.add_argument("--n-members", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=50)
    p.add_argument("--exp-prefix", default=None, help="experiment prefix (default L63B, or L63B20 for obs interval 20)")
    p.add_argument("--out", default=OUT_DIR)
    a = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    stats = load_norm_stats(NORM_STATS)
    models = {k: load_model(c, cfg, device=device)[0] for k, (c, cfg) in checkpoints(a.exp_prefix or ("L63B" if a.obs_interval == 10 else f"L63B{a.obs_interval}")).items()}
    os.makedirs(a.out, exist_ok=True)
    path = os.path.join(a.out, f"table2_l63_neural_obs{a.obs_interval}.json")
    results = json.load(open(path)) if os.path.exists(path) else {}

    val = {c: make_dataset(c, 99, a.num_val, a.obs_interval) for c in a.cases}
    test = {c: make_dataset(c, 123 if c == "s0" else 131, a.num_test, a.obs_interval) for c in a.cases}
    clim = {c: climatology(np.stack([w["true_state"].numpy() for w in test[c].windows])) for c in a.cases}

    def tune(method, grid_keys, grids):
        trace = {}
        for combo in itertools.product(*grids):
            kw = dict(zip(grid_keys, combo))
            r = np.mean([val_rmse(run(method, models, val[c], device, stats, 1, a.batch_size, **kw)) for c in a.cases])
            trace[combo] = float(r)
            print(f"    tune {method} {kw}: val RMSE {r:.4f}", flush=True)
        best = min(trace, key=trace.get)
        return dict(zip(grid_keys, best)), {"/".join(map(str, k)): v for k, v in trace.items()}

    for method in a.methods:
        hp, trace = {}, None
        if method == "SDA2":
            hp, trace = tune(method, ["gw"], [GW_GRID])
        elif method == "D-Unet+SDA2":
            hp, trace = tune(method, ["tau0", "gw"], [HYBRID_TAU0_GRID, HYBRID_GW_GRID])
        for c in a.cases:
            t0 = time.time()
            n = 1 if method == "D-Unet" else a.n_members
            res = run(method, models, test[c], device, stats, n, a.batch_size, **hp)
            row = score(res, *clim[c])
            row.update(hyperparameters=hp, tuning=trace, obs_interval=a.obs_interval, seconds=time.time() - t0)
            results.setdefault(c, {})[method] = row
            print(f"  {c.upper()} {method:<12} RMSE {row['rmse']:.3f} EV {row['ev']:.3f} "
                  f"CRPSS {row.get('crpss', row.get('mae_skill')):.3f} SS {row.get('ss', float('nan')):.2f} {hp}", flush=True)
            json.dump(results, open(path, "w"), indent=2)
    print(f"saved {path}")


if __name__ == "__main__":
    main()
