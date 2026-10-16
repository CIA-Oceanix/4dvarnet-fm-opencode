import argparse
import json
import os
import time

import numpy as np
import torch

from data.lorenz63 import Lorenz63Config
from data.random_bias_dataset import RandomBiasLorenz63Dataset
from data.random_param_dataset import RandomParamLorenz63Dataset
from evaluation.baselines import ETKF, ETKS, EnKF, Strong4DVar, Weak4DVar
from models.lorenz63_dynamics import Lorenz63Dynamics

BASE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(BASE, "experiments", "l63_da_table2")
DT = 0.01
DEFAULT_OBS_INTERVAL = 10
WINDOW_STEPS = 300
CASES = {"s0": 1.6, "s1": 1.0}
ENS_METHODS = ("ETKF", "EnKF", "ETKS")
VAR_METHODS = ("Strong-4DVar", "Weak-4DVar")
INFLATION_GRID = [1.0, 1.05, 1.1, 1.15, 1.2, 1.3, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0]
Q_GRID = [0.003, 0.01, 0.03, 0.1, 0.3, 1.0]


def make_dataset(case, seed, num_windows, obs_interval=DEFAULT_OBS_INTERVAL):
    cfg = Lorenz63Config(case=1, seed=seed, num_windows=num_windows, T_max=3.0, obs_interval=obs_interval,
                         param_bias=0.0 if case == "s0" else 0.15,
                         forcing_state_bias=0.0 if case == "s0" else 0.1)
    if case == "s0":
        return RandomParamLorenz63Dataset(cfg, param_noise=0.2)
    return RandomBiasLorenz63Dataset(cfg, param_noise=0.2, bias_mode="fixed")


def force_key(case):
    return "forcing_true" if case == "s0" else "forcing_corrupted"


def make_method(name, case, device, inflation=1.0, q_var=0.05, n_members=30):
    expo = CASES[case]
    dyn = Lorenz63Dynamics(dt=DT, coupling_exponent=expo)
    common = dict(dt=DT, device=device, coupling_exponent=expo, dynamics=dyn)
    if name == "ETKF":
        m = ETKF(N_ensemble=n_members, inflation=inflation, **common)
        m.store_ensemble = True
        return m
    if name == "ETKS":
        return ETKS(N_ensemble=n_members, inflation=inflation, **common)
    if name == "EnKF":
        m = EnKF(N_ensemble=n_members, inflation=inflation, **common)
        m.store_ensemble = True
        return m
    if name == "Strong-4DVar":
        return Strong4DVar(da_window_steps=WINDOW_STEPS, **common)
    if name == "Weak-4DVar":
        return Weak4DVar(da_window_steps=WINDOW_STEPS, Q_var=q_var, **common)
    raise ValueError(name)


def run_method(method, ds, case, device, batch_size):
    trajs, members, vars_ = [], [], []
    for i in range(0, len(ds), batch_size):
        batch = [ds[j] for j in range(i, min(i + batch_size, len(ds)))]
        stack = lambda k: torch.stack([w[k].to(device) for w in batch])
        results = method.assimilate_batch(
            stack("obs"), stack("obs_mask"), stack(force_key(case)),
            torch.stack([w["true_state"] for w in batch]),
            sigma=torch.tensor([w["sigma"] for w in batch], device=device),
            rho=torch.tensor([w["rho"] for w in batch], device=device),
            beta=torch.tensor([w["beta"] for w in batch], device=device),
            c1=torch.tensor([w.get("c1", 1.0) for w in batch], device=device),
        )
        for r in results:
            trajs.append(np.asarray(r.trajectory, dtype=np.float64))
            if r.ensemble is not None:
                members.append(np.asarray(r.ensemble, dtype=np.float32))
                vars_.append(np.var(members[-1], axis=0, ddof=1))
    truth = np.stack([ds[j]["true_state"].numpy() for j in range(len(ds))]).astype(np.float64)
    out = {"traj": np.stack(trajs), "truth": truth}
    if members:
        out["members"] = np.stack(members)
        out["var"] = np.stack(vars_)
    return out


def gini_half(x):
    x = np.sort(x)
    n = x.size
    return float(np.sum((2 * np.arange(1, n + 1) - n - 1) * x) / (n * n))


def climatology(truth):
    flat = truth.reshape(-1, truth.shape[-1])
    return flat.var(axis=0), np.array([gini_half(flat[:, j]) for j in range(flat.shape[1])])


def ensemble_crps(members, truth):
    W, M = members.shape[0], members.shape[1]
    total = np.zeros(truth.shape[-1])
    coeff = (2 * np.arange(M) - M + 1).reshape(1, M, 1, 1)
    for w in range(W):
        m = members[w].astype(np.float64)
        mae = np.abs(m - truth[w][None]).mean(0)
        pair = 2 * (coeff[0] * np.sort(m, axis=0)).sum(0) / (M * M)
        total += (mae - 0.5 * pair).mean(0)
    return total / W


def score(res, clim_var, clim_crps):
    truth = res["truth"]
    mse = ((res["traj"] - truth) ** 2).mean(axis=(0, 1))
    ev_j = 1.0 - mse / clim_var
    row = {"ev": float(ev_j.mean()), "ev_per_var": ev_j.tolist(),
           "rmse": float(np.sqrt(mse).mean())}
    if "members" in res:
        crps_j = ensemble_crps(res["members"], truth)
        row["crpss"] = float((1.0 - crps_j / clim_crps).mean())
        row["crpss_per_var"] = (1.0 - crps_j / clim_crps).tolist()
        err2 = ((res["traj"] - truth) ** 2).sum()
        row["ss"] = float(np.sqrt(res["var"].sum() / err2))
        row["ss_per_var"] = np.sqrt(res["var"].sum(axis=(0, 1)) / ((res["traj"] - truth) ** 2).sum(axis=(0, 1))).tolist()
        row["n_members"] = int(res["members"].shape[1])
    else:
        mae_j = np.abs(res["traj"] - truth).mean(axis=(0, 1))
        row["mae_skill"] = float((1.0 - mae_j / clim_crps).mean())
    return row


def tune(name, case, device, batch_size, val_ds, grid, key):
    best, trace = None, {}
    for v in grid:
        kwargs = {key: v}
        m = make_method(name, case, device, **kwargs)
        res = run_method(m, val_ds, case, device, batch_size)
        rmse = float(np.sqrt(((res["traj"] - res["truth"]) ** 2).mean(axis=(0, 1))).mean())
        if not np.isfinite(rmse):
            rmse = float("inf")
        trace[str(v)] = rmse
        print(f"    tune {name}/{case} {key}={v}: val RMSE {rmse:.4f}", flush=True)
        if best is None or rmse < trace[str(best)]:
            best = v
    return best, trace


def main():
    p = argparse.ArgumentParser(description="L63 Table-2 metrics (EV, CRPSS, SS) for the model-based DA schemes")
    p.add_argument("--cases", nargs="+", default=["s0", "s1"], choices=["s0", "s1"])
    p.add_argument("--methods", nargs="+", default=list(ENS_METHODS + VAR_METHODS))
    p.add_argument("--num-test", type=int, default=200)
    p.add_argument("--num-val", type=int, default=50)
    p.add_argument("--n-members", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=50)
    p.add_argument("--obs-interval", type=int, default=DEFAULT_OBS_INTERVAL)
    p.add_argument("--no-tune", action="store_true")
    p.add_argument("--out", default=OUT_DIR)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.out, exist_ok=True)
    results_path = os.path.join(args.out, f"table2_l63_da_obs{args.obs_interval}.json")
    results = json.load(open(results_path)) if os.path.exists(results_path) else {}

    for case in args.cases:
        print(f"== {case.upper()}: building datasets", flush=True)
        val_ds = make_dataset(case, 99, args.num_val, args.obs_interval)
        test_ds = make_dataset(case, 123 if case == "s0" else 131, args.num_test, args.obs_interval)
        clim_var, clim_crps = climatology(np.stack([w["true_state"].numpy() for w in test_ds.windows]))
        res_case = results.setdefault(case, {})
        etkf_inflation = None
        for name in args.methods:
            t0 = time.time()
            hp = {}
            if name in ("ETKF", "EnKF") and not args.no_tune:
                lam, trace = tune(name, case, device, args.batch_size, val_ds, INFLATION_GRID, "inflation")
                hp = {"inflation": lam, "val_rmse": trace}
            elif name == "ETKS":
                lam = etkf_inflation or res_case.get("ETKF", {}).get("hyperparameters", {}).get("inflation", 1.0)
                hp = {"inflation": lam, "note": "inflation of the ETKF filter pass"}
            elif name == "Weak-4DVar" and not args.no_tune:
                q, trace = tune(name, case, device, args.batch_size, val_ds, Q_GRID, "q_var")
                hp = {"q_var": q, "val_rmse": trace}
            if name == "ETKF":
                etkf_inflation = hp.get("inflation")
            kw = {k: v for k, v in hp.items() if k in ("inflation", "q_var")}
            method = make_method(name, case, device, n_members=args.n_members, **kw)
            res = run_method(method, test_ds, case, device, args.batch_size)
            row = score(res, clim_var, clim_crps)
            row["hyperparameters"] = hp
            row["obs_interval"] = args.obs_interval
            row["seconds"] = time.time() - t0
            res_case[name] = row
            print(f"  {case.upper()} {name:<13} EV {row['ev']:.3f}  "
                  f"CRPSS {row.get('crpss', row.get('mae_skill', float('nan'))):.3f}  "
                  f"SS {row.get('ss', float('nan')):.2f}  [{row['seconds']:.0f}s]", flush=True)
            with open(results_path, "w") as f:
                json.dump(results, f, indent=2)
    print(f"saved {results_path}")


if __name__ == "__main__":
    main()
