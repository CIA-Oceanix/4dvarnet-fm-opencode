import argparse
import json
import os
import time
from multiprocessing import Pool

import numpy as np
import torch

from eval_da_l63 import (DEFAULT_OBS_INTERVAL, climatology, force_key, make_dataset, make_method,
                         score)

BASE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(BASE, "experiments", "l63_da_table2")
SIZES = [1, 10, 25, 50, 100, 150, 200]
INFLATION_GRID = [1.0, 1.05, 1.1, 1.2, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 8.0]
_STATE = {}


def solve_window(args):
    idx, case, n_members, inflation, seed = args
    torch.set_num_threads(1)
    torch.manual_seed(seed + idx)
    w = _STATE["ds"][idx]
    method = make_method("ETKS", case, torch.device("cpu"), inflation=inflation, n_members=n_members)
    r = method.assimilate_batch(
        w["obs"].unsqueeze(0), w["obs_mask"].unsqueeze(0), w[force_key(case)].unsqueeze(0),
        w["true_state"].unsqueeze(0),
        sigma=torch.tensor([w["sigma"]]), rho=torch.tensor([w["rho"]]),
        beta=torch.tensor([w["beta"]]), c1=torch.tensor([w.get("c1", 1.0)]),
    )[0]
    return np.asarray(r.trajectory, dtype=np.float64), np.asarray(r.ensemble, dtype=np.float32)


def run(procs, ds, case, n_members, inflation, seed):
    _STATE["ds"] = ds
    with Pool(procs) as pool:
        out = pool.map(solve_window, [(i, case, n_members, inflation, seed) for i in range(len(ds))], chunksize=1)
    truth = np.stack([ds[j]["true_state"].numpy() for j in range(len(ds))]).astype(np.float64)
    members = np.stack([o[1] for o in out])
    return {"traj": np.stack([o[0] for o in out]), "truth": truth, "members": members,
            "var": members.astype(np.float64).var(axis=1, ddof=1) if n_members > 1 else None}


def val_rmse(res):
    e = np.sqrt(((res["traj"] - res["truth"]) ** 2).mean(axis=(0, 1))).mean()
    return float(e) if np.isfinite(e) else float("inf")


def main():
    p = argparse.ArgumentParser(description="ETKS ensemble-size sweep on L63 (EV, CRPSS, SS, RMSE)")
    p.add_argument("--cases", nargs="+", default=["s0", "s1"], choices=["s0", "s1"])
    p.add_argument("--sizes", nargs="+", type=int, default=SIZES)
    p.add_argument("--obs-interval", type=int, default=DEFAULT_OBS_INTERVAL)
    p.add_argument("--num-test", type=int, default=200)
    p.add_argument("--num-val", type=int, default=50)
    p.add_argument("--procs", type=int, default=56)
    p.add_argument("--seed", type=int, default=1000)
    p.add_argument("--out", default=OUT_DIR)
    a = p.parse_args()
    os.makedirs(a.out, exist_ok=True)
    path = os.path.join(a.out, f"etks_ensemble_size_obs{a.obs_interval}.json")
    results = json.load(open(path)) if os.path.exists(path) else {}

    for case in a.cases:
        print(f"== {case.upper()} obs_interval={a.obs_interval}", flush=True)
        val_ds = make_dataset(case, 99, a.num_val, a.obs_interval)
        test_ds = make_dataset(case, 123 if case == "s0" else 131, a.num_test, a.obs_interval)
        clim = climatology(np.stack([w["true_state"].numpy() for w in test_ds.windows]))
        out = results.setdefault(case, {})
        for n in a.sizes:
            if str(n) in out:
                continue
            t0 = time.time()
            if n < 2:
                out[str(n)] = {"undefined": "a 1-member ensemble has no anomalies, so the ETKF analysis cannot be computed"}
                print(f"  {case.upper()} N={n}: undefined (no ensemble anomalies)", flush=True)
                json.dump(results, open(path, "w"), indent=2)
                continue
            trace = {}
            for lam in INFLATION_GRID:
                trace[str(lam)] = val_rmse(run(a.procs, val_ds, case, n, lam, a.seed))
                print(f"    tune N={n} inflation={lam}: val RMSE {trace[str(lam)]:.4f}", flush=True)
            lam = float(min(trace, key=trace.get))
            res = run(a.procs, test_ds, case, n, lam, a.seed)
            row = score(res, *clim)
            row.update(hyperparameters={"inflation": lam, "val_rmse": trace}, n_members=n,
                       obs_interval=a.obs_interval, seconds=time.time() - t0)
            out[str(n)] = row
            print(f"  {case.upper()} N={n:<3} RMSE {row['rmse']:.3f} EV {row['ev']:.4f} CRPSS {row['crpss']:.3f} "
                  f"SS {row['ss']:.2f} lambda={lam} [{row['seconds']:.0f}s]", flush=True)
            json.dump(results, open(path, "w"), indent=2)
    print(f"saved {path}")


if __name__ == "__main__":
    main()
