import argparse
import json
import os
import time
from multiprocessing import Pool

import numpy as np
import torch

from eval_da_l63 import CASES, DEFAULT_OBS_INTERVAL, DT, Q_GRID, climatology, force_key, make_dataset, score
from evaluation.baselines import L96Weak4DVar, ObsOperator
from models.lorenz63_dynamics import Lorenz63Dynamics

BASE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(BASE, "experiments", "l63_da_table2")
SUBWINDOW_GRID = [50, 100, 150, 300]
_STATE = {}


class L63Variational(L96Weak4DVar):
    """L-BFGS strong/weak 4D-Var cycled over sub-windows, L63 parameters."""

    def assimilate(self, observations, obs_mask, forcing, true_state=None,
                   sigma=10.0, rho=28.0, beta=8 / 3, c1=1.0, **kwargs):
        return super().assimilate(observations, obs_mask, forcing, true_state=true_state,
                                  **{"F": None}, _l63=dict(sigma=sigma, rho=rho, beta=beta, c1=c1))

    def _solve_window(self, xb, win_obs, win_mask, start, win_force, params):
        return super()._solve_window(xb, win_obs, win_mask, start, win_force, params["_l63"])


def build(case, mode, subwindow, q):
    dyn = Lorenz63Dynamics(dt=DT, coupling_exponent=CASES[case])
    return L63Variational(dt=DT, dynamics=dyn, obs_operator=ObsOperator(3), da_window_steps=subwindow,
                          optimizer="lbfgs", max_iter=40, lr=1.0, mode=mode, q_var_scale=q,
                          r_var=0.5, device=torch.device("cpu"))


def solve_one(args):
    idx, case, mode, subwindow, q = args
    torch.set_num_threads(1)
    w = _STATE["ds"][idx]
    m = build(case, mode, subwindow, q)
    r = m.assimilate(w["obs"], w["obs_mask"], w[force_key(case)], w["true_state"],
                     sigma=w["sigma"], rho=w["rho"], beta=w["beta"], c1=w.get("c1", 1.0))
    return np.asarray(r.trajectory, dtype=np.float64)


def run(procs, ds, case, mode, subwindow, q):
    _STATE["ds"] = ds
    with Pool(procs) as pool:
        trajs = pool.map(solve_one, [(i, case, mode, subwindow, q) for i in range(len(ds))], chunksize=1)
    truth = np.stack([ds[j]["true_state"].numpy() for j in range(len(ds))]).astype(np.float64)
    return {"traj": np.stack(trajs), "truth": truth}


def val_rmse(res):
    e = np.sqrt(((res["traj"] - res["truth"]) ** 2).mean(axis=(0, 1))).mean()
    return float(e) if np.isfinite(e) else float("inf")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cases", nargs="+", default=["s0", "s1"])
    p.add_argument("--obs-interval", type=int, default=DEFAULT_OBS_INTERVAL)
    p.add_argument("--num-test", type=int, default=200)
    p.add_argument("--num-val", type=int, default=50)
    p.add_argument("--procs", type=int, default=56)
    p.add_argument("--out", default=OUT_DIR)
    a = p.parse_args()
    os.makedirs(a.out, exist_ok=True)
    path = os.path.join(a.out, f"table2_l63_4dvar_obs{a.obs_interval}.json")
    results = {}
    for case in a.cases:
        print(f"== {case.upper()} obs_interval={a.obs_interval}", flush=True)
        val_ds = make_dataset(case, 99, a.num_val, a.obs_interval)
        test_ds = make_dataset(case, 123 if case == "s0" else 131, a.num_test, a.obs_interval)
        clim_var, clim_crps = climatology(np.stack([w["true_state"].numpy() for w in test_ds.windows]))
        out = results.setdefault(case, {})
        trace = {}
        for sw in SUBWINDOW_GRID:
            trace[sw] = val_rmse(run(a.procs, val_ds, case, "strong", sw, 0.0))
            print(f"    tune Strong sub-window={sw}: val RMSE {trace[sw]:.4f}", flush=True)
        sw_s = min(trace, key=trace.get)
        hp_s = {"subwindow": sw_s, "max_iter": 40, "val_rmse": trace}
        qtrace = {}
        for sw in SUBWINDOW_GRID:
            for q in Q_GRID:
                qtrace[f"{sw}/{q}"] = val_rmse(run(a.procs, val_ds, case, "weak", sw, q))
                print(f"    tune Weak sub-window={sw} q={q}: val RMSE {qtrace[f'{sw}/{q}']:.4f}", flush=True)
        best = min(qtrace, key=qtrace.get)
        sw_w, q_w = int(best.split("/")[0]), float(best.split("/")[1])
        hp_w = {"subwindow": sw_w, "q_var": q_w, "max_iter": 40, "val_rmse": qtrace}
        for name, mode, sw, q, hp in (("Strong-4DVar", "strong", sw_s, 0.0, hp_s),
                                      ("Weak-4DVar", "weak", sw_w, q_w, hp_w)):
            t0 = time.time()
            row = score(run(a.procs, test_ds, case, mode, sw, q), clim_var, clim_crps)
            row["hyperparameters"] = hp
            row["obs_interval"] = a.obs_interval
            row["seconds"] = time.time() - t0
            out[name] = row
            print(f"  {case.upper()} {name:<13} RMSE {row['rmse']:.3f} EV {row['ev']:.3f} "
                  f"MAEskill {row['mae_skill']:.3f}", flush=True)
            json.dump(results, open(path, "w"), indent=2)
    print(f"saved {path}")


if __name__ == "__main__":
    main()
