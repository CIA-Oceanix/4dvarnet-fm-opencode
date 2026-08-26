import argparse
import json
import os
import sys
import time

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from data.qg import QGConfig, make_qg_s0_s1_datasets
from evaluation.run_qg_baselines import run


def _device(name):
    return torch.device(name) if name else torch.device(
        "cuda" if torch.cuda.is_available() else "cpu")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nx", type=int, default=64)
    ap.add_argument("--num-windows", type=int, default=5)
    ap.add_argument("--window-days", type=float, default=60.0)
    ap.add_argument("--spinup-years", type=float, default=2.0)
    ap.add_argument("--ensemble", type=int, default=80)
    ap.add_argument("--inflation-list", default="1.0,1.15,1.3")
    ap.add_argument("--loc-list", default="6,10,14")
    ap.add_argument("--method-list", default="etkf,enkf")
    ap.add_argument("--init", default="lagged", choices=["lagged", "white"])
    ap.add_argument("--geometry", default="random_columns")
    ap.add_argument("--init-lag-days", type=float, default=2.0)
    ap.add_argument("--cols-per-day", type=int, default=3)
    ap.add_argument("--scenarios", default="test_s0,test_s1a,test_s1b")
    ap.add_argument("--outdir", default="reports/outputs/figs")
    ap.add_argument("--device", default=None)
    ap.add_argument("--tag", default="sweep")
    args = ap.parse_args()

    device = _device(args.device)
    cfg = QGConfig(nx=args.nx, window_days=args.window_days,
                   spinup_years=args.spinup_years, num_windows=args.num_windows,
                   obs_geometry=args.geometry, cols_per_day=args.cols_per_day,
                   seed=7)
    scenarios = tuple(args.scenarios.split(","))
    os.makedirs(args.outdir, exist_ok=True)

    print(f"device={device} building dataset once", flush=True)
    t0 = time.time()
    ds = make_qg_s0_s1_datasets(cfg)
    print(f"dataset built in {time.time() - t0:.1f}s", flush=True)

    infls = [float(x) for x in args.inflation_list.split(",")]
    locs = [float(x) for x in args.loc_list.split(",")]
    methods = args.method_list.split(",")
    for method in methods:
        for infl in infls:
            for loc in locs:
                tag = f"{method}_i{infl}_l{loc}"
                t1 = time.time()
                p = run(method, cfg, device=device, N_ensemble=args.ensemble,
                        inflation=infl, loc_radius=loc, init=args.init,
                        geometry=args.geometry, scenarios=scenarios,
                        out_path=None, ds=ds, init_lag_days=args.init_lag_days)
                dt = time.time() - t1
                rows = " ".join(
                    f"{s}:{p['scenarios'][s]['expvar_full']:.3f}"
                    for s in p["scenarios"])
                print(f"[{args.tag}|{tag}] {dt:.1f}s {rows}", flush=True)
                with open(os.path.join(
                        args.outdir, f"qg_{args.tag}_{tag}.json"), "w") as f:
                    json.dump(p, f, indent=2)
    print("SWEEP DONE", flush=True)


if __name__ == "__main__":
    main()
