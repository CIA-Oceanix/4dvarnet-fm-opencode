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


def nzlab(noise):
    return "def" if noise is None else f"{noise:g}".replace(".", "p")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nx", type=int, default=64)
    ap.add_argument("--num-windows", type=int, default=5)
    ap.add_argument("--window-days", type=float, default=60.0)
    ap.add_argument("--spinup-years", type=float, default=2.0)
    ap.add_argument("--ensemble", type=int, default=80)
    ap.add_argument("--ensemble-list", default=None)
    ap.add_argument("--inflation-list", default="1.0,1.15,1.3")
    ap.add_argument("--loc-list", default="6,10,14")
    ap.add_argument("--method-list", default="etkf,enkf")
    ap.add_argument("--init", default="lagged", choices=["lagged", "white"])
    ap.add_argument("--geometry", default="random_columns")
    ap.add_argument("--init-lag-days-list", default="2.0")
    ap.add_argument("--band", dest="band_half", type=float, default=0.25)
    ap.add_argument("--cols-per-day", type=int, default=3)
    ap.add_argument("--cols-per-day-list", default=None)
    ap.add_argument("--obs-noise-frac-list", default=None)
    ap.add_argument("--obs-var", choices=["q", "psi"], default="q")
    ap.add_argument("--scenarios", default="test_s0,test_s1a,test_s1b")
    ap.add_argument("--outdir", default="reports/outputs/figs")
    ap.add_argument("--device", default=None)
    ap.add_argument("--tag", default="sweep")
    ap.add_argument("--disp-frac", type=float, default=1.0)
    ap.add_argument("--disp-frac-list", default=None)
    ap.add_argument("--etkf-ridge-list", default=None)
    ap.add_argument("--etkf-additive-list", default=None)
    args = ap.parse_args()

    device = _device(args.device)
    scenarios = tuple(args.scenarios.split(","))
    os.makedirs(args.outdir, exist_ok=True)

    infls = [float(x) for x in args.inflation_list.split(",")]
    locs = [float(x) for x in args.loc_list.split(",")]
    methods = args.method_list.split(",")
    lags = args.init_lag_days_list.split(",")
    ns = ([int(x) for x in args.ensemble_list.split(",")]
          if args.ensemble_list else [args.ensemble])
    disps = ([float(x) for x in args.disp_frac_list.split(",")]
             if args.disp_frac_list else [args.disp_frac])
    ridges = ([float(x) for x in args.etkf_ridge_list.split(",")]
              if args.etkf_ridge_list else [0.0])
    addit = ([float(x) for x in args.etkf_additive_list.split(",")]
             if args.etkf_additive_list else [0.0])
    colss = ([int(x) for x in args.cols_per_day_list.split(",")]
             if args.cols_per_day_list else [args.cols_per_day])
    noises = ([float(x) for x in args.obs_noise_frac_list.split(",")]
              if args.obs_noise_frac_list else [None])
    for cols in colss:
        for noise in noises:
            if noise is None:
                cfg = QGConfig(
                    nx=args.nx, window_days=args.window_days,
                    spinup_years=args.spinup_years, num_windows=args.num_windows,
                    obs_geometry=args.geometry, cols_per_day=cols, seed=7)
            else:
                cfg = QGConfig(
                    nx=args.nx, window_days=args.window_days,
                    spinup_years=args.spinup_years, num_windows=args.num_windows,
                    obs_geometry=args.geometry, cols_per_day=cols,
                    obs_noise_std_frac=noise, seed=7)
            print(f"device={device} building dataset (cols={cols},"
                  f"noise={noise}) once", flush=True)
            t0 = time.time()
            ds = make_qg_s0_s1_datasets(cfg)
            print(f"dataset built in {time.time() - t0:.1f}s", flush=True)
            for method in methods:
                for lag in lags:
                    for infl in infls:
                        for loc in locs:
                            for n in ns:
                                for disp in disps:
                                    for ridge in ridges:
                                        for a in addit:
                                            tag = (f"{method}_c{cols}"
                                                   f"_nz{nzlab(noise)}"
                                                   f"_i{infl}_l{loc}_lag{lag}"
                                                   f"_n{n}_d{disp}_r{ridge}_a{a}")
                                            t1 = time.time()
                                            p = run(method, cfg, device=device,
                                                    N_ensemble=n,
                                                    inflation=infl, loc_radius=loc,
                                                    init=args.init,
                                                    geometry=args.geometry,
                                                    scenarios=scenarios,
                                                    out_path=None, ds=ds,
                                                    init_lag_days=float(lag),
                                                    band_half=args.band_half,
                                                    obs_var=args.obs_var,
                                                    disp_frac=disp,
                                                    etkf_ridge=ridge,
                                                    etkf_additive=a)
                                            dt = time.time() - t1
                                            rows = " ".join(
                                                f"{s}:EV{p['scenarios'][s]['expvar_full']:.3f}"
                                                f"/FF{p['scenarios'][s].get('expvar_free'):.3f}"
                                                for s in p["scenarios"])
                                            print(f"[{args.tag}|{tag}] {dt:.1f}s {rows}",
                                                  flush=True)
                                            with open(os.path.join(
                                                    args.outdir,
                                                    f"qg_{args.tag}_{tag}.json"),
                                                    "w") as f:
                                                json.dump(p, f, indent=2)
    print("SWEEP DONE", flush=True)


if __name__ == "__main__":
    main()
