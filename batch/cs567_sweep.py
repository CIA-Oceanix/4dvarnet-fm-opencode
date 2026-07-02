#!/usr/bin/env python3
"""CS5/CS6/CS7 baseline sweep driver.

Usage:
    python batch/cs567_sweep.py --dws 40              # 4DVar sweep (all 4 methods)
    python batch/cs567_sweep.py --method enkf --inflation 1.2
    python batch/cs567_sweep.py --method etkf --inflation 1.6
"""
import os, sys, argparse, time, json
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.lorenz63 import Lorenz63Config, make_mixed_datasets
from evaluation.run import run_and_cache_baselines

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXP_DIR = os.path.join(BASE, "experiments")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dws", type=int, default=None)
    parser.add_argument("--method", choices=["enkf", "etkf"], default=None)
    parser.add_argument("--inflation", type=float, default=None)
    args = parser.parse_args()

    if args.dws is None and args.method is None:
        parser.error("Specify --dws or --method")
    if args.method and args.inflation is None:
        parser.error("--method requires --inflation")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dev_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    print(f"Device: {device} ({dev_name})")

    base_cfg = Lorenz63Config(
        dt=0.01, T_max=3.0, obs_interval=20,
        R_var=0.5, B_var=2.0,
        num_windows=2000, window_spacing=2000,
        spinup_steps=10000, seed=42,
        sigma_true=10.0, rho_true=28.0, beta_true=2.6666666666666665,
        gamma=0.05, W_L_bar=0.0, c1=1.0, c2=0.1,
        sigma_0=0.08, sigma_L=0.20,
        tau_eta=5.0, sigma_eta=0.7071067811865476,
        param_bias=0.0, forcing_state_bias=0.0, forcing_coupling="linear",
    )

    t0 = time.time()
    datasets = make_mixed_datasets(
        base_cfg, num_test_windows=200,
        include_randparam_test=False,
        include_sparse_obs_test=True,
    )
    print(f"Datasets generated in {time.time()-t0:.1f}s")
    for key in sorted(datasets):
        print(f"  {key}: {len(datasets[key])} windows")

    dws = args.dws if args.dws is not None else 50

    if args.method == "enkf":
        enkf_config = {"inflation": args.inflation}
        etkf_config = {"inflation": 1.0}
        print(f"EnKF inflation={args.inflation}, ETKF default=1.0, DWS={dws}")
    elif args.method == "etkf":
        enkf_config = {"inflation": 1.0}
        etkf_config = {"inflation": args.inflation}
        print(f"ETKF inflation={args.inflation}, EnKF default=1.0, DWS={dws}")
    else:
        enkf_config = {"inflation": 1.0}
        etkf_config = {"inflation": 1.0}
        print(f"DWS sweep: DWS={dws}, all methods (EnKF/ETKF at default inf=1.0)")

    suffix = "_cs567"
    dws_suffix = f"_dws{dws}"
    our_cache = os.path.join(EXP_DIR, f"baselines{dws_suffix}{suffix}.json")

    t1 = time.time()
    run_and_cache_baselines(
        datasets, device,
        batch_size=1,
        da_window_steps=dws,
        enkf_config=enkf_config,
        etkf_config=etkf_config,
        suffix=suffix,
    )
    print(f"Total time: {time.time()-t1:.1f}s")
    print("Done.")


if __name__ == "__main__":
    main()
