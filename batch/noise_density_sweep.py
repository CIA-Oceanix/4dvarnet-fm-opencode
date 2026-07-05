#!/usr/bin/env python3
"""Run one observation-noise or density config for S0/S1 sensitivity sweep.

The true trajectory is identical across configs (same seed -> same params ->
same trajectory); only the observation process changes (R_var controls noise
variance, obs_interval controls how many steps between observations).

Usage:
    python batch/noise_density_sweep.py --rvar 0.5
    python batch/noise_density_sweep.py --obs-interval 20
"""
import os, sys, argparse, time
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.lorenz63 import Lorenz63Config, make_mixed_datasets
from evaluation.run import run_and_cache_baselines, BASE

EXP_DIR = os.path.join(BASE, "experiments")
os.makedirs(EXP_DIR, exist_ok=True)


def main():
    parser = argparse.ArgumentParser()
    g = parser.add_mutually_exclusive_group(required=True)
    g.add_argument("--rvar", type=float, default=None,
                   help="Observation noise variance (R_var) to sweep")
    g.add_argument("--obs-interval", type=int, default=None,
                   help="Observation interval in steps to sweep")
    args = parser.parse_args()

    sweep_type = "rvar" if args.rvar is not None else "obsint"
    sweep_val = args.rvar if args.rvar is not None else args.obs_interval

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dev_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    print(f"Device: {device} ({dev_name})")
    print(f"Sweep: {sweep_type}={sweep_val}")

    r_var = args.rvar if args.rvar is not None else 0.5
    obs_interval = args.obs_interval if args.obs_interval is not None else 20

    base_cfg = Lorenz63Config(
        dt=0.01, T_max=3.0, obs_interval=obs_interval,
        R_var=r_var, B_var=2.0,
        num_windows=2000, window_spacing=2000,
        spinup_steps=10000, seed=42,
    )

    t0 = time.time()
    cache_tag = f"r{r_var}_oi{obs_interval}"
    datasets_cache = os.path.join(EXP_DIR, f"datasets_{cache_tag}.pt")
    datasets = None
    if os.path.exists(datasets_cache):
        try:
            loaded = torch.load(datasets_cache)
            if "test_s0" in loaded and "test_s1" in loaded:
                datasets = loaded
                print(f"Loaded cached datasets in {time.time()-t0:.1f}s")
        except Exception:
            pass
    if datasets is None:
        datasets = make_mixed_datasets(
            base_cfg, num_test_windows=200,
            include_s1_test=True, param_noise=0.2,
        )
        print(f"Datasets generated in {time.time()-t0:.1f}s")
        torch.save(datasets, datasets_cache)
    for key in datasets:
        print(f"  {key}: {len(datasets[key])} windows")

    suffix = f"_{sweep_type}{sweep_val}"

    enkf_config = {"inflation": 1.2, "R_var": r_var}
    etkf_config = {"inflation": 1.6, "R_var": r_var}
    weak_config = {"R_var": r_var}
    strong_config = {"R_var": r_var}

    t1 = time.time()
    run_and_cache_baselines(
        datasets, device,
        batch_size=200,
        da_window_steps=50,
        weak_config=weak_config,
        strong_config=strong_config,
        enkf_config=enkf_config,
        etkf_config=etkf_config,
        suffix=suffix,
    )
    print(f"Total time: {time.time()-t1:.1f}s")
    print("Done.")


if __name__ == "__main__":
    main()
