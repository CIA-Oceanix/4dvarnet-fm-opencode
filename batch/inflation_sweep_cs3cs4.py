#!/usr/bin/env python3
"""Run one inflation value for CS3/CS4 baseline sweep.

Usage:
    python batch/inflation_sweep_cs3cs4.py --method enkf --inflation 1.2
    python batch/inflation_sweep_cs3cs4.py --method etkf --inflation 1.6
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
    parser.add_argument("--method", required=True, choices=["enkf", "etkf"])
    parser.add_argument("--inflation", required=True, type=float)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dev_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    print(f"Device: {device} ({dev_name})")
    print(f"Method: {args.method}, Inflation: {args.inflation}")

    base_cfg = Lorenz63Config(
        dt=0.01, T_max=3.0, obs_interval=20,
        R_var=0.5, B_var=2.0,
        num_windows=2000, window_spacing=2000,
        spinup_steps=10000, seed=42,
    )

    t0 = time.time()
    datasets_cache = os.path.join(EXP_DIR, "datasets.pt")
    datasets = None
    if os.path.exists(datasets_cache):
        try:
            loaded = torch.load(datasets_cache)
            if "test_cs3" in loaded and "test_cs4" in loaded:
                datasets = loaded
                print(f"Loaded cached datasets in {time.time()-t0:.1f}s")
            else:
                print("Cached datasets lack CS3/CS4, regenerating...")
        except Exception:
            print("Failed to load cached datasets, regenerating...")
    if datasets is None:
        datasets = make_mixed_datasets(
            base_cfg, num_test_windows=200,
            include_randparam_test=True, param_noise=0.2,
        )
        print(f"Datasets generated in {time.time()-t0:.1f}s")
        torch.save(datasets, datasets_cache)
    for key in datasets:
        print(f"  {key}: {len(datasets[key])} windows")

    if args.method == "enkf":
        enkf_config = {"inflation": args.inflation}
        etkf_config = {"inflation": 1.6}
    else:
        enkf_config = {"inflation": 1.2}
        etkf_config = {"inflation": args.inflation}

    t1 = time.time()
    run_and_cache_baselines(
        datasets, device,
        batch_size=1,
        da_window_steps=50,
        enkf_config=enkf_config,
        etkf_config=etkf_config,
        suffix="_cs3cs4",
    )
    print(f"Total time: {time.time()-t1:.1f}s")
    print("Done.")


if __name__ == "__main__":
    main()
