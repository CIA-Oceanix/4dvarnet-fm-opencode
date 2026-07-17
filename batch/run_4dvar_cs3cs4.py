#!/usr/bin/env python3
"""Weak-4DVar and Strong-4DVar evaluation on CS3+CS4 with per-window params.

Usage:
    python batch/run_4dvar_cs3cs4.py
"""
import os, sys, time, json
import torch
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.lorenz63 import Lorenz63Config, make_mixed_datasets
from data.lorenz63 import Lorenz63Config as L63Config
from evaluation.baselines import Weak4DVar, Strong4DVar
from evaluation.run import evaluate_baseline, fmt_rmse, EXP_DIR
from evaluation.metrics import rmse

DWS = 50
SUFFIX = "_cs3cs4_4dvar"


def main():
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
        include_randparam_test=True, param_noise=0.2,
    )
    print(f"Datasets generated in {time.time()-t0:.1f}s")
    for key in sorted(datasets):
        print(f"  {key}: {len(datasets[key])} windows")

    cases = [
        ("cs3", "test_cs3", 1, 0.0, "CS3", "linear"),
        ("cs4", "test_cs4", 2, 0.15, "CS4", "quartic"),
    ]

    weak4dvar_pool = {
        "linear": Weak4DVar(
            dt=0.01, da_window_steps=DWS, device=device, coupling_type="linear",
            opt_steps=150, lr=0.02,
        ),
        "quartic": Weak4DVar(
            dt=0.01, da_window_steps=DWS, device=device, coupling_type="quartic",
            opt_steps=150, lr=0.02,
        ),
    }
    strong4dvar_pool = {
        "linear": Strong4DVar(
            dt=0.01, da_window_steps=DWS, device=device, coupling_type="linear",
            max_iter=40, lr=0.1,
        ),
        "quartic": Strong4DVar(
            dt=0.01, da_window_steps=DWS, device=device, coupling_type="quartic",
            max_iter=40, lr=0.1,
        ),
    }

    cfg_cs3 = L63Config(case=1, param_bias=0.0, T_max=3.0, seed=125)
    cfg_cs4 = L63Config(case=2, param_bias=0.15, forcing_state_bias=0.15,
                         forcing_coupling="quartic", T_max=3.0, seed=126)

    results = {"config": {"T_max": 3.0, "da_window_steps": DWS}}
    total_t0 = time.time()

    for case_name, ds_key, case_val, bias, label, coupling_type in cases:
        if ds_key not in datasets:
            print(f"  {ds_key} not found, skipping")
            continue
        ds = datasets[ds_key]
        cfg = cfg_cs3 if case_name == "cs3" else cfg_cs4

        for method_name, pool in [("Weak-4DVar", weak4dvar_pool), ("Strong-4DVar", strong4dvar_pool)]:
            method = pool[coupling_type]
            print(f"    {label}/{method_name:<15} ...", end=" ", flush=True)
            t1 = time.time()
            (rmse_stats, expvar_stats), bl_results = evaluate_baseline(
                method, ds, cfg, device, return_trajs=True, batch_size=200,
            )
            m, s = rmse_stats
            elapsed = time.time() - t1

            if case_name not in results:
                results[case_name] = {}
            results[case_name][method_name] = fmt_rmse(m, s)
            results["total_time_seconds"] = time.time() - total_t0

            print(f"X={m[0]:.4f} Y={m[1]:.4f} Z={m[2]:.4f}"
                  f"  mean={np.mean(m):.4f} [{elapsed:.1f}s]")

            trajs = np.stack([r.trajectory for r in bl_results], axis=0)
            truths = np.stack([ds[i]["true_state"].numpy() for i in range(len(ds))], axis=0)
            traj_data = {"trajectories": trajs, "truths": truths}
            traj_path = os.path.join(
                EXP_DIR,
                f"baselines_trajs_dws{DWS}{SUFFIX}_{case_name}_{method_name.replace('-', '_').replace(' ', '_')}.npz"
            )
            np.savez_compressed(traj_path, **traj_data)

    cache_path = os.path.join(EXP_DIR, f"baselines_dws{DWS}{SUFFIX}.json")
    with open(cache_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nSaved results to {cache_path}")
    print(f"Total time: {time.time()-total_t0:.1f}s")
    print("Done.")


if __name__ == "__main__":
    main()
