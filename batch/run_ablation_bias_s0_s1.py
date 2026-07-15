"""Ablation: sweep param_bias and forcing_state_bias independently around the
S0 and S1 operating points, as those are actually defined in
data/lorenz63.py::make_s0_s1_trainval (test_s0_cfg / test_s1_cfg):
  S0: param_bias=0.0,  forcing_state_bias=0.0,  seed=123
  S1: param_bias=0.15, forcing_state_bias=0.1,  seed=131
data/lorenz63.py never ties a coupling exponent to the S0/S1 case -- truth
always uses coupling_exponent_truth=1.6 and DA always uses the
coupling_exponent_da default (1.0), regardless of scenario -- so both
scenarios share one DA forward model here.

Each individual sweep holds the OTHER bias fixed at that scenario's own
baseline value (0.0/0.0 for S0, 0.15/0.1 for S1) so the sweep is anchored at
the scenario's actual operating point, plus a joint sweep where both biases
are varied together from 0 at matching values.

Unlike evaluation/run.py::run_and_cache_baselines (which always feeds DA the
TRUE forcing, making forcing_state_bias a no-op), this script feeds DA the
CORRUPTED forcing so forcing_state_bias actually perturbs what DA sees.
Truth trajectories are unaffected by either bias (RandomBiasLorenz63Dataset
bias_mode="fixed" only biases the DA-facing sigma/rho/beta and the corrupted
forcing), so varying bias here does not change the physical windows.

Usage: python batch/run_ablation_bias_s0_s1.py [--num-windows N] [--quick]
"""
import argparse
import os
import sys
import json
import time

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from data.lorenz63 import Lorenz63Config
from data.random_bias_dataset import RandomBiasLorenz63Dataset
from evaluation.baselines import (
    Weak4DVar, Strong4DVar, EnKF, ETKF,
    JointWeak4DVar, JointStrong4DVar, JointEnKF, JointETKF,
)
from evaluation.run import evaluate_baseline, fmt_rmse
from evaluation.metrics import param_rmse

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXP_DIR = os.path.join(BASE, "experiments")
os.makedirs(EXP_DIR, exist_ok=True)

DA_WINDOW_STEPS = 50
PARAM_NOISE = 0.2

PARAM_BIAS_GRID = [0.0, 0.05, 0.10, 0.15, 0.20, 0.30]
FORCING_BIAS_GRID = [0.0, 0.05, 0.10, 0.15, 0.20, 0.30]
# Joint sweep: param_bias and forcing_state_bias vary together (same value on both).
JOINT_BIAS_GRID = [0.0, 0.05, 0.10, 0.15, 0.20, 0.30]

# (label, seed, base_param_bias, base_forcing_state_bias) -- the S0/S1
# operating points from data/lorenz63.py::make_s0_s1_trainval.
SCENARIOS = [
    ("s0", 123, 0.0, 0.0),
    ("s1", 131, 0.15, 0.1),
]
COUPLING_EXPONENT_DA = 1.0  # matches Lorenz63Config.coupling_exponent_da default

METHODS = [
    "Weak-4DVar", "Joint-Weak-4DVar",
    "Strong-4DVar", "Joint-Strong-4DVar",
    "EnKF", "Joint-EnKF",
    "ETKF", "Joint-ETKF",
]

WEAK_CFG = {"opt_steps": 150, "lr": 0.02}
STRONG_CFG = {"max_iter": 40, "lr": 0.1}
ENKF_CFG = {"N_ensemble": 30, "inflation": 2.0}
ETKF_CFG = {"N_ensemble": 30, "inflation": 2.0}


def build_dataset(seed, num_windows, param_bias, forcing_state_bias):
    cfg = Lorenz63Config(
        case=1, param_bias=param_bias, forcing_state_bias=forcing_state_bias,
        T_max=3.0, seed=seed, num_windows=num_windows,
    )
    return RandomBiasLorenz63Dataset(cfg, param_noise=PARAM_NOISE, bias_mode="fixed")


def build_methods(coupling_exponent, device):
    return {
        "Weak-4DVar": Weak4DVar(dt=0.01, da_window_steps=DA_WINDOW_STEPS, device=device,
                                 coupling_exponent=coupling_exponent, **WEAK_CFG),
        "Joint-Weak-4DVar": JointWeak4DVar(dt=0.01, da_window_steps=DA_WINDOW_STEPS, device=device,
                                            coupling_exponent=coupling_exponent, **WEAK_CFG),
        "Strong-4DVar": Strong4DVar(dt=0.01, da_window_steps=DA_WINDOW_STEPS, device=device,
                                     coupling_exponent=coupling_exponent, **STRONG_CFG),
        "Joint-Strong-4DVar": JointStrong4DVar(dt=0.01, da_window_steps=DA_WINDOW_STEPS, device=device,
                                                coupling_exponent=coupling_exponent, **STRONG_CFG),
        "EnKF": EnKF(dt=0.01, device=device, coupling_exponent=coupling_exponent, **ENKF_CFG),
        "Joint-EnKF": JointEnKF(dt=0.01, device=device, coupling_exponent=coupling_exponent, **ENKF_CFG),
        "ETKF": ETKF(dt=0.01, device=device, coupling_exponent=coupling_exponent, **ETKF_CFG),
        "Joint-ETKF": JointETKF(dt=0.01, device=device, coupling_exponent=coupling_exponent, **ETKF_CFG),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-windows", type=int, default=50)
    parser.add_argument("--quick", action="store_true",
                         help="tiny smoke test: 3 windows, 2 methods, 1 scenario, 2 grid points")
    parser.add_argument("--out", default=os.path.join(EXP_DIR, "ablation_bias_s0_s1.json"))
    args = parser.parse_args()

    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print("============================================")
    print(" Baseline S0/S1 Noise Ablations (local)")
    print("============================================")

    scenarios = SCENARIOS
    methods = METHODS
    param_grid = PARAM_BIAS_GRID
    forcing_grid = FORCING_BIAS_GRID
    joint_grid = JOINT_BIAS_GRID
    num_windows = args.num_windows

    if args.quick:
        scenarios = SCENARIOS[:1]
        methods = ["Weak-4DVar", "Joint-Weak-4DVar"]
        param_grid = [0.0, 0.15]
        forcing_grid = [0.0, 0.10]
        joint_grid = [0.0, 0.15]
        num_windows = 3

    cache_path = args.out
    partial = {}
    if os.path.exists(cache_path):
        with open(cache_path) as f:
            partial = json.load(f)
        print(f"Found partial results ({cache_path}), resuming...")

    total_t0 = time.time()

    method_pool = build_methods(COUPLING_EXPONENT_DA, device)

    for exp_label, seed, base_p_bias, base_f_bias in scenarios:
        # use_corrupted_forcing=True (case=2) so evaluate_baseline picks "forcing_corrupted".
        # seed matches this scenario's dataset seed (123=s0, 131=s1), same convention as
        # run_and_cache_baselines, so DA-method RNG (ensemble/background init, obs
        # perturbations) differs between s0 and s1 but stays fixed across the bias sweep
        # within a scenario (common random numbers -> isolates the bias effect).
        da_cfg = Lorenz63Config(case=2, seed=seed)

        # Each sweep is anchored at this scenario's own (base_p_bias, base_f_bias)
        # operating point -- e.g. sweeping forcing_state_bias for S1 holds
        # param_bias at S1's 0.15, not at 0.0.
        sweeps = [
            ("param_bias", param_grid, base_f_bias),
            ("forcing_state_bias", forcing_grid, base_p_bias),
            ("joint_bias", joint_grid, None),
        ]

        for axis, grid, other_fixed in sweeps:
            for value in grid:
                if axis == "param_bias":
                    p_bias, f_bias = value, other_fixed
                elif axis == "forcing_state_bias":
                    p_bias, f_bias = other_fixed, value
                else:  # joint_bias: both biases move together
                    p_bias, f_bias = value, value

                run_key = f"{exp_label}__{axis}__{value}"
                if run_key not in partial:
                    partial[run_key] = {
                        "scenario": exp_label,
                        "base_param_bias": base_p_bias,
                        "base_forcing_state_bias": base_f_bias,
                        "coupling_exponent_da": COUPLING_EXPONENT_DA,
                        "axis": axis,
                        "value": value,
                        "param_bias": p_bias,
                        "forcing_state_bias": f_bias,
                        "num_windows": num_windows,
                        "results": {},
                    }

                entry = partial[run_key]
                if all(m in entry["results"] for m in methods):
                    print(f"  {run_key} already done, skipping")
                    continue

                dataset = build_dataset(seed, num_windows, p_bias, f_bias)
                true_params = np.stack([
                    np.array([w["true_sigma"], w["true_rho"], w["true_beta"], w["true_c1"]])
                    for w in dataset
                ], axis=0)

                for name in methods:
                    if name in entry["results"]:
                        continue
                    method = method_pool[name]
                    is_joint = name.startswith("Joint-")
                    t1 = time.time()
                    (m, s), results = evaluate_baseline(
                        method, dataset, da_cfg, device, return_trajs=True, batch_size=num_windows)
                    elapsed = time.time() - t1
                    result = fmt_rmse(m, s)
                    if is_joint and results[0].params is not None:
                        pred_params = np.stack([r.params[-1] for r in results], axis=0)
                        prmse = param_rmse(pred_params, true_params)
                        result["param_rmse"] = {
                            "sigma": float(prmse[0]), "rho": float(prmse[1]),
                            "beta": float(prmse[2]), "c1": float(prmse[3]),
                        }
                    entry["results"][name] = result
                    with open(cache_path, "w") as f:
                        json.dump(partial, f, indent=2)
                    param_str = ""
                    if is_joint and "param_rmse" in result:
                        p = result["param_rmse"]
                        param_str = (f"  params s={p['sigma']:.4f} r={p['rho']:.4f} "
                                     f"b={p['beta']:.4f} c1={p['c1']:.4f}")
                    print(f"  {run_key}/{name:<20} mean={np.mean(m):.4f}{param_str} [{elapsed:.1f}s]")

    print(f"\nTotal time: {time.time() - total_t0:.1f}s")
    print(f"Results saved to {cache_path}")


if __name__ == "__main__":
    main()
