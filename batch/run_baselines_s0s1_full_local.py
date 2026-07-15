import os
import sys
import json
import time
import torch
import numpy as np

sys.path.insert(0, ".")
from data.lorenz63 import Lorenz63Config, make_s0_s1_trainval
from evaluation.baselines import (
    Weak4DVar, Strong4DVar, EnKF, ETKF,
    JointWeak4DVar, JointStrong4DVar, JointEnKF, JointETKF,
)
from evaluation.metrics import param_rmse
from evaluation.run import evaluate_baseline

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXP_DIR = os.path.join(BASE, "experiments")
os.makedirs(EXP_DIR, exist_ok=True)

print("============================================")
print(" Joint Baseline S0/S1 Full Run (local)")
print("============================================")

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

base_cfg = Lorenz63Config()
datasets = make_s0_s1_trainval(
    base_cfg,
    num_train_windows=2000,
    num_val_windows=100,
    num_test_windows=200,
    param_noise=0.2,
    bias_range=(0.0, 0.2),
)
print(f"test_s0: {len(datasets['test_s0'])} windows")
print(f"test_s1: {len(datasets['test_s1'])} windows")

DA_WINDOW_STEPS = 50
ENKF_INFLATION = 2.0
ETKF_INFLATION = 2.0
BATCH_SIZE = 200

# case_name, ds_key, coupling_exponent, DA cfg -- matches evaluation/run.py's
# _BASELINE_CASES / cfg_map so state RMSE stays comparable to the vanilla run.
cfg_s0 = Lorenz63Config(case=1, param_bias=0.0, forcing_state_bias=0.0, T_max=3.0, seed=123)
cfg_s1 = Lorenz63Config(case=2, param_bias=0.15, forcing_state_bias=0.1, T_max=3.0, seed=131)
CASES = [("s0", "test_s0", cfg_s0, 1.6), ("s1", "test_s1", cfg_s1, 1.0)]

method_factories = {
    "Weak-4DVar":         lambda ce: Weak4DVar(dt=0.01, da_window_steps=DA_WINDOW_STEPS, device=device, coupling_exponent=ce, opt_steps=150, lr=0.02),
    "Joint-Weak-4DVar":   lambda ce: JointWeak4DVar(dt=0.01, da_window_steps=DA_WINDOW_STEPS, device=device, coupling_exponent=ce, opt_steps=150, lr=0.02),
    "Strong-4DVar":       lambda ce: Strong4DVar(dt=0.01, da_window_steps=DA_WINDOW_STEPS, device=device, coupling_exponent=ce, max_iter=40, lr=0.1),
    "Joint-Strong-4DVar": lambda ce: JointStrong4DVar(dt=0.01, da_window_steps=DA_WINDOW_STEPS, device=device, coupling_exponent=ce, max_iter=40, lr=0.1),
    "EnKF":               lambda ce: EnKF(dt=0.01, device=device, coupling_exponent=ce, N_ensemble=30, inflation=ENKF_INFLATION),
    "Joint-EnKF":         lambda ce: JointEnKF(dt=0.01, device=device, coupling_exponent=ce, N_ensemble=30, inflation=ENKF_INFLATION),
    "ETKF":               lambda ce: ETKF(dt=0.01, device=device, coupling_exponent=ce, N_ensemble=30, inflation=ETKF_INFLATION),
    "Joint-ETKF":         lambda ce: JointETKF(dt=0.01, device=device, coupling_exponent=ce, N_ensemble=30, inflation=ETKF_INFLATION),
}

cache_path = os.path.join(
    EXP_DIR,
    f"baselines_joint_dws{DA_WINDOW_STEPS}_s0s1_inf{ENKF_INFLATION}_etkf_inf{ETKF_INFLATION}.json",
)
samples_path = os.path.join(
    EXP_DIR,
    f"baselines_joint_dws{DA_WINDOW_STEPS}_s0s1_inf{ENKF_INFLATION}_etkf_inf{ETKF_INFLATION}_samples.npz",
)

partial = {}
if os.path.exists(cache_path):
    with open(cache_path) as f:
        partial = json.load(f)
    print(f"  Found partial results ({cache_path}), resuming...")
else:
    print(f"  Running joint baselines (da_window_steps={DA_WINDOW_STEPS})...")

samples = dict(np.load(samples_path)) if os.path.exists(samples_path) else {}

total_t0 = time.time()

for case_name, ds_key, cfg, coupling_exponent in CASES:
    ds = datasets[ds_key]
    if case_name not in partial:
        partial[case_name] = {}

    all_true_params = np.stack([
        np.array([w["true_sigma"], w["true_rho"], w["true_beta"], w["true_c1"]])
        for w in ds
    ], axis=0)

    pairs = [
        ("Weak-4DVar", "Joint-Weak-4DVar"),
        ("Strong-4DVar", "Joint-Strong-4DVar"),
        ("EnKF", "Joint-EnKF"),
        ("ETKF", "Joint-ETKF"),
    ]

    for base_name, joint_name in pairs:
        base_key = f"{case_name}__{base_name.replace('-', '_')}"
        joint_key = f"{case_name}__{joint_name.replace('-', '_')}"
        pair_done = (
            partial[case_name].get(base_name) is not None
            and partial[case_name].get(joint_name) is not None
            and f"{base_key}__best_traj" in samples
            and f"{joint_key}__best_traj" in samples
        )
        if pair_done:
            print(f"    {case_name}/{base_name:<20} + {joint_name:<20} already done, skipping")
            continue

        # Both variants of a pair are (re)computed together: the trajectory
        # samples below are selected from the JOINT method's own RMSE ranking,
        # so Default's bl_results must still be in memory when Joint is
        # processed rather than relying on a possibly-stale cache.
        bl_results_by_name = {}
        for method_name in (base_name, joint_name):
            factory = method_factories[method_name]
            method = factory(coupling_exponent)
            print(f"    {case_name}/{method_name:<20} ...", end=" ", flush=True)
            t1 = time.time()
            (m, s), bl_results = evaluate_baseline(
                method, ds, cfg, device, return_trajs=True, batch_size=BATCH_SIZE)
            elapsed = time.time() - t1
            bl_results_by_name[method_name] = bl_results

            entry = {
                "state_rmse": {
                    "X": {"mean": float(m[0]), "std": float(s[0])},
                    "Y": {"mean": float(m[1]), "std": float(s[1])},
                    "Z": {"mean": float(m[2]), "std": float(s[2])},
                    "mean": float(np.mean(m)),
                    "std": float(np.sqrt(np.mean(np.square(s)))),
                },
            }

            is_joint = method_name.startswith("Joint-")
            if is_joint and bl_results[0].params is not None:
                pred_params = np.stack([r.params[-1] for r in bl_results], axis=0)
                prmse = param_rmse(pred_params, all_true_params)
                perr_std = np.std(np.abs(pred_params - all_true_params), axis=0)
                entry["param_rmse"] = {
                    "sigma": {"mean": float(prmse[0]), "std": float(perr_std[0])},
                    "rho": {"mean": float(prmse[1]), "std": float(perr_std[1])},
                    "beta": {"mean": float(prmse[2]), "std": float(perr_std[2])},
                    "c1": {"mean": float(prmse[3]), "std": float(perr_std[3])},
                }
            else:
                entry["param_rmse"] = None

            partial[case_name][method_name] = entry
            partial["total_time_seconds"] = time.time() - total_t0
            with open(cache_path, "w") as f:
                json.dump(partial, f, indent=2)

            state_str = (f"X={m[0]:.4f}±{s[0]:.4f} Y={m[1]:.4f}±{s[1]:.4f} "
                         f"Z={m[2]:.4f}±{s[2]:.4f} mean={np.mean(m):.4f}±{np.sqrt(np.mean(np.square(s))):.4f}")
            if entry["param_rmse"]:
                p = entry["param_rmse"]
                param_str = (f" | params s={p['sigma']['mean']:.4f}±{p['sigma']['std']:.4f} "
                             f"r={p['rho']['mean']:.4f}±{p['rho']['std']:.4f} "
                             f"b={p['beta']['mean']:.4f}±{p['beta']['std']:.4f} "
                             f"c1={p['c1']['mean']:.4f}±{p['c1']['std']:.4f}")
            else:
                param_str = ""
            print(f"{state_str}{param_str} [{elapsed:.1f}s]")

        # Cache best/median/worst reconstructions (by per-window mean RMSE)
        # for the trajectory-sample pages in reports/generate_baseline_report.py.
        # Windows are selected by the JOINT method's own RMSE ranking (the
        # report's focus is how well joint estimation does); Default's
        # reconstruction on those same windows is stored alongside for a
        # correctly-paired comparison -- both plotted against one shared
        # ground truth, never two different test windows overlaid as if they
        # were the same (that was a prior bug: Default/Joint each picking
        # their own best/median/worst window independently).
        joint_bl = bl_results_by_name[joint_name]
        default_bl = bl_results_by_name[base_name]
        per_window_rmse_joint = np.array([np.mean(r.rmse) for r in joint_bl])
        per_window_rmse_default = np.array([np.mean(r.rmse) for r in default_bl])
        order = np.argsort(per_window_rmse_joint)
        picks = {"best": int(order[0]), "median": int(order[len(order) // 2]), "worst": int(order[-1])}

        for tag, idx in picks.items():
            truth = ds[idx]["true_state"].cpu().numpy()
            obs_mask = ds[idx]["obs_mask"].cpu().numpy()

            samples[f"{joint_key}__{tag}_idx"] = np.array(idx)
            samples[f"{joint_key}__{tag}_traj"] = joint_bl[idx].trajectory
            samples[f"{joint_key}__{tag}_truth"] = truth
            samples[f"{joint_key}__{tag}_obs_mask"] = obs_mask
            samples[f"{joint_key}__{tag}_rmse"] = np.array(per_window_rmse_joint[idx])

            samples[f"{base_key}__{tag}_idx"] = np.array(idx)
            samples[f"{base_key}__{tag}_traj"] = default_bl[idx].trajectory
            samples[f"{base_key}__{tag}_truth"] = truth
            samples[f"{base_key}__{tag}_obs_mask"] = obs_mask
            samples[f"{base_key}__{tag}_rmse"] = np.array(per_window_rmse_default[idx])
        np.savez(samples_path, **samples)

print(f"\n{'=' * 150}")
print(f"  {'Case':<6} {'Method':<20} {'State RMSE (X/Y/Z = mean, as mean±std)':<70} {'Param RMSE (s/r/b/c1, as mean±std)':<70} {'Ratio':<8}")
print(f"  {'-' * 148}")
all_finite = True
for case_name, ds_key, cfg, coupling_exponent in CASES:
    case_results = partial[case_name]
    vanilla_states = {
        bn: case_results[bn]["state_rmse"]["mean"]
        for bn in ["Weak-4DVar", "Strong-4DVar", "EnKF", "ETKF"] if bn in case_results
    }
    for method_name, entry in case_results.items():
        st = entry["state_rmse"]
        state_str = (f"{st['X']['mean']:.4f}±{st['X']['std']:.4f}/"
                     f"{st['Y']['mean']:.4f}±{st['Y']['std']:.4f}/"
                     f"{st['Z']['mean']:.4f}±{st['Z']['std']:.4f} = "
                     f"{st['mean']:.4f}±{st['std']:.4f}")
        for v in (st["X"]["mean"], st["Y"]["mean"], st["Z"]["mean"]):
            if not np.isfinite(v):
                all_finite = False

        if entry["param_rmse"]:
            p = entry["param_rmse"]
            param_str = (f"{p['sigma']['mean']:.4f}±{p['sigma']['std']:.4f}/"
                         f"{p['rho']['mean']:.4f}±{p['rho']['std']:.4f}/"
                         f"{p['beta']['mean']:.4f}±{p['beta']['std']:.4f}/"
                         f"{p['c1']['mean']:.4f}±{p['c1']['std']:.4f}")
            for v in (p["sigma"]["mean"], p["rho"]["mean"], p["beta"]["mean"], p["c1"]["mean"]):
                if not np.isfinite(v):
                    all_finite = False
        else:
            param_str = "N/A"

        if method_name.startswith("Joint-"):
            vn = method_name.replace("Joint-", "")
            v_mean = vanilla_states.get(vn, 0)
            ratio_str = f"{st['mean'] / v_mean:.4f}" if v_mean > 0 else "N/A"
        else:
            ratio_str = "N/A"

        print(f"  {case_name:<6} {method_name:<20} {state_str:<70} {param_str:<70} {ratio_str:<8}")
print(f"  {'-' * 148}")

if all_finite:
    print("\nAll results finite!")
else:
    print("\nNaN detected!")

print(f"\nSaved joint baseline results to {cache_path}")
