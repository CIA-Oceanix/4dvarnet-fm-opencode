import os, sys, json, time, subprocess
import torch
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from data.lorenz63 import Lorenz63Config
from evaluation.baselines import Weak4DVar, Strong4DVar, EnKF, ETKF
from evaluation.metrics import rmse

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXP_DIR = os.path.join(BASE, "experiments")
os.makedirs(EXP_DIR, exist_ok=True)

_BASELINE_METHODS = ["Weak-4DVar", "Strong-4DVar", "EnKF", "ETKF"]
_BASELINE_CASES = [
    ("cs1", "test_cs1", 1, 0.0, "CS1", "linear"),
    ("cs2", "test_cs2", 2, 0.15, "CS2", "quartic"),
    ("cs3", "test_cs3", 1, 0.0, "CS3", "linear"),
    ("cs4", "test_cs4", 2, 0.15, "CS4", "quartic"),
]


def _baseline_traj_path(case_name, method_name, dws_suffix="", param_suffix=""):
    key = f"{case_name}_{method_name.replace('-', '_').replace(' ', '_')}"
    return os.path.join(EXP_DIR, f"baselines_trajs{dws_suffix}{param_suffix}_{key}.npz")


def fmt_rmse(mean_arr, std_arr):
    return {
        "X": {"mean": float(mean_arr[0]), "std": float(std_arr[0])},
        "Y": {"mean": float(mean_arr[1]), "std": float(std_arr[1])},
        "Z": {"mean": float(mean_arr[2]), "std": float(std_arr[2])},
        "mean": float(np.mean(mean_arr)),
    }


def evaluate_baseline(method, dataset, cfg, device, return_trajs=False, batch_size=1):
    sig, rho, bet = cfg.da_params
    rmse_list = []
    results_list = []

    use_corrupted = getattr(cfg, 'use_corrupted_forcing', True)
    force_key = "forcing_corrupted" if use_corrupted else "forcing_true"

    if batch_size > 1 and hasattr(method, 'assimilate_batch'):
        for i in range(0, len(dataset), batch_size):
            batch = [dataset[j] for j in range(i, min(i + batch_size, len(dataset)))]
            obs = torch.stack([w["obs"].to(device) for w in batch], dim=0)
            mask = torch.stack([w["obs_mask"].to(device) for w in batch], dim=0)
            truth = torch.stack([w["true_state"] for w in batch], dim=0)
            force = torch.stack([w[force_key].to(device) for w in batch], dim=0)

            results = method.assimilate_batch(obs, mask, force, truth, sigma=sig, rho=rho, beta=bet)
            for result in results:
                rmse_list.append(result.rmse)
                results_list.append(result)
    else:
        for i in range(len(dataset)):
            w = dataset[i]
            obs = w["obs"].to(device)
            mask = w["obs_mask"].to(device)
            truth = w["true_state"]
            force = w[force_key].to(device)
            result = method.assimilate(obs, mask, force, truth, sigma=sig, rho=rho, beta=bet)
            rmse_list.append(result.rmse)
            results_list.append(result)

    all_rmse = np.stack(rmse_list, axis=0)
    stats = (np.mean(all_rmse, axis=0), np.std(all_rmse, axis=0))
    if return_trajs:
        return stats, results_list
    return stats


def run_and_cache_baselines(datasets, device, batch_size=1, da_window_steps=None,
                             weak_config=None, strong_config=None, enkf_config=None,
                             etkf_config=None, suffix=""):
    if da_window_steps is None:
        N = int(3.0 / 0.01)
    else:
        N = da_window_steps
    dws_suffix = f"_dws{N}"
    param_suffix = suffix
    if enkf_config and enkf_config.get("inflation", 1.0) != 1.0:
        param_suffix += f"_inf{enkf_config['inflation']}"
    if etkf_config and etkf_config.get("inflation", 1.0) != 1.0:
        param_suffix += f"_etkf_inf{etkf_config['inflation']}"
    cache_path = os.path.join(EXP_DIR, f"baselines{dws_suffix}{param_suffix}.json")

    partial = {}
    if os.path.exists(cache_path):
        with open(cache_path) as f:
            partial = json.load(f)
        print(f"  Found partial results ({cache_path}), resuming...")
    else:
        print(f"  Running baselines (da_window_steps={N})...")

    weak_cfg = weak_config or {}
    strong_cfg = strong_config or {}
    enkf_cfg = enkf_config or {}
    etkf_cfg = etkf_config or {}

    baseline_pool = {}
    for coupling_type in ("linear", "quartic"):
        baseline_pool[coupling_type] = {
            "Weak-4DVar": Weak4DVar(dt=0.01, da_window_steps=N, device=device,
                                     coupling_type=coupling_type, **weak_cfg),
            "Strong-4DVar": Strong4DVar(dt=0.01, da_window_steps=N, device=device,
                                         coupling_type=coupling_type, **strong_cfg),
            "EnKF": EnKF(dt=0.01, device=device, coupling_type=coupling_type, **enkf_cfg),
            "ETKF": ETKF(dt=0.01, device=device, coupling_type=coupling_type, **etkf_cfg),
        }

    cfg_cs1 = Lorenz63Config(case=1, param_bias=0.0, T_max=3.0, seed=123)
    cfg_cs2 = Lorenz63Config(case=2, param_bias=0.15, forcing_state_bias=0.15,
                              forcing_coupling="quartic", T_max=3.0, seed=124)
    cfg_cs3 = Lorenz63Config(case=1, param_bias=0.0, T_max=3.0, seed=125)
    cfg_cs4 = Lorenz63Config(case=2, param_bias=0.15, forcing_state_bias=0.15,
                              forcing_coupling="quartic", T_max=3.0, seed=126)
    cfg_map = {"cs1": cfg_cs1, "cs2": cfg_cs2, "cs3": cfg_cs3, "cs4": cfg_cs4}

    if "config" not in partial:
        partial["config"] = {"T_max": 3.0, "da_window_steps": N}

    total_t0 = time.time()

    for case_name, ds_key, case_val, bias, label, coupling_type in _BASELINE_CASES:
        ds = datasets[ds_key]
        cfg = cfg_map[case_name]
        method_map = baseline_pool[coupling_type]
        for name in _BASELINE_METHODS:
            if partial.get(case_name, {}).get(name) is not None:
                print(f"    {label}/{name:<15} already done, skipping")
                continue

            method = method_map[name]
            print(f"    {label}/{name:<15} ...", end=" ", flush=True)
            t1 = time.time()
            (m, s), bl_results = evaluate_baseline(method, ds, cfg, device, return_trajs=True, batch_size=batch_size)
            elapsed = time.time() - t1

            if case_name not in partial:
                partial[case_name] = {}
            partial[case_name][name] = fmt_rmse(m, s)
            partial["total_time_seconds"] = time.time() - total_t0
            with open(cache_path, "w") as f:
                json.dump(partial, f, indent=2)

            trajs = np.stack([r.trajectory for r in bl_results], axis=0)
            truths = np.stack([ds[i]["true_state"].numpy() for i in range(len(ds))], axis=0)
            traj_data = {"trajectories": trajs, "truths": truths}
            if bl_results[0].ensemble_variance is not None:
                traj_data["ensemble_variance"] = np.stack(
                    [r.ensemble_variance for r in bl_results], axis=0)
            np.savez_compressed(_baseline_traj_path(case_name, name, dws_suffix, param_suffix), **traj_data)

            print(f"X={m[0]:.4f} Y={m[1]:.4f} Z={m[2]:.4f}"
                  f"  mean={np.mean(m):.4f} [{elapsed:.1f}s]")

    # Combine per-method trajectory files into final .npz
    traj_path = os.path.join(EXP_DIR, f"baselines_trajectories{dws_suffix}{param_suffix}.npz")
    all_present = all(
        os.path.exists(_baseline_traj_path(case_name, name, dws_suffix, param_suffix))
        for case_name, _, _, _, _, _ in _BASELINE_CASES
        for name in _BASELINE_METHODS
    )

    if all_present:
        print("  Combining trajectories...")
        traj_arrays = {}
        for case_name, _, _, _, _, _ in _BASELINE_CASES:
            for name in _BASELINE_METHODS:
                src = _baseline_traj_path(case_name, name, dws_suffix, param_suffix)
                data = np.load(src)
                prefix = f"{case_name}_{name.replace('-', '_').replace(' ', '_')}"
                for key in data.files:
                    traj_arrays[f"{prefix}_{key}"] = data[key]
                data.close()
                os.remove(src)
        np.savez_compressed(traj_path, **traj_arrays)
        print(f"    Saved: {traj_path}")
    else:
        print("    (incomplete — skipping trajectory combination)")

    report_script = os.path.join(BASE, "reports", "generate_report.py")
    subprocess.run([sys.executable, report_script,
                    "--baselines", cache_path], capture_output=True)
    return partial
