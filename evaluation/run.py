import os
import sys
import json
import time
import subprocess
import torch
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from data.lorenz63 import Lorenz63Config
from evaluation.baselines import Weak4DVar, Strong4DVar, EnKF, ETKF
from models.lorenz63_dynamics import Lorenz63Dynamics

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXP_DIR = os.path.join(BASE, "experiments")
L63_EXP_DIR = os.path.join(EXP_DIR, "l63")
os.makedirs(L63_EXP_DIR, exist_ok=True)

_BASELINE_METHODS = ["Weak-4DVar", "Strong-4DVar", "EnKF", "ETKF"]
_BASELINE_CASES = [
    ("s0", "test_s0", 1, 0.0, "S0", 1.6),
    ("s1", "test_s1", 2, 0.15, "S1", 1.0),
]


def _baseline_traj_path(case_name, method_name, suffix=""):
    key = f"{case_name}_{method_name.replace('-', '_').replace(' ', '_')}"
    return os.path.join(L63_EXP_DIR, f"baselines_trajs{suffix}_{key}.npz")


def fmt_rmse(mean_arr, std_arr, ensemble_spread_arr=None, r2_arr=None,
             crps_mean_arr=None, crps_std_arr=None, elapsed_seconds=None):
    """`mean`/`std` are the per-window RMSE mean/std (over the test windows).

    `ensemble_spread_arr`, when given (EnKF/ETKF only), is the RMS ensemble
    spread sqrt(mean_variance) -- a measure of the filter's own uncertainty
    estimate, not of accuracy variability across windows.

    `r2_arr` is R^2 (coefficient of determination, 1 - SS_res/SS_tot) pooled
    over all windows/timesteps together, so (unlike the other metrics) it
    has no window-to-window std.

    `crps_mean_arr`/`crps_std_arr` are the per-window CRPS mean/std across
    test windows. A point forecast (4DVar) reduces CRPS to MAE; an ensemble
    forecast (EnKF/ETKF) uses the standard ensemble CRPS estimator.

    `elapsed_seconds` is the wall-clock time spent evaluating this method
    over all test windows.
    """
    dims = ("X", "Y", "Z")
    out = {
        "X": {"rmse": {"mean": float(mean_arr[0]), "std": float(std_arr[0])}},
        "Y": {"rmse": {"mean": float(mean_arr[1]), "std": float(std_arr[1])}},
        "Z": {"rmse": {"mean": float(mean_arr[2]), "std": float(std_arr[2])}},
        "rmse": {"mean": float(np.mean(mean_arr)), "std": float(np.mean(std_arr))},
    }
    if ensemble_spread_arr is not None:
        for i, d in enumerate(dims):
            out[d]["ensemble_spread"] = float(ensemble_spread_arr[i])
        out["ensemble_spread"] = float(np.mean(ensemble_spread_arr))
    if r2_arr is not None:
        for i, d in enumerate(dims):
            out[d]["r2"] = float(r2_arr[i])
        out["r2"] = float(np.mean(r2_arr))
    if crps_mean_arr is not None:
        for i, d in enumerate(dims):
            out[d]["crps"] = {"mean": float(crps_mean_arr[i]), "std": float(crps_std_arr[i])}
        out["crps"] = {"mean": float(np.mean(crps_mean_arr)), "std": float(np.mean(crps_std_arr))}
    if elapsed_seconds is not None:
        out["elapsed_seconds"] = float(elapsed_seconds)
    return out


def evaluate_baseline(method, dataset, cfg, device, return_trajs=False, batch_size=1,
                       use_corrupted_forcing=None):
    """Run `method` over every window in `dataset`.

    The baseline is fed the dataset's (possibly per-window, possibly biased)
    sigma/rho/beta/c1 values when present, falling back to `cfg.da_params`
    otherwise.

    `use_corrupted_forcing` selects which forcing array (`forcing_true` vs
    `forcing_corrupted`) is fed to the baseline. Defaults to
    `cfg.use_corrupted_forcing` (i.e. `cfg.case == 2`) when not given
    explicitly -- pass it explicitly to avoid coupling the forcing choice to
    `cfg.case`, which also controls `cfg.da_params`'s bias.
    """
    sig, rho, bet = cfg.da_params
    rmse_list = []
    results_list = []
    all_sq_err = []
    all_ref = []

    if use_corrupted_forcing is None:
        use_corrupted_forcing = getattr(cfg, 'use_corrupted_forcing', True)
    force_key = "forcing_corrupted" if use_corrupted_forcing else "forcing_true"
    has_per_window_params = "sigma" in dataset[0]

    if batch_size > 1 and hasattr(method, 'assimilate_batch'):
        for i in range(0, len(dataset), batch_size):
            batch = [dataset[j] for j in range(i, min(i + batch_size, len(dataset)))]
            obs = torch.stack([w["obs"].to(device) for w in batch], dim=0)
            mask = torch.stack([w["obs_mask"].to(device) for w in batch], dim=0)
            truth = torch.stack([w["true_state"] for w in batch], dim=0)
            force = torch.stack([w[force_key].to(device) for w in batch], dim=0)
            if has_per_window_params:
                sigma = torch.tensor([w["sigma"] for w in batch], device=device)
                rho = torch.tensor([w["rho"] for w in batch], device=device)
                beta = torch.tensor([w["beta"] for w in batch], device=device)
                c1 = torch.tensor([w.get("c1", 1.0) for w in batch], device=device)
            else:
                sigma, rho, beta = sig, rho, bet
                c1 = 1.0
            results = method.assimilate_batch(obs, mask, force, truth, sigma=sigma, rho=rho, beta=beta, c1=c1)
            for result_idx, result in enumerate(results):
                analysis = result.trajectory
                ref = truth[result_idx].detach().cpu().numpy()
                all_sq_err.append((analysis - ref) ** 2)
                all_ref.append(ref)
                rmse_list.append(result.rmse)
                results_list.append(result)
    else:
        for i in range(len(dataset)):
            w = dataset[i]
            obs = w["obs"].to(device)
            mask = w["obs_mask"].to(device)
            truth = w["true_state"]
            force = w[force_key].to(device)
            s = w.get("sigma", sig)
            r = w.get("rho", rho)
            b = w.get("beta", bet)
            c1_val = w.get("c1", 1.0)
            result = method.assimilate(obs, mask, force, truth, sigma=s, rho=r, beta=b, c1=c1_val)
            analysis = result.trajectory
            ref = truth.numpy()
            all_sq_err.append((analysis - ref) ** 2)
            all_ref.append(ref)
            rmse_list.append(result.rmse)
            results_list.append(result)

    all_rmse = np.stack(rmse_list, axis=0)
    rmse_stats = (np.mean(all_rmse, axis=0), np.std(all_rmse, axis=0))
    all_sq_err = np.concatenate(all_sq_err, axis=0)
    all_ref = np.concatenate(all_ref, axis=0)
    pooled_mse = np.mean(all_sq_err, axis=0)
    pooled_var = np.var(all_ref, axis=0)
    pooled_var = np.maximum(pooled_var, 1e-12)
    expvar = 1 - pooled_mse / pooled_var
    expvar_stats = (expvar, np.zeros_like(expvar))
    if return_trajs:
        return (rmse_stats, expvar_stats), results_list
    return rmse_stats, expvar_stats


def run_and_cache_baselines(datasets, device, batch_size=1, da_window_steps=None,
                             weak_config=None, strong_config=None, enkf_config=None,
                             etkf_config=None, suffix=""):
    if da_window_steps is None:
        N = int(3.0 / 0.01)
    else:
        N = da_window_steps
    cache_path = os.path.join(L63_EXP_DIR, f"baselines{suffix}.json")

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

    dynamics_pool = {}
    for expo in {c[5] for c in _BASELINE_CASES}:
        dynamics_pool[expo] = Lorenz63Dynamics(dt=0.01, coupling_exponent=expo)
    baseline_pool = {}
    for expo in {c[5] for c in _BASELINE_CASES}:
        dynamics = dynamics_pool[expo]
        baseline_pool[expo] = {
            "Weak-4DVar": Weak4DVar(dt=0.01, da_window_steps=N, device=device,
                                     coupling_exponent=expo, dynamics=dynamics, **weak_cfg),
            "Strong-4DVar": Strong4DVar(dt=0.01, da_window_steps=N, device=device,
                                         coupling_exponent=expo, dynamics=dynamics, **strong_cfg),
            "EnKF": EnKF(dt=0.01, device=device, coupling_exponent=expo, dynamics=dynamics, **enkf_cfg),
            "ETKF": ETKF(dt=0.01, device=device, coupling_exponent=expo, dynamics=dynamics, **etkf_cfg),
        }

    cfg_s0 = Lorenz63Config(param_bias=0.0, forcing_state_bias=0.0, T_max=3.0, seed=123)
    cfg_s1 = Lorenz63Config(param_bias=0.15, forcing_state_bias=0.1, T_max=3.0, seed=131)
    cfg_map = {"s0": cfg_s0, "s1": cfg_s1}

    if "config" not in partial:
        partial["config"] = {
            "T_max": 3.0, "da_window_steps": N,
            "enkf_inflation": enkf_cfg.get("inflation", 1.0),
            "etkf_inflation": etkf_cfg.get("inflation", 1.0),
        }

    total_t0 = time.time()

    for case_name, ds_key, case_val, bias, label, coupling_exponent in _BASELINE_CASES:
        if ds_key not in datasets:
            continue
        ds = datasets[ds_key]
        cfg = cfg_map[case_name]
        method_map = baseline_pool[coupling_exponent]
        for name in _BASELINE_METHODS:
            if partial.get(case_name, {}).get(name) is not None:
                print(f"    {label}/{name:<15} already done, skipping")
                continue

            method = method_map[name]
            print(f"    {label}/{name:<15} ...", end=" ", flush=True)
            t1 = time.time()
            ((m, s), (r2, _)), bl_results = evaluate_baseline(method, ds, cfg, device, return_trajs=True,
                                                                batch_size=batch_size,
                                                                use_corrupted_forcing=(case_val == 2))
            elapsed = time.time() - t1

            all_ens_var = None
            if bl_results[0].ensemble_variance is not None:
                all_ens_var = np.stack([r.ensemble_variance for r in bl_results], axis=0)
            ensemble_spread = (np.sqrt(np.mean(all_ens_var, axis=(0, 1)))
                                if all_ens_var is not None else None)

            crps_mean = crps_std = None
            if bl_results[0].es is not None:
                all_es = np.stack([r.es for r in bl_results], axis=0)
                crps_mean = np.mean(all_es, axis=0)
                crps_std = np.std(all_es, axis=0)

            if case_name not in partial:
                partial[case_name] = {}
            partial[case_name][name] = fmt_rmse(m, s, ensemble_spread, r2, crps_mean, crps_std, elapsed)
            partial["total_time_seconds"] = time.time() - total_t0
            with open(cache_path, "w") as f:
                json.dump(partial, f, indent=2)

            trajs = np.stack([r.trajectory for r in bl_results], axis=0)
            truths = np.stack([ds[i]["true_state"].numpy() for i in range(len(ds))], axis=0)
            traj_data = {"trajectories": trajs, "truths": truths}
            if all_ens_var is not None:
                traj_data["ensemble_variance"] = all_ens_var
            np.savez_compressed(_baseline_traj_path(case_name, name, suffix), **traj_data)

            print(f"X={m[0]:.4f} Y={m[1]:.4f} Z={m[2]:.4f}"
                  f"  mean={np.mean(m):.4f} [{elapsed:.1f}s]")

    # Combine per-method trajectory files into final .npz
    traj_path = os.path.join(L63_EXP_DIR, f"baselines_trajectories{suffix}.npz")
    all_present = all(
        os.path.exists(_baseline_traj_path(case_name, name, suffix))
        for case_name, _, _, _, _, _ in _BASELINE_CASES
        for name in _BASELINE_METHODS
    )

    if all_present:
        print("  Combining trajectories...")
        traj_arrays = {}
        for case_name, _, _, _, _, _ in _BASELINE_CASES:
            for name in _BASELINE_METHODS:
                src = _baseline_traj_path(case_name, name, suffix)
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
