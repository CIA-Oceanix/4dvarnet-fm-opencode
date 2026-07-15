"""Baseline runner for the rotating shallow-water case study.

Runs Weak-4DVar, Strong-4DVar, EnKF, and ETKF on SW S0/S1 datasets
and computes per-component (ocean / atmosphere) metrics.

Observation model
-----------------
Ocean (layer 1): ``obs_stride_ocean`` spatial sub-sampling.
Atmosphere (layer 2): ``obs_stride_atmos`` spatial sub-sampling.
Observations are available at *every* DA-timestep (no temporal gaps).
"""

import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from data.shallow_water import (
    ShallowWaterConfig,
    ShallowWaterDataset,
    make_sw_obs_indices,
    make_sw_s0_s1_datasets,
)
from evaluation.baselines import Weak4DVar, Strong4DVar, EnKF, ETKF, ObsOperator
from evaluation.metrics import compute_sw_component_metrics
from models.shallow_water_dynamics import ShallowWaterDynamics

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXP_DIR = os.path.join(BASE, "experiments")
os.makedirs(EXP_DIR, exist_ok=True)

_BASELINE_METHODS = ["Weak-4DVar", "Strong-4DVar", "EnKF", "ETKF"]
_SW_CASES = [
    ("s0", "test_s0", "S0"),
    ("s1", "test_s1", "S1"),
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _create_sw_dynamics(
    config: ShallowWaterConfig, scenario: str = "S0"
) -> ShallowWaterDynamics:
    """Create a ShallowWaterDynamics instance for the given scenario.

    The S0/S1 perturbation is embedded in the dataset (different
    Bickley jet amplitudes), so the dynamics itself is the same.
    """
    return ShallowWaterDynamics(
        Nx=config.Nx,
        Ny=config.Ny,
        dt=config.dt,
        K=config.K,
        tau0=config.tau0,
        f_cor=config.f_cor,
        g1=config.g1,
        g2=config.g2,
        coupling=config.coupling,
        friction=config.friction,
        viscosity=config.viscosity,
        land_mask_type=config.land_mask_type,
    )


def _extract_obs_values(
    full_obs: torch.Tensor, obs_indices: torch.Tensor
) -> torch.Tensor:
    """Extract observed values from the full state vector.

    Parameters
    ----------
    full_obs : Tensor (..., state_dim)
        Full observation vector (NaN at unobserved positions).
    obs_indices : Tensor (n_obs,)
        Flat indices of observed state components.

    Returns
    -------
    Tensor (..., n_obs)
    """
    return full_obs[..., obs_indices]


def _temporal_mask(batch_size: int, K: int, device: torch.device) -> torch.Tensor:
    """Boolean temporal mask – all True (observations at every timestep)."""
    return torch.ones(batch_size, K, dtype=torch.bool, device=device)


def _average_sw_metrics(all_metrics: list) -> dict:
    """Average per-component metric dicts across multiple DA windows."""
    if not all_metrics:
        return {}

    result: dict = {}
    layer_names = ("ocean", "atmosphere")
    field_names = ("h", "u", "v")

    for layer_name in layer_names:
        layer_metrics: dict = {}
        for fname in field_names:
            rmse_vals = [m[layer_name][fname]["rmse"] for m in all_metrics]
            ev_vals = [m[layer_name][fname]["ev"] for m in all_metrics]
            layer_metrics[fname] = {
                "rmse": float(np.mean(rmse_vals)),
                "ev": float(np.mean(ev_vals)),
            }
        agg_rmse = [m[layer_name]["aggregate"]["rmse"] for m in all_metrics]
        agg_ev = [m[layer_name]["aggregate"]["ev"] for m in all_metrics]
        layer_metrics["aggregate"] = {
            "rmse": float(np.mean(agg_rmse)),
            "ev": float(np.mean(agg_ev)),
        }
        result[layer_name] = layer_metrics

    overall_rmse = [m["overall"]["rmse"] for m in all_metrics]
    overall_ev = [m["overall"]["ev"] for m in all_metrics]
    result["overall"] = {
        "rmse": float(np.mean(overall_rmse)),
        "ev": float(np.mean(overall_ev)),
    }
    return result


# ---------------------------------------------------------------------------
# Per-method evaluation
# ---------------------------------------------------------------------------

def evaluate_sw_baseline(
    method,
    dataset: ShallowWaterDataset,
    config: ShallowWaterConfig,
    obs_indices: torch.Tensor,
    device: torch.device,
    return_trajs: bool = False,
    batch_size: int = 1,
) -> dict | tuple:
    """Run a single DA method on a SW dataset.

    The method receives *pre-extracted* observation values of shape
    ``(T, n_obs)`` together with a boolean temporal mask (all True)
    and an ``ObsOperator`` that maps the full state back to the observed
    sub-space internally.

    Returns
    -------
    dict  (or (dict, list) when *return_trajs* is True)
        Averaged per-component metrics (ocean, atmosphere, overall).
    """
    K = config.K
    results_list: list = []

    if batch_size > 1 and hasattr(method, "assimilate_batch"):
        for i in range(0, len(dataset), batch_size):
            batch = [dataset[j] for j in range(i, min(i + batch_size, len(dataset)))]
            B = len(batch)

            obs_full = torch.stack(
                [w["obs"] for w in batch], dim=0
            )  # (B, K, state_dim)
            obs_vals = _extract_obs_values(obs_full, obs_indices)  # (B, K, n_obs)
            tmask = _temporal_mask(B, K, device)
            truth = torch.stack(
                [w["true_state"] for w in batch], dim=0
            )  # (B, K, state_dim)
            force = torch.stack(
                [w["forcing"] for w in batch], dim=0
            )  # (B, K, 2)

            batch_results = method.assimilate_batch(
                obs_vals.to(device), tmask, force.to(device), truth,
            )
            results_list.extend(batch_results)
    else:
        for i in range(len(dataset)):
            w = dataset[i]
            obs_full = w["obs"]  # (K, state_dim)
            obs_vals = _extract_obs_values(obs_full, obs_indices)  # (K, n_obs)
            tmask = torch.ones(K, dtype=torch.bool, device=device)
            truth = w["true_state"]  # (K, state_dim)
            force = w["forcing"]  # (K, 2)

            result = method.assimilate(
                obs_vals.to(device), tmask, force.to(device), truth,
            )
            results_list.append(result)

    # ---- per-window metrics → average ----
    all_metrics: list = []
    for idx in range(len(results_list)):
        analysis = results_list[idx].trajectory  # (K, state_dim)
        truth_np = dataset[idx]["true_state"].numpy()  # (K, state_dim)
        comp = compute_sw_component_metrics(
            analysis, truth_np, config.Nx, config.Ny,
        )
        all_metrics.append(comp)

    avg = _average_sw_metrics(all_metrics)

    if return_trajs:
        return avg, results_list
    return avg


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run_sw_baselines(
    config: ShallowWaterConfig,
    num_test_windows: int = 200,
    da_window_steps: int = 500,
    batch_size: int = 200,
    t_max: float = 3.0,
    enkf_inflation: float = 2.0,
    etkf_inflation: float = 2.0,
    output_dir: str = "outputs/sw_baselines",
    methods: list[str] | None = None,
) -> dict:
    """Run all four DA baselines on the rotating SW S0/S1 scenarios.

    Parameters
    ----------
    config : ShallowWaterConfig
        Base configuration (Nx, Ny, physical params, etc.).
        ``K`` is overridden by *da_window_steps* for dataset generation.
    num_test_windows : int
        Number of independent DA windows per scenario.
    da_window_steps : int
        Timesteps inside each DA window (also sets ``config.K``).
    batch_size : int
        Number of windows processed in parallel.  Use 1 for large grids
        to keep memory usage manageable.
    t_max : float
        Trajectory length in time units (stored for metadata only).
    enkf_inflation : float
        Multiplicative inflation for EnKF.
    etkf_inflation : float
        Multiplicative inflation for ETKF.
    output_dir : str
        Directory for cached artefacts (created if absent).
    methods : list[str], optional
        Subset of DA methods to run.  Defaults to all four.

    Returns
    -------
    dict
        ``{"S0": {method: component_metrics, ...},
          "S1": {method: component_metrics, ...},
          "config": {...}, "total_time_seconds": ...}``
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ----- dataset config (override K to match DA window) -----
    ds_config = ShallowWaterConfig(
        **{**config.__dict__, "K": da_window_steps, "num_windows": num_test_windows}
    )

    # ----- generate S0 / S1 datasets -----
    print("\n── Generating SW S0/S1 datasets ──")
    t0 = time.time()
    datasets = make_sw_s0_s1_datasets(ds_config, num_test_windows=num_test_windows)
    print(f"  test_s0: {len(datasets['test_s0'])} windows")
    print(f"  test_s1: {len(datasets['test_s1'])} windows")
    print(f"  Dataset generation: {time.time() - t0:.1f}s")

    # ----- observation operator -----
    obs_indices = make_sw_obs_indices(config)
    pct = 100.0 * len(obs_indices) / config.state_dim
    print(f"  Observed dims: {len(obs_indices)} / {config.state_dim} ({pct:.1f}%)")

    # ----- results container -----
    results: dict = {
        "config": {
            "T_max": t_max,
            "da_window_steps": da_window_steps,
            "Nx": config.Nx,
            "Ny": config.Ny,
            "n_obs": int(len(obs_indices)),
        }
    }

    total_t0 = time.time()

    for case_name, ds_key, label in _SW_CASES:
        if ds_key not in datasets:
            print(f"  Skipping {label}: dataset not found")
            continue

        ds = datasets[ds_key]
        print(f"\n── {label} baselines ({len(ds)} windows) ──")

        dynamics = _create_sw_dynamics(ds_config, scenario=case_name)
        obs_operator = ObsOperator(config.state_dim, obs_indices)

        _run_methods = methods or _BASELINE_METHODS

        method_instances = {
            "Weak-4DVar": Weak4DVar(
                dt=config.dt,
                da_window_steps=da_window_steps,
                device=device,
                dynamics=dynamics,
                obs_operator=obs_operator,
            ),
            "Strong-4DVar": Strong4DVar(
                dt=config.dt,
                da_window_steps=da_window_steps,
                device=device,
                dynamics=dynamics,
                obs_operator=obs_operator,
            ),
            "EnKF": EnKF(
                dt=config.dt,
                device=device,
                dynamics=dynamics,
                obs_operator=obs_operator,
                inflation=enkf_inflation,
                noise_init_std=0.05,
            ),
            "ETKF": ETKF(
                dt=config.dt,
                device=device,
                dynamics=dynamics,
                obs_operator=obs_operator,
                inflation=etkf_inflation,
                noise_init_std=0.05,
            ),
        }

        if case_name not in results:
            results[case_name] = {}

        for name in _run_methods:
            method = method_instances[name]
            print(f"    {label}/{name:<15} ...", end=" ", flush=True)
            t1 = time.time()

            avg_metrics = evaluate_sw_baseline(
                method,
                ds,
                ds_config,
                obs_indices,
                device,
                batch_size=batch_size,
            )

            elapsed = time.time() - t1
            rmse_val = avg_metrics.get("overall", {}).get("rmse", float("nan"))
            ev_val = avg_metrics.get("overall", {}).get("ev", float("nan"))
            print(f"RMSE={rmse_val:.4f}  EV={ev_val:.4f}  [{elapsed:.1f}s]")

            results[case_name][name] = avg_metrics

    results["total_time_seconds"] = time.time() - total_t0
    return results
