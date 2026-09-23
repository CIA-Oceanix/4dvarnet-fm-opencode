"""ETKF/EnKF under a sweep of the number of observation times per window, at
random observation times (the first step always observed), on a subset of the
P1 200-window L96 test set, S0 and S1.

Everything but the observation times goes through the exact P1 baseline path
(``evaluation.run_l96.run_and_cache_baselines``: 30 members, inflation 2.0,
same S0/S1 dynamics and 24D obs operator, batched assimilation), so the
``reg30`` cell -- the cached regular 30-time obs of the same windows -- is
directly comparable to the P1 benchmark rows. Each ``n_obs`` cell observes
every selected window ``--n-draws`` times with independent obs-time/noise
draws. Per-window metrics follow ``reports/l96/generate_p1_l96_benchmark.py``
(RMSE, MAE, spread = mean ensemble std, grouped slow/obs_fast/all_obs on the
24D observed subspace).

  python eval_da_obs_count_l96.py --n-obs 10           # one sweep cell
  python eval_da_obs_count_l96.py --n-obs reg30        # P1 reference cell
"""
from __future__ import annotations

import argparse
import json
import os
import zlib

import numpy as np
import torch

from evaluation.estimate_metrics import _groups_from_per_window
from evaluation.run_l96 import EXP_DIR, make_obs_j_indices, run_and_cache_baselines

DEFAULT_CACHE = "/Odyssey/private/rfablet/Python/4dvarnet-fm-opencode/experiments/l96_datasets_obsj2_int100_nwin200.pt"
OUT_DIR = os.path.join(EXP_DIR, "l96_da_obs_count")
METHODS = ("ETKF", "EnKF")
CASES = {"s0": "test_s0", "s1": "test_s1"}
INFLATION = 2.0
R_VAR = 0.5


def random_obs_times(num_steps: int, n_obs: int, rng: np.random.Generator) -> np.ndarray:
    """Step 0 plus ``n_obs - 1`` distinct steps drawn uniformly from
    ``1..num_steps-1``, sorted."""
    if not 1 <= n_obs <= num_steps:
        raise ValueError(f"n_obs={n_obs} out of range [1, {num_steps}]")
    rest = rng.choice(np.arange(1, num_steps), size=n_obs - 1, replace=False)
    return np.concatenate([[0], np.sort(rest)])


def observe_at_times(true_state: torch.Tensor, times: np.ndarray, obs_var_indices,
                     R_var: float, rng: np.random.Generator) -> tuple[torch.Tensor, torch.Tensor]:
    """(T, od) obs, NaN except at ``times``, and the (T,) boolean mask."""
    truth = true_state[:, list(obs_var_indices)]
    obs = torch.full_like(truth, float("nan"))
    noise = torch.from_numpy(rng.standard_normal((len(times), truth.shape[1]))).to(truth.dtype)
    obs[times] = truth[times] + noise * float(np.sqrt(R_var))
    mask = torch.zeros(truth.shape[0], dtype=torch.bool)
    mask[times] = True
    return obs, mask


def _cell_seed(case: str, window: int, n_obs: int, draw: int) -> int:
    return zlib.crc32(f"{case}/{window}/{n_obs}/{draw}".encode())


def build_cell(datasets: dict, windows: list[int], n_obs: str, n_draws: int,
               obs_var_indices) -> dict:
    """{'test_s0': [...], 'test_s1': [...]} of shallow-copied window dicts with
    their obs replaced (window-major, draw-minor). ``reg30`` keeps the cached
    regular obs, one copy per window."""
    cell = {}
    for case, key in CASES.items():
        items = []
        for wi in windows:
            w = datasets[key][wi]
            if n_obs == "reg30":
                items.append(dict(w))
                continue
            for d in range(n_draws):
                rng = np.random.default_rng(_cell_seed(case, wi, int(n_obs), d))
                times = random_obs_times(w["true_state"].shape[0], int(n_obs), rng)
                obs, mask = observe_at_times(w["true_state"], times, obs_var_indices, R_VAR, rng)
                items.append(dict(w, obs=obs, obs_mask=mask))
        cell[key] = items
    return cell


def per_window_metrics(traj: np.ndarray, truth: np.ndarray, ens_var: np.ndarray | None,
                       idx: np.ndarray) -> dict:
    """P1-convention per-window metrics on the 24D observed subspace."""
    if traj.shape[-1] > len(idx):
        traj = traj[..., idx]
    out = {
        "rmse": _groups_from_per_window(np.sqrt(((traj - truth) ** 2).mean(axis=1))),
        "mae": _groups_from_per_window(np.abs(traj - truth).mean(axis=1)),
    }
    if ens_var is not None:
        if ens_var.shape[-1] > len(idx):
            ens_var = ens_var[..., idx]
        out["spread"] = _groups_from_per_window(np.sqrt(np.clip(ens_var, 0, None)).mean(axis=1))
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--n-obs", required=True, help="number of obs times per window, or 'reg30'")
    p.add_argument("--n-windows", type=int, default=10)
    p.add_argument("--n-draws", type=int, default=5)
    p.add_argument("--data-cache", default=DEFAULT_CACHE)
    p.add_argument("--device", default=None)
    p.add_argument("--da-fast-weights", action="store_true",
                   help="pass each window's fast_weights (S0 true, S1 *_da) to the DA model; "
                        "P1 did not, so its DA rows use unweighted fast coupling")
    args = p.parse_args()

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    datasets = torch.load(args.data_cache, weights_only=False)
    n_total = len(datasets["test_s0"])
    windows = list(np.linspace(0, n_total, args.n_windows, endpoint=False).astype(int))
    obs_var_indices = make_obs_j_indices(8, 4, 2)
    idx = np.array(obs_var_indices)
    cell = build_cell(datasets, windows, args.n_obs, args.n_draws, obs_var_indices)
    draws = 1 if args.n_obs == "reg30" else args.n_draws
    tag = f"_obscount_{args.n_obs}_w{args.n_windows}_d{draws}" + ("_dafw" if args.da_fast_weights else "")
    print(f"n_obs={args.n_obs} windows={windows} draws={draws} -> {len(cell['test_s0'])} runs per case")

    run_and_cache_baselines(
        cell, device, batch_size=200, da_window_steps=500,
        enkf_config={"inflation": INFLATION}, etkf_config={"inflation": INFLATION},
        suffix=tag, exclude_methods=["Weak-4DVar", "Strong-4DVar"],
        obs_j=2, obs_interval=None, fw_randomized=True, da_fast_weights=args.da_fast_weights,
    )
    traj_path = os.path.join(
        EXP_DIR, f"l96_baselines_trajectories_dws500{tag}_inf{INFLATION}_etkf_inf{INFLATION}_obsj2_fw"
                 f"{'_dafw' if args.da_fast_weights else ''}.npz")
    z = np.load(traj_path)

    os.makedirs(OUT_DIR, exist_ok=True)
    summary = {"n_obs": args.n_obs, "windows": [int(w) for w in windows], "n_draws": draws,
               "inflation": INFLATION, "N_ensemble": 30, "da_fast_weights": args.da_fast_weights, "source": os.path.basename(traj_path),
               "cases": {}}
    arrays = {}
    for case, key in CASES.items():
        truth = np.stack([w["true_state"].numpy()[:, idx] for w in cell[key]]).astype(np.float64)
        summary["cases"][case] = {}
        for method in METHODS:
            k = f"{case}_{method}"
            vkey = f"{k}_ensemble_variance"
            m = per_window_metrics(z[f"{k}_trajectories"].astype(np.float64), truth,
                                   z[vkey].astype(np.float64) if vkey in z.files else None, idx)
            summary["cases"][case][method] = {
                metric: {g: {"mean": float(v.mean()), "std": float(v.std(ddof=1))} for g, v in groups.items()}
                for metric, groups in m.items()
            }
            for metric, groups in m.items():
                for g, v in groups.items():
                    arrays[f"{k}_{metric}_{g}"] = v
            r = summary["cases"][case][method]["rmse"]["all_obs"]
            print(f"  {case}/{method}: RMSE {r['mean']:.4f} ± {r['std']:.4f}")
    arrays["window_index"] = np.repeat(windows, draws)
    np.savez_compressed(os.path.join(OUT_DIR, f"per_window{tag}.npz"), **arrays)
    with open(os.path.join(OUT_DIR, f"summary{tag}.json"), "w") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
