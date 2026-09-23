"""ETKF/EnKF/Strong-4DVar under a random observing system per window, drawn
with the training sampler (``data/obs_times.py::resample_obs_variable``,
``data.obs_random_layout``) except that step 0 is always observed: n_obs
uniform in ``--n-obs-range`` at stratified times (first block pinned to step
0), k uniform in ``--fast-range`` observed fast channels per window (subset
redrawn per obs time), the 8 slow channels always observed, R_var=0.5.

The background is built from the step-0 obs; the 16-k observed-space fast
channels missing at step 0 are filled by linear interpolation along the fast
ring (``evaluation.run_l96.make_fast_ring_fill``), the 16 never-observed fast
channels keep the P1 initialization. Every DA window (``--da-window-steps``)
holds at least one obs time: a draw leaving one empty is redrawn whole
(rejection; only binds for n_obs < 2*T/da_window_steps). Everything else is the P1 DA path
(30 members, Strong-4DVar max_iter=10 lr=0.2, DA window 500, batched,
per-window fast_weights). A degenerate range (``--n-obs-range 20 20
--fast-range 8 8``) gives one cell of the factorial grid.

The drawn layouts are saved (``layouts<tag>.pt``) so learned schemes can be
scored on the exact same observations (identical for every inflation).

  python eval_da_random_layout_l96.py                           # arm A, 200 windows
  python eval_da_random_layout_l96.py --n-windows 20 --n-draws 3 --n-obs-range 5 5 --fast-range 16 16
"""
from __future__ import annotations

import argparse
import json
import os
import zlib

import numpy as np
import torch

from data.obs_times import resample_obs_variable
from eval_da_obs_count_l96 import CASES, DEFAULT_CACHE, R_VAR, per_window_metrics
from evaluation.run_l96 import EXP_DIR, make_fast_ring_fill, make_obs_j_indices, run_and_cache_baselines

OUT_DIR = os.path.join(EXP_DIR, "l96_da_random_layout")
METHODS = ("ETKF", "EnKF", "Strong-4DVar")
NUM_SLOW = 8
MAX_REDRAWS = 1000


def _layout_seed(case: str, window: int, n_obs_range, fast_range, draw: int) -> int:
    key = f"{case}/{window}/{n_obs_range[0]}-{n_obs_range[1]}/{fast_range[0]}-{fast_range[1]}/{draw}"
    return zlib.crc32(key.encode())


def covers_da_windows(mask: torch.Tensor, da_window_steps: int) -> bool:
    """Every ``da_window_steps`` chunk of the (T,) mask has an obs time."""
    T = mask.shape[0]
    return all(bool(mask[s:s + da_window_steps].any()) for s in range(0, T, da_window_steps))


def draw_layout(truth_obs: torch.Tensor, n_obs_range, fast_range, seed: int,
                da_window_steps: int) -> tuple[torch.Tensor, torch.Tensor]:
    """(T, od) obs and (T,) mask for one window's observed-space truth, with
    at least one obs time in every DA window."""
    T = truth_obs.shape[0]
    if n_obs_range[0] < -(-T // da_window_steps):
        raise ValueError(f"n_obs >= {n_obs_range[0]} cannot cover {-(-T // da_window_steps)} DA windows")
    g = torch.Generator().manual_seed(seed)
    for _ in range(MAX_REDRAWS):
        obs, mask = resample_obs_variable(
            truth_obs.unsqueeze(0), torch.zeros(1, T, dtype=torch.bool), R_VAR,
            tuple(n_obs_range), tuple(fast_range), num_slow=NUM_SLOW, generator=g, pin_first=True)
        if covers_da_windows(mask[0], da_window_steps):
            return obs[0], mask[0]
    raise RuntimeError(f"no DA-window-covering layout in {MAX_REDRAWS} draws")


def build_cell(datasets: dict, windows: list[int], n_draws: int, n_obs_range, fast_range,
               obs_var_indices, da_window_steps: int) -> tuple[dict, dict]:
    """Window dicts with their obs replaced (window-major, draw-minor) and the
    per-run layout record {case: {n_obs, k, obs, obs_mask, window_index}}."""
    idx = list(obs_var_indices)
    cell, layouts = {}, {}
    for case, key in CASES.items():
        items, rec = [], {"n_obs": [], "k": [], "obs": [], "obs_mask": [], "window_index": []}
        for wi in windows:
            w = datasets[key][wi]
            for d in range(n_draws):
                obs, mask = draw_layout(w["true_state"][:, idx].float(), n_obs_range, fast_range,
                                        _layout_seed(case, wi, n_obs_range, fast_range, d), da_window_steps)
                items.append(dict(w, obs=obs.to(w["true_state"].dtype), obs_mask=mask))
                rec["n_obs"].append(int(mask.sum()))
                rec["k"].append(int(torch.isfinite(obs[0, NUM_SLOW:]).sum()))
                rec["obs"].append(obs)
                rec["obs_mask"].append(mask)
                rec["window_index"].append(wi)
        cell[key] = items
        layouts[case] = {"n_obs": np.array(rec["n_obs"]), "k": np.array(rec["k"]),
                         "window_index": np.array(rec["window_index"]),
                         "obs": torch.stack(rec["obs"]), "obs_mask": torch.stack(rec["obs_mask"])}
    return cell, layouts


def _traj_path(tag: str, inflation: float, da_window_steps: int) -> str:
    inf = f"_inf{inflation}_etkf_inf{inflation}" if inflation != 1.0 else ""
    return os.path.join(EXP_DIR, f"l96_baselines_trajectories_dws{da_window_steps}{tag}{inf}_obsj2_fw_dafw.npz")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--n-obs-range", type=int, nargs=2, default=[5, 50])
    p.add_argument("--fast-range", type=int, nargs=2, default=[4, 16])
    p.add_argument("--n-windows", type=int, default=200)
    p.add_argument("--n-draws", type=int, default=1)
    p.add_argument("--inflation", type=float, default=2.0)
    p.add_argument("--da-window-steps", type=int, default=500)
    p.add_argument("--methods", nargs="+", default=list(METHODS), choices=METHODS)
    p.add_argument("--data-cache", default=DEFAULT_CACHE)
    p.add_argument("--device", default=None)
    args = p.parse_args()

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    datasets = torch.load(args.data_cache, weights_only=False)
    n_total = len(datasets["test_s0"])
    windows = list(np.linspace(0, n_total, args.n_windows, endpoint=False).astype(int))
    obs_var_indices = make_obs_j_indices(8, 4, 2)
    idx = np.array(obs_var_indices)
    (lo, hi), (flo, fhi) = args.n_obs_range, args.fast_range
    cell, layouts = build_cell(datasets, windows, args.n_draws, (lo, hi), (flo, fhi), obs_var_indices,
                               args.da_window_steps)
    tag = f"_rlayout_n{lo}-{hi}_k{flo}-{fhi}_w{args.n_windows}_d{args.n_draws}"
    print(f"n_obs {lo}-{hi}, k {flo}-{fhi}, inflation {args.inflation}, {len(cell['test_s0'])} runs per case")

    out_tag = f"{tag}_inf{args.inflation}"
    os.makedirs(OUT_DIR, exist_ok=True)
    torch.save(layouts, os.path.join(OUT_DIR, f"layouts{out_tag}.pt"))

    fill = make_fast_ring_fill(8, 4, 2)
    run_and_cache_baselines(
        cell, device, batch_size=200, da_window_steps=args.da_window_steps,
        enkf_config={"inflation": args.inflation, "init_fill": fill},
        etkf_config={"inflation": args.inflation, "init_fill": fill},
        strong_config={"max_iter": 10, "lr": 0.2, "init_fill": fill},
        suffix=tag, exclude_methods=["Weak-4DVar"] + [m for m in METHODS if m not in args.methods],
        obs_j=2, obs_interval=None, fw_randomized=True, da_fast_weights=True,
    )
    z = np.load(_traj_path(tag, args.inflation, args.da_window_steps))

    summary = {"n_obs_range": [lo, hi], "fast_range": [flo, fhi], "windows": [int(w) for w in windows],
               "n_draws": args.n_draws, "inflation": args.inflation, "N_ensemble": 30,
               "da_fast_weights": True, "step0_observed": True, "init_fill": "fast_ring_linear",
               "da_window_steps": args.da_window_steps, "min_obs_per_da_window": 1,
               "source": os.path.basename(_traj_path(tag, args.inflation, args.da_window_steps)), "cases": {}}
    arrays = {}
    for case, key in CASES.items():
        truth = np.stack([w["true_state"].numpy()[:, idx] for w in cell[key]]).astype(np.float64)
        summary["cases"][case] = {}
        arrays[f"{case}_n_obs"] = layouts[case]["n_obs"]
        arrays[f"{case}_k"] = layouts[case]["k"]
        arrays[f"{case}_window_index"] = layouts[case]["window_index"]
        for method in args.methods:
            k = f"{case}_{method.replace('-', '_')}"
            vkey = f"{k}_ensemble_variance"
            m = per_window_metrics(z[f"{k}_trajectories"].astype(np.float64), truth,
                                   z[vkey].astype(np.float64) if vkey in z.files else None, idx)
            summary["cases"][case][method] = {
                metric: {g: {"mean": float(v.mean()), "std": float(v.std(ddof=1)) if v.size > 1 else 0.0}
                         for g, v in groups.items()}
                for metric, groups in m.items()
            }
            for metric, groups in m.items():
                for g, v in groups.items():
                    arrays[f"{k}_{metric}_{g}"] = v
            r = summary["cases"][case][method]["rmse"]["all_obs"]
            print(f"  {case}/{method}: RMSE {r['mean']:.4f} ± {r['std']:.4f}")
    np.savez_compressed(os.path.join(OUT_DIR, f"per_window{out_tag}.npz"), **arrays)
    with open(os.path.join(OUT_DIR, f"summary{out_tag}.json"), "w") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
