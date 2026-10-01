"""Per-window FM-operator score of the rows scorable from stored files (CPU).

- point estimates (FMS = squared error at every tau): DirectUNet-M (1200 ep, seeds 1-3),
  Strong-4DVar (regular grid only; the random-layout trajectories were not kept);
- Gaussian FMS of the benchmark-file ETKF / EnKF / ETKS from their stored per-time
  means and ensemble variances (both layouts).

Writes ``<out>/<name>_<layout>_<case>.npz`` (per-window arrays) and ``.json`` (pooled),
in the format of ``probe_fm_score_eval.py``.

usage: python reports/l96/probe_fm_score_stored.py OUT_DIR
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import _inputs  # noqa: E402
from evaluation.run_l96 import make_obs_j_indices  # noqa: E402
from reports.l96.probe_fm_score import norm_stats, pooled_from_windows, window_gaussian  # noqa: E402

EXT = _inputs.root("benchmark_extended", "here")
DA_REG = _inputs.root("p1_benchmark", "here") / "da_current_2026-09-29"
DA_RAND = EXT / "da_inflation_2026-09-28"
NEWINF = "_infs0-1.2_s1-3.0_etkf_infs0-1.15_s1-2.5"
ETKS = "etks-correct-Lfull-infs0-1.15_s1-2.5"
DU = {"reg": EXT, "rand": EXT / "eval_rlayout_n10-100_k4-16_w200"}
DA = {
    "reg": {
        "ETKF": DA_REG / f"l96_baselines_trajectories_dws500_s0c_crps{NEWINF}_obsj2_int100_fw_dafw.npz",
        "EnKF": DA_REG / f"l96_baselines_trajectories_dws500_s0c_crps{NEWINF}_obsj2_int100_fw_dafw.npz",
        "ETKS": DA_REG / f"l96_baselines_trajectories_dws500_s0c_test_{ETKS}{NEWINF}_obsj2_int100_fw_dafw.npz",
        "Strong_4DVar": DA_REG / "l96_baselines_trajectories_dws500_s0c_inf2.0_etkf_inf2.0_obsj2_int100_fw_dafw.npz",
    },
    "rand": {
        "ETKF": DA_RAND / f"l96_baselines_trajectories_dws500_rlayout_n10-100_k4-16_w200_d1{NEWINF}_obsj2_fw_dafw.npz",
        "EnKF": DA_RAND / f"l96_baselines_trajectories_dws500_rlayout_n10-100_k4-16_w200_d1{NEWINF}_obsj2_fw_dafw.npz",
        "ETKS": EXT / (f"l96_baselines_trajectories_dws500_rlayout_n10-100_k4-16_w200_d1_{ETKS}"
                       "_infs0-1.15_s1-2.5_etkf_infs0-1.15_s1-2.5_obsj2_fw_dafw.npz"),
    },
}


def save(out: Path, stem: str, arrays: dict) -> None:
    np.savez(out / f"{stem}.npz", **{k: v.astype(np.float32) for k, v in arrays.items()})
    stats = pooled_from_windows(arrays)
    (out / f"{stem}.json").write_text(json.dumps(stats, indent=1))
    print(f"{stem}: rmse {stats['rmse']:.4f}", flush=True)


def main(out: Path) -> None:
    mu, sd = norm_stats()
    idx = np.array(make_obs_j_indices(8, 4, 2))
    out.mkdir(parents=True, exist_ok=True)
    for lay in ("reg", "rand"):
        for case in ("s0", "s1"):
            truth = None
            for s in (1, 2, 3):
                z = np.load(DU[lay] / f"L96B_directunet_monaiM_ep1200_seed{s}/ens1_no1/estimates_{case}.npz")
                t = z["truth"].astype(np.float64)
                if truth is None:
                    truth = t
                assert np.allclose(t, truth, atol=1e-4)
                err = (t - z["trajectories"].astype(np.float64)) / sd
                save(out, f"directunet_ep1200_seed{s}_{lay}_{case}", window_gaussian(err, np.zeros_like(err), sd))
            for method, path in DA[lay].items():
                z = np.load(path)
                sel = (lambda a: a[..., idx] if a.shape[-1] > 24 else a)  # noqa: E731
                mean = sel(z[f"{case}_{method}_trajectories"].astype(np.float64))
                vkey = f"{case}_{method}_ensemble_variance"
                var = (np.clip(sel(z[vkey].astype(np.float64)), 0, None) / sd ** 2 if vkey in z.files
                       else np.zeros_like(mean))
                save(out, f"dastored_{lay}_{method}_{case}", window_gaussian((truth - mean) / sd, var, sd))


if __name__ == "__main__":
    main(Path(sys.argv[1]))
