"""Assemble the Weak-4DVar benchmark chunks into one benchmark-layout trajectory file.

``batch/run_l96_weak4dvar_bench.sbatch`` writes 20 chunks of 10 windows per case
(``<case>_q<q>_it40_w<start>.npz`` with ``windows`` and ``<case>_Weak_4DVar_trajectories``).
This script checks that every case covers windows 0-199 exactly once, stacks them in window
order, rescores them against the test-set truth with the benchmark convention (per-window
RMSE on the 24 observed channels, averaged over channels) and checks the result against the
per-chunk JSONs. It writes ``<out>/<NAME>`` (keys ``s0_Weak_4DVar_trajectories`` /
``s1_Weak_4DVar_trajectories``, the layout of the other DA trajectory files) and a summary JSON.

usage: assemble_weak4dvar_bench.py <chunk_dir> <out_dir>
"""
import glob
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.getcwd())
from evaluation.run_l96 import make_obs_j_indices  # noqa: E402

TEST = "/Odyssey/private/rfablet/Python/4dvarnet-fm-opencode/experiments/l96_datasets_obsj2_int100_nwin200.pt"
NAME = "l96_baselines_trajectories_dws500_s0c_test_weak4dvar-qs0-0.03_s1-0.3-it40_obsj2_int100_fw_dafw.npz"
Q = {"s0": "0.03", "s1": "0.3"}
N_WIN = 200


def main(chunk_dir: str, out_dir: str) -> None:
    idx = np.array(make_obs_j_indices(8, 4, 2))
    data = torch.load(TEST, weights_only=False)
    out, summary = {}, {"test_set": TEST, "q_var_scale": Q, "max_iter": 40, "cases": {}}
    for case, q in Q.items():
        files = sorted(glob.glob(os.path.join(chunk_dir, f"{case}_q{q}_it40_w*.npz")))
        windows, trajs, chunk_rmse, resets = [], [], [], 0
        for f in files:
            z = np.load(f)
            t = z[f"{case}_Weak_4DVar_trajectories"]
            windows += list(z["windows"])
            trajs.append(t[..., idx] if t.shape[-1] > len(idx) else t)
            meta = json.load(open(f.replace(".npz", ".json")))
            chunk_rmse.append((len(z["windows"]), meta["cases"][case]["rmse"]))
            resets += meta["n_resets"]
        order = np.argsort(windows)
        if sorted(windows) != list(range(N_WIN)):
            sys.exit(f"{case}: windows do not cover 0-{N_WIN - 1} exactly once")
        traj = np.concatenate(trajs)[order].astype(np.float32)
        truth = np.stack([w["true_state"].numpy()[:, idx] for w in data[f"test_{case}"]]).astype(np.float64)
        per_window = np.sqrt(((traj - truth) ** 2).mean(1)).mean(1)
        rmse = float(per_window.mean())
        from_chunks = sum(n * r for n, r in chunk_rmse) / N_WIN
        if abs(rmse - from_chunks) > 1e-4:
            sys.exit(f"{case}: rescored RMSE {rmse:.5f} differs from the chunks' {from_chunks:.5f}")
        out[f"{case}_Weak_4DVar_trajectories"] = traj
        summary["cases"][case] = {"rmse": rmse, "rmse_sd_windows": float(per_window.std(ddof=1)),
                                  "n_chunks": len(files), "n_resets": resets}
        print(f"{case}: q {q}, {len(files)} chunks, RMSE {rmse:.4f} ± {per_window.std(ddof=1):.4f}, resets {resets}")
    os.makedirs(out_dir, exist_ok=True)
    np.savez_compressed(os.path.join(out_dir, NAME), **out)
    json.dump(summary, open(os.path.join(out_dir, NAME.replace(".npz", ".json")), "w"), indent=1)
    print(f"wrote {os.path.join(out_dir, NAME)}")


if __name__ == "__main__":
    main(*sys.argv[1:3])
