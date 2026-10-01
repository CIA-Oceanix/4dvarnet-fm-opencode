"""FM-operator score of DA ensemble dumps (``*_members.npz`` from --save-members).

Scores every members file under ``root`` with ``probe_fm_score.window_members``
and writes ``<out>/<prefix><file stem>.npz`` (per window) and ``.json`` (pooled).
Run in the job that wrote the dumps: they live on node-local storage (``evaluation/members_store.py``).

usage: python reports/l96/probe_fm_score_da.py ROOT OUT_DIR [PREFIX]
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from reports.l96.probe_fm_score import norm_stats, pooled_from_windows, window_members  # noqa: E402


def main(root: Path, out: Path, prefix: str = "") -> None:
    mu, sd = norm_stats()
    out.mkdir(parents=True, exist_ok=True)
    files = sorted(root.rglob("*_members.npz"))
    assert files, f"no *_members.npz under {root}"
    for f in files:
        z = np.load(f)
        members, truth = z["members"], z["truth"]
        assert members.shape[2] == 24 and truth.shape == members.shape[:3], (f, members.shape, truth.shape)
        arrays = window_members(members, truth, mu, sd, label=f.name)
        np.savez(out / f"{prefix}{f.stem}.npz", **{k: v.astype(np.float32) for k, v in arrays.items()})
        stats = pooled_from_windows(arrays)
        dest = out / f"{prefix}{f.stem}.json"
        dest.write_text(json.dumps(stats, indent=1))
        print(f"FMSCORE {f.name}: rmse {stats['rmse']:.4f} -> {dest}", flush=True)


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3] if len(sys.argv) > 3 else "")
