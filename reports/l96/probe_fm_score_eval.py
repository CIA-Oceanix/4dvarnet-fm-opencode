"""FM-operator score of an L96 ensemble eval, members kept in memory.

Runs an eval script (eval_neural_l96 / eval_sda_l96 /
eval_sda_directunet_hybrid_l96) with its members-saving hook replaced by one
writing ``fmscore_<case>.json`` (pooled) and ``fmscore_<case>.npz`` (per-window
arrays of ``probe_fm_score.window_members``) next to --output. No members_*.npz
is written.

usage (from the repo root, PYTHONPATH=.):
  python reports/l96/probe_fm_score_eval.py eval_neural_l96 --checkpoint ... --output out/eval.json
"""
import importlib
import json
import sys
from pathlib import Path

import numpy as np

from reports.l96.probe_fm_score import norm_stats, pooled_from_windows, window_members


def hook(out_dir, case, members, truth, mode="full"):
    mu, sd = norm_stats()
    arrays = window_members(members, truth, mu, sd, label=case)
    np.savez(Path(out_dir) / f"fmscore_{case}.npz", **{k: v.astype(np.float32) for k, v in arrays.items()})
    stats = pooled_from_windows(arrays)
    path = Path(out_dir) / f"fmscore_{case}.json"
    path.write_text(json.dumps(stats, indent=1))
    print(f"FMSCORE {case}: rmse {stats['rmse']:.4f} gauss "
          + json.dumps({t: round(v["all"], 5) for t, v in stats["gauss"].items()}), flush=True)
    return str(path)


if __name__ == "__main__":
    mod_name = sys.argv[1]
    sys.argv = [mod_name + ".py"] + sys.argv[2:]
    mod = importlib.import_module(mod_name)
    mod.save_members_or_scores = hook
    mod.members_store.members_dir = lambda out, keep=False, mode="full": Path(out)
    mod.main()
