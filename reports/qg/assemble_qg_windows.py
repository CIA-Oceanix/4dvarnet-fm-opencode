"""Assemble per-window .pt files (from generate_qg_window_chunk.py's array
tasks) into the single combined truth-cache file that
`make_qg_s0_s1_datasets(cfg, num_test_windows=n, cache_dir=...)` expects, so
downstream code hits the cache instead of regenerating.
"""
import argparse
import glob
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from data.qg import QGConfig, _truth_cache_path
from generate_qg_window_chunk import SPLIT_SEED_BASE, SPLIT_SIZE  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--split", required=True, choices=["train", "val", "test"])
    p.add_argument("--windows-dir", required=True,
                    help="Directory containing window_*.pt files for this split "
                         "(i.e. the --out-dir passed to generate_qg_window_chunk.py)")
    p.add_argument("--cache-dir", required=True)
    args = p.parse_args()

    n_total = SPLIT_SIZE[args.split]
    split_dir = os.path.join(args.windows_dir, args.split)
    files = sorted(glob.glob(os.path.join(split_dir, "window_*.pt")))
    if len(files) != n_total:
        found = {int(os.path.basename(f)[7:12]) for f in files}
        missing = sorted(set(range(n_total)) - found)
        raise RuntimeError(
            f"[{args.split}] expected {n_total} windows, found {len(files)}; "
            f"missing indices: {missing[:20]}{'...' if len(missing) > 20 else ''}")

    windows = [torch.load(f, map_location="cpu") for f in files]

    cfg = QGConfig(nx=64, dt=7200.0, seed=SPLIT_SEED_BASE[args.split],
                   num_windows=n_total)
    os.makedirs(args.cache_dir, exist_ok=True)
    cache_path = _truth_cache_path(cfg, n_total, args.cache_dir)
    torch.save(windows, cache_path)
    print(f"[{args.split}] assembled {len(windows)} windows -> {cache_path}")


if __name__ == "__main__":
    main()
