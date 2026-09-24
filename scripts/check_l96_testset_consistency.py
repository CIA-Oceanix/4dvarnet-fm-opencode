"""Check that benchmark results were produced on one canonical L96 test set.

Given the manifest written by ``scripts/make_l96_canonical_testset.py``:

- the test-set file still hashes to the manifest's SHA-256;
- every learned-scheme result dir (``neural_eval.json`` / ``sda_eval.json``
  plus ``estimates_{s0,s1}.npz``) names that exact file as its dataset, and
  its stored truth equals the test set's observed-space truth, window for
  window;
- every DA ``per_window*.npz`` (``eval_da_random_layout_l96.py``) covers
  windows 0..N-1 in order and was drawn from layouts whose file hashes are in
  the manifest (the files the test set was built from).

Exit status 1 if anything fails.

  python scripts/check_l96_testset_consistency.py --manifest <testset>.manifest.json \\
      --results experiments/eval_rlayout/*/* --da <per_window npz> ...
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from eval_da_obs_count_l96 import CASES  # noqa: E402
from evaluation.run_l96 import make_obs_j_indices  # noqa: E402
from scripts.make_l96_canonical_testset import fingerprint, sha256_file  # noqa: E402


def check_learned(result_dir: str, testset: str, truths: dict) -> list[str]:
    errors = []
    js = [os.path.join(result_dir, n) for n in ("neural_eval.json", "sda_eval.json", "hybrid_eval.json")
          if os.path.exists(os.path.join(result_dir, n))]
    if not js:
        return [f"{result_dir}: no neural_eval.json / sda_eval.json / hybrid_eval.json"]
    ds_path = json.load(open(js[0])).get("dataset", {}).get("path")
    if ds_path is None or os.path.realpath(ds_path) != os.path.realpath(testset):
        errors.append(f"{result_dir}: dataset {ds_path} is not the canonical {testset}")
    for case in CASES:
        est = os.path.join(result_dir, f"estimates_{case}.npz")
        if not os.path.exists(est):
            errors.append(f"{result_dir}: missing estimates_{case}.npz")
            continue
        with np.load(est) as z:
            t = z["truth"]
        ref = truths[case].astype(t.dtype)
        if t.shape != ref.shape or not np.array_equal(t, ref):
            errors.append(f"{result_dir}: {case} truth differs from the canonical test set")
    return errors


def check_da(path: str, manifest: dict) -> list[str]:
    errors = []
    expected = np.array(manifest.get("window_index", range(manifest["n_windows"])))
    with np.load(path) as z:
        for case in CASES:
            key = f"{case}_window_index"
            if key in z.files and not np.array_equal(z[key], expected):
                errors.append(f"{path}: {case} window order differs from the test set's")
    tag = os.path.basename(path).replace("per_window", "layouts").replace(".npz", ".pt")
    lp = os.path.join(os.path.dirname(path), tag)
    if tag in manifest["source_layouts"]:
        if os.path.exists(lp) and sha256_file(lp) != manifest["source_layouts"][tag]:
            errors.append(f"{path}: {tag} changed since the test set was built")
    elif not os.path.exists(lp):
        errors.append(f"{path}: layouts file {tag} missing and not a source of the canonical test set")
    else:
        errors += layouts_match_testset(lp, manifest)
    return errors


def layouts_match_testset(layouts_path: str, manifest: dict) -> list[str]:
    """A DA run whose layouts file is not one of the test set's sources is
    accepted iff its obs/mask equal the canonical test set's, window by window."""
    layouts = torch.load(layouts_path, weights_only=False)
    data = torch.load(manifest["testset"], weights_only=False)
    errors = []
    for case, key in CASES.items():
        if case not in layouts:
            continue
        rec = layouts[case]
        for i in range(manifest["n_windows"]):
            w = data[key][i]
            obs = rec["obs"][i].to(w["obs"].dtype)
            if not (torch.equal(rec["obs_mask"][i].to(w["obs_mask"].dtype), w["obs_mask"])
                    and torch.equal(torch.isnan(obs), torch.isnan(w["obs"]))
                    and torch.equal(torch.nan_to_num(obs), torch.nan_to_num(w["obs"]))):
                errors.append(f"{layouts_path}: {case} window {i} obs differ from the canonical test set")
                break
    return errors


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--manifest", required=True)
    p.add_argument("--results", nargs="*", default=[], help="learned-scheme result dirs (globs ok)")
    p.add_argument("--da", nargs="*", default=[], help="DA per_window*.npz files (globs ok)")
    args = p.parse_args()

    manifest = json.load(open(args.manifest))
    testset = manifest["testset"]
    errors = []
    if sha256_file(testset) != manifest["sha256"]:
        errors.append(f"{testset} no longer matches the manifest SHA-256")
    data = torch.load(testset, weights_only=False)
    idx = list(make_obs_j_indices(8, 4, 2))
    truths = {}
    for case, key in CASES.items():
        ws = [data[key][i] for i in range(manifest["n_windows"])]
        if fingerprint(ws) != manifest["fields"][case]:
            errors.append(f"{case}: test-set fields do not match the manifest")
        truths[case] = np.stack([w["true_state"].numpy()[:, idx] for w in ws])

    dirs = sorted({d for g in args.results for d in glob.glob(g)})
    das = sorted({f for g in args.da for f in glob.glob(g)})
    for d in dirs:
        e = check_learned(d, testset, truths)
        errors += e
        print(f"{'FAIL' if e else 'ok  '} {d}")
    for f in das:
        e = check_da(f, manifest)
        errors += e
        print(f"{'FAIL' if e else 'ok  '} {f}")
    for e in errors:
        print("  -", e)
    print(f"{len(dirs)} learned + {len(das)} DA results checked: {'FAILED' if errors else 'all consistent'}")
    sys.exit(1 if errors else 0)


if __name__ == "__main__":
    main()
