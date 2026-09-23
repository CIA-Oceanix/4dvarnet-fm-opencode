"""Build the canonical L96 random-observing-system test set from the layouts
the DA baselines were scored on (``eval_da_random_layout_l96.py``), so every
scheme -- DA, deterministic, flow, SDA -- sees identical inputs.

Output = the P1 200-window test cache with ONLY ``obs``/``obs_mask`` replaced
by the stored layouts; truth, forcings (true and corrupted) and parameters
(true and ``*_da``) are the cache's own objects. Refuses to write unless:

- every stored copy of a case's layouts (one per inflation run) is identical,
- the layouts regenerate bit for bit from their per-window seeds and the
  cache's truth (so they belong to THIS cache),
- windows are 0..N-1 in order, one draw each,
- the written file reloads with obs/mask equal to the layouts and every other
  field equal to the cache.

A manifest (``<output>.manifest.json``) records the file's SHA-256 and a
per-case SHA-256 of each field, for ``scripts/check_l96_testset_consistency.py``.

  python scripts/make_l96_canonical_testset.py --tag rlayout_n10-100_k4-16_w200_d1
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import re
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from eval_da_obs_count_l96 import CASES, DEFAULT_CACHE  # noqa: E402
from eval_da_random_layout_l96 import _layout_seed, draw_layout  # noqa: E402
from evaluation.run_l96 import make_obs_j_indices  # noqa: E402

DEFAULT_LAYOUT_DIR = ("/Odyssey/private/rfablet/Python/4dvarnet-fm-opencode/4dvarnet-fm-da-random-layout/"
                      "experiments/l96_da_random_layout")
TENSOR_FIELDS = ("true_state", "forcing_true", "forcing_corrupted")


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _sha_tensors(ts) -> str:
    h = hashlib.sha256()
    for t in ts:
        h.update(t.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def _scalars(w: dict) -> dict:
    return {k: v for k, v in w.items() if not isinstance(v, torch.Tensor)}


def fingerprint(windows) -> dict:
    """Per-field SHA-256 over a case's windows, in order."""
    out = {f: _sha_tensors([w[f] for w in windows]) for f in ("obs", "obs_mask") + TENSOR_FIELDS}
    params = json.dumps([_scalars(w) for w in windows], sort_keys=True, default=repr)
    out["params"] = hashlib.sha256(params.encode()).hexdigest()
    return out


def parse_tag(tag: str) -> tuple[tuple[int, int], tuple[int, int], int, int]:
    m = re.fullmatch(r"rlayout_n(\d+)-(\d+)_k(\d+)-(\d+)_w(\d+)_d(\d+)", tag)
    if m is None:
        raise ValueError(f"unrecognized layout tag {tag!r}")
    a, b, c, d, w, n = (int(x) for x in m.groups())
    return (a, b), (c, d), w, n


def load_layouts(layout_dir: str, tag: str) -> tuple[dict, list[str]]:
    files = sorted(glob.glob(os.path.join(layout_dir, f"layouts_{tag}_*.pt")))
    if not files:
        raise FileNotFoundError(f"no layouts_{tag}_*.pt in {layout_dir}")
    merged: dict = {}
    for f in files:
        for case, rec in torch.load(f, weights_only=False).items():
            if case not in merged:
                merged[case] = rec
                continue
            ref = merged[case]
            same = (np.array_equal(ref["window_index"], rec["window_index"])
                    and torch.equal(ref["obs_mask"], rec["obs_mask"])
                    and torch.equal(torch.nan_to_num(ref["obs"], nan=1e30), torch.nan_to_num(rec["obs"], nan=1e30))
                    and torch.equal(torch.isnan(ref["obs"]), torch.isnan(rec["obs"])))
            if not same:
                raise ValueError(f"{case} layouts differ between {files[0]} and {f}")
    missing = set(CASES) - set(merged)
    if missing:
        raise ValueError(f"cases {sorted(missing)} missing from {files}")
    return merged, files


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--tag", default="rlayout_n10-100_k4-16_w200_d1")
    p.add_argument("--layout-dir", default=DEFAULT_LAYOUT_DIR)
    p.add_argument("--data-cache", default=DEFAULT_CACHE)
    p.add_argument("--da-window-steps", type=int, default=500)
    p.add_argument("--output", default=None)
    args = p.parse_args()

    n_range, k_range, n_windows, n_draws = parse_tag(args.tag)
    layouts, layout_files = load_layouts(args.layout_dir, args.tag)
    datasets = torch.load(args.data_cache, weights_only=False)
    idx = list(make_obs_j_indices(8, 4, 2))
    windows = [int(w) for w in np.linspace(0, len(datasets["test_s0"]), n_windows, endpoint=False)]
    expected_index = np.repeat(windows, n_draws)

    out, manifest_cases = {}, {}
    for case, key in CASES.items():
        rec, ds = layouts[case], datasets[key]
        if not np.array_equal(rec["window_index"], expected_index):
            raise ValueError(f"{case}: layout window order is not {n_windows} windows x {n_draws} draws")
        source = [ds[i] for i in windows]
        before = fingerprint(source)
        entries = []
        for j, wi in enumerate(expected_index):
            w = ds[int(wi)]
            d = j % n_draws
            obs, mask = draw_layout(w["true_state"][:, idx].float(), n_range, k_range,
                                    _layout_seed(case, int(wi), n_range, k_range, d), args.da_window_steps)
            if not (torch.equal(mask, rec["obs_mask"][j])
                    and torch.equal(torch.isnan(obs), torch.isnan(rec["obs"][j]))
                    and torch.equal(torch.nan_to_num(obs), torch.nan_to_num(rec["obs"][j]))):
                raise ValueError(f"{case} window {wi} draw {d}: stored layout does not regenerate from its seed")
            entries.append(dict(w, obs=rec["obs"][j].to(w["obs"].dtype),
                                obs_mask=rec["obs_mask"][j].to(w["obs_mask"].dtype)))
        if fingerprint([ds[i] for i in windows]) != before:
            raise AssertionError(f"{case}: source windows changed while building the test set")
        out[key] = ds if n_draws == 1 and windows == list(range(len(ds))) else entries
        if out[key] is ds:
            for j, e in enumerate(entries):
                ds[j]["obs"], ds[j]["obs_mask"] = e["obs"], e["obs_mask"]
        after = fingerprint([out[key][j] for j in range(len(entries))])
        for f in TENSOR_FIELDS + ("params",):
            ref = fingerprint([ds[int(wi)] for wi in expected_index])[f] if out[key] is not ds else before[f]
            if after[f] != ref:
                raise AssertionError(f"{case}: field {f} differs from the source windows")
        manifest_cases[case] = after

    path = args.output or os.path.join(os.path.dirname(args.data_cache), f"l96_testset_{args.tag}.pt")
    torch.save(out, path)
    reloaded = torch.load(path, weights_only=False)
    n_entries = len(expected_index)
    for case, key in CASES.items():
        if fingerprint([reloaded[key][i] for i in range(n_entries)]) != manifest_cases[case]:
            raise AssertionError(f"{case}: reloaded file does not match what was written")
    manifest = {
        "testset": os.path.abspath(path), "sha256": sha256_file(path), "tag": args.tag,
        "n_obs_range": list(n_range), "fast_range": list(k_range), "n_windows": n_entries,
        "window_index": [int(w) for w in expected_index], "n_draws": n_draws,
        "source_cache": os.path.abspath(args.data_cache), "source_cache_sha256": sha256_file(args.data_cache),
        "source_layouts": {os.path.basename(f): sha256_file(f) for f in layout_files},
        "fields": manifest_cases,
    }
    with open(path + ".manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
    for case, key in CASES.items():
        m = reloaded[key]
        n = [int(m[i]["obs_mask"].sum()) for i in range(n_entries)]
        print(f"{case}: n_obs {min(n)}-{max(n)} (mean {np.mean(n):.1f}), step 0 observed in all: "
              f"{all(bool(m[i]['obs_mask'][0]) for i in range(n_entries))}")
    print(f"wrote {path}\n  sha256 {manifest['sha256']}")


if __name__ == "__main__":
    main()
