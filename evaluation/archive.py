"""Canonical resolution of archived L96 run artifacts.

Every archived run is a directory holding the artifacts needed to reproduce its
row in a report:

    <run>/checkpoints/{stage1.pt,stage1_best.ckpt}
    <run>/resolved_config.yaml
    <run>/estimates_{s0,s1}.npz
    <run>/neural_eval.json          (optional; metrics sidecar)
    <run>/manifest.json             (optional; provenance, see write_manifest)

Two layouts coexist:

* **canonical**  ``experiments/l96/<run>/`` -- where checkpoints were consolidated
  on 2026-09-10 so every worktree can reach them by one absolute path.
* **legacy**     ``experiments/<run>/``     -- where runs were originally written,
  and where the large ``estimates_*.npz`` were left behind by that consolidation.

This module is the single place that knows about either. Nothing else should
compute an artifact path, and in particular nothing should derive one by walking
a fixed number of parents from another path: ``eval_monai_l96.py`` used
``Path(ckpt).resolve().parents[2] / "l96_norm_stats_obsj2.pt"``, which silently
broke for every archived checkpoint the moment the ``experiments/l96/`` level was
introduced -- the stats file was not missing, it was one directory up from where
the arithmetic looked.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Optional

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS = ROOT / "experiments"
ARCHIVE = EXPERIMENTS / "l96"

CASES = ("s0", "s1")
DEFAULT_NORM_STATS = "l96_norm_stats_obsj2.pt"


def candidate_run_dirs(name: str) -> list[Path]:
    """Directories that may hold run ``name``, canonical location first.

    ``name`` may contain a subdirectory (the report's ens30 entries look like
    ``L3_vanilla_cfm_s0s1/ens30_no10``), which is preserved in both layouts.
    """
    return [ARCHIVE / name, EXPERIMENTS / name]


def resolve_run_dir(name: str, *, required: bool = True) -> Optional[Path]:
    """First existing directory for ``name``; ``None`` (or raise) if there is none."""
    for d in candidate_run_dirs(name):
        if d.is_dir():
            return d
    if required:
        raise FileNotFoundError(
            f"No run directory for {name!r}. Looked in: "
            + ", ".join(str(d) for d in candidate_run_dirs(name))
        )
    return None


def resolve_artifact(name: str, filename: str, *, required: bool = True) -> Optional[Path]:
    """First existing ``<run>/<filename>``, searching canonical then legacy.

    Resolved per artifact rather than per run: the 2026-09-10 consolidation moved
    checkpoints and configs into the archive but deliberately left the large
    ``estimates_*.npz`` behind, so a single run's artifacts can legitimately span
    both layouts.
    """
    for d in candidate_run_dirs(name):
        p = d / filename
        if p.is_file():
            return p
    if required:
        raise FileNotFoundError(
            f"No {filename!r} for run {name!r}. Looked in: "
            + ", ".join(str(d / filename) for d in candidate_run_dirs(name))
        )
    return None


def resolve_estimates(name: str, case: str, *, filename: Optional[str] = None,
                      required: bool = True) -> Optional[Path]:
    """Estimate arrays for ``name``/``case``.

    ``filename`` overrides the default ``estimates_<case>.npz`` and may contain
    a ``{case}`` placeholder. Not every run writes the standard name: the
    obs-density hybrids write ``estimates_directunet_sda3_<case>_keep16.npz``
    (the keep16 file being the full-density one the consolidated benchmark
    reports), so assuming one convention silently loses those rows.
    """
    if case not in CASES:
        raise ValueError(f"case must be one of {CASES}, got {case!r}")
    fname = filename.format(case=case) if filename else f"estimates_{case}.npz"
    return resolve_artifact(name, fname, required=required)


def resolve_config(name: str, *, required: bool = True) -> Optional[Path]:
    return resolve_artifact(name, "resolved_config.yaml", required=required)


def resolve_checkpoint(name: str, prefer: str = "stage1.pt",
                       *, required: bool = True) -> Optional[Path]:
    """Resolve a checkpoint, trying ``prefer`` then the other known filename.

    Both forms exist per run and hold the same weights, but they are *not*
    interchangeable to every loader: ``stage1_best.ckpt`` is a Lightning
    checkpoint (has a ``state_dict`` key) while ``stage1.pt`` is a bare
    ``OrderedDict``. ``evaluation.neural_inference.load_model`` accepts only the
    former.
    """
    order = [prefer] + [f for f in ("stage1.pt", "stage1_best.ckpt") if f != prefer]
    for filename in order:
        p = resolve_artifact(name, f"checkpoints/{filename}", required=False)
        if p is not None:
            return p
    if required:
        raise FileNotFoundError(f"No checkpoint for run {name!r} (tried {order})")
    return None


def resolve_norm_stats(cfg=None, *, name: Optional[str] = None,
                       required: bool = False) -> Optional[Path]:
    """Locate the per-channel normalization stats.

    Order of precedence, most explicit first:

    1. ``cfg.data.norm_stats_path`` -- what the run actually recorded.
    2. ``<run>/l96_norm_stats_obsj2.pt`` -- stats kept beside the run.
    3. ``experiments/l96_norm_stats_obsj2.pt`` -- the shared default.

    Never derived by parent-walking from a checkpoint path.
    """
    if cfg is not None:
        data = getattr(cfg, "data", None)
        recorded = None
        if data is not None:
            recorded = data.get("norm_stats_path") if hasattr(data, "get") else getattr(
                data, "norm_stats_path", None)
        if recorded:
            p = Path(recorded)
            if not p.is_absolute():
                p = ROOT / p
            if p.is_file():
                return p
    if name is not None:
        p = resolve_artifact(name, DEFAULT_NORM_STATS, required=False)
        if p is not None:
            return p
    shared = EXPERIMENTS / DEFAULT_NORM_STATS
    if shared.is_file():
        return shared
    if required:
        raise FileNotFoundError(
            f"No normalization stats found (cfg.data.norm_stats_path, "
            f"<run>/{DEFAULT_NORM_STATS}, {shared})")
    return None


def missing_artifacts(name: str, *, need_estimates: bool = True,
                      need_config: bool = False,
                      need_checkpoint: bool = False) -> list[str]:
    """Artifacts a reproducible run should have but does not. Empty == complete.

    Defaults to what *regenerating the report* needs, which is the estimate
    arrays only. ``resolved_config.yaml`` is opt-in because runs predating the
    config-persistence work (PR #171, 2026-09-08) legitimately have none -- they
    are still perfectly scoreable, just not re-runnable without reconstructing a
    config by hand.
    """
    missing = []
    if need_config and resolve_config(name, required=False) is None:
        missing.append("resolved_config.yaml")
    if need_estimates:
        for case in CASES:
            if resolve_estimates(name, case, required=False) is None:
                missing.append(f"estimates_{case}.npz")
    if need_checkpoint and resolve_checkpoint(name, required=False) is None:
        missing.append("checkpoints/*")
    return missing


def audit(names: Iterable[str], **kwargs) -> dict[str, list[str]]:
    """Map run name -> missing artifacts, for every incomplete run."""
    return {n: m for n in names if (m := missing_artifacts(n, **kwargs))}


def write_manifest(name: str, payload: dict) -> Path:
    """Record provenance beside a run so a later re-score can be checked.

    The motivating failure: an archived ``neural_eval.json`` whose numbers could
    not be reproduced by re-scoring the ``estimates_*.npz`` sitting next to it,
    with no record of which inputs produced either.
    """
    run_dir = resolve_run_dir(name)
    path = run_dir / "manifest.json"
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return path


def read_manifest(name: str) -> Optional[dict]:
    p = resolve_artifact(name, "manifest.json", required=False)
    return json.loads(p.read_text()) if p is not None else None
