"""Canonical resolution of archived run artifacts (L96 and QG).

Every archived run is a directory holding the artifacts needed to reproduce its
row in a report:

    <run>/checkpoints/{stage1.pt,stage1_best.ckpt}   (L96)
    <run>/stage1_best.pt                             (QG: bare state dict)
    <run>/resolved_config.yaml
    <run>/estimates_{s0,s1}.npz
    <run>/neural_eval.json          (optional; metrics sidecar)
    <run>/manifest.json             (optional; provenance, see write_manifest)

Two layouts coexist per case study:

* **canonical**  ``experiments/<system>/<run>/`` -- where checkpoints were
  consolidated (L96 on 2026-09-10, QG on 2026-09-14) so every worktree can
  reach them by one absolute path.
* **legacy**     ``experiments/<run>/``     -- where runs were originally written,
  and where the large ``estimates_*.npz`` were left behind by that consolidation.

This module is the single place that knows about either. Nothing else should
compute an artifact path, and in particular nothing should derive one by walking
a fixed number of parents from another path: ``eval_monai_l96.py`` used
``Path(ckpt).resolve().parents[2] / "l96_norm_stats_obsj2.pt"``, which silently
broke for every archived checkpoint the moment the ``experiments/l96/`` level was
introduced -- the stats file was not missing, it was one directory up from where
the arithmetic looked.

``L96`` and ``QG`` are the two layouts; the module-level functions are the L96
ones, kept as the names every existing L96 caller already imports.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Optional

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS = ROOT / "experiments"
ARCHIVE = EXPERIMENTS / "l96"

CASES = ("s0", "s1")
DEFAULT_NORM_STATS = "l96_norm_stats_obsj2.pt"


@dataclass(frozen=True)
class RunArchive:
    """One case study's archive layout.

    ``experiments``/``archive_dir`` are resolved through this module's globals on
    every call rather than captured at import, so tests can redirect the whole
    tree with ``monkeypatch.setattr(archive, "EXPERIMENTS", ...)``.
    """

    system: str
    default_norm_stats: str
    checkpoints: tuple[str, ...]
    cases: tuple[str, ...] = CASES
    shared_stats: tuple[str, ...] = field(default_factory=tuple)

    @property
    def experiments(self) -> Path:
        return EXPERIMENTS

    @property
    def archive_dir(self) -> Path:
        return EXPERIMENTS / self.system

    def candidate_run_dirs(self, name: str) -> list[Path]:
        """Directories that may hold run ``name``, canonical location first.

        ``name`` may contain a subdirectory (the L96 report's ens30 entries look
        like ``L3_vanilla_cfm_s0s1/ens30_no10``), which is preserved in both
        layouts.
        """
        return [self.archive_dir / name, self.experiments / name]

    def resolve_run_dir(self, name: str, *, required: bool = True) -> Optional[Path]:
        """First existing directory for ``name``; ``None`` (or raise) if there is none."""
        for d in self.candidate_run_dirs(name):
            if d.is_dir():
                return d
        if required:
            raise FileNotFoundError(
                f"No run directory for {name!r}. Looked in: "
                + ", ".join(str(d) for d in self.candidate_run_dirs(name))
            )
        return None

    def resolve_artifact(self, name: str, filename: str,
                         *, required: bool = True) -> Optional[Path]:
        """First existing ``<run>/<filename>``, searching canonical then legacy.

        Resolved per artifact rather than per run: the consolidations moved
        checkpoints and configs into the archive but deliberately left the large
        ``estimates_*.npz`` behind, so a single run's artifacts can legitimately
        span both layouts.
        """
        for d in self.candidate_run_dirs(name):
            p = d / filename
            if p.is_file():
                return p
        if required:
            raise FileNotFoundError(
                f"No {filename!r} for run {name!r}. Looked in: "
                + ", ".join(str(d / filename) for d in self.candidate_run_dirs(name))
            )
        return None

    def resolve_estimates(self, name: str, case: str, *, filename: Optional[str] = None,
                          required: bool = True) -> Optional[Path]:
        """Estimate arrays for ``name``/``case``.

        ``filename`` overrides the default ``estimates_<case>.npz`` and may contain
        a ``{case}`` placeholder. Not every run writes the standard name: the L96
        obs-density hybrids write ``estimates_directunet_sda3_<case>_keep16.npz``
        (the keep16 file being the full-density one the consolidated benchmark
        reports), so assuming one convention silently loses those rows.
        """
        if case not in self.cases:
            raise ValueError(f"case must be one of {self.cases}, got {case!r}")
        fname = filename.format(case=case) if filename else f"estimates_{case}.npz"
        return self.resolve_artifact(name, fname, required=required)

    def resolve_config(self, name: str, *, required: bool = True) -> Optional[Path]:
        return self.resolve_artifact(name, "resolved_config.yaml", required=required)

    def resolve_checkpoint(self, name: str, prefer: Optional[str] = None,
                           *, required: bool = True) -> Optional[Path]:
        """Resolve a checkpoint, trying ``prefer`` first then the other known names.

        The known forms hold the same weights but are *not* interchangeable to
        every loader: a Lightning ``*.ckpt`` nests the weights under a
        ``state_dict`` key while a ``*.pt`` written at the end of training is a
        bare ``OrderedDict``. ``evaluation.neural_inference.load_model`` accepts
        only the former; ``eval_qg_neural_s0_s1.py`` accepts both.
        """
        order = list(self.checkpoints)
        if prefer is not None:
            order.sort(key=lambda rel: Path(rel).name != prefer)
        for rel in order:
            p = self.resolve_artifact(name, rel, required=False)
            if p is not None:
                return p
        if required:
            raise FileNotFoundError(f"No checkpoint for run {name!r} (tried {order})")
        return None

    def resolve_norm_stats(self, cfg=None, *, name: Optional[str] = None,
                           key: str = "norm_stats_path", filename: Optional[str] = None,
                           required: bool = False) -> Optional[Path]:
        """Locate a normalization-stats file.

        Order of precedence, most explicit first:

        1. ``cfg.data[key]`` -- what the run actually recorded.
        2. ``<run>/<filename>`` -- stats kept beside the run.
        3. ``experiments/<system>/<filename>`` -- stats kept with the archive, so
           they travel with the checkpoints into any worktree.
        4. ``experiments/<filename>`` -- the shared default.

        Never derived by parent-walking from a checkpoint path. Step 3 exists
        because QG's stats lived only in the training worktree: the archived
        checkpoints were reachable from master and still unusable there, since
        the psi z-scoring they were trained under was not.
        """
        fname = filename or self.default_norm_stats
        if cfg is not None:
            data = getattr(cfg, "data", None)
            recorded = None
            if data is not None:
                recorded = data.get(key) if hasattr(data, "get") else getattr(data, key, None)
            if recorded:
                p = Path(recorded)
                if not p.is_absolute():
                    p = ROOT / p
                if p.is_file():
                    return p
        if name is not None:
            p = self.resolve_artifact(name, fname, required=False)
            if p is not None:
                return p
        for cand in (self.archive_dir / fname, self.experiments / fname):
            if cand.is_file():
                return cand
        if required:
            raise FileNotFoundError(
                f"No normalization stats found (cfg.data.{key}, <run>/{fname}, "
                f"{self.archive_dir / fname}, {self.experiments / fname})")
        return None

    def missing_artifacts(self, name: str, *, need_estimates: bool = True,
                          need_config: bool = False,
                          need_checkpoint: bool = False) -> list[str]:
        """Artifacts a reproducible run should have but does not. Empty == complete.

        Defaults to what *regenerating the report* needs, which is the estimate
        arrays only. ``resolved_config.yaml`` is opt-in because runs predating the
        config-persistence work (L96: PR #171, 2026-09-08; QG: 2026-09-19)
        legitimately have none -- they are still perfectly scoreable, just not
        re-runnable without reconstructing a config by hand.
        """
        missing = []
        if need_config and self.resolve_config(name, required=False) is None:
            missing.append("resolved_config.yaml")
        if need_estimates:
            for case in self.cases:
                if self.resolve_estimates(name, case, required=False) is None:
                    missing.append(f"estimates_{case}.npz")
        if need_checkpoint and self.resolve_checkpoint(name, required=False) is None:
            missing.append("checkpoints/*")
        return missing

    def audit(self, names: Iterable[str], **kwargs) -> dict[str, list[str]]:
        """Map run name -> missing artifacts, for every incomplete run."""
        return {n: m for n in names if (m := self.missing_artifacts(n, **kwargs))}

    def write_manifest(self, name: str, payload: dict) -> Path:
        """Record provenance beside a run so a later re-score can be checked.

        The motivating failure: an archived ``neural_eval.json`` whose numbers could
        not be reproduced by re-scoring the ``estimates_*.npz`` sitting next to it,
        with no record of which inputs produced either.
        """
        run_dir = self.resolve_run_dir(name)
        path = run_dir / "manifest.json"
        path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        return path

    def read_manifest(self, name: str) -> Optional[dict]:
        p = self.resolve_artifact(name, "manifest.json", required=False)
        return json.loads(p.read_text()) if p is not None else None


L96 = RunArchive(
    system="l96",
    default_norm_stats=DEFAULT_NORM_STATS,
    checkpoints=("checkpoints/stage1.pt", "checkpoints/stage1_best.ckpt"),
    shared_stats=(DEFAULT_NORM_STATS,),
)

# QG writes its final weights as a bare state dict at the run root
# (``train_qg_neural.py``'s ``torch.save(lit.model.state_dict(), ...)``) and keeps
# Lightning's own monitored checkpoints under ``checkpoints/``; a run killed
# before it finished (the Q5 sweep's 24h timeouts) has only the latter. Its
# per-run estimates cover S0 only -- the S0/S1 cross-scenario numbers are
# produced by ``eval_qg_neural_s0_s1.py`` into a single report JSON, not per run.
QG = RunArchive(
    system="qg",
    default_norm_stats="qg_psi_norm_stats.pt",
    checkpoints=("stage1_best.pt", "checkpoints/stage1_best.ckpt",
                 "checkpoints/stage1_last.ckpt"),
    cases=("s0",),
    shared_stats=("qg_psi_norm_stats.pt", "qg_param_norm_stats.pt",
                  "qg_forcing_norm_stats.pt"),
)


def candidate_run_dirs(name: str) -> list[Path]:
    return L96.candidate_run_dirs(name)


def resolve_run_dir(name: str, *, required: bool = True) -> Optional[Path]:
    return L96.resolve_run_dir(name, required=required)


def resolve_artifact(name: str, filename: str, *, required: bool = True) -> Optional[Path]:
    return L96.resolve_artifact(name, filename, required=required)


def resolve_estimates(name: str, case: str, *, filename: Optional[str] = None,
                      required: bool = True) -> Optional[Path]:
    return L96.resolve_estimates(name, case, filename=filename, required=required)


def resolve_config(name: str, *, required: bool = True) -> Optional[Path]:
    return L96.resolve_config(name, required=required)


def resolve_checkpoint(name: str, prefer: str = "stage1.pt",
                       *, required: bool = True) -> Optional[Path]:
    return L96.resolve_checkpoint(name, prefer, required=required)


def resolve_norm_stats(cfg=None, *, name: Optional[str] = None,
                       required: bool = False) -> Optional[Path]:
    return L96.resolve_norm_stats(cfg, name=name, required=required)


def missing_artifacts(name: str, **kwargs) -> list[str]:
    return L96.missing_artifacts(name, **kwargs)


def audit(names: Iterable[str], **kwargs) -> dict[str, list[str]]:
    return L96.audit(names, **kwargs)


def write_manifest(name: str, payload: dict) -> Path:
    return L96.write_manifest(name, payload)


def read_manifest(name: str) -> Optional[dict]:
    return L96.read_manifest(name)
