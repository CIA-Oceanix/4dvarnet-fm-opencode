"""Where the L96 benchmark report generators read their inputs.

Each report reads a few named input roots (``here``, ``bench``, ``p1``, ...). By
default a root resolves to that report's **input bundle** in the canonical
archive, ``<main checkout>/experiments/l96/report_inputs/<report>/<root>/``:
hard links of exactly the files the report was last generated from, so the
report regenerates from any checkout and survives the topic worktrees being
pruned. ``shared()`` is the main checkout's ``experiments/`` (test sets, DA
caches), which is canonical already and is not bundled.

``FDV_REPORT_INPUT_ROOTS`` (a JSON object ``{root: path}``) overrides the
roots; ``scripts/bundle_report_inputs.py`` uses it to record a generator
against the directories the bundle is built from. Nothing else should.
"""
from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path

ROOTS_ENV = "FDV_REPORT_INPUT_ROOTS"
REPORTS = {
    "p1_benchmark": ("here",),
    "benchmark_default": ("here", "bench", "p1", "da_random_layout", "random_obs_times"),
    "benchmark_extended": ("here", "bench", "p1", "n20", "da_random_layout", "random_obs_times"),
}


@lru_cache(maxsize=1)
def main_checkout() -> Path:
    checkout = Path(__file__).resolve().parents[2]
    dot_git = checkout / ".git"
    if not dot_git.is_file():
        return checkout
    gitdir = Path(dot_git.read_text().split("gitdir:", 1)[1].strip())
    if not gitdir.is_absolute():
        gitdir = (checkout / gitdir).resolve()
    commondir = gitdir / "commondir"
    common = (gitdir / commondir.read_text().strip()).resolve() if commondir.is_file() else gitdir
    return common.parent


def shared() -> Path:
    return main_checkout() / "experiments"


def bundle(report: str) -> Path:
    return shared() / "l96" / "report_inputs" / report


def root(report: str, name: str) -> Path:
    if name not in REPORTS[report]:
        raise KeyError(f"{report!r} has no input root {name!r}; known: {REPORTS[report]}")
    override = os.environ.get(ROOTS_ENV)
    if override:
        return Path(json.loads(override)[name])
    return bundle(report) / name
