"""SLURM scripts keep their scratch files off the node's root disk and clean them up.

A full root filesystem on sl-mee-br-207 (``/tmp`` is on it) made DataLoader workers fail
with ENOSPC and hung a 1200-epoch FDV run for ~15 h. Scripts now put TMPDIR and
MPLCONFIGDIR under /SCRATCH (falling back to a per-job /tmp dir) and remove it at exit.
"""
from pathlib import Path

BATCH = Path(__file__).resolve().parents[1] / "batch"
SCRIPTS = sorted(p for p in BATCH.iterdir() if p.is_file() and p.suffix in (".sbatch", ".sh"))


def test_no_uncleaned_per_job_mplconfig_under_tmp():
    offenders = [p.name for p in SCRIPTS if "/tmp/mplconfig_" in p.read_text()]
    assert not offenders, f"use the NODE_TMP snippet instead of /tmp/mplconfig_*: {offenders}"


def test_node_tmp_is_exported_and_cleaned():
    for p in SCRIPTS:
        text = p.read_text()
        if "NODE_TMP=" in text:
            assert 'export TMPDIR="$NODE_TMP"' in text and "trap 'rm -rf \"$NODE_TMP\"' EXIT" in text, p.name
