"""Where full ensemble dumps (``members_<case>.npz``) are written.

A members array is ``(W, T, D, M)`` float32 -- 1.6-3.3 GB per case at the L96
benchmark size -- and on 2026-09-24 these dumps were ~475 GB of the project's
~490 GB on the shared /Odyssey volume, which was full. Sweeps and diagnostics
only need the metrics, which every eval script computes in-process from the
in-memory members, so by default the dump goes to the GPU node's local /tmp.

Keep it next to the run's other outputs only when a report generator reads it
(the P1, benchmark-default and consolidated rows): pass ``--keep-members`` or
export ``FDV_KEEP_MEMBERS=1``. ``FDV_MEMBERS_TMP`` overrides the node-local root.

Node-local /tmp is neither visible from other nodes nor guaranteed to outlive
the job, so anything that must re-read the members runs in the same job.
"""
from __future__ import annotations

import getpass
import os
from pathlib import Path

KEEP_ENV = "FDV_KEEP_MEMBERS"
ROOT_ENV = "FDV_MEMBERS_TMP"


def keep_members(flag: bool = False) -> bool:
    return flag or os.environ.get(KEEP_ENV, "").strip() not in ("", "0", "false", "False")


def node_local_root() -> Path:
    override = os.environ.get(ROOT_ENV)
    if override:
        return Path(override)
    job = os.environ.get("SLURM_JOB_ID", f"pid{os.getpid()}")
    return Path("/tmp") / getpass.getuser() / f"members_{job}"


def members_path(output_dir: str | Path, case: str, keep: bool = False) -> Path:
    output_dir = Path(output_dir)
    name = f"members_{case}.npz"
    if keep_members(keep):
        return output_dir / name
    local = node_local_root() / output_dir.resolve().relative_to("/")
    local.mkdir(parents=True, exist_ok=True)
    return local / name
