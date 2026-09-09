import hashlib
import json
import os
import shutil
from datetime import datetime, timezone

from omegaconf import DictConfig, OmegaConf

FINGERPRINT_KEYS = ("model", "data", "training")


def config_fingerprint(cfg: DictConfig) -> str:
    subtree = {
        key: OmegaConf.to_container(cfg[key], resolve=True)
        for key in FINGERPRINT_KEYS
        if key in cfg
    }
    payload = json.dumps(subtree, sort_keys=True, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _has_checkpoint(exp_dir: str) -> bool:
    ckpt_dir = os.path.join(exp_dir, "checkpoints")
    if not os.path.isdir(ckpt_dir):
        return False
    return any(f.endswith(".ckpt") for f in os.listdir(ckpt_dir))


def _archive_experiment_dir(exp_dir: str) -> str:
    runs_dir = exp_dir.rstrip("/") + "_runs"
    os.makedirs(runs_dir, exist_ok=True)
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    dest = os.path.join(runs_dir, run_id)
    shutil.move(exp_dir, dest)
    return dest


def resolve_experiment_dir(exp_dir: str, cfg: DictConfig, fresh: bool = False) -> str | None:
    """Archive a stale/incompatible `exp_dir` before this run starts.

    Returns the path it was archived to, or None if `exp_dir` is safe to
    reuse/resume as-is (including the case where it doesn't exist yet).
    Never deletes anything -- a mismatched or --fresh'd experiment directory
    is moved to `<exp_dir>_runs/<run_id>/`, never removed.
    """
    if not os.path.isdir(exp_dir):
        return None
    if not _has_checkpoint(exp_dir):
        return None

    results_path = os.path.join(exp_dir, "results.json")
    finished = os.path.exists(results_path)
    if finished and fresh:
        archived_to = _archive_experiment_dir(exp_dir)
        print(f"  --fresh requested on a finished experiment: archived old run to {archived_to}")
        return archived_to

    old_config_path = os.path.join(exp_dir, "resolved_config.yaml")
    if not os.path.exists(old_config_path):
        archived_to = _archive_experiment_dir(exp_dir)
        print(f"  Existing checkpoint has no resolved_config.yaml to verify against: "
              f"archived old run to {archived_to}")
        return archived_to

    old_cfg = OmegaConf.load(old_config_path)
    if config_fingerprint(old_cfg) != config_fingerprint(cfg):
        archived_to = _archive_experiment_dir(exp_dir)
        print(f"  Config changed since the last run of this experiment: "
              f"archived old run to {archived_to}")
        return archived_to

    return None


def resume_ckpt_path(stage: int) -> str | None:
    """Path to the last checkpoint for `stage`, relative to the (already
    chdir'd-into) experiment directory, or None if there isn't one yet."""
    path = os.path.join("checkpoints", f"stage{stage}_last.ckpt")
    return path if os.path.exists(path) else None
