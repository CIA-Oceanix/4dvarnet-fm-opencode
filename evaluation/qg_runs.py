"""What a QG evaluation needs to know about an archived training run.

An evaluation must rebuild *the architecture the checkpoint was trained with*.
Re-declaring it at the call site -- which `eval_qg_neural_s0_s1.py` did, as a
hand-maintained `param_dim`/`cond_extra_dim`/`ic_dim` literal per scheme -- is a
silent-failure shape: nothing connects the literal to the run, so a config
change makes the eval load weights into the wrong model or (at best) raise a
shape error far from the cause.

`architecture(run)` reads it from the run instead, in this order:

1. `resolved_config.yaml` -- written by `train_qg_neural.py` since 2026-09-19,
   the post-CLI-override config the model was actually built from.
2. `results.json` -- its `config` block, for the Q1-Q4 runs archived before (1)
   existed. It records every architecture field, so those runs remain
   evaluable without reconstructing anything by hand.
3. `config/experiment/<run>.yaml` -- the source config, last because it predates
   any CLI override and cannot know which ones were applied.

**`cond_mode` is deliberately not part of this.** It is an evaluation-time
choice, not a property of the checkpoint: the schemes are trained with
`"true"`/`"noisy"` (which ignore the scenario label) and evaluated with
`"scenario"` (which reads each window's own believed model), and taking the
training value here would silently turn the S1 comparison into a meaningless
one. See `eval_qg_neural_s0_s1.py`'s module docstring.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from omegaconf import OmegaConf

from evaluation.archive import QG, ROOT

FIELDS = ("model_type", "param_dim", "cond_extra_dim", "ic_dim")
EXPERIMENT_CONFIGS = ROOT / "config" / "experiment"


def _from_resolved_config(path: Path) -> dict:
    cfg = OmegaConf.load(path)
    model = cfg.get("model", {})
    return {
        "model_type": model.get("model_type"),
        "param_dim": int(model.get("param_dim", 0)),
        "cond_extra_dim": int(model.get("cond_extra_dim", 0)),
        "ic_dim": int(model.get("ic_dim", 0)),
        "source": "resolved_config.yaml",
    }


def _from_results_json(path: Path) -> dict:
    d = json.loads(path.read_text())
    cfg = d.get("config", {})
    return {
        "model_type": d.get("model_type"),
        "param_dim": int(cfg.get("param_dim", 0)),
        "cond_extra_dim": int(cfg.get("cond_extra_dim", 0)),
        # Older runs record include_ic and derive ic_dim from it (2 psi layers).
        "ic_dim": int(cfg.get("ic_dim", 2 if cfg.get("include_ic") else 0)),
        "source": "results.json",
    }


def _from_experiment_yaml(path: Path) -> dict:
    cfg = OmegaConf.load(path)
    model = cfg.get("model", {})
    return {
        "model_type": cfg.get("model_type"),
        "param_dim": int(model.get("param_dim", 0)),
        "cond_extra_dim": int(model.get("cond_extra_dim", 0)),
        "ic_dim": 2 if cfg.get("data", {}).get("include_ic", False) else 0,
        "source": "config/experiment",
    }


def architecture(run: str, *, experiment_id: Optional[str] = None) -> dict:
    """Model-construction arguments for archived run ``run``.

    ``experiment_id`` names the source YAML for fallback 3 when it differs from
    the run directory name. Returns the four `FIELDS` plus ``source``, naming
    which of the three the answer came from -- callers print it, so a run that
    silently fell through to the weakest source is visible in the eval log.
    """
    p = QG.resolve_config(run, required=False)
    if p is not None:
        return _from_resolved_config(p)
    p = QG.resolve_artifact(run, "results.json", required=False)
    if p is not None:
        return _from_results_json(p)
    yaml_path = EXPERIMENT_CONFIGS / f"{experiment_id or run}.yaml"
    if yaml_path.is_file():
        return _from_experiment_yaml(yaml_path)
    raise FileNotFoundError(
        f"No config for run {run!r}: no resolved_config.yaml or results.json in "
        + " or ".join(str(d) for d in QG.candidate_run_dirs(run))
        + f", and no {yaml_path}")


def checkpoint(run: str, *, required: bool = True) -> Optional[Path]:
    """The run's weights: the finished bare state dict, else Lightning's own.

    A run killed before it finished has only `checkpoints/stage1_{best,last}.ckpt`
    -- still loadable (`eval_qg_neural_s0_s1.py` strips the `model.` prefix), so
    an interrupted sweep stays evaluable instead of looking like a missing run.
    """
    return QG.resolve_checkpoint(run, required=required)


def norm_stats_paths(run: str) -> dict:
    """The three stats files a QG evaluation may need, config-first.

    Absent entries are ``None``: only conditioned schemes have param/forcing
    stats, and a run trained with ``--no-normalize`` has no psi stats.
    """
    p = QG.resolve_config(run, required=False)
    cfg = OmegaConf.load(p) if p is not None else None
    return {
        "psi": QG.resolve_norm_stats(cfg, name=run, key="norm_stats_path",
                                     filename="qg_psi_norm_stats.pt"),
        "param": QG.resolve_norm_stats(cfg, name=run, key="param_norm_stats_path",
                                       filename="qg_param_norm_stats.pt"),
        "forcing": QG.resolve_norm_stats(cfg, name=run, key="forcing_norm_stats_path",
                                         filename="qg_forcing_norm_stats.pt"),
    }
