"""Spectral-wind QG windows for the QG neural pipeline, with train-time regeneration (G5).

Bridges the G2 generator (`data.qg_datasets`) to `data.qg_neural.QGNeuralDataset`,
which expects legacy full-resolution window dicts (`true_state`,
`init_lead_truth`, `true_params`, optionally `wind_curl`) and draws
observations itself through `QGS01Dataset._generate_obs_ic`:

* `SpecWindTrainSource.draw(k)` generates a fresh set of train windows for
  regeneration round k: new spawn keys in the train namespace starting at
  index 10**9 (disjoint from the stored train split, val and test by
  construction) and an independent Latin-hypercube design per round.
  Nothing is written to disk.
* `materialize_split` returns a stored split at full resolution. Val (stored
  every 12 h) is regenerated deterministically with its stored shard layout
  and checked against the stored frames; test (stored at full resolution) is
  read with purpose='test' only.
* `RegenerateTrainWindows` is a Lightning callback that swaps the train
  dataset's windows every `every` epochs.

Only the `none` and `true` conditioning modes are supported: `noisy` and
`scenario` rely on the legacy storm corruption and need the spectral S1
corruption of the Option B plan's PR-2.
"""
from __future__ import annotations

import json
import os
import time

import numpy as np
import pytorch_lightning as pl
import torch

from data.normalization import compute_channel_stats
from data.qg import QGConfig, QGS01Dataset
from data.qg_datasets import (
    DESIGN_KEY,
    QGDatasetSpec,
    QGGeneratedSplit,
    design_factors,
    generate_windows,
    split_dir,
)
from data.qg_neural import (
    _daily_mean_field,
    _true_params_vector,
    layer_split,
    psi_daily,
    steps_per_day,
)
from models.qg_wind_modes import FourierWindBasis

REGEN_BASE_INDEX = 10**9
SUPPORTED_COND_MODES = ("none", "true")


def check_compatible(spec: QGDatasetSpec, cfg: QGConfig) -> None:
    pairs = {"nx": (spec.nx, cfg.nx), "L": (spec.L, cfg.L), "dt": (spec.dt, cfg.dt),
             "window_days": (spec.window_days, cfg.window_days),
             "lead_days": (spec.lead_days, cfg.init_lead_days), "delta": (spec.delta, cfg.delta)}
    bad = {k: v for k, v in pairs.items() if float(v[0]) != float(v[1])}
    if bad:
        raise ValueError(f"dataset spec and QGConfig disagree on {bad}")


def legacy_windows(result: dict, spec: QGDatasetSpec, with_wind_curl: bool = False) -> list[dict]:
    if result["keep_every"] != 1:
        raise ValueError("legacy windows need full time resolution (keep_every=1)")
    lead = spec.n_lead + 1
    n_win = spec.n_window
    basis = FourierWindBasis(nx=spec.nx, L=spec.L, kmax=spec.kmax) if with_wind_curl else None
    out = []
    for b in range(result["true_state"].shape[0]):
        frames = result["true_state"][b].reshape(result["true_state"].shape[1], -1)
        amps = result["wind_amplitudes"][b]
        win_amps = torch.cat([amps[lead:], amps[-1:]], dim=0)[:n_win]
        f = {k: float(v[b]) for k, v in result["factors"].items()}
        w = {
            "true_state": frames[lead:lead + n_win].clone(),
            "init_lead_truth": frames[:lead].clone(),
            "true_params": {"U1": f["U1"], "rd": f["rd"], "rek": f["rek"],
                            "beta": spec.beta, "U2": spec.U2},
            "wind_state_true": win_amps.clone(),
            "wind_amp": f["level"],
            "wind_seed": int(result["window_seed"][b]) % 2**31,
            "specwind": {"spec": spec.name, "index": int(result["indices"][b]),
                         "factors": f, "regime": int(result["regime"][b]),
                         "r_cf": float(spec.r_cf), "kmax": int(spec.kmax)},
        }
        if with_wind_curl:
            w["wind_curl"] = basis.curl_field(win_amps.double()).float()
        out.append(w)
    return out


class SpecWindTrainSource:
    def __init__(self, spec: QGDatasetSpec, n_windows: int, device: torch.device | str = "cpu",
                 batch_size: int = 256, with_wind_curl: bool = False):
        self.spec = spec
        self.n_windows = int(n_windows)
        self.device = device
        self.batch_size = batch_size
        self.with_wind_curl = with_wind_curl

    def indices(self, round_: int) -> list[int]:
        start = REGEN_BASE_INDEX + int(round_) * self.n_windows
        return list(range(start, start + self.n_windows))

    def factors(self, round_: int) -> dict:
        return design_factors(self.spec, "train", n=self.n_windows,
                              design_key=(DESIGN_KEY, 1 + int(round_)))

    def draw(self, round_: int) -> list[dict]:
        res = generate_windows(self.spec, "train", self.indices(round_), device=self.device,
                               batch_size=self.batch_size, factors=self.factors(round_),
                               keep_every=1)
        return legacy_windows(res, self.spec, self.with_wind_curl)


def materialize_split(spec: QGDatasetSpec, split: str, root: str,
                      device: torch.device | str = "cpu", with_wind_curl: bool = False,
                      check: bool = True) -> tuple[list[dict], dict]:
    d = split_dir(root, spec, split)
    purpose = {"train": "train", "val": "eval", "test": "test"}[split]
    stored = QGGeneratedSplit(d, purpose=purpose)
    report = {"split": split, "n": len(stored), "regenerated": False, "max_abs_diff": None}
    if stored.manifest["keep_every"] == 1:
        windows = []
        for shard in stored.manifest["shards"]:
            data = torch.load(os.path.join(d, shard["file"]), map_location="cpu", weights_only=False)
            windows.extend(legacy_windows(data, spec, with_wind_curl))
        return windows, report
    windows, max_diff = [], 0.0
    keep = stored.manifest["keep_every"]
    for shard in stored.manifest["shards"]:
        data = torch.load(os.path.join(d, shard["file"]), map_location="cpu", weights_only=False)
        res = generate_windows(spec, split, data["indices"], device=device,
                               batch_size=shard["batch_size"], keep_every=1)
        if check:
            regen = res["true_state"][:, ::keep]
            max_diff = max(max_diff, float((regen - data["true_state"]).abs().max()))
        windows.extend(legacy_windows(res, spec, with_wind_curl))
    report.update(regenerated=True, max_abs_diff=max_diff if check else None)
    return windows, report


def load_full_res_windows(spec: QGDatasetSpec, split: str, root: str, indices: list[int],
                          device: torch.device | str = "cpu", with_wind_curl: bool = False,
                          check_rtol: float = 1e-4) -> tuple[list[dict], dict]:
    """Full-resolution legacy windows for selected `indices` of a stored split.

    Splits stored at full resolution (test) are read directly, one shard at a
    time. Thinned splits (train, val) are regenerated for those indices and
    compared with the stored frames at the stored resolution.
    """
    d = split_dir(root, spec, split)
    purpose = {"train": "train", "val": "eval", "test": "test"}[split]
    stored = QGGeneratedSplit(d, purpose=purpose)
    by_index = {w["index"]: w for w in stored.windows}
    missing = [i for i in indices if i not in by_index]
    if missing:
        raise KeyError(f"indices not in the stored {split} split: {missing[:5]}")
    keep = stored.manifest["keep_every"]
    order = {i: k for k, i in enumerate(indices)}
    out: list = [None] * len(indices)
    report = {"split": split, "n": len(indices), "regenerated": keep != 1, "max_rel_diff": None}
    by_file: dict[str, list[int]] = {}
    for i in indices:
        by_file.setdefault(by_index[i]["file"], []).append(i)
    max_diff = 0.0
    for name, idx in by_file.items():
        data = torch.load(os.path.join(d, name), map_location="cpu", weights_only=False)
        rows = [by_index[i]["row"] for i in idx]
        if keep == 1:
            sub = {**data, "true_state": data["true_state"][rows],
                   "wind_amplitudes": data["wind_amplitudes"][rows],
                   "window_seed": [data["window_seed"][r] for r in rows],
                   "regime": data["regime"][rows], "indices": [data["indices"][r] for r in rows],
                   "factors": {k: np.asarray(v)[rows] for k, v in data["factors"].items()}}
        else:
            sub = generate_windows(spec, split, idx, device=device, keep_every=1)
            ref = data["true_state"][rows]
            diff = float((sub["true_state"][:, ::keep] - ref).abs().max() / ref.abs().max())
            max_diff = max(max_diff, diff)
        for w in legacy_windows(sub, spec, with_wind_curl):
            out[order[w["specwind"]["index"]]] = w
    if keep != 1:
        report["max_rel_diff"] = max_diff
        if max_diff > check_rtol:
            raise RuntimeError(f"regenerated {split} windows differ from storage by {max_diff:.3e} "
                               "(relative to the stored field's max)")
    return out, report


def with_fixed_obs(windows: list[dict], cfg: QGConfig, indices: list[int] | None = None) -> list[dict]:
    """Fixed observations and initial states; `indices` (the windows' dataset
    indices) seed them per window, so the draw does not depend on which subset
    or shard a window is loaded with."""
    idx = list(range(len(windows))) if indices is None else [int(i) for i in indices]
    obs_ic = QGS01Dataset._generate_obs_ic(cfg, windows, idx)
    return [{**w, **o} for w, o in zip(windows, obs_ic)]


def train_norm_stats(windows: list[dict], cfg: QGConfig, with_forcing: bool = False) -> dict:
    split = layer_split(cfg)
    psi1, psi2, forcing = [], [], []
    params = torch.stack([_true_params_vector(w) for w in windows])
    for w in windows:
        ps = psi_daily(w, cfg)
        psi1.append(ps[:, :split].reshape(-1))
        psi2.append(ps[:, split:].reshape(-1))
        if with_forcing:
            forcing.append(_daily_mean_field(w["wind_curl"], steps_per_day(cfg)).reshape(-1))
    out = {"psi": compute_channel_stats(torch.stack([torch.cat(psi1), torch.cat(psi2)], dim=-1)),
           "params": compute_channel_stats(params)}
    if with_forcing:
        out["forcing"] = compute_channel_stats(torch.cat(forcing).unsqueeze(-1))
    return out


class RegenerateTrainWindows(pl.Callback):
    def __init__(self, dataset, source: SpecWindTrainSource, every: int, log_path: str | None = None):
        if every < 1:
            raise ValueError("regeneration interval must be >= 1 epoch")
        self.dataset = dataset
        self.source = source
        self.every = int(every)
        self.log_path = log_path

    def on_train_epoch_start(self, trainer, pl_module) -> None:
        epoch = trainer.current_epoch
        if epoch == 0 or epoch % self.every:
            return
        round_ = epoch // self.every
        t0 = time.time()
        self.dataset.windows = self.source.draw(round_)
        entry = {"epoch": epoch, "round": round_, "n": len(self.dataset.windows),
                 "seconds": round(time.time() - t0, 1),
                 "first_index": self.source.indices(round_)[0]}
        if self.log_path:
            with open(self.log_path, "a") as fh:
                fh.write(json.dumps(entry) + "\n")


def regeneration_rounds_disjoint(source: SpecWindTrainSource, rounds: int, stored_n: int) -> bool:
    seen = set()
    for r in range(rounds):
        idx = source.indices(r)
        if seen & set(idx) or min(idx) < stored_n:
            return False
        seen |= set(idx)
    return True


def summarize_rounds(source: SpecWindTrainSource, rounds: int) -> dict:
    lv = [source.factors(r)["level"] for r in range(rounds)]
    return {"rounds": rounds, "calm_fraction": [float(np.mean(x == 0)) for x in lv]}
