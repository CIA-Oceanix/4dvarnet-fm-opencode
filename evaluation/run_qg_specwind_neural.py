"""Learned rows of the spectral-wind gyrostat QG benchmark: score a trained network like the DA rows.

P1 of `docs/plans/analysis/qg_specwind_neural.md`. The windows, their fixed
observations and the truth are the DA driver's own (`run_qg_specwind_da.s0_windows`
with its `build_cfg`, so the obs are the DA rows' exact draw). A network trained by
`train_qg_neural.py --specwind-spec` (DirectUNet or Vanilla CFM, T-channels) is rebuilt
from the run's `resolved_config.yaml`, estimates the 30 daily-mean psi fields, and the
estimate is placed back on the 2-hourly truth grid:

- the daily means sit at the centres of their 12-step bins (step 12 d + 5.5);
- a not-a-knot cubic spline in time, per grid point and layer, evaluates them at every
  step (the first and last half-days are extrapolated by the end pieces);
- psi is inverted to PV with the window's own rd at each step (linear, so this equals
  interpolating q).

Scores are the DA driver's `window_metrics` (EV / RMSE of psi1, psi2, q1, q2 against the
2-hourly truth), plus for ensembles the fair CRPS and spread / RMSE per layer computed on
q, exactly as `run_qg_baselines` does for the DA ensembles. The daily-mean diagnostic
(the network output against the daily-mean truth, no interpolation) is stored next to them
as `daily_*`. The per-window JSON is written as `<out>_meta.json` in the DA driver's schema,
so `reports/qg/generate_qg_specwind_da_report.py` reads the rows without a new code path.

Run as a module from the repo root:
    python -m evaluation.run_qg_specwind_neural --run-dir experiments/qg_specwind_neural/G1_..._s0 \
        --spec qg_specwind_gyrostat_v1 --n-windows 100 --n-members 1 \
        --out experiments/qg_specwind_da/qg_specwind_gyrostat_v1/test/neural_G1_s0/shard_0_of_1.json
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import os
import time
from types import SimpleNamespace

import numpy as np
import torch
from omegaconf import OmegaConf
from scipy.interpolate import CubicSpline

from data.normalization import load_norm_stats
from data.qg import QGConfig
from data.qg_datasets import SPECS, shard_indices
from data.normalization import normalize
from data.qg_neural import (
    QGBatch,
    QGNeuralDataset,
    compose_days,
    denorm_psi,
    layer_split,
    slice_days,
    psi_to_q,
    qg_collate,
    steps_per_day,
    windowed_estimate,
)
from evaluation.metrics import crps as ensemble_crps
from evaluation.qg_specwind_s1 import REALISTIC_SELECTED, REALISTIC_VARIANTS
from evaluation.run_qg_specwind_da import (
    FIELDS,
    OBS_SEED,
    apply_s1,
    build_cfg,
    s0_windows,
    window_metrics,
)
from train_qg_neural import DIRECT_TYPES, SDA_TYPES, build_model

INTERP_KINDS = ("cubic", "linear")


def bin_centres(n_days: int, spd: int) -> np.ndarray:
    """Step-time centre of each daily bin (the mean of steps spd*d ... spd*d + spd - 1)."""
    return np.arange(n_days) * spd + (spd - 1) / 2.0


def daily_to_steps(daily: np.ndarray, spd: int, n_steps: int, kind: str = "cubic") -> np.ndarray:
    """Interpolate (..., days, D) daily means to (..., n_steps, D) model steps along the day axis."""
    if kind not in INTERP_KINDS:
        raise ValueError(f"unknown interpolation {kind!r}")
    x = bin_centres(daily.shape[-2], spd)
    t = np.arange(n_steps, dtype=float)
    if kind == "cubic":
        return CubicSpline(x, daily, axis=-2, bc_type="not-a-knot")(t)
    slope_lo = (daily[..., 1, :] - daily[..., 0, :]) / (x[1] - x[0])
    slope_hi = (daily[..., -1, :] - daily[..., -2, :]) / (x[-1] - x[-2])
    k = np.clip(np.searchsorted(x, t, side="right") - 1, 0, len(x) - 2)
    w = (t - x[k]) / (x[k + 1] - x[k])
    out = (1.0 - w)[:, None] * daily[..., k, :] + w[:, None] * daily[..., k + 1, :]
    lo, hi = t < x[0], t > x[-1]
    out[..., lo, :] = daily[..., :1, :] + (t[lo] - x[0])[:, None] * slope_lo[..., None, :]
    out[..., hi, :] = daily[..., -1:, :] + (t[hi] - x[-1])[:, None] * slope_hi[..., None, :]
    return out


def load_run(run_dir: str, cfg: QGConfig, device: torch.device) -> tuple[torch.nn.Module, str, dict | None]:
    """The trained network, rebuilt at the architecture its resolved_config.yaml records."""
    rc = OmegaConf.load(os.path.join(run_dir, "resolved_config.yaml"))
    m = rc.model
    model_type = str(m.model_type)
    if model_type not in (*DIRECT_TYPES, "vanilla_cfm_tchannels", *SDA_TYPES):
        raise ValueError(f"{model_type!r} is not scored by this driver")
    days = rc.data.get("train_window_days", None)
    model_cfg = dataclasses.replace(cfg, window_days=float(days)) if days is not None else cfg
    model = build_model(model_type, model_cfg, param_dim=int(m.get("param_dim", 0)),
                        cond_extra_dim=int(m.get("cond_extra_dim", 0)), ic_dim=int(m.get("ic_dim", 0)),
                        use_obs_mask=bool(m.get("use_obs_mask", False)),
                        hidden_channels=list(m.get("hidden_channels", [64, 128, 256])),
                        num_res_blocks=int(m.get("num_res_blocks", 2)))
    state = torch.load(os.path.join(run_dir, "stage1_best.pt"), map_location="cpu", weights_only=False)
    state = state["state_dict"] if isinstance(state, dict) and "state_dict" in state else state
    model.load_state_dict({(k[6:] if k.startswith("model.") else k): v for k, v in state.items()},
                          strict=True)
    norm = None
    if rc.data.get("normalize", True):
        recorded = rc.data.get("norm_stats_path")
        beside = os.path.join(run_dir, os.path.basename(recorded or "specwind_psi_norm_stats.pt"))
        norm_path = beside if os.path.exists(beside) else recorded
        if not norm_path or not os.path.exists(norm_path):
            raise FileNotFoundError(f"psi norm stats of {run_dir} not found ({beside}, {recorded})")
        norm = load_norm_stats(norm_path)
    return model.to(device).eval(), model_type, norm


def _sub_window(batch, start: int, stop: int, n_days: int):
    """Days [start, stop) of a QGBatch, or of every per-day tensor of a sampler's batch shim."""
    if isinstance(batch, QGBatch):
        return slice_days(batch, start, stop)
    return SimpleNamespace(**{k: v[:, start:stop] if torch.is_tensor(v) and v.dim() > 1
                              and v.shape[1] == n_days else v for k, v in vars(batch).items()})


class ComposedFlow(torch.nn.Module):
    """A flow network built for `inner.T` days run on an `n_days` state: at every flow step
    its velocity is computed on overlapping sub-windows and blended (`compose_days`), as SDA
    composes a local score over a long trajectory (Rozet & Louppe 2023). Sampling then draws
    one coherent `n_days` trajectory -- unlike blending independently sampled sub-windows,
    which averages independent members and shrinks the ensemble spread where they overlap.
    Exposes what `VanillaCFM.sample` and `sda_guided_sample` read (`forward`, `sample`,
    `state_dim`, `sigma_prior`, `N_outer`, `interpolant`, `T`).
    """

    def __init__(self, inner: torch.nn.Module, n_days: int, stride: int):
        super().__init__()
        self.inner = inner
        self.T = n_days
        self.stride = stride
        for name in ("state_dim", "sigma_prior", "N_outer", "interpolant", "train_tau_0_only"):
            setattr(self, name, getattr(inner, name))

    def forward(self, x: torch.Tensor, batch, tau: torch.Tensor) -> torch.Tensor:
        t, n = self.inner.T, x.shape[1]
        return compose_days(lambda s: self.inner(x[:, s:s + t], _sub_window(batch, s, s + t, n), tau),
                            n, t, self.stride)

    def sample(self, batch, N_outer=None):
        from models.vanilla_cfm import VanillaCFM
        return VanillaCFM.sample(self, batch, N_outer=N_outer)


def renorm_psi(psi_phys: torch.Tensor, cfg: QGConfig, norm: dict | None) -> torch.Tensor:
    """Physical daily psi -> the global per-layer z-score of `norm` (inverse of `denorm_psi`)."""
    if norm is None:
        return psi_phys
    split = layer_split(cfg)
    out = psi_phys.clone()
    for k, sl in enumerate((slice(None, split), slice(split, None))):
        out[..., sl] = normalize(psi_phys[..., sl], {"mean": norm["mean"][k], "std": norm["std"][k]})
    return out


def sda_members(model: torch.nn.Module, batch, n_members: int, guidance_weight: float,
                n_steps: int = 10, mean_estimate: torch.Tensor | None = None,
                tau0: float = 0.0) -> torch.Tensor:
    """(n_members, B, days, D) SDA1 samples: the unconditional prior's Euler ODE nudged by the
    normalized gradient of the observation misfit at x_hat_1 (`evaluation.sda_sampler`).

    The QG observations are per cell (`batch.obs_mask` has the state's own (B, days, D)
    shape), so they go in as the per-cell `obs_channel_mask` with an all-true temporal mask;
    NaN never occurs in the zero-filled QG obs, the mask alone selects the observed cells.
    The guidance step is normalized by the gradient norm, so the scalar R cancels and only
    `guidance_weight` (an L2 step over one window's whole normalized state) matters.
    `mean_estimate` (normalized like the prior's state) warm-starts every member at flow time
    `tau0` from the interpolant between noise and that estimate (SDEdit), the remaining
    steps guided as usual.
    """
    from evaluation.sda_sampler import sda_guided_sample
    b, t, _ = batch.obs.shape
    shim = SimpleNamespace(obs=batch.obs,
                           obs_mask=torch.ones(b, t, dtype=torch.bool, device=batch.obs.device))
    out, _ = sda_guided_sample(model, shim, R_var=1.0, N_outer=n_steps,
                               guidance_weight=guidance_weight, n_members=n_members,
                               obs_channel_mask=batch.obs_mask, mean_estimate=mean_estimate,
                               tau0=tau0)
    return out[None] if n_members == 1 else out.permute(3, 0, 1, 2)


@torch.no_grad()
def estimate_members(model: torch.nn.Module, model_type: str, batch, n_members: int,
                     guidance_weight: float = 100.0, sda_steps: int = 10,
                     mean_estimate: torch.Tensor | None = None, tau0: float = 0.0) -> torch.Tensor:
    """(n_members, B, days, D) normalized daily psi; one member for a DirectUNet."""
    if model_type in DIRECT_TYPES:
        return model(batch)[None]
    if model_type in SDA_TYPES:
        return sda_members(model, batch, n_members, guidance_weight, sda_steps, mean_estimate,
                           tau0)
    return torch.stack([model.sample(batch) for _ in range(n_members)])


def _daily_mean(x: np.ndarray, spd: int) -> np.ndarray:
    return x.reshape(x.shape[0] // spd, spd, -1).mean(1)


def score_window(members_psi_daily: np.ndarray, window: dict, cfg: QGConfig, device: torch.device,
                 interp: str = "cubic") -> dict:
    """Scores of one window from its (N, days, D) physical daily psi members."""
    spd = steps_per_day(cfg)
    truth = window["true_state"].cpu().numpy()
    rd = float(window["true_params"]["rd"])
    steps = daily_to_steps(members_psi_daily, spd, truth.shape[0], interp)
    n, t, d = steps.shape
    q = psi_to_q(torch.from_numpy(steps.reshape(n * t, d)).float().to(device), rd, cfg, device)
    q = q.cpu().numpy().reshape(n, t, d)
    q_mean = q.mean(0)
    m = window_metrics(q_mean, q_mean, truth, window["true_params"], cfg, device)
    out = {k: v for k, v in m.items() if not k.startswith(("ev_free", "rmse_free"))}
    q_daily_est = psi_to_q(torch.from_numpy(members_psi_daily.mean(0)).float().to(device), rd, cfg,
                           device).cpu().numpy()
    md = window_metrics(q_daily_est, q_daily_est, _daily_mean(truth, spd), window["true_params"], cfg,
                        device)
    out.update({f"daily_{k}": v for k, v in md.items() if k.startswith(("ev_da", "rmse_da"))})
    out["daily_score"] = md["score"]
    out["crps"] = float(np.mean(ensemble_crps(q, truth)))
    if n > 1:
        per = d // 2
        var = q.var(0, ddof=1)
        for li in range(2):
            sl = slice(li * per, (li + 1) * per)
            err = float(np.sqrt(np.mean((q_mean[:, sl] - truth[:, sl]) ** 2)))
            out[f"spread_ratio_q{li + 1}"] = float(np.sqrt(np.mean(var[:, sl]))) / max(err, 1e-30)
    return out


def evaluate(model, model_type: str, norm: dict | None, windows: list[dict], cfg: QGConfig,
             device: torch.device, n_members: int = 1, interp: str = "cubic",
             batch_size: int = 4, guidance_weight: float = 100.0,
             sda_steps: int = 10, window_stride: int = 5, warm_start=None,
             tau0: float = 0.5, full_window: bool = False) -> list[dict]:
    """Per-window scores. A network built for fewer days than the window covers it with
    overlapping sub-windows `window_stride` days apart: a DirectUNet blends its sub-window
    estimates (`windowed_estimate`), a flow blends its velocity at every step
    (`ComposedFlow`), so each member stays one coherent trajectory. `warm_start`
    `(model, model_type, norm)` -- a deterministic run -- gives the SDA members their
    starting estimate at flow time `tau0`. `full_window` runs a network that accepts any
    window length (`any_length`, the 3D prior) on the whole window instead."""
    if full_window and not getattr(model, "any_length", False):
        raise ValueError("full_window needs a network that accepts any window length")
    ds = QGNeuralDataset(windows, cfg, norm, on_the_fly_obs=False)
    per_window = []
    for start in range(0, len(windows), batch_size):
        idx = list(range(start, min(start + batch_size, len(windows))))
        batch = qg_collate([ds[i] for i in idx]).to(device)
        t_model = batch.T if full_window else getattr(model, "T", batch.T)
        mean_estimate = None
        if warm_start is not None:
            w_model, w_type, w_norm = warm_start
            w_est = windowed_estimate(lambda b: estimate_members(w_model, w_type, b, 1)[0], batch,
                                      getattr(w_model, "T", batch.T), window_stride)
            mean_estimate = renorm_psi(denorm_psi(w_est, cfg, w_norm), cfg, norm)
        if model_type in DIRECT_TYPES:
            members = windowed_estimate(
                lambda b: estimate_members(model, model_type, b, n_members), batch, t_model,
                window_stride)
        else:
            flow = model if t_model == batch.T else ComposedFlow(model, batch.T, window_stride)
            members = estimate_members(flow, model_type, batch, n_members, guidance_weight,
                                       sda_steps, mean_estimate, tau0)
        members = members.cpu()
        for b, i in enumerate(idx):
            w = windows[i]
            phys = denorm_psi(members[:, b], cfg, norm).numpy()
            m = score_window(phys, w, cfg, device, interp)
            m.update({"index": int(w["init_seed_key"]), "level": w["specwind"]["factors"]["level"],
                      "n_members": int(phys.shape[0])})
            per_window.append(m)
    return per_window


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--run-dir", required=True, help="train_qg_neural.py experiment directory")
    p.add_argument("--spec", required=True, choices=sorted(SPECS))
    p.add_argument("--root", default="experiments/qg_datasets")
    p.add_argument("--split", default="test", choices=("test", "val"))
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--n-windows", type=int, default=100)
    p.add_argument("--shard", type=int, default=0)
    p.add_argument("--n-shards", type=int, default=1)
    p.add_argument("--cols-per-day", type=int, default=3)
    p.add_argument("--obs-noise-frac", type=float, default=0.05)
    p.add_argument("--n-members", type=int, default=30, help="flow samples per window (ens30)")
    p.add_argument("--s1", action="store_true",
                   help="score on the S1 observations: the DA driver's realistic S1 corruption "
                        "(--s1-variant), so the obs carry the DA S1 rows' exact altimetry-error "
                        "draws; the obs-only networks use nothing else of S1")
    p.add_argument("--s1-variant", default=REALISTIC_SELECTED, choices=sorted(REALISTIC_VARIANTS))
    p.add_argument("--guidance-weight", type=float, default=100.0,
                   help="SDA: L2 guidance step per Euler step over one window's normalized state")
    p.add_argument("--sda-steps", type=int, default=10, help="SDA: guided Euler steps")
    p.add_argument("--warm-start-run", default=None,
                   help="SDA: a deterministic run (DirectUNet) whose estimate starts every member "
                        "at flow time --tau0 instead of pure noise")
    p.add_argument("--tau0", type=float, default=0.5, help="SDA warm start: starting flow time")
    p.add_argument("--full-window", action="store_true",
                   help="run a network that accepts any window length (3D prior) on the whole "
                        "window instead of composing it over its training window")
    p.add_argument("--window-stride", type=int, default=None,
                   help="days between the overlapping sub-windows of a network trained on a "
                        "shorter window (default: the run's data.window_stride, else 5)")
    p.add_argument("--interp", default="cubic", choices=INTERP_KINDS)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--out", required=True, help="output path; per-window scores go to <out>_meta.json")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()
    spec = SPECS[args.spec]
    device = torch.device(args.device)
    all_idx = list(range(args.start, args.start + args.n_windows))
    idx = [all_idx[k] for k in shard_indices(len(all_idx), args.shard, args.n_shards)]
    cfg = build_cfg(spec, args.cols_per_day, args.obs_noise_frac, init_lag_days=5.0)
    t0 = time.time()
    windows, report = s0_windows(spec, args.split, args.root, idx, cfg, device)
    levels = REALISTIC_VARIANTS[args.s1_variant] if args.s1 else None
    if levels is not None:
        windows = apply_s1(windows, spec, levels)
    load_s = time.time() - t0
    model, model_type, norm = load_run(args.run_dir, cfg, device)
    t0 = time.time()
    rc = OmegaConf.load(os.path.join(args.run_dir, "resolved_config.yaml"))
    stride = args.window_stride or int(rc.data.get("window_stride", None) or 5)
    warm = None
    if args.warm_start_run:
        warm = load_run(args.warm_start_run, cfg, device)
        if warm[1] not in DIRECT_TYPES:
            raise ValueError(f"--warm-start-run must be a DirectUNet run, got {warm[1]!r}")
    per_window = evaluate(model, model_type, norm, windows, cfg, device, args.n_members,
                          args.interp, args.batch_size, args.guidance_weight, args.sda_steps,
                          window_stride=stride, warm_start=warm, tau0=args.tau0,
                          full_window=args.full_window)
    meta = {"spec": spec.name, "split": args.split, "indices": idx,
            "method": f"neural_{model_type}", "run_dir": os.path.abspath(args.run_dir),
            "cols_per_day": args.cols_per_day, "obs_noise_frac": args.obs_noise_frac,
            "N": per_window[0]["n_members"] if per_window else None, "interp": args.interp,
            "network_window_days": rc.data.get("train_window_days", None),
            "window_stride": (stride if rc.data.get("train_window_days", None)
                              and not args.full_window else None),
            "full_window": args.full_window,
            "sda": ({"guidance_weight": args.guidance_weight, "steps": args.sda_steps,
                     "warm_start_run": (os.path.abspath(args.warm_start_run)
                                        if args.warm_start_run else None),
                     "tau0": args.tau0 if args.warm_start_run else None}
                    if model_type in SDA_TYPES else None),
            "load": report, "load_seconds": round(load_s, 1),
            "eval_seconds": round(time.time() - t0, 1), "obs_seed": OBS_SEED,
            "s1_kappa": 1.0 if args.s1 else 0.0,
            "s1_preset": "realistic" if args.s1 else None,
            "s1_variant": args.s1_variant if args.s1 else None,
            "s1_levels": levels.as_dict() if levels is not None else None,
            "s1_note": "obs-only network: of S1 it sees only the corrupted observations",
            "per_window": per_window}
    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    with open(os.path.splitext(args.out)[0] + "_meta.json", "w") as fh:
        json.dump(meta, fh, indent=1)
    summary = {k: float(np.mean([w[k] for w in per_window]))
               for k in ("score", "daily_score", "crps", *[f"ev_da_{f}" for f in FIELDS])}
    with open(args.out, "w") as fh:
        json.dump({"summary": summary, "n": len(per_window)}, fh, indent=1)
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
