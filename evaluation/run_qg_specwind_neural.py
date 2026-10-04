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
import json
import os
import time

import numpy as np
import torch
from omegaconf import OmegaConf
from scipy.interpolate import CubicSpline

from data.normalization import load_norm_stats
from data.qg import QGConfig
from data.qg_datasets import SPECS, shard_indices
from data.qg_neural import QGNeuralDataset, denorm_psi, psi_to_q, qg_collate, steps_per_day
from evaluation.metrics import crps as ensemble_crps
from evaluation.run_qg_specwind_da import FIELDS, OBS_SEED, build_cfg, s0_windows, window_metrics
from train_qg_neural import DIRECT_TYPES, build_model

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
    if model_type not in (*DIRECT_TYPES, "vanilla_cfm_tchannels"):
        raise ValueError(f"{model_type!r} is not scored by this driver (SDA rows: P6)")
    model = build_model(model_type, cfg, param_dim=int(m.get("param_dim", 0)),
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


@torch.no_grad()
def estimate_members(model: torch.nn.Module, model_type: str, batch, n_members: int) -> torch.Tensor:
    """(n_members, B, days, D) normalized daily psi; one member for a DirectUNet."""
    if model_type in DIRECT_TYPES:
        return model(batch)[None]
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
             batch_size: int = 4) -> list[dict]:
    ds = QGNeuralDataset(windows, cfg, norm, on_the_fly_obs=False)
    per_window = []
    for start in range(0, len(windows), batch_size):
        idx = list(range(start, min(start + batch_size, len(windows))))
        batch = qg_collate([ds[i] for i in idx]).to(device)
        members = estimate_members(model, model_type, batch, n_members).cpu()
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
    load_s = time.time() - t0
    model, model_type, norm = load_run(args.run_dir, cfg, device)
    t0 = time.time()
    per_window = evaluate(model, model_type, norm, windows, cfg, device, args.n_members,
                          args.interp, args.batch_size)
    meta = {"spec": spec.name, "split": args.split, "indices": idx,
            "method": f"neural_{model_type}", "run_dir": os.path.abspath(args.run_dir),
            "cols_per_day": args.cols_per_day, "obs_noise_frac": args.obs_noise_frac,
            "N": per_window[0]["n_members"] if per_window else None, "interp": args.interp,
            "load": report, "load_seconds": round(load_s, 1),
            "eval_seconds": round(time.time() - t0, 1), "obs_seed": OBS_SEED,
            "s1_kappa": 0.0, "s1_note": "obs-only network: S1 = S0 by construction",
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
