"""S0 ETKF/EnKF on the spectral-wind QG datasets: ocean state only, true wind prescribed.

DA-1 of `docs/plans/analysis/qg_specwind_da_s0.md`. Windows of a stored split
(forced `qg_specwind_gyrostat_v1` or coupled `qg_coupled_gyrostat_v1`) are
loaded at full resolution, given fixed nadir-like observations (upper-layer
psi on `cols_per_day` random meridional columns, each once per day) and the
S0 scenario fields (true parameters, the truth's own wind amplitudes), then
passed to `evaluation.run_qg_baselines.run`. The DA model is `QGDynamics` with
the spectral wind hook and the truth's eddy drag, so S0 is a perfect model for
both datasets. The defaults are the DA-2 val-tuned settings
(`docs/results/qg_specwind_da2_val_tuning.md`), with the exact localized
EnSRF update since 2026-09-28 (`docs/results/qg_specwind_etkf_loc_update.md`):
ETKF radius 8, ridge 0.1, `loc_mode="ensrf"`, cross-layer weight 1, bred
initial ensemble.

Run as a module from the repo root:
    python -m evaluation.run_qg_specwind_da --spec qg_specwind_gyrostat_v1 --n-windows 100 \
        --method etkf --out experiments/qg_specwind_da/etkf_forced_c3.json
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import os
import shutil
import tempfile
import time

import numpy as np
import torch

from data.qg import QGConfig
from data.qg_datasets import SPECS, QGDatasetSpec, shard_indices
from data.qg_specwind_neural import check_compatible, load_full_res_windows, with_fixed_obs
from evaluation.qg_specwind_s1 import (
    COMPONENTS,
    REALISTIC_GROUPS,
    REALISTIC_SELECTED,
    REALISTIC_VARIANTS,
    REFERENCE,
    S1Levels,
    obs_error_frac,
    s1_windows,
)
from evaluation.run_qg_baselines import run
from models.qg_dynamics import QGDynamics
from models.qg_wind_modes import FourierWindBasis

OBS_SEED = 20_260_926


def build_cfg(spec: QGDatasetSpec, cols_per_day: int, obs_noise_std_frac: float,
              init_lag_days: float) -> QGConfig:
    cfg = QGConfig(nx=spec.nx, L=spec.L, dt=spec.dt, beta=spec.beta, delta=spec.delta,
                   U2=spec.U2, window_days=spec.window_days, init_lead_days=spec.lead_days,
                   obs_geometry="random_columns", obs_field="psi", cols_per_day=cols_per_day,
                   obs_noise_std_frac=obs_noise_std_frac, init_lag_days=init_lag_days,
                   seed=OBS_SEED, wind_sigma=spec.sigma)
    check_compatible(spec, cfg)
    return cfg


def s0_windows(spec: QGDatasetSpec, split: str, root: str, indices: list[int], cfg: QGConfig,
               device: torch.device | str = "cpu") -> tuple[list[dict], dict]:
    windows, report = load_full_res_windows(spec, split, root, indices, device=device)
    windows = with_fixed_obs(windows, cfg, indices)
    per_layer = cfg.ny * cfg.nx
    out = []
    for idx, w in zip(indices, windows):
        tp = w["true_params"]
        inv = QGDynamics(nx=cfg.nx, L=cfg.L, dt=cfg.dt, beta=tp["beta"], rd=tp["rd"],
                         delta=cfg.delta, U1=tp["U1"], U2=tp["U2"], rek=tp["rek"])
        psi1 = inv.streamfunctions(w["true_state"])[:, 0].reshape(w["true_state"].shape[0], -1)
        out.append({
            **w,
            "da_model": "qg2l", "da_nx": cfg.nx, "da_params": dict(tp),
            "init_seed_key": idx,
            "wind_state_corrupted": w["wind_state_true"],
            "target_state_psi": psi1, "target_state_q": w["true_state"][:, :per_layer].clone(),
        })
    return out, report


def s1_levels_from(kappa: float, components=COMPONENTS, base: S1Levels = REFERENCE) -> S1Levels:
    return base.scaled(kappa).only(components)


def apply_s1(windows: list[dict], spec: QGDatasetSpec, levels: S1Levels) -> list[dict]:
    basis = FourierWindBasis(nx=spec.nx, L=spec.L, kmax=spec.kmax)
    return s1_windows(windows, levels, spec.dt, basis.wavevectors)


def da_cfg(cfg: QGConfig, levels: S1Levels | None) -> QGConfig:
    """The filter's config: its observation-error fraction follows the S1 `obs` component."""
    if levels is None or not (levels.obs_white_frac or levels.obs_corr_frac):
        return cfg
    return dataclasses.replace(cfg, obs_noise_std_frac=obs_error_frac(levels))


FIELDS = ("psi1", "psi2", "q1", "q2")


def _layers(q: np.ndarray, tp: dict, cfg: QGConfig, device) -> dict:
    inv = QGDynamics(nx=cfg.nx, L=cfg.L, dt=cfg.dt, beta=tp["beta"], rd=tp["rd"], delta=cfg.delta,
                     U1=tp["U1"], U2=tp["U2"], rek=tp["rek"]).to(device)
    psi = inv.streamfunctions(torch.from_numpy(q).to(device)).cpu().numpy()
    psi = psi.reshape(q.shape[0], 2, -1)
    per = q.shape[1] // 2
    return {"psi1": psi[:, 0], "psi2": psi[:, 1], "q1": q[:, :per], "q2": q[:, per:]}


def window_metrics(analysis: np.ndarray, free: np.ndarray, truth: np.ndarray, tp: dict,
                   cfg: QGConfig, device) -> dict:
    """Per-window EV and RMSE of psi1/psi2/q1/q2 for the analysis and the free forecast.

    EV = 1 - MSE / Var(truth) over the window's space-time, with psi inverted
    using the window's own true parameters.
    """
    a, f, t = (_layers(x, tp, cfg, device) for x in (analysis, free, truth))
    out = {}
    for k in FIELDS:
        var = float(((t[k] - t[k].mean()) ** 2).mean())
        for tag, est in (("da", a[k]), ("free", f[k])):
            mse = float(((est - t[k]) ** 2).mean())
            out[f"ev_{tag}_{k}"] = 1.0 - mse / max(var, 1e-30)
            out[f"rmse_{tag}_{k}"] = mse ** 0.5
    out["score"] = float(np.mean([out[f"ev_da_{k}"] for k in FIELDS]))
    return out


def evaluate(windows: list[dict], cfg: QGConfig, method: str, device: torch.device, N: int = 80,
             inflation: float = 1.0, loc_radius: float = 8.0, etkf_ridge: float = 0.1,
             loc_cross_layer: float = 1.0, init_ensemble: str = "bred", breed_days: float = 3.0,
             disp_frac: float = 1.0, init_lag_days: float = 5.0, out_path: str | None = None,
             save_traj: str | None = None, etkf_loc_mode: str = "ensrf",
             enks_lag: int | None = None, r_scale: float = 1.0,
             enks_taper_days: float | None = None, **var_kwargs) -> tuple[dict, list[dict]]:
    """Run S0/S1 DA on prepared windows; return run()'s payload and per-window metrics."""
    traj_dir = save_traj or tempfile.mkdtemp(prefix="qgda_", dir=os.environ.get("TMPDIR", "/tmp"))
    try:
        payload = run(method, cfg, device=device, N_ensemble=N, inflation=inflation,
                      loc_radius=loc_radius, scenarios=("test_s0",), out_path=out_path,
                      init="lagged", geometry="random_columns", obs_var="psi",
                      init_lag_days=init_lag_days, ds={"test_s0": windows}, etkf_ridge=etkf_ridge,
                      save_traj=traj_dir, loc_cross_layer=loc_cross_layer,
                      init_ensemble_kind=init_ensemble, breed_days=breed_days, disp_frac=disp_frac,
                      etkf_loc_mode=etkf_loc_mode, enks_lag=enks_lag, obs_var_r_scale=r_scale,
                      enks_taper_steps=(enks_taper_days * 86400.0 / cfg.dt
                                        if enks_taper_days is not None else None),
                      **var_kwargs)
        s = payload["scenarios"]["test_s0"]
        traj = np.load(s["traj_path"])
        per_window = []
        for k, w in enumerate(windows):
            m = window_metrics(traj["analyses"][k], traj["free_forecast"][k], traj["refs"][k],
                               w["true_params"], cfg, device)
            m.update({"index": int(w["init_seed_key"]), "crps": s["crps_list"][k],
                      "level": w["specwind"]["factors"]["level"],
                      **{f"spread_ratio_{f}": v for f, v in s["spread_ratio_list"][k].items()},
                      "n_fallback": s["fallback_list"][k]})
            per_window.append(m)
    finally:
        if save_traj is None:
            shutil.rmtree(traj_dir, ignore_errors=True)
    return payload, per_window


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--spec", required=True, choices=sorted(SPECS))
    p.add_argument("--root", default="experiments/qg_datasets")
    p.add_argument("--split", default="test", choices=("test", "val"))
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--n-windows", type=int, default=100)
    p.add_argument("--shard", type=int, default=0)
    p.add_argument("--n-shards", type=int, default=1)
    p.add_argument("--method", default="etkf",
                   choices=("etkf", "enkf", "enks", "strong4dvar", "weak4dvar"),
                   help="enks = localized EnKS smoother on the EnSRF ETKF")
    p.add_argument("--cols-per-day", type=int, default=3)
    p.add_argument("--obs-noise-frac", type=float, default=0.05)
    p.add_argument("--init-lag-days", type=float, default=5.0)
    p.add_argument("--N", type=int, default=80)
    p.add_argument("--inflation", type=float, default=1.0)
    p.add_argument("--loc-radius", type=float, default=8.0)
    p.add_argument("--etkf-ridge", type=float, default=0.1)
    p.add_argument("--loc-cross-layer", type=float, default=1.0,
                   help="cross-layer localization weight (0: the lower layer is not updated directly)")
    p.add_argument("--init-ensemble", default="bred", choices=("white", "bred"))
    p.add_argument("--breed-days", type=float, default=3.0)
    p.add_argument("--disp-frac", type=float, default=1.0)
    p.add_argument("--enks-taper-days", type=float, default=8.0,
                   help="EnKS time taper width in days (Gaspari-Cohn, zero beyond twice it; val-tuned 8; "
                        "<= 0 disables it)")
    p.add_argument("--da-window-steps", type=int, default=60,
                   help="4D-Var sub-window length in model steps (60 = 5 days)")
    p.add_argument("--fourdvar-optimizer", default="lbfgs", choices=("lbfgs", "adam"))
    p.add_argument("--fourdvar-max-iter", type=int, default=60)
    p.add_argument("--fourdvar-opt-steps", type=int, default=150)
    p.add_argument("--fourdvar-lr", type=float, default=1.0)
    p.add_argument("--fourdvar-grad-clip", type=float, default=1000.0)
    p.add_argument("--b-var-scale", type=float, default=1.0)
    p.add_argument("--fourdvar-b", default="diag", choices=("diag", "spectral"),
                   help="4D-Var background covariance: diagonal (legacy) or spectral climatological "
                        "(from the lead buffer, correlated between layers)")
    p.add_argument("--q-var-scale", type=float, default=0.1)
    p.add_argument("--r-scale", type=float, default=1.0,
                   help="multiply the filter's observation-error variance (absorbs model error)")
    p.add_argument("--enks-lag", type=int, default=0,
                   help="EnKS hard lag in analyses (0 = none; the benchmark uses the time taper instead, "
                        "which beat lag 12 on val)")
    p.add_argument("--etkf-loc-mode", default="ensrf", choices=("square_root", "ensrf"),
                   help="localized ETKF update: legacy square_root, or the exact EnSRF (ensrf)")
    p.add_argument("--s1-kappa", type=float, default=0.0,
                   help="S1 intensity: multiplies the reference error levels (0 = S0)")
    p.add_argument("--s1-preset", default="reference", choices=("reference", "realistic"),
                   help="reference: legacy-analogue levels scaled by --s1-kappa; realistic: the "
                        "realism-anchored levels (--s1-variant), switched on by --s1-kappa 1")
    p.add_argument("--s1-variant", default=REALISTIC_SELECTED, choices=sorted(REALISTIC_VARIANTS),
                   help="realistic-preset level set (default: the calibrated one)")
    p.add_argument("--s1-components", default=None,
                   help=f"comma-separated components (reference: {COMPONENTS}; "
                        f"realistic: {REALISTIC_GROUPS}); default all")
    for comp_arg in ("amp-bias", "noise-frac", "shift-frac", "param-bias"):
        p.add_argument(f"--s1-{comp_arg}", type=float, default=None,
                       help="override the reference level of this component (before --s1-kappa)")
    p.add_argument("--out", required=True, help="summary JSON path (run() output)")
    p.add_argument("--save-traj", default=None,
                   help="directory for run()'s trajectory npz (analysis mean, free forecast, truth)")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()
    spec = SPECS[args.spec]
    device = torch.device(args.device)
    all_idx = list(range(args.start, args.start + args.n_windows))
    idx = [all_idx[k] for k in shard_indices(len(all_idx), args.shard, args.n_shards)]
    cfg = build_cfg(spec, args.cols_per_day, args.obs_noise_frac, args.init_lag_days)
    t0 = time.time()
    windows, report = s0_windows(spec, args.split, args.root, idx, cfg, device)
    realistic = args.s1_preset == "realistic"
    comps = (args.s1_components.split(",") if args.s1_components
             else list(REALISTIC_GROUPS if realistic else COMPONENTS))
    preset = REALISTIC_VARIANTS[args.s1_variant] if realistic else REFERENCE
    base = S1Levels(**{**preset.as_dict(), **{
        k: getattr(args, f"s1_{k}") for k in ("amp_bias", "noise_frac", "shift_frac", "param_bias")
        if getattr(args, f"s1_{k}") is not None}})
    levels = base.only(comps) if realistic else s1_levels_from(args.s1_kappa, comps, base)
    if args.s1_kappa:
        windows = apply_s1(windows, spec, levels)
    cfg_da = da_cfg(cfg, levels if args.s1_kappa else None)
    load_s = time.time() - t0
    t0 = time.time()
    summary, per_window = evaluate(
        windows, cfg_da, args.method, device, N=args.N, inflation=args.inflation,
        loc_radius=args.loc_radius, etkf_ridge=args.etkf_ridge,
        loc_cross_layer=args.loc_cross_layer, init_ensemble=args.init_ensemble,
        breed_days=args.breed_days, disp_frac=args.disp_frac, init_lag_days=args.init_lag_days,
        out_path=args.out, save_traj=args.save_traj, etkf_loc_mode=args.etkf_loc_mode,
        enks_lag=args.enks_lag or None, r_scale=args.r_scale,
        enks_taper_days=args.enks_taper_days if args.enks_taper_days and args.enks_taper_days > 0 else None,
        **({k: getattr(args, k) for k in ("da_window_steps", "b_var_scale", "q_var_scale")}
           | {"optimizer": args.fourdvar_optimizer, "fourdvar_max_iter": args.fourdvar_max_iter,
              "fourdvar_opt_steps": args.fourdvar_opt_steps, "fourdvar_lr": args.fourdvar_lr,
              "fourdvar_grad_clip": args.fourdvar_grad_clip, "fourdvar_b": args.fourdvar_b}
           if "4dvar" in args.method else {}))
    meta = {"spec": spec.name, "split": args.split, "indices": idx, "method": args.method,
            "cols_per_day": args.cols_per_day, "obs_noise_frac": args.obs_noise_frac,
            "init_lag_days": args.init_lag_days, "N": args.N, "inflation": args.inflation,
            "loc_radius": args.loc_radius, "etkf_ridge": args.etkf_ridge,
            "loc_cross_layer": args.loc_cross_layer, "init_ensemble": args.init_ensemble,
            "breed_days": args.breed_days, "disp_frac": args.disp_frac, "etkf_loc_mode": args.etkf_loc_mode, "enks_lag": args.enks_lag, "r_scale": args.r_scale, "enks_taper_days": args.enks_taper_days,
            "fourdvar": ({"da_window_steps": args.da_window_steps, "optimizer": args.fourdvar_optimizer,
                          "max_iter": args.fourdvar_max_iter, "lr": args.fourdvar_lr,
                          "b_var_scale": args.b_var_scale, "q_var_scale": args.q_var_scale,
                          "b": args.fourdvar_b}
                         if "4dvar" in args.method else None),
            "load": report, "load_seconds": round(load_s, 1),
            "da_seconds": round(time.time() - t0, 1), "obs_seed": OBS_SEED,
            "s1_kappa": args.s1_kappa, "s1_preset": args.s1_preset if args.s1_kappa else None,
            "s1_variant": args.s1_variant if args.s1_kappa and realistic else None,
            "s1_levels": levels.as_dict() if args.s1_kappa else None,
            "per_window": per_window}
    with open(os.path.splitext(args.out)[0] + "_meta.json", "w") as fh:
        json.dump(meta, fh, indent=1)
    print(json.dumps({"meta": meta, "summary": summary.get("scenarios", {}).get("test_s0")},
                     default=str)[:2000])


if __name__ == "__main__":
    main()
