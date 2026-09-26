"""S0 ETKF/EnKF on the spectral-wind QG datasets: ocean state only, true wind prescribed.

DA-1 of `docs/plans/analysis/qg_specwind_da_s0.md`. Windows of a stored split
(forced `qg_specwind_gyrostat_v1` or coupled `qg_coupled_gyrostat_v1`) are
loaded at full resolution, given fixed nadir-like observations (upper-layer
psi on `cols_per_day` random meridional columns, each once per day) and the
S0 scenario fields (true parameters, the truth's own wind amplitudes), then
passed to `evaluation.run_qg_baselines.run`. The DA model is `QGDynamics` with
the spectral wind hook and the truth's eddy drag, so S0 is a perfect model for
both datasets.

Run as a module from the repo root:
    python -m evaluation.run_qg_specwind_da --spec qg_specwind_gyrostat_v1 --n-windows 100 \
        --method etkf --out experiments/qg_specwind_da/etkf_forced_c3.json
"""
from __future__ import annotations

import argparse
import json
import os
import time

import torch

from data.qg import QGConfig
from data.qg_datasets import SPECS, QGDatasetSpec, shard_indices
from data.qg_specwind_neural import check_compatible, load_full_res_windows, with_fixed_obs
from evaluation.run_qg_baselines import run
from models.qg_dynamics import QGDynamics

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
    for w in windows:
        tp = w["true_params"]
        inv = QGDynamics(nx=cfg.nx, L=cfg.L, dt=cfg.dt, beta=tp["beta"], rd=tp["rd"],
                         delta=cfg.delta, U1=tp["U1"], U2=tp["U2"], rek=tp["rek"])
        psi1 = inv.streamfunctions(w["true_state"])[:, 0].reshape(w["true_state"].shape[0], -1)
        out.append({
            **w,
            "da_model": "qg2l", "da_nx": cfg.nx, "da_params": dict(tp),
            "wind_state_corrupted": w["wind_state_true"],
            "target_state_psi": psi1, "target_state_q": w["true_state"][:, :per_layer].clone(),
        })
    return out, report


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--spec", required=True, choices=sorted(SPECS))
    p.add_argument("--root", default="experiments/qg_datasets")
    p.add_argument("--split", default="test", choices=("test", "val"))
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--n-windows", type=int, default=100)
    p.add_argument("--shard", type=int, default=0)
    p.add_argument("--n-shards", type=int, default=1)
    p.add_argument("--method", default="etkf", choices=("etkf", "enkf"))
    p.add_argument("--cols-per-day", type=int, default=3)
    p.add_argument("--obs-noise-frac", type=float, default=0.05)
    p.add_argument("--init-lag-days", type=float, default=5.0)
    p.add_argument("--N", type=int, default=80)
    p.add_argument("--inflation", type=float, default=1.0)
    p.add_argument("--loc-radius", type=float, default=2.0)
    p.add_argument("--etkf-ridge", type=float, default=0.1)
    p.add_argument("--out", required=True, help="summary JSON path (run() output)")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()
    spec = SPECS[args.spec]
    device = torch.device(args.device)
    all_idx = list(range(args.start, args.start + args.n_windows))
    idx = [all_idx[k] for k in shard_indices(len(all_idx), args.shard, args.n_shards)]
    cfg = build_cfg(spec, args.cols_per_day, args.obs_noise_frac, args.init_lag_days)
    t0 = time.time()
    windows, report = s0_windows(spec, args.split, args.root, idx, cfg, device)
    load_s = time.time() - t0
    t0 = time.time()
    summary = run(args.method, cfg, device=device, N_ensemble=args.N, inflation=args.inflation,
                  loc_radius=args.loc_radius, scenarios=("test_s0",), out_path=args.out,
                  init="lagged", geometry="random_columns", obs_var="psi",
                  init_lag_days=args.init_lag_days, ds={"test_s0": windows},
                  etkf_ridge=args.etkf_ridge)
    meta = {"spec": spec.name, "split": args.split, "indices": idx, "method": args.method,
            "cols_per_day": args.cols_per_day, "obs_noise_frac": args.obs_noise_frac,
            "init_lag_days": args.init_lag_days, "N": args.N, "inflation": args.inflation,
            "loc_radius": args.loc_radius, "etkf_ridge": args.etkf_ridge,
            "load": report, "load_seconds": round(load_s, 1),
            "da_seconds": round(time.time() - t0, 1), "obs_seed": OBS_SEED}
    with open(os.path.splitext(args.out)[0] + "_meta.json", "w") as fh:
        json.dump(meta, fh, indent=1)
    print(json.dumps({"meta": meta, "summary": summary.get("scenarios", {}).get("test_s0")},
                     default=str)[:2000])


if __name__ == "__main__":
    main()
