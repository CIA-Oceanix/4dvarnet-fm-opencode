"""DA-2: S0 ETKF/EnKF tuning on val for the spectral-wind QG datasets.

One task = one configuration over the first `--n-windows` val windows of the
forced dataset. Axes (`docs/plans/analysis/qg_specwind_da_s0.md` §4.1):
localization radius, ETKF ridge, the cross-layer localization weight
(vertical localization) and the initial ensemble (white noise, or bred along
the lagged truth). Metrics are computed per window from the saved
trajectories with each window's own parameters (`run()`'s per-field psi
metrics invert every window with window 0's parameters).

    python -m evaluation.qg_specwind_da2_sweep --list
    python -m evaluation.qg_specwind_da2_sweep --task 7 --out-dir experiments/qg_specwind_da2
    python -m evaluation.qg_specwind_da2_sweep --summarize --out-dir experiments/qg_specwind_da2
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import shutil
import tempfile
import time

import numpy as np
import torch

from data.qg_datasets import SPECS
from evaluation.run_qg_baselines import run
from evaluation.run_qg_specwind_da import build_cfg, s0_windows
from models.qg_dynamics import QGDynamics

SPEC = "qg_specwind_gyrostat_v1"
BASELINE = {"method": "etkf", "loc_radius": 2.0, "etkf_ridge": 0.1, "loc_cross_layer": 0.0,
            "init_ensemble": "white"}
FIELDS = ("psi1", "psi2", "q1", "q2")


def configs() -> list[dict]:
    out = []
    for loc, ridge, cross, init in itertools.product((1.0, 2.0, 3.0), (0.0, 0.1, 1.0),
                                                     (0.0, 0.5, 1.0), ("white", "bred")):
        out.append({"method": "etkf", "loc_radius": loc, "etkf_ridge": ridge,
                    "loc_cross_layer": cross, "init_ensemble": init})
    for loc, cross, init in itertools.product((1.0, 2.0, 3.0), (0.0, 0.5, 1.0), ("white", "bred")):
        out.append({"method": "enkf", "loc_radius": loc, "etkf_ridge": 0.1,
                    "loc_cross_layer": cross, "init_ensemble": init})
    return out


def config_name(c: dict) -> str:
    return (f"{c['method']}_loc{c['loc_radius']:g}_r{c['etkf_ridge']:g}"
            f"_x{c['loc_cross_layer']:g}_{c['init_ensemble']}")


def _layers(q: np.ndarray, tp: dict, cfg, device) -> dict:
    inv = QGDynamics(nx=cfg.nx, L=cfg.L, dt=cfg.dt, beta=tp["beta"], rd=tp["rd"], delta=cfg.delta,
                     U1=tp["U1"], U2=tp["U2"], rek=tp["rek"]).to(device)
    psi = inv.streamfunctions(torch.from_numpy(q).to(device)).cpu().numpy()
    psi = psi.reshape(q.shape[0], 2, -1)
    per = q.shape[1] // 2
    return {"psi1": psi[:, 0], "psi2": psi[:, 1], "q1": q[:, :per], "q2": q[:, per:]}


def window_metrics(analysis: np.ndarray, free: np.ndarray, truth: np.ndarray, tp: dict, cfg,
                   device) -> dict:
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


def run_task(task: int, out_dir: str, root: str, n_windows: int, device: torch.device,
             N: int, cols_per_day: int, breed_days: float) -> str:
    c = configs()[task]
    spec = SPECS[SPEC]
    cfg = build_cfg(spec, cols_per_day, 0.05, 5.0)
    idx = list(range(n_windows))
    t0 = time.time()
    windows, report = s0_windows(spec, "val", root, idx, cfg, device)
    load_s = time.time() - t0
    tmp = tempfile.mkdtemp(prefix="qgda2_", dir=os.environ.get("TMPDIR", "/tmp"))
    try:
        t0 = time.time()
        payload = run(c["method"], cfg, device=device, N_ensemble=N, inflation=1.0,
                      loc_radius=c["loc_radius"], scenarios=("test_s0",), out_path=None,
                      init="lagged", geometry="random_columns", obs_var="psi", init_lag_days=5.0,
                      ds={"test_s0": windows}, etkf_ridge=c["etkf_ridge"], save_traj=tmp,
                      loc_cross_layer=c["loc_cross_layer"], init_ensemble_kind=c["init_ensemble"],
                      breed_days=breed_days)
        da_s = time.time() - t0
        s = payload["scenarios"]["test_s0"]
        traj = np.load(s["traj_path"])
        per_window = []
        for k, w in enumerate(windows):
            m = window_metrics(traj["analyses"][k], traj["free_forecast"][k], traj["refs"][k],
                               w["true_params"], cfg, device)
            m.update({"index": idx[k], "crps": s["crps_list"][k],
                      "level": w["specwind"]["factors"]["level"]})
            per_window.append(m)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    rec = {"config": c, "name": config_name(c), "N": N, "cols_per_day": cols_per_day,
           "breed_days": breed_days, "indices": idx, "load": report,
           "load_seconds": round(load_s, 1), "da_seconds": round(da_s, 1),
           "spread_t0_mean": s["spread_t0_mean"], "per_window": per_window}
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{config_name(c)}.json")
    with open(path, "w") as fh:
        json.dump(rec, fh, indent=1)
    return path


def _boot_ci(diff: np.ndarray, n_boot: int = 10000, seed: int = 0) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    means = diff[rng.integers(0, len(diff), (n_boot, len(diff)))].mean(1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def summarize(out_dir: str) -> dict:
    recs = {}
    for f in sorted(os.listdir(out_dir)):
        if f.endswith(".json") and f != "summary.json":
            with open(os.path.join(out_dir, f)) as fh:
                r = json.load(fh)
            recs[r["name"]] = r
    base_name = config_name(BASELINE)
    base = recs.get(base_name)
    rows = []
    for name, r in recs.items():
        pw = r["per_window"]
        row = {"name": name, **r["config"], "n": len(pw), "da_seconds": r["da_seconds"]}
        for key in ["score"] + [f"ev_{t}_{k}" for t in ("da", "free") for k in FIELDS]:
            row[key] = float(np.mean([w[key] for w in pw]))
        row["psi2_worse_than_free"] = int(sum(w["ev_da_psi2"] < w["ev_free_psi2"] for w in pw))
        if base is not None and [w["index"] for w in base["per_window"]] == [w["index"] for w in pw]:
            diff = np.array([w["score"] for w in pw]) - np.array([w["score"] for w in base["per_window"]])
            row["d_score"] = float(diff.mean())
            row["d_score_ci"] = _boot_ci(diff)
        rows.append(row)
    rows.sort(key=lambda x: -x["score"])
    out = {"baseline": base_name, "rows": rows}
    with open(os.path.join(out_dir, "summary.json"), "w") as fh:
        json.dump(out, fh, indent=1)
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--list", action="store_true")
    p.add_argument("--task", type=int)
    p.add_argument("--summarize", action="store_true")
    p.add_argument("--out-dir", default="experiments/qg_specwind_da2")
    p.add_argument("--root", default="experiments/qg_datasets")
    p.add_argument("--n-windows", type=int, default=20)
    p.add_argument("--N", type=int, default=80)
    p.add_argument("--cols-per-day", type=int, default=3)
    p.add_argument("--breed-days", type=float, default=3.0)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()
    if args.list:
        for k, c in enumerate(configs()):
            print(k, config_name(c))
    elif args.task is not None:
        print(run_task(args.task, args.out_dir, args.root, args.n_windows, torch.device(args.device),
                       args.N, args.cols_per_day, args.breed_days))
    elif args.summarize:
        s = summarize(args.out_dir)
        for r in s["rows"]:
            ci = r.get("d_score_ci")
            print(f"{r['name']:34s} score {r['score']:.3f}  psi1 {r['ev_da_psi1']:.3f}  "
                  f"psi2 {r['ev_da_psi2']:.3f} (free {r['ev_free_psi2']:.3f}, worse in "
                  f"{r['psi2_worse_than_free']}/{r['n']})  q1 {r['ev_da_q1']:.3f}  q2 {r['ev_da_q2']:.3f}"
                  + (f"  d {r['d_score']:+.3f} [{ci[0]:+.3f}, {ci[1]:+.3f}]" if ci else ""))


if __name__ == "__main__":
    main()
