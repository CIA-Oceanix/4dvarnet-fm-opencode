"""Animate DA reconstructions on the spectral-wind QG datasets: obs vs truth vs analysis.

Reads the trajectory npz written by `evaluation.run_qg_specwind_da --save-traj`
(analysis mean, free forecast and truth, PV for every step), rebuilds the same
windows and observations, and writes per window:

* a daily-frame GIF. Top row: that day's observed psi_1 columns, true psi_1,
  analysis psi_1, free-forecast psi_1. Bottom row: analysis error in psi_1,
  true psi_2, analysis psi_2, free-forecast psi_2 (psi_2 is never observed);
* daily explained-variance curves (analysis and free forecast, both layers)
  and the run's summary metrics, in `data.json` for the overview page.

Windows must be regenerated on the same GPU type they were generated on; the
loader checks this against the stored frames.
"""
import argparse
import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from data.qg_datasets import SPECS  # noqa: E402
from evaluation.run_qg_specwind_da import build_cfg, s0_windows  # noqa: E402
from models.qg_dynamics import QGDynamics  # noqa: E402


def _psi(q: np.ndarray, tp: dict, cfg, device) -> np.ndarray:
    inv = QGDynamics(nx=cfg.nx, L=cfg.L, dt=cfg.dt, beta=tp["beta"], rd=tp["rd"], delta=cfg.delta,
                     U1=tp["U1"], U2=tp["U2"], rek=tp["rek"]).to(device)
    out = inv.streamfunctions(torch.from_numpy(q).to(device)).cpu().numpy()
    return out.reshape(q.shape[0], 2, cfg.ny, cfg.nx)


def _ev(est: np.ndarray, ref: np.ndarray) -> np.ndarray:
    ref_c = ref - ref.mean(axis=(-2, -1), keepdims=True)
    err = ((est - ref) ** 2).mean(axis=(-2, -1))
    var = (ref_c ** 2).mean(axis=(-2, -1))
    return 1.0 - err / np.maximum(var, 1e-30)


def _animate(w, psi_true, psi_da, psi_free, spd: int, path: str, scale: float = 0.5) -> dict:
    T = psi_true.shape[0]
    days = T // spd
    v1 = float(np.percentile(np.abs(psi_true[:, 0]), 99.5))
    v2 = float(np.percentile(np.abs(psi_true[:, 1]), 99.5))
    ve = float(np.percentile(np.abs(psi_da[:, 0] - psi_true[:, 0]), 99.5)) or v1
    obs, mask, cols = w["obs"].numpy(), w["obs_mask"].numpy(), w["obs_columns"].numpy()
    frames = []
    for d in range(days):
        t = (d + 1) * spd - 1
        obs_map = np.full(psi_true.shape[-2:], np.nan)
        for s in range(d * spd, (d + 1) * spd):
            if mask[s]:
                obs_map[:, int(cols[s])] = obs[s]
        fig, axes = plt.subplots(2, 4, figsize=(15, 7.6))
        panels = [
            (obs_map, "observed ψ₁ (that day's columns)", v1, "RdBu_r"),
            (psi_true[t, 0], "true ψ₁", v1, "RdBu_r"),
            (psi_da[t, 0], "ETKF analysis ψ₁", v1, "RdBu_r"),
            (psi_free[t, 0], "free forecast ψ₁", v1, "RdBu_r"),
            (psi_da[t, 0] - psi_true[t, 0], "analysis − truth, ψ₁", ve, "PuOr_r"),
            (psi_true[t, 1], "true ψ₂ (never observed)", v2, "RdBu_r"),
            (psi_da[t, 1], "ETKF analysis ψ₂", v2, "RdBu_r"),
            (psi_free[t, 1], "free forecast ψ₂", v2, "RdBu_r"),
        ]
        for ax, (fld, ttl, vm, cmap) in zip(axes.ravel(), panels):
            cm = matplotlib.colormaps[cmap].copy()
            cm.set_bad("#d9dde0")
            ax.imshow(np.ma.masked_invalid(fld), cmap=cm, vmin=-vm, vmax=vm, origin="lower")
            ax.set_title(ttl, fontsize=10)
            ax.set_xticks([])
            ax.set_yticks([])
        fig.suptitle(f"day {d + 1} of 30", fontsize=12)
        fig.tight_layout(rect=(0, 0, 1, 0.96))
        fig.canvas.draw()
        img = Image.fromarray(np.asarray(fig.canvas.buffer_rgba())).convert("RGB")
        frames.append(img.resize((int(img.width * scale), int(img.height * scale)), Image.LANCZOS))
        plt.close(fig)
    frames[0].save(path, save_all=True, append_images=frames[1:], duration=350, loop=0, optimize=True)
    return {"file": os.path.basename(path), "width": frames[0].width, "height": frames[0].height,
            "frames": len(frames)}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--traj", required=True)
    p.add_argument("--meta", required=True, help="the *_meta.json written next to the run summary")
    p.add_argument("--summary", required=True, help="the run summary JSON")
    p.add_argument("--root", default="experiments/qg_datasets")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--device", default="cuda")
    args = p.parse_args()
    with open(args.meta) as fh:
        meta = json.load(fh)
    with open(args.summary) as fh:
        summary = json.load(fh)["scenarios"]["test_s0"]
    spec = SPECS[meta["spec"]]
    device = torch.device(args.device)
    cfg = build_cfg(spec, meta["cols_per_day"], meta["obs_noise_frac"], meta["init_lag_days"])
    windows, report = s0_windows(spec, meta["split"], args.root, meta["indices"], cfg, device)
    traj = np.load(args.traj)
    spd = round(86400.0 / cfg.dt)
    os.makedirs(os.path.join(args.out_dir, "gifs"), exist_ok=True)
    data = {"meta": meta, "load": report, "metrics": summary["metrics_per_field"], "windows": []}
    for k, w in enumerate(windows):
        tp = w["true_params"]
        truth = _psi(traj["refs"][k], tp, cfg, device)
        da = _psi(traj["analyses"][k], tp, cfg, device)
        free = _psi(traj["free_forecast"][k], tp, cfg, device)
        gif = _animate(w, truth, da, free, spd,
                       os.path.join(args.out_dir, "gifs", f"{meta['split']}_{meta['indices'][k]:04d}.gif"))
        daily = lambda x: x[: (len(x) // spd) * spd].reshape(-1, spd).mean(1).tolist()  # noqa: E731
        f = w["specwind"]["factors"]
        data["windows"].append({
            "index": meta["indices"][k], "gif": gif, "factors": f, "regime": w["specwind"]["regime"],
            "ev_daily": {"da_psi1": daily(_ev(da[:, 0], truth[:, 0])),
                         "free_psi1": daily(_ev(free[:, 0], truth[:, 0])),
                         "da_psi2": daily(_ev(da[:, 1], truth[:, 1])),
                         "free_psi2": daily(_ev(free[:, 1], truth[:, 1]))},
            "obs_events": int(w["obs_mask"].sum()),
        })
    with open(os.path.join(args.out_dir, "data.json"), "w") as fh:
        json.dump(data, fh)
    print(json.dumps({"windows": len(data["windows"]), "out": args.out_dir}))


if __name__ == "__main__":
    main()
