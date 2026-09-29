"""Figures for the spectral-wind QG DA report (`generate_qg_specwind_da_report.py`).

In the style of `generate_qg_reconstruction_figs.py` (the legacy QG DA report):

* reconstruction panels for 3 test windows of the forced dataset (best / median
  / worst by the ETKF's S0 per-window score) in S0 and in the realistic S1
  (base): rows truth | free forecast | ETKF | EnKF | EnKS, columns psi1 | psi2 | q1 | q2
  at day 27;
* a DA-cycle animation (ETKF) per window and scenario: observed column, true
  and DA wind-stress curl, truth / analysis for q1, psi1, psi2, q2;
* summary figures: observation-density curve (S0), S0 vs S1 per field, and the
  S1 error budget (Shapley attribution on val).

The DA is re-run for the 3 windows only (same seeds and settings as the report
runs), so it needs a GPU; the summary figures read the report's run records.

    python reports/qg/generate_qg_specwind_da_figs.py --device cuda
"""
from __future__ import annotations

import argparse
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data.qg_datasets import SPECS  # noqa: E402
from evaluation.qg_specwind_s1 import REALISTIC_GROUPS, REALISTIC_VARIANTS  # noqa: E402
from evaluation.qg_specwind_s1_sweep import _load, realistic_name, shapley_over  # noqa: E402
from evaluation.run_qg_specwind_da import (  # noqa: E402
    FIELDS,
    apply_s1,
    build_cfg,
    da_cfg,
    evaluate,
    s0_windows,
)
from generate_qg_specwind_da_report import _find, _values, load_runs  # noqa: E402
from models.qg_dynamics import QGDynamics  # noqa: E402
from models.qg_wind_modes import FourierWindBasis  # noqa: E402

SPEC = "qg_specwind_gyrostat_v1"
S1_VARIANT = "base"
METHODS = (("ETKF", "etkf", {"loc_radius": 8.0, "etkf_ridge": 0.1, "etkf_loc_mode": "ensrf"}),
           ("EnKF", "enkf", {"loc_radius": 6.0, "etkf_ridge": 0.1}),
           ("EnKS (smoother)", "enks", {"loc_radius": 8.0, "etkf_ridge": 0.1, "etkf_loc_mode": "ensrf",
                                        "enks_lag": 12}))
LABEL = {"psi1": "ψ₁", "psi2": "ψ₂", "q1": "q₁", "q2": "q₂"}
CMAP = "RdBu_r"
INK, MUTED, GRID = "#1f2a30", "#5b6b72", "#e3e8ea"
SERIES = {"etkf": "#2a78d6", "enkf": "#1baf7a", "free": "#8a959b", "s0": "#2a78d6",
          "s1": "#eb6834"}
GROUP_COLORS = {"forcing": "#2a78d6", "rd": "#eb6834", "drag": "#9b59b6", "obs": "#1baf7a",
                "res": "#d4a017"}


def select_windows(runs: list[dict]) -> dict[str, int]:
    r = _find(runs, SPEC, "etkf", 3)
    if r is None:
        raise SystemExit("needs the complete forced ETKF S0 run (3 columns per day)")
    order = sorted(r["per_window"], key=lambda w: w["score"], reverse=True)
    pick = {"best": order[0], "median": order[len(order) // 2], "worst": order[-1]}
    for k, w in pick.items():
        print(f"{k}: window {w['index']} (S0 ETKF score {w['score']:.3f})")
    return {k: int(w["index"]) for k, w in pick.items()}


def rerun(indices: list[int], device: torch.device, ds_root: str) -> dict:
    spec = SPECS[SPEC]
    cfg = build_cfg(spec, 3, 0.05, 5.0)
    out = {}
    for scen in ("s0", "s1"):
        windows, _ = s0_windows(spec, "test", ds_root, indices, cfg, device)
        levels = REALISTIC_VARIANTS[S1_VARIANT] if scen == "s1" else None
        if levels is not None:
            windows = apply_s1(windows, spec, levels)
        for _, method, kw in METHODS:
            tmp = os.path.join(os.environ.get("TMPDIR", "/tmp"), f"qgfig_{scen}_{method}_{os.getpid()}")
            payload, per_window = evaluate(windows, da_cfg(cfg, levels), method, device,
                                           save_traj=tmp, **kw)
            traj = np.load(payload["scenarios"]["test_s0"]["traj_path"])
            out[(scen, method)] = {"analyses": traj["analyses"], "free": traj["free_forecast"],
                                   "refs": traj["refs"], "per_window": per_window,
                                   "windows": windows}
    return out


def _psi(q: np.ndarray, tp: dict, cfg, device) -> np.ndarray:
    inv = QGDynamics(nx=cfg.nx, L=cfg.L, dt=cfg.dt, beta=tp["beta"], rd=tp["rd"], delta=cfg.delta,
                     U1=tp["U1"], U2=tp["U2"], rek=tp["rek"]).to(device)
    return inv.streamfunctions(torch.from_numpy(q).to(device)).cpu().numpy().reshape(
        q.shape[0], 2, cfg.ny, cfg.nx)


def _layers(q: np.ndarray, tp: dict, cfg, device) -> dict:
    psi = _psi(q, tp, cfg, device)
    per = cfg.ny * cfg.nx
    return {"psi1": psi[:, 0], "psi2": psi[:, 1],
            "q1": q[:, :per].reshape(-1, cfg.ny, cfg.nx), "q2": q[:, per:].reshape(-1, cfg.ny, cfg.nx)}


def _style(ax) -> None:
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_color(GRID)


def recon_figure(scen: str, label: str, k: int, runs: dict, cfg, device, out_dir: str) -> str:
    base = runs[(scen, "etkf")]
    w = base["windows"][k]
    tp = w["true_params"]
    t = int(base["refs"].shape[1] * 0.9)
    rows = [("truth", base["refs"][k], None), ("free forecast", base["free"][k], None)]
    rows += [(name, runs[(scen, m)]["analyses"][k], runs[(scen, m)]["per_window"][k])
             for name, m, _ in METHODS]
    fields = [(name, _layers(traj[t:t + 1], tp, cfg, device), pw) for name, traj, pw in rows]
    truth = fields[0][1]
    vmax = {f: float(np.percentile(np.abs(truth[f]), 99.5)) or 1.0 for f in FIELDS}
    fig, axes = plt.subplots(len(fields), 4, figsize=(12, 3.0 * len(fields)))
    for i, (name, lay, pw) in enumerate(fields):
        for j, f in enumerate(FIELDS):
            ax = axes[i, j]
            ax.imshow(lay[f][0], cmap=CMAP, vmin=-vmax[f], vmax=vmax[f], origin="lower")
            _style(ax)
            if i == 0:
                ax.set_title(LABEL[f], fontsize=12, color=INK)
            if pw is not None:
                ax.text(0.03, 0.04, f"EV {pw[f'ev_da_{f}']:.2f}", transform=ax.transAxes,
                        fontsize=8, color=INK, bbox={"fc": "white", "ec": "none", "alpha": 0.75})
            elif name == "free forecast":
                ax.text(0.03, 0.04, f"EV {base['per_window'][k][f'ev_free_{f}']:.2f}",
                        transform=ax.transAxes, fontsize=8, color=INK,
                        bbox={"fc": "white", "ec": "none", "alpha": 0.75})
            if j == 0:
                ax.set_ylabel(name, fontsize=11, color=INK)
    title = "S0" if scen == "s0" else f"S1 realistic ({S1_VARIANT})"
    fig.suptitle(f"{title} — {label} window (test {w['init_seed_key']}), day {t / 12:.0f}; "
                 "EV = whole-window explained variance", fontsize=11, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    path = os.path.join(out_dir, f"qg_specwind_da_recon_{scen}_{label}.png")
    fig.savefig(path, dpi=90)
    plt.close(fig)
    return path


def cycle_animation(scen: str, label: str, k: int, runs: dict, cfg, device, out_dir: str,
                    scale: float = 0.55) -> str:
    r = runs[(scen, "etkf")]
    w = r["windows"][k]
    tp = w["true_params"]
    spec = SPECS[SPEC]
    basis = FourierWindBasis(nx=spec.nx, L=spec.L, kmax=spec.kmax)
    true_curl = basis.curl_field(w["wind_state_true"].double()).numpy()
    da_curl = basis.curl_field(w["wind_state_corrupted"].double()).numpy()
    truth = _layers(r["refs"][k], tp, cfg, device)
    ana = _layers(r["analyses"][k], tp, cfg, device)
    obs, mask = w["obs"].numpy(), w["obs_mask"].numpy()
    cols = w["obs_columns"].numpy()
    spd = 12
    days = r["refs"].shape[1] // spd
    vmax = {f: float(np.percentile(np.abs(truth[f]), 99.5)) or 1.0 for f in FIELDS}
    vw = float(np.percentile(np.abs(true_curl), 99.5)) or float(np.abs(da_curl).max()) or 1.0
    frames = []
    for d in range(days):
        t = (d + 1) * spd - 1
        omap = np.full((cfg.ny, cfg.nx), np.nan)
        for s in range(d * spd, (d + 1) * spd):
            if mask[s]:
                omap[:, int(cols[s])] = obs[s]
        fig, axes = plt.subplots(2, 5, figsize=(17, 7.2))
        panels = [
            (omap, vmax["psi1"], "observed ψ₁ columns (that day)"),
            (true_curl[t], vw, "true wind-stress curl"),
            (da_curl[t], vw, "DA-model wind-stress curl"),
            (truth["q1"][t], vmax["q1"], "truth q₁"),
            (ana["q1"][t], vmax["q1"], "ETKF analysis q₁"),
            (truth["psi1"][t], vmax["psi1"], "truth ψ₁"),
            (ana["psi1"][t], vmax["psi1"], "ETKF analysis ψ₁"),
            (truth["psi2"][t], vmax["psi2"], "truth ψ₂ (never observed)"),
            (ana["psi2"][t], vmax["psi2"], "ETKF analysis ψ₂"),
            (ana["q2"][t] - truth["q2"][t], vmax["q2"], "analysis − truth, q₂"),
        ]
        for ax, (fld, vm, title) in zip(axes.ravel(), panels):
            cm = matplotlib.colormaps[CMAP].copy()
            cm.set_bad("#d9dde0")
            ax.imshow(np.ma.masked_invalid(fld), cmap=cm, vmin=-vm, vmax=vm, origin="lower")
            ax.set_title(title, fontsize=10, color=INK)
            _style(ax)
        name = "S0" if scen == "s0" else f"S1 realistic ({S1_VARIANT})"
        fig.suptitle(f"{name} DA cycle — {label} window (test {w['init_seed_key']}) — day {d + 1} of {days}",
                     fontsize=12, color=INK)
        fig.tight_layout(rect=(0, 0, 1, 0.95))
        fig.canvas.draw()
        img = Image.fromarray(np.asarray(fig.canvas.buffer_rgba())).convert("RGB")
        frames.append(img.resize((int(img.width * scale), int(img.height * scale)), Image.LANCZOS))
        plt.close(fig)
    path = os.path.join(out_dir, f"qg_specwind_da_cycle_{scen}_{label}.gif")
    frames[0].save(path, save_all=True, append_images=frames[1:], duration=350, loop=0, optimize=True)
    return path


def _axes_style(ax) -> None:
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTED)
    ax.tick_params(colors=MUTED)


def density_figure(runs: list[dict], out_dir: str) -> str:
    cols = sorted({r["cols"] for r in runs if r["spec"] == SPEC})
    pts = [(c, _find(runs, SPEC, "etkf", c)) for c in cols]
    pts = [(c, r) for c, r in pts if r is not None]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), sharex=True)
    for ax, fs in zip(axes, (("psi1", "psi2"), ("q1", "q2"))):
        for f, color, dy in zip(fs, ("#2a78d6", "#1baf7a"), (6, -8)):
            ys = [_values(r, "da", f).mean() for _, r in pts]
            ax.plot([c for c, _ in pts], ys, marker="o", ms=6, lw=2, color=color,
                    label=f"{LABEL[f]} ({'upper' if f.endswith('1') else 'lower'} layer)")
            ax.annotate(f"{ys[-1]:.3f}", (pts[-1][0], ys[-1]), textcoords="offset points",
                        xytext=(6, dy), va="center", fontsize=9, color=INK)
        _axes_style(ax)
        ax.set_xlabel("observed columns per day", color=INK)
        ax.set_ylabel("explained variance", color=INK)
        ax.legend(frameon=False, fontsize=9)
    fig.suptitle("S0 observation density, ETKF (forced test, 100 windows)", fontsize=11, color=INK)
    fig.tight_layout()
    path = os.path.join(out_dir, "qg_specwind_da_density.png")
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return path


def s0_s1_figure(runs: list[dict], out_dir: str) -> str:
    fig, axes = plt.subplots(1, len(METHODS), figsize=(5.5 * len(METHODS), 3.9), sharey=True)
    x = np.arange(len(FIELDS))
    for ax, (name, method, _) in zip(axes, METHODS):
        s0 = _find(runs, SPEC, method, 3)
        s1 = _find(runs, SPEC, method, 3, f"realistic {S1_VARIANT}")
        for off, r, key, lab in ((-0.2, s0, "s0", "S0"), (0.2, s1, "s1", f"S1 realistic {S1_VARIANT}")):
            ys = [_values(r, "da", f).mean() for f in FIELDS]
            ax.bar(x + off, ys, width=0.38, color=SERIES[key], label=lab)
            for xi, y in zip(x + off, ys):
                ax.text(xi, y + 0.01, f"{y:.2f}", ha="center", va="bottom", fontsize=8, color=INK)
        _axes_style(ax)
        ax.set_xticks(x, [LABEL[f] for f in FIELDS])
        ax.set_title(f"{name}", fontsize=11, color=INK)
        ax.axhline(0, color=MUTED, lw=0.8)
    axes[0].set_ylabel("explained variance (analysis)", color=INK)
    axes[0].legend(frameon=False, fontsize=9, loc="lower left")
    fig.suptitle("S0 vs realistic S1 (forced test, 3 columns per day, 100 windows)", fontsize=11, color=INK)
    fig.tight_layout()
    path = os.path.join(out_dir, "qg_specwind_da_s0_s1.png")
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return path


def budget_figure(s1_root: str, out_dir: str) -> tuple[str, dict]:
    sh = shapley_over(_load(s1_root), REALISTIC_GROUPS, lambda c: realistic_name(S1_VARIANT, c))
    if sh is None:
        raise SystemExit(f"incomplete realistic-{S1_VARIANT} Shapley runs under {s1_root}")
    metrics = ("psi1", "psi2", "q1", "q2", "score")
    fig, ax = plt.subplots(figsize=(10, 3.9))
    y = np.arange(len(metrics))[::-1]
    for yi, m in zip(y, metrics):
        left = 0.0
        for g in REALISTIC_GROUPS:
            v = sh[m]["components"][g]["shapley"]
            ax.barh(yi, v, left=left, color=GROUP_COLORS[g], edgecolor="white", lw=1.5,
                    label=g if m == metrics[0] else None)
            if v > 0.02:
                ax.text(left + v / 2, yi, f"{100 * v / sh[m]['loss']:.0f}%", ha="center",
                        va="center", fontsize=8, color="white")
            left += max(v, 0.0)
        ax.text(left + 0.005, yi, f"loss {sh[m]['loss']:.3f}", va="center", fontsize=8, color=INK)
    ax.set_yticks(y, [LABEL.get(m, "score") for m in metrics])
    _axes_style(ax)
    ax.grid(axis="x", color=GRID, lw=0.8)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("EV loss S0 → S1, Shapley share per error group (val, 20 windows)", color=INK)
    ax.legend(frameon=False, fontsize=9, ncol=5, loc="upper center", bbox_to_anchor=(0.5, 1.18))
    fig.tight_layout()
    path = os.path.join(out_dir, "qg_specwind_da_s1_budget.png")
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return path, sh


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--root", default="experiments/qg_specwind_da")
    p.add_argument("--s1-root", default="experiments/qg_specwind_s1_ensrf")
    p.add_argument("--datasets", default="experiments/qg_datasets")
    p.add_argument("--out-dir", default="reports/qg/outputs/figs")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--summary-only", action="store_true", help="skip the DA re-runs and animations")
    args = p.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    runs = load_runs(args.root)
    print(density_figure(runs, args.out_dir))
    print(s0_s1_figure(runs, args.out_dir))
    print(budget_figure(args.s1_root, args.out_dir)[0])
    if args.summary_only:
        return
    device = torch.device(args.device)
    picks = select_windows(runs)
    cfg = build_cfg(SPECS[SPEC], 3, 0.05, 5.0)
    reruns = rerun(list(picks.values()), device, args.datasets)
    for scen in ("s0", "s1"):
        for k, label in enumerate(picks):
            print(recon_figure(scen, label, k, reruns, cfg, device, args.out_dir))
            print(cycle_animation(scen, label, k, reruns, cfg, device, args.out_dir))


if __name__ == "__main__":
    main()
