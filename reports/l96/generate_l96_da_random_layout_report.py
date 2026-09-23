"""Report for the L96 DA baselines under the random observing system
(eval_da_random_layout_l96.py, batch/run_l96_da_random_layout.sbatch).

Reads ``experiments/l96_da_random_layout/{summary,per_window}_rlayout_*`` and
writes ``reports/l96/outputs/l96_da_random_layout.{md,png}``:
  arms A / A' -- n_obs U{6..50} / U{10..100}, k U{4..16}, 200 windows: RMSE and
            spread per method next to the P1 regular-grid rows, then RMSE binned by n_obs and k;
  arm B  -- n_obs x k factorial grid (RMSE tables + figure);
  inflation check -- ETKF/EnKF at n_obs 10 / 100 for inflation 1.5 / 2.0 / 2.5.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "experiments" / "l96_da_random_layout"
OUT = ROOT / "reports" / "l96" / "outputs"
METHODS = ("ETKF", "EnKF", "Strong-4DVar")
CASES = ("s0", "s1")
GROUP = "all_obs"
K_COLORS = {4: "#86b6ef", 8: "#3987e5", 12: "#1c5cab", 16: "#0d366b"}
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
P1_REGULAR = {
    "s0": {"ETKF": 0.8213, "EnKF": 0.8339, "Strong-4DVar": 0.7028},
    "s1": {"ETKF": 1.4786, "EnKF": 1.5140, "Strong-4DVar": 1.4362},
}
N_BINS = ((6, 10), (11, 20), (21, 35), (36, 50), (51, 75), (76, 100))
K_BINS = ((4, 7), (8, 11), (12, 16))


def load() -> list[dict]:
    runs = []
    for p in sorted(SRC.glob("summary_rlayout_*.json")):
        s = json.loads(p.read_text())
        s["per_window"] = np.load(SRC / p.name.replace("summary", "per_window").replace(".json", ".npz"))
        runs.append(s)
    return runs


def rmse(s: dict, case: str, method: str) -> dict | None:
    return s["cases"][case].get(method, {}).get("rmse", {}).get(GROUP)


def is_arm_a(s: dict) -> bool:
    return s["n_obs_range"][0] < s["n_obs_range"][1] and s["fast_range"] == [4, 16] and s["inflation"] == 2.0


def is_grid(s: dict) -> bool:
    return (s["n_obs_range"][0] == s["n_obs_range"][1] and s["fast_range"][0] == s["fast_range"][1]
            and s["inflation"] == 2.0)


def is_infl(s: dict) -> bool:
    return s["n_obs_range"][0] == s["n_obs_range"][1] and s["fast_range"] == [4, 16]


def arm_a_section(s: dict) -> list[str]:
    pw = s["per_window"]
    lo, hi = s["n_obs_range"]
    lines = [f"## Random observing system, n_obs U{{{lo}..{hi}}}: {len(s['windows'])} windows x {s['n_draws']} draw",
             "",
             "| Method | S0 RMSE | S0 P1 regular | S0 sp/RMSE | S1 RMSE | S1 P1 regular | S1 sp/RMSE |",
             "|---|---|---|---|---|---|---|"]
    for m in METHODS:
        row = [m]
        for case in CASES:
            r = rmse(s, case, m)
            if r is None:
                row += ["–"] * 3
                continue
            sp = s["cases"][case][m].get("spread", {}).get(GROUP)
            row += [f"{r['mean']:.4f} ± {r['std']:.4f}", f"{P1_REGULAR[case][m]:.4f}",
                    f"{sp['mean'] / r['mean']:.3f}" if sp else "–"]
        lines.append("| " + " | ".join(row) + " |")
    lines += ["", "P1 regular = regular 30-time grid, all 16 observed fast channels, 200 windows "
              "(corrected DA-fast-weights rerun, job 54716).", ""]
    for label, key, bins in (("n_obs", "n_obs", N_BINS), ("k (fast channels observed)", "k", K_BINS)):
        lines += [f"### RMSE by {label}", "",
                  "| Method | Case | " + " | ".join(f"{lo}–{hi}" for lo, hi in bins) + " |",
                  "|---|---|" + "---|" * len(bins)]
        for m in METHODS:
            for case in CASES:
                col = f"{case}_{m.replace('-', '_')}_rmse_{GROUP}"
                if col not in pw.files:
                    continue
                v, x = pw[col], pw[f"{case}_{key}"]
                cells = []
                for lo, hi in bins:
                    sel = (x >= lo) & (x <= hi)
                    cells.append(f"{v[sel].mean():.3f} (n={sel.sum()})" if sel.any() else "–")
                if all(c == "–" for c in cells):
                    continue
                lines.append(f"| {m} | {case.upper()} | " + " | ".join(cells) + " |")
        lines.append("")
    return lines


def grid_section(grid: list[dict]) -> list[str]:
    ns = sorted({s["n_obs_range"][0] for s in grid})
    ks = sorted({s["fast_range"][0] for s in grid})
    cell = {(s["n_obs_range"][0], s["fast_range"][0]): s for s in grid}
    s0 = next(iter(grid))
    lines = [f"## Arm B: n_obs x k grid, {len(s0['windows'])} windows x {s0['n_draws']} draws", "",
             "RMSE (all observed channels), rows n_obs, columns k.", ""]
    for m in METHODS:
        for case in CASES:
            lines += [f"**{m}, {case.upper()}**", "", "| n_obs | " + " | ".join(f"k={k}" for k in ks) + " |",
                      "|---|" + "---|" * len(ks)]
            for n in ns:
                vals = [rmse(cell[(n, k)], case, m) if (n, k) in cell else None for k in ks]
                lines.append(f"| {n} | " + " | ".join(f"{v['mean']:.3f}" if v else "–" for v in vals) + " |")
            lines.append("")
    return lines


def infl_section(infl: list[dict]) -> list[str]:
    ns = sorted({s["n_obs_range"][0] for s in infl})
    infs = sorted({s["inflation"] for s in infl})
    cell = {(s["n_obs_range"][0], s["inflation"]): s for s in infl}
    s0 = next(iter(infl))
    lines = [f"## Inflation check: k U{{4..16}}, {len(s0['windows'])} windows x {s0['n_draws']} draws", "",
             "| Method | Case | n_obs | " + " | ".join(f"inf={i}" for i in infs) + " |",
             "|---|---|---|" + "---|" * len(infs)]
    for m in ("ETKF", "EnKF"):
        for case in CASES:
            for n in ns:
                vals = [rmse(cell[(n, i)], case, m) if (n, i) in cell else None for i in infs]
                best = min((v["mean"] for v in vals if v), default=None)
                txt = [("**" if v and v["mean"] == best else "") + (f"{v['mean']:.3f}" if v else "–")
                       + ("**" if v and v["mean"] == best else "") for v in vals]
                lines.append(f"| {m} | {case.upper()} | {n} | " + " | ".join(txt) + " |")
    return lines + [""]


def grid_figure(grid: list[dict], path: Path) -> None:
    ns = sorted({s["n_obs_range"][0] for s in grid})
    ks = sorted({s["fast_range"][0] for s in grid})
    cell = {(s["n_obs_range"][0], s["fast_range"][0]): s for s in grid}
    fig, axes = plt.subplots(2, 3, figsize=(12, 6.5), sharex=True, sharey="row")
    for i, case in enumerate(CASES):
        for j, m in enumerate(METHODS):
            ax = axes[i, j]
            for k in ks:
                ys = [rmse(cell[(n, k)], case, m) if (n, k) in cell else None for n in ns]
                pts = [(n, y["mean"]) for n, y in zip(ns, ys) if y]
                if not pts:
                    continue
                x, y = zip(*pts)
                ax.plot(x, y, color=K_COLORS.get(k, INK2), lw=2, marker="o", ms=5, label=f"k={k}")
                if j == len(METHODS) - 1:
                    ax.annotate(f"k={k}", (x[-1], y[-1]), xytext=(6, 0), textcoords="offset points",
                                va="center", fontsize=8, color=INK2)
            ax.axhline(P1_REGULAR[case][m], color=INK2, lw=1, ls="--")
            ax.set_title(f"{m} · {case.upper()}", fontsize=10, color=INK)
            ax.grid(color=GRID, lw=0.8)
            ax.set_axisbelow(True)
            for side in ("top", "right"):
                ax.spines[side].set_visible(False)
            ax.tick_params(colors=INK2, labelsize=8)
            if i == 1:
                ax.set_xlabel("obs times per window (n_obs)", fontsize=9, color=INK2)
            if j == 0:
                ax.set_ylabel("RMSE (all observed)", fontsize=9, color=INK2)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    handles.append(plt.Line2D([], [], color=INK2, lw=1, ls="--"))
    labels.append("P1 regular grid (30 obs, k=16)")
    fig.legend(handles, labels, loc="upper center", ncol=len(labels), frameon=False, fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main() -> None:
    runs = load()
    arm_a = [s for s in runs if is_arm_a(s)]
    grid = [s for s in runs if is_grid(s)]
    infl = [s for s in runs if is_infl(s)]
    lines = ["# L96 DA baselines under the random observing system", "",
             "Observing system drawn with the training sampler (`data.obs_random_layout`): n_obs stratified "
             "obs times per window with the first block pinned to step 0 and at least one obs time in every "
             "500-step DA window (draws violating it are redrawn), k of the 16 observed-space fast "
             "channels per window (subset redrawn per obs time), 8 slow channels always observed, R_var=0.5. "
             "Background from the step-0 obs, the 16-k missing fast channels filled by linear interpolation "
             "along the fast ring, the 16 never-observed fast channels as in P1. P1 DA settings (30 members, "
             "Strong-4DVar max_iter=10 lr=0.2, DA window 500) with per-window DA fast weights; metrics on the "
             "24 observed channels, mean ± std across runs.", ""]
    for s in sorted(arm_a, key=lambda s: s["n_obs_range"]):
        lines += arm_a_section(s)
    if grid:
        OUT.mkdir(parents=True, exist_ok=True)
        grid_figure(grid, OUT / "l96_da_random_layout.png")
        lines += ["![RMSE vs n_obs per k](l96_da_random_layout.png)", ""] + grid_section(grid)
    if infl:
        lines += infl_section(infl)
    (OUT / "l96_da_random_layout.md").write_text("\n".join(lines) + "\n")
    print(f"wrote {OUT / 'l96_da_random_layout.md'} ({len(arm_a)} random-layout, {len(grid)} grid, {len(infl)} inflation runs)")


if __name__ == "__main__":
    main()
