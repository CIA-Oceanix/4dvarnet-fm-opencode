"""Report for the L96 ETKF/EnKF obs-count sweep (eval_da_obs_count_l96.py).

Reads every ``experiments/l96_da_obs_count/summary_obscount_*.json`` and writes
``reports/l96/outputs/l96_da_obs_count.md`` + ``l96_da_obs_count.png``: RMSE /
MAE / spread-over-RMSE per n_obs, S0 and S1 with the S1/S0 ratio, and the
``reg30`` (cached regular 30-time grid, same windows) reference row.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "experiments" / "l96_da_obs_count"
OUT = ROOT / "reports" / "l96" / "outputs"
METHODS = ("ETKF", "EnKF")
COLORS = {"ETKF": "#2a78d6", "EnKF": "#eb6834"}
CASES = ("s0", "s1")
GROUP = "all_obs"


def load(da_fast_weights: bool) -> tuple[list[dict], dict | None]:
    sweep, ref = [], None
    for p in sorted(SRC.glob("summary_obscount_*.json")):
        s = json.loads(p.read_text())
        if s.get("da_fast_weights", False) != da_fast_weights:
            continue
        if s["n_obs"] == "reg30":
            ref = s
        else:
            sweep.append(s)
    sweep.sort(key=lambda s: int(s["n_obs"]))
    return sweep, ref


def _cell(s: dict, case: str, method: str, metric: str) -> dict:
    return s["cases"][case][method][metric][GROUP]


def table(sweep: list[dict], ref: dict | None) -> list[str]:
    lines = []
    for method in METHODS:
        lines += [f"\n### {method}\n",
                  "| n_obs | S0 RMSE | S1 RMSE | S1/S0 | S0 MAE | S1 MAE | sp/RMSE (S0) | sp/RMSE (S1) |",
                  "|---|---|---|---|---|---|---|---|"]
        rows = [(s["n_obs"], s) for s in sweep] + ([("reg30 (regular grid)", ref)] if ref else [])
        for label, s in rows:
            r0, r1 = _cell(s, "s0", method, "rmse"), _cell(s, "s1", method, "rmse")
            m0, m1 = _cell(s, "s0", method, "mae"), _cell(s, "s1", method, "mae")
            sp0, sp1 = _cell(s, "s0", method, "spread"), _cell(s, "s1", method, "spread")
            lines.append(
                f"| {label} | {r0['mean']:.4f} ± {r0['std']:.4f} | {r1['mean']:.4f} ± {r1['std']:.4f} | "
                f"{r1['mean'] / r0['mean']:.3f} | {m0['mean']:.4f} | {m1['mean']:.4f} | "
                f"{sp0['mean'] / r0['mean']:.3f} | {sp1['mean'] / r1['mean']:.3f} |")
    return lines


def figure(sweep: list[dict], ref: dict | None, path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    x = [int(s["n_obs"]) for s in sweep]
    for ax, case in zip(axes, CASES):
        for method in METHODS:
            mean = [_cell(s, case, method, "rmse")["mean"] for s in sweep]
            std = [_cell(s, case, method, "rmse")["std"] for s in sweep]
            c = COLORS[method]
            ax.fill_between(x, [m - d for m, d in zip(mean, std)], [m + d for m, d in zip(mean, std)],
                            color=c, alpha=0.12, linewidth=0)
            ax.plot(x, mean, color=c, linewidth=2, marker="o", markersize=5, label=f"{method} (random times)")
            ax.annotate(method, (x[-1], mean[-1]), xytext=(6, 0), textcoords="offset points",
                        color="#52514e", fontsize=9, va="center")
            if ref:
                ax.plot([30], [_cell(ref, case, method, "rmse")["mean"]], marker="D", markersize=8,
                        markerfacecolor="white", markeredgecolor=c, markeredgewidth=2, linestyle="none",
                        label=f"{method} regular grid (30)")
        ax.set_xscale("log")
        ax.set_xticks(x)
        ax.xaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter())
        ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
        ax.set_xlabel("obs times per window (log)")
        ax.set_title(case.upper(), color="#0b0b0b")
        ax.grid(True, color="#e6e5e0", linewidth=0.8)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
    axes[0].set_ylabel("RMSE (24D observed, mean ± std over runs)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, fontsize=8, loc="lower center", ncol=len(labels))
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> None:
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--da-fast-weights", action="store_true")
    args = p.parse_args()
    sweep, ref = load(args.da_fast_weights)
    if not sweep:
        sys.exit(f"no sweep summaries in {SRC}")
    OUT.mkdir(parents=True, exist_ok=True)
    stem = "l96_da_obs_count" + ("_dafw" if args.da_fast_weights else "")
    png = OUT / f"{stem}.png"
    figure(sweep, ref, png)
    s = sweep[0]
    lines = [
        "# L96 ETKF/EnKF vs number of observation times per window\n",
        f"{len(s['windows'])} windows of the P1 200-window test set (indices {s['windows']}), "
        f"{s['n_draws']} independent obs-time/noise draws per window per n_obs; obs times = step 0 "
        "plus n_obs-1 distinct steps drawn uniformly over the 3000-step window, all 24 observed "
        f"channels at each time, R_var=0.5. P1 DA settings: {s['N_ensemble']} members, inflation "
        f"{s['inflation']}. Metrics follow the P1 benchmark (group `{GROUP}`, physical units, "
        "mean ± std across runs). `reg30` = the cached regular 30-time obs of the same windows "
        "(one run per window), the direct link to the P1 table.\n",
        ("DA forward model uses each window's fast_weights (S0 true, S1 biased `_da`)."
         if args.da_fast_weights else
         "DA forward model uses UNWEIGHTED fast coupling, exactly as the P1 DA rows (the "
         "per-window fast_weights are not passed), so S0 is not strictly perfect-model.") + "\n",
        f"![RMSE vs n_obs]({png.name})",
    ]
    lines += table(sweep, ref)
    (OUT / f"{stem}.md").write_text("\n".join(lines) + "\n")
    print(f"wrote {OUT / f'{stem}.md'}")


if __name__ == "__main__":
    main()
