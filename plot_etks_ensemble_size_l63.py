import argparse
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(BASE, "experiments", "l63_da_table2")
SURFACE, INK, INK_SOFT, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
COLORS = {"s0": "#2a78d6", "s1": "#eb6834"}
LABELS = {"s0": "S0 (perfect model)", "s1": "S1 (imperfect model)"}
PANELS = [("rmse", "RMSE (physical units)"), ("ev", "Explained variance"),
          ("crpss", "CRPS skill score"), ("ss", "Spread–skill ratio")]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--obs-interval", type=int, default=10)
    p.add_argument("--out", default=OUT_DIR)
    a = p.parse_args()
    data = json.load(open(os.path.join(a.out, f"etks_ensemble_size_obs{a.obs_interval}.json")))

    plt.rcParams.update({"font.size": 10, "axes.edgecolor": GRID, "axes.labelcolor": INK_SOFT,
                         "xtick.color": INK_SOFT, "ytick.color": INK_SOFT, "text.color": INK})
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), facecolor=SURFACE)
    sizes = sorted(int(n) for n in data["s0"] if "undefined" not in data["s0"][n])
    for ax, (key, title) in zip(axes.flat, PANELS):
        ax.set_facecolor(SURFACE)
        for case in ("s0", "s1"):
            ys = [data[case][str(n)][key] for n in sizes]
            ax.plot(sizes, ys, color=COLORS[case], linewidth=2, marker="o", markersize=6,
                    markeredgecolor=SURFACE, markeredgewidth=1.5, label=LABELS[case])
            ax.annotate(case.upper(), (sizes[-1], ys[-1]), xytext=(6, 0), textcoords="offset points",
                        color=INK_SOFT, va="center", fontsize=9)
        if key == "ss":
            ax.axhline(1.0, color=INK_SOFT, linewidth=1, linestyle="--")
            ax.annotate("calibrated", (sizes[0], 1.0), xytext=(0, 4), textcoords="offset points",
                        color=INK_SOFT, fontsize=8)
            ax.set_ylim(0.55, 1.05)
        ax.set_xscale("log")
        ax.set_xticks(sizes)
        ax.set_xticklabels([str(n) for n in sizes])
        ax.minorticks_off()
        ax.set_xlim(sizes[0] * 0.85, sizes[-1] * 1.35)
        ax.set_title(title, loc="left", fontsize=11, color=INK)
        ax.set_xlabel("ensemble members N")
        ax.grid(axis="y", color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    axes[0, 0].legend(frameon=False, loc="center right", bbox_to_anchor=(1.0, 0.5))
    fig.suptitle(f"L63 ETKS vs ensemble size (obs interval {a.obs_interval}, inflation re-tuned per N; "
                 "N = 1 undefined: no anomalies)", x=0.01, ha="left", fontsize=11, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    path = os.path.join(a.out, f"etks_ensemble_size_obs{a.obs_interval}.png")
    fig.savefig(path, dpi=160, facecolor=SURFACE)
    print(path)


if __name__ == "__main__":
    main()
