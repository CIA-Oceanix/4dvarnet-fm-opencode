#!/usr/bin/env python3
"""
Generate synthesis PDF report from experiment results.
Usage: python generate_report.py [--output path/to/report.pdf]
"""
import os, sys, json, datetime
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXP_DIR = os.path.join(BASE, "experiments")
OUTPUT_DIR = os.path.join(BASE, "reports", "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

COLORS_BL = {"Weak-4DVar": "#ff7f0e", "Strong-4DVar": "#2ca02c", "EnKF": "#d62728"}
COLORS_FM = "#1f77b4"

def load_baselines(path=None):
    if path is None:
        path = os.path.join(EXP_DIR, "baselines.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)

def load_experiments():
    results = []
    for d in sorted(os.listdir(EXP_DIR)):
        rpath = os.path.join(EXP_DIR, d, "results.json")
        if os.path.exists(rpath):
            with open(rpath) as f:
                results.append(json.load(f))
    results.sort(key=lambda r: r.get("experiment_id", ""))
    return results

def pick_best_fm(experiments):
    if not experiments:
        return None
    return min(experiments, key=lambda e: e["fm_cs2"]["mean"])

# ── Pages ───────────────────────────────────────────────────────

def page_title(pdf, experiments, baselines, best):
    fig, ax = plt.subplots(figsize=(11, 8))
    ax.axis("off")

    bl1 = {}
    bl2 = {}
    if baselines:
        bl1 = {n: baselines["cs1"][n]["mean"]
               for n in ["Weak-4DVar", "Strong-4DVar", "EnKF"] if n in baselines.get("cs1", {})}
        bl2 = {n: baselines["cs2"][n]["mean"]
               for n in ["Weak-4DVar", "Strong-4DVar", "EnKF"] if n in baselines.get("cs2", {})}
    best_bl_cs1 = min(bl1.values()) if bl1 else 0
    best_bl_cs2 = min(bl2.values()) if bl2 else 0

    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lines = [
        "4DVarNet-FM: Experiment Synthesis Report  (live-updated)",
        "=" * 110,
        f"  Generated: {now}  |  T_max = 3.0s  |  dt = 0.01  |  15 obs/window",
        "",
        "Part 1 — 4DVarNet-FM Variants:",
        "-" * 110,
        f"{'Config':<20} {'CS1 X':<8} {'CS1 Y':<8} {'CS1 Z':<8} {'CS1 μ':<8} "
        f"{'CS2 X':<8} {'CS2 Y':<8} {'CS2 Z':<8} {'CS2 μ':<8} {'Deg':<7} {'Time':<7}",
        "-" * 110,
    ]
    if experiments:
        for e in experiments:
            c1, c2 = e["fm_cs1"], e["fm_cs2"]
            t = f"{e['total_time_seconds']/60:.0f}m"
            deg = e["fm_degradation"]
            marker = " ◀" if best and e["experiment_id"] == best["experiment_id"] else ""
            lines.append(
                f"{e['experiment_id']:<20} "
                f"{c1['X']['mean']:<8.3f} {c1['Y']['mean']:<8.3f} {c1['Z']['mean']:<8.3f} {c1['mean']:<8.3f} "
                f"{c2['X']['mean']:<8.3f} {c2['Y']['mean']:<8.3f} {c2['Z']['mean']:<8.3f} {c2['mean']:<8.3f} "
                f"{deg:<7.2f}x {t:<7}{marker}"
            )
    else:
        lines.append(f"{'  (no FM experiments completed yet)':<20}")
    lines += [
        "-" * 110,
        "",
        "Part 2 — DA Baselines (same test data):",
        "-" * 60,
        f"{'Method':<20} {'CS1 μ':<10} {'CS2 μ':<10} {'Deg':<8}",
        "-" * 60,
    ]
    if bl1 and bl2:
        for name in ["Weak-4DVar", "Strong-4DVar", "EnKF"]:
            if name in bl1 and name in bl2:
                m1, m2 = bl1[name], bl2[name]
                d = m2 / (m1 + 1e-10)
                lines.append(f"{name:<20} {m1:<10.4f} {m2:<10.4f} {d:<8.2f}x")
    else:
        lines.append(f"{'  (baselines not yet run)'}")
    lines += ["-" * 60, ""]
    if best:
        lines += [
            f"Best 4DVarNet-FM: {best['experiment_id']}  (CS2 μ={best['fm_cs2']['mean']:.4f})",
            f"Best baseline CS1: {min(bl1, key=bl1.get) if bl1 else 'N/A'} μ={best_bl_cs1:.4f}",
            f"Best baseline CS2: {min(bl2, key=bl2.get) if bl2 else 'N/A'} μ={best_bl_cs2:.4f}",
        ]
    else:
        lines += ["Best 4DVarNet-FM: (pending)"]
    ax.text(0.05, 0.98, "\n".join(lines), transform=ax.transAxes,
            fontsize=7.5, fontfamily="monospace", verticalalignment="top")
    fig.suptitle("4DVarNet-FM Synthesis Report", fontsize=16, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    pdf.savefig(fig)
    plt.close()
    print("  Page 1: Title + tables")

def _bar_chart_page(pdf, experiments, baselines, cs_key, title):
    fig, ax = plt.subplots(figsize=(10, 5.5))
    fm_ids = [e["experiment_id"] for e in experiments]
    fm_vals = [e[cs_key]["mean"] for e in experiments]
    bl1_key = "cs1" if "CS1" in title else "cs2"
    bl_names = []
    bl_vals = []
    if baselines:
        bl_names = [n for n in ["Weak-4DVar", "Strong-4DVar", "EnKF"] if n in baselines.get(bl1_key, {})]
        bl_vals = [baselines[bl1_key][n]["mean"] for n in bl_names]

    labels = fm_ids + bl_names
    vals = fm_vals + bl_vals
    if not vals:
        ax.text(0.5, 0.5, "No data available yet", ha="center", va="center", fontsize=14)
        ax.set_title(title, fontsize=14, fontweight="bold")
        plt.tight_layout()
        pdf.savefig(fig)
        plt.close()
        return

    colors = [COLORS_FM] * len(fm_vals) + [COLORS_BL[n] for n in bl_names]
    x = np.arange(len(labels))
    bars = ax.bar(x, vals, color=colors, width=0.55, edgecolor="white", linewidth=0.3)

    if bl_vals:
        bl_min = min(bl_vals)
        ax.axhline(bl_min, color="green", ls="--", lw=1.5, alpha=0.6,
                   label=f"Best baseline: {bl_min:.4f}")
    if fm_vals:
        fm_min = min(fm_vals)
        ax.axhline(fm_min, color=COLORS_FM, ls="-.", lw=1.5, alpha=0.6,
                   label=f"Best 4DVarNet-FM: {fm_min:.4f}")

    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                f"{val:.3f}", ha="center", va="bottom", fontsize=7, rotation=0)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8, rotation=40, ha="right")
    ax.set_ylabel("Mean RMSE", fontsize=12, fontweight="bold")
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.legend(fontsize=9, loc="upper right")
    ax.grid(True, axis="y", alpha=0.3, ls="--")
    ax.set_ylim(0, max(vals) * 1.25)
    plt.tight_layout()
    pdf.savefig(fig)
    plt.close()

def page_cs1_rmse(pdf, experiments, baselines):
    _bar_chart_page(pdf, experiments, baselines, "fm_cs1",
                    "Case Study 1 — RMSE (lower is better)")
    print("  Page 2: CS1 RMSE")

def page_cs2_rmse(pdf, experiments, baselines):
    _bar_chart_page(pdf, experiments, baselines, "fm_cs2",
                    "Case Study 2 — RMSE (lower is better)")
    print("  Page 3: CS2 RMSE")

def page_degradation(pdf, experiments, baselines):
    fig, ax = plt.subplots(figsize=(10, 5.5))
    fm_ids = [e["experiment_id"] for e in experiments]
    fm_vals = [e["fm_degradation"] for e in experiments]
    bl_names = []
    bl_vals = []
    if baselines and "cs1" in baselines and "cs2" in baselines:
        bl_names = [n for n in ["Weak-4DVar", "Strong-4DVar", "EnKF"]
                    if n in baselines.get("cs1", {}) and n in baselines.get("cs2", {})]
        bl_vals = [baselines["cs2"][n]["mean"] / (baselines["cs1"][n]["mean"] + 1e-10) for n in bl_names]

    labels = fm_ids + bl_names
    vals = fm_vals + bl_vals
    if not vals:
        ax.text(0.5, 0.5, "No data available yet", ha="center", va="center", fontsize=14)
        ax.set_title("Robustness — Lower Degradation = More Robust", fontsize=14, fontweight="bold")
        plt.tight_layout()
        pdf.savefig(fig)
        plt.close()
        print("  Page 4: Degradation ratios (skipped — no data)")
        return

    colors = [COLORS_FM] * len(fm_vals) + [COLORS_BL[n] for n in bl_names]
    x = np.arange(len(labels))
    bars = ax.bar(x, vals, color=colors, width=0.55, edgecolor="white", linewidth=0.3)

    if bl_vals:
        bl_min = min(bl_vals)
        ax.axhline(bl_min, color="green", ls="--", lw=1.5, alpha=0.6,
                   label=f"Best baseline: {bl_min:.2f}x")
    ax.axhline(1.0, color="gray", ls=":", lw=1, alpha=0.5, label="Ideal (1.0x)")

    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{val:.2f}x", ha="center", va="bottom", fontsize=7, rotation=0)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8, rotation=40, ha="right")
    ax.set_ylabel("Degradation Ratio (CS2 μ / CS1 μ)", fontsize=12, fontweight="bold")
    ax.set_title("Robustness — Lower Degradation = More Robust", fontsize=14, fontweight="bold")
    ax.legend(fontsize=9, loc="upper right")
    ax.grid(True, axis="y", alpha=0.3, ls="--")
    ax.set_ylim(0, max(vals) * 1.35)
    plt.tight_layout()
    pdf.savefig(fig)
    plt.close()
    print("  Page 4: Degradation ratios")

def page_training_time(pdf, experiments):
    fig, ax = plt.subplots(figsize=(10, 5))
    if not experiments:
        ax.text(0.5, 0.5, "No experiments completed yet", ha="center", va="center", fontsize=14)
        ax.set_title("Training Time Breakdown", fontsize=14, fontweight="bold")
        plt.tight_layout()
        pdf.savefig(fig)
        plt.close()
        print("  Page 5: Training time (skipped — no data)")
        return

    ids = [e["experiment_id"] for e in experiments]
    s1 = [e.get("stage1_time_seconds", 0) / 60 for e in experiments]
    s2 = [e.get("stage2_time_seconds", 0) / 60 for e in experiments]
    total = [e["total_time_seconds"] / 60 for e in experiments]

    x = np.arange(len(ids))
    w = 0.35
    bars1 = ax.bar(x - w / 2, s1, w, label="Stage 1", color="#4ecdc4", edgecolor="white")
    bars2 = ax.bar(x + w / 2, s2, w, label="Stage 2", color="#f7dc6f", edgecolor="white")

    for bar, val in zip(bars1, s1):
        if val > 0.5:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                    f"{val:.0f}m", ha="center", va="bottom", fontsize=7)
    for bar, val in zip(bars2, s2):
        if val > 0.5:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                    f"{val:.0f}m", ha="center", va="bottom", fontsize=7)

    ax.set_xticks(x)
    ax.set_xticklabels(ids, fontsize=9, rotation=30, ha="right")
    ax.set_ylabel("Time (minutes)", fontsize=12, fontweight="bold")
    ax.set_title("Training Time Breakdown", fontsize=14, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, axis="y", alpha=0.3, ls="--")

    for i, t in enumerate(total):
        ax.annotate(f"Total: {t:.0f}m", (x[i], s1[i] + s2[i] + 0.5),
                    fontsize=7, ha="center", va="bottom", alpha=0.7)

    plt.tight_layout()
    pdf.savefig(fig)
    plt.close()
    print("  Page 5: Training time")

def page_conclusion(pdf, experiments, baselines, best):
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.axis("off")

    bl1 = {}
    bl2 = {}
    if baselines:
        bl1 = {n: baselines["cs1"][n]["mean"]
               for n in ["Weak-4DVar", "Strong-4DVar", "EnKF"] if n in baselines.get("cs1", {})}
        bl2 = {n: baselines["cs2"][n]["mean"]
               for n in ["Weak-4DVar", "Strong-4DVar", "EnKF"] if n in baselines.get("cs2", {})}

    lines = [
        "Conclusion & Recommendations  (live-updated)",
        "=" * 90,
        "",
    ]

    if best:
        beats_cs1 = best["fm_cs1"]["mean"] < min(bl1.values()) if bl1 else False
        beats_cs2 = best["fm_cs2"]["mean"] < min(bl2.values()) if bl2 else False
        best_bl_cs1_name = min(bl1, key=bl1.get) if bl1 else "N/A"
        best_bl_cs1_val = bl1[best_bl_cs1_name] if best_bl_cs1_name != "N/A" else 0
        best_bl_cs2_name = min(bl2, key=bl2.get) if bl2 else "N/A"
        best_bl_cs2_val = bl2[best_bl_cs2_name] if best_bl_cs2_name != "N/A" else 0

        lines += [
            f"Best 4DVarNet-FM variant:  {best['experiment_id']}",
            f"  CS1 mean RMSE:  {best['fm_cs1']['mean']:.4f}  "
            f"(best baseline {best_bl_cs1_name}: {best_bl_cs1_val:.4f})"
            f"  {'✓ BEATS' if beats_cs1 else '✗ Does NOT beat'} baseline",
            f"  CS2 mean RMSE:  {best['fm_cs2']['mean']:.4f}  "
            f"(best baseline {best_bl_cs2_name}: {best_bl_cs2_val:.4f})"
            f"  {'✓ BEATS' if beats_cs2 else '✗ Does NOT beat'} baseline",
            f"  Degradation:    {best['fm_degradation']:.2f}x",
            "",
            "Comparison across all variants:",
            "-" * 60,
        ]

        for e in experiments:
            c1, c2 = e["fm_cs1"], e["fm_cs2"]
            beats1 = c1["mean"] < min(bl1.values()) if bl1 else False
            beats2 = c2["mean"] < min(bl2.values()) if bl2 else False
            status = ""
            if beats1 and beats2:
                status = "★ BEATS ALL BASELINES"
            elif beats2:
                status = "~ Beats CS2 baselines only"
            elif not bl1:
                status = "(no baselines for comparison)"
            else:
                status = "Below baselines"
            lines.append(f"  {e['experiment_id']:<20} CS1 μ={c1['mean']:<8.4f}  "
                         f"CS2 μ={c2['mean']:<8.4f}  Deg={e['fm_degradation']:.2f}x  {status}")

        lines += ["", "Key Insights:", "-" * 90]
        by_id = {e["experiment_id"]: e for e in experiments}
        if "A1_baseline" in by_id and "C4_stage1_only" in by_id:
            a1, c4 = by_id["A1_baseline"], by_id["C4_stage1_only"]
            lines.append(f"  • Non-Gaussian residual: CS2 μ {a1['fm_cs2']['mean']:.4f} vs "
                         f"{c4['fm_cs2']['mean']:.4f} (stage1-only) "
                         f"→ {'helps' if a1['fm_cs2']['mean'] < c4['fm_cs2']['mean'] else 'hurts'} robustness")
        if "D1_cs2_only" in by_id and "A1_baseline" in by_id:
            d1, a1 = by_id["D1_cs2_only"], by_id["A1_baseline"]
            lines.append(f"  • CS2-only training: CS2 μ {d1['fm_cs2']['mean']:.4f} vs "
                         f"{a1['fm_cs2']['mean']:.4f} (mixed) "
                         f"→ {'beneficial' if d1['fm_cs2']['mean'] < a1['fm_cs2']['mean'] else 'worse'} robustness")
        if "C1_longer_train" in by_id and "A1_baseline" in by_id:
            c1, a1 = by_id["C1_longer_train"], by_id["A1_baseline"]
            lines.append(f"  • Longer training: CS2 μ {c1['fm_cs2']['mean']:.4f} vs "
                         f"{a1['fm_cs2']['mean']:.4f} (default) "
                         f"→ {'improves' if c1['fm_cs2']['mean'] < a1['fm_cs2']['mean'] else 'no benefit from'} more epochs")
        if "B1_small_unet" in by_id and "A1_baseline" in by_id:
            b1, a1 = by_id["B1_small_unet"], by_id["A1_baseline"]
            lines.append(f"  • Smaller U-Net: CS2 μ {b1['fm_cs2']['mean']:.4f} vs "
                         f"{a1['fm_cs2']['mean']:.4f} (default) "
                         f"→ {'better' if b1['fm_cs2']['mean'] < a1['fm_cs2']['mean'] else 'worse'} generalization")

        lines += [
            "-" * 90,
            "",
            "Recommendation:",
            f"  Use {best['experiment_id']} for production (lowest CS2 RMSE).",
        ]
    else:
        lines += ["  (No FM experiments completed yet — pending results)"]

    ax.text(0.05, 0.98, "\n".join(lines), transform=ax.transAxes,
            fontsize=8.5, fontfamily="monospace", verticalalignment="top")
    fig.suptitle("4DVarNet-FM Synthesis Report — Conclusion",
                 fontsize=16, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    pdf.savefig(fig)
    plt.close()
    print("  Page 6: Conclusion")

# ── Main ────────────────────────────────────────────────────────

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default=os.path.join(OUTPUT_DIR, "synthesis_report.pdf"))
    parser.add_argument("--baselines", default=None,
                        help="Path to baselines JSON (e.g. experiments/baselines_dws50_inf1.2.json)")
    args = parser.parse_args()

    experiments = load_experiments()
    baselines = load_baselines(args.baselines)
    best = pick_best_fm(experiments)

    print(f"Experiments found: {len(experiments)}")
    if best:
        print(f"Best (lowest CS2 RMSE): {best['experiment_id']}")

    print(f"\nGenerating PDF: {args.output}")

    plt.rcParams.update({"font.size": 11, "axes.titlesize": 13, "axes.labelsize": 11})

    with PdfPages(args.output) as pdf:
        page_title(pdf, experiments, baselines, best)
        page_cs1_rmse(pdf, experiments, baselines)
        page_cs2_rmse(pdf, experiments, baselines)
        page_degradation(pdf, experiments, baselines)
        page_training_time(pdf, experiments)
        page_conclusion(pdf, experiments, baselines, best)

    print(f"\n✔ Report saved: {args.output}")

if __name__ == "__main__":
    main()
