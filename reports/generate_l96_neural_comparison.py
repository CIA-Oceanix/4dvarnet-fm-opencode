#!/usr/bin/env python3
"""Compare L96 neural models vs DA baselines on S0/S1 (all-5-param test config).

Reads:
  - experiments/L{1,2}_*/results.json  (neural fm_s0 / fm_s1 RMSE)
  - experiments/evaluate_all_l96_dws500_all5params.json  (DA baselines)

Prints and saves a markdown comparison table to reports/outputs/l96_neural_comparison.md
"""
import os, json, sys

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXP_DIR = os.path.join(BASE, "experiments")

NEURAL_CONFIGS = ["L1_direct_unet_s0s1", "L2_vanilla_cfm_s0s1"]
DA_BASELINE_FILE = "evaluate_all_l96_dws500_all5params.json"


def load_neural(exp_id):
    path = os.path.join(EXP_DIR, exp_id, "results.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        r = json.load(f)
    out = {"experiment_id": exp_id, "model_type": r.get("model_type", "?")}
    for k in ("fm_s0", "fm_s1"):
        if k in r:
            out[k] = r[k]["mean"]
    if "fm_s0" in r and "fm_s1" in r:
        out["degradation"] = r["fm_s1"]["mean"] / max(r["fm_s0"]["mean"], 1e-10)
    return out


def load_da():
    path = os.path.join(EXP_DIR, DA_BASELINE_FILE)
    if not os.path.exists(path):
        return None
    with open(path) as f:
        data = json.load(f)
    out = {}
    bl = data.get("baselines", data)
    for case in ("s0", "s1"):
        out[case] = {}
        if case not in bl:
            continue
        for m in bl[case]:
            if m == "config":
                continue
            entry = bl[case][m]
            if entry and "mean" in entry:
                out[case][m] = entry["mean"]
    return out


def main():
    neural = [load_neural(e) for e in NEURAL_CONFIGS]
    neural = [n for n in neural if n is not None]
    da = load_da()

    lines = []
    lines.append("# L96 Neural vs DA Baseline Comparison (S0/S1, all-5-param config)\n")
    lines.append("Test config: all 5 params (F, c1, h, hx, eps) randomized; "
                 "S0 ±20%, S1 ±20% + 10% bias.\n")

    if da:
        da_methods = sorted(da.get("s0", {}).keys())
        lines.append("## DA Baselines (mean RMSE)\n")
        lines.append("| Case | " + " | ".join(da_methods) + " |")
        lines.append("|------|" + "------|" * len(da_methods))
        for case in ("s0", "s1"):
            lines.append("| " + case + " | " +
                         " | ".join(f"{da[case].get(m, float('nan')):.4f}" for m in da_methods) + " |")
        lines.append("")
    else:
        lines.append(f"## DA Baselines\n(not found: {DA_BASELINE_FILE})\n")

    lines.append("## Neural Models\n")
    lines.append("| Experiment | Type | S0 RMSE | S1 RMSE | Degradation (S1/S0) |")
    lines.append("|------------|------|---------|---------|---------------------|")
    for n in neural:
        lines.append(f"| {n['experiment_id']} | {n['model_type']} | "
                     f"{n.get('fm_s0', float('nan')):.4f} | {n.get('fm_s1', float('nan')):.4f} | "
                     f"{n.get('degradation', float('nan')):.3f} |")
    lines.append("")

    out_dir = os.path.join(BASE, "reports", "outputs")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "l96_neural_comparison.md")
    with open(out_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
