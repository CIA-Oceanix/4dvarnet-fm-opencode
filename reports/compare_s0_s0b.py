#!/usr/bin/env python3
"""Compare S0/S0b and S1/S1b DA baseline results (S0b/S1b = fast_weights randomized)."""
import json, os, sys
import numpy as np

EXP_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "experiments")


def load(path):
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def compare_tables(s0_path, s0b_path, methods, label_pair):
    s0 = load(s0_path)
    s0b = load(s0b_path)
    if s0 is None or s0b is None:
        missing = []
        if s0 is None: missing.append(f"legacy ({s0_path})")
        if s0b is None: missing.append(f"fw-rand ({s0b_path})")
        print(f"  SKIP {label_pair}: missing {' + '.join(missing)}")
        return

    print(f"\n{'='*80}")
    print(f"  {label_pair}: legacy vs fast_weights-randomized")
    print(f"{'='*80}")

    header = f"{'Method':<16} {'Case':<6} {'Legacy':>10} {'FW-rand':>10} {'Delta':>10} {'Rel%':>8}"
    print(header)
    print("-" * len(header))

    for method in methods:
        for case in ["s0", "s1"]:
            label = case.upper()
            leg = s0.get("baselines", {}).get(case, {}).get(method, {})
            fw = s0b.get("baselines", {}).get(case, {}).get(method, {})
            leg_rmse = leg.get("mean", float("nan"))
            fw_rmse = fw.get("mean", float("nan"))
            delta = fw_rmse - leg_rmse
            rel = (delta / leg_rmse * 100) if leg_rmse != 0 else float("nan")
            print(f"{method:<16} {label:<6} {leg_rmse:>10.4f} {fw_rmse:>10.4f} {delta:>+10.4f} {rel:>+7.1f}%")


def main():
    dws = 500
    inf = 2.0
    etkf_inf = 2.0
    methods = ["EnKF", "ETKF", "Strong-4DVar"]

    s0_path = os.path.join(EXP_DIR, f"l96_baselines_dws{dws}_inf{inf}_etkf_inf{etkf_inf}_obsj2_int200.json")
    s0b_path = os.path.join(EXP_DIR, f"l96_baselines_dws{dws}_inf{inf}_etkf_inf{etkf_inf}_obsj2_int200_fw.json")

    compare_tables(s0_path, s0b_path, methods, "S0 (legacy) vs S0b (fw-randomized)")

    print("\n")
    print("Note: S1 vs S1b comparison requires biased fast_weights DA runs (not yet available).")
    print("The current S0b run uses unbiased fast_weights randomization.")


if __name__ == "__main__":
    main()
