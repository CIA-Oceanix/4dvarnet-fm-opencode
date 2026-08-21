#!/usr/bin/env python3
"""Repro gate: compare legacy S0/S1 DA baselines against cached reference values.

Usage:
    python reports/repro_gate_b2.py [--ref PATH] [--new PATH] [--tolerance 0.01]
"""
import argparse
import json
import os
import sys

EXP_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "experiments")


def load(path):
    with open(path) as f:
        return json.load(f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ref", type=str, default=None, help="Reference cache path")
    parser.add_argument("--new", type=str, default=None, help="New run cache path")
    parser.add_argument("--tolerance", type=float, default=0.01, help="Relative tolerance (default 1%)")
    args = parser.parse_args()

    ref_path = args.ref or os.path.join(EXP_DIR, "l96_baselines_dws500_inf2.0_etkf_inf2.0_obsj2.json")
    new_path = args.new

    if new_path is None:
        candidates = [
            os.path.join(EXP_DIR, "l96_baselines_dws500_inf2.0_etkf_inf2.0_obsj2_int200.json"),
            os.path.join(EXP_DIR, "l96_baselines_dws500_inf2.0_etkf_inf2.0_obsj2.json"),
        ]
        for c in candidates:
            if c != ref_path and os.path.exists(c):
                new_path = c
                break
        if new_path is None:
            print("ERROR: No new cache found. Run legacy baselines first.")
            sys.exit(1)

    print(f"Reference: {os.path.basename(ref_path)}")
    print(f"New:       {os.path.basename(new_path)}")
    print(f"Tolerance: {args.tolerance*100:.1f}%")
    print()

    ref = load(ref_path)
    new = load(new_path)

    methods = ["EnKF", "ETKF", "Strong-4DVar"]
    all_pass = True

    header = f"{'Method':<16} {'Case':<6} {'Ref':>10} {'New':>10} {'Delta':>10} {'Rel%':>8} {'Pass':>6}"
    print(header)
    print("-" * len(header))

    for case in ["s0", "s1"]:
        for method in methods:
            ref_rmse = ref.get(case, {}).get(method, {}).get("mean", float("nan"))
            new_rmse = new.get(case, {}).get(method, {}).get("mean", float("nan"))
            if ref_rmse == 0 or ref_rmse != ref_rmse:
                status = "SKIP"
                print(f"{method:<16} {case.upper():<6} {ref_rmse:>10.4f} {new_rmse:>10.4f} {'N/A':>10} {'N/A':>8} {status:>6}")
                continue
            delta = new_rmse - ref_rmse
            rel = abs(delta / ref_rmse)
            passed = rel <= args.tolerance
            status = "PASS" if passed else "FAIL"
            if not passed:
                all_pass = False
            print(f"{method:<16} {case.upper():<6} {ref_rmse:>10.4f} {new_rmse:>10.4f} {delta:>+10.4f} {rel*100:>+7.2f}% {status:>6}")

    print()
    if all_pass:
        print("VERDICT: PASS — all methods within tolerance")
    else:
        print("VERDICT: FAIL — some methods exceed tolerance")
    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
