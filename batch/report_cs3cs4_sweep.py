#!/usr/bin/env python3
"""Summarise EnKF/ETKF inflation sweep results for CS3/CS4.

Usage:
    python batch/report_cs3cs4_sweep.py
"""
import os, sys, json, glob

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXP_DIR = os.path.join(BASE, "experiments")


def extract_method_inflation(fname: str):
    name = os.path.basename(fname).replace(".json", "")
    if "_etkf_inf" in name:
        parts = name.split("_etkf_inf")
        method = "ETKF"
        inf = float(parts[1].replace("_cs3cs4", ""))
    elif "_inf" in name and "cs3cs4" in name:
        parts = name.split("_inf")
        inf_str = parts[1]
        if "_etkf" in inf_str:
            return None, None
        inf = float(inf_str.replace("_cs3cs4", ""))
        method = "EnKF"
    else:
        return None, None
    return method, inf


def main():
    pattern = os.path.join(EXP_DIR, "baselines_dws50_cs3cs4_*.json")
    files = sorted(glob.glob(pattern))
    if not files:
        pattern2 = os.path.join(EXP_DIR, "baselines_dws50_*_cs3cs4.json")
        files = sorted(glob.glob(pattern2))

    if not files:
        print("No CS3/CS4 sweep result files found in experiments/")
        sys.exit(1)

    rows_enkf = []
    rows_etkf = []

    for fpath in files:
        method, inf = extract_method_inflation(fpath)
        if method is None:
            continue
        with open(fpath) as f:
            data = json.load(f)

        cs3 = data.get("cs3", {})
        cs4 = data.get("cs4", {})

        enkf_cs3 = cs3.get("EnKF", {})
        enkf_cs4 = cs4.get("EnKF", {})
        etkf_cs3 = cs3.get("ETKF", {})
        etkf_cs4 = cs4.get("ETKF", {})

        if method == "EnKF":
            rows_enkf.append(
                (inf, enkf_cs3.get("mean"), enkf_cs4.get("mean"))
            )
        else:
            rows_etkf.append(
                (inf, etkf_cs3.get("mean"), etkf_cs4.get("mean"))
            )

    rows_enkf.sort(key=lambda r: r[0])
    rows_etkf.sort(key=lambda r: r[0])

    print("=" * 72)
    print("  CS3/CS4 Inflation Sensitivity — EnKF")
    print("=" * 72)
    print(f"  {'Inflation':<12} {'CS3 μ':<12} {'CS4 μ':<12} {'CS3+CS4 μ':<12}")
    print(f"  {'-'*10}   {'-'*8}   {'-'*8}   {'-'*10}")
    best = None
    best_val = float("inf")
    for inf, c3, c4 in rows_enkf:
        if c3 is None or c4 is None:
            row = f"  {inf:<12.2f} {'—':<12} {'—':<12} {'—':<12}"
        else:
            both = (c3 + c4) / 2
            if both < best_val:
                best_val = both
                best = (inf, c3, c4, both)
            row = f"  {inf:<12.2f} {c3:<12.4f} {c4:<12.4f} {both:<12.4f}"
        print(row)
    if best:
        print(f"  {'─'*48}")
        print(f"  ★ Best: inflation={best[0]:.2f}  "
              f"CS3={best[1]:.4f}  CS4={best[2]:.4f}  avg={best[3]:.4f}")
    print()

    print("=" * 72)
    print("  CS3/CS4 Inflation Sensitivity — ETKF")
    print("=" * 72)
    print(f"  {'Inflation':<12} {'CS3 μ':<12} {'CS4 μ':<12} {'CS3+CS4 μ':<12}")
    print(f"  {'-'*10}   {'-'*8}   {'-'*8}   {'-'*10}")
    best = None
    best_val = float("inf")
    for inf, c3, c4 in rows_etkf:
        if c3 is None or c4 is None:
            row = f"  {inf:<12.2f} {'—':<12} {'—':<12} {'—':<12}"
        else:
            both = (c3 + c4) / 2
            if both < best_val:
                best_val = both
                best = (inf, c3, c4, both)
            row = f"  {inf:<12.2f} {c3:<12.4f} {c4:<12.4f} {both:<12.4f}"
        print(row)
    if best:
        print(f"  {'─'*48}")
        print(f"  ★ Best: inflation={best[0]:.2f}  "
              f"CS3={best[1]:.4f}  CS4={best[2]:.4f}  avg={best[3]:.4f}")
    print()

    print("─" * 72)
    print("  Note: EnKF sweep holds ETKF at 1.6; ETKF sweep holds EnKF at 1.2.")
    print()


if __name__ == "__main__":
    main()
