#!/usr/bin/env python3
"""Compare S0b vs S0c DA baseline results — isolating the effect of h randomization.

S0b: all params randomized (F, c1, h, hx, eps, fast_weights)
S0c: all params randomized EXCEPT h (h fixed at reference)

Usage:
    python reports/compare_s0b_s0c.py [--dws 500] [--obs-interval 200]
"""
import argparse, glob, json, os, re, sys

EXP_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "experiments")


def load(path):
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def find_cache(dws, obs_interval, tag):
    pattern = os.path.join(EXP_DIR, f"l96_baselines_dws{dws}_inf2.0_etkf_inf2.0_obsj2_int{obs_interval}*{tag}.json")
    matches = sorted(glob.glob(pattern))
    return matches[-1] if matches else None


def compare_tables(path_a, path_b, methods, label_a, label_b, label_pair):
    a = load(path_a)
    b = load(path_b)
    if a is None or b is None:
        missing = []
        if a is None: missing.append(f"{label_a} ({os.path.basename(path_a)})")
        if b is None: missing.append(f"{label_b} ({os.path.basename(path_b)})")
        print(f"  SKIP {label_pair}: missing {' + '.join(missing)}")
        return

    print(f"\n{'='*80}")
    print(f"  {label_pair}")
    print(f"  {label_a}:  {os.path.basename(path_a)}")
    print(f"  {label_b}:  {os.path.basename(path_b)}")
    print(f"{'='*80}")

    header = f"{'Method':<16} {'Case':<6} {label_a:>10} {label_b:>10} {'Delta':>10} {'Rel%':>8}"
    print(header)
    print("-" * len(header))

    for method in methods:
        for case in ["s0"]:
            a_rmse = a.get(case, {}).get(method, {}).get("mean", float("nan"))
            b_rmse = b.get(case, {}).get(method, {}).get("mean", float("nan"))
            delta = b_rmse - a_rmse
            rel = (delta / a_rmse * 100) if a_rmse != 0 else float("nan")
            print(f"{method:<16} {case.upper():<6} {a_rmse:>10.4f} {b_rmse:>10.4f} {delta:>+10.4f} {rel:>+7.1f}%")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dws", type=int, default=500)
    parser.add_argument("--obs-interval", type=int, default=200, help="200=Obs15, 100=Obs30")
    args = parser.parse_args()

    methods = ["EnKF", "ETKF", "Strong-4DVar"]
    obs_label = "Obs15" if args.obs_interval == 200 else "Obs30"

    s0b_path = find_cache(args.dws, args.obs_interval, "_fw")
    s0c_path = find_cache(args.dws, args.obs_interval, "_fw_s0c")

    compare_tables(s0b_path, s0c_path, methods, "S0b (all rand)", "S0c (no h)", f"S0b vs S0c ({obs_label})")


if __name__ == "__main__":
    main()
