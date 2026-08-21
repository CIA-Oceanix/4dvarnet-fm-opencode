#!/usr/bin/env python3
"""Compare S0b vs S0c and S1b vs S1c DA baseline results — isolating h randomization.

S0b: all params randomized, DA uses true params (no bias)
S0c: all params randomized EXCEPT h (h fixed), DA uses true params
S1b: same as S0b/S0c but with DA forward biased on non-h params (+10%)

Usage:
    python reports/compare_s0b_s0c.py [--dws 500] [--obs-interval 200]
"""
import argparse
import glob
import json
import os

EXP_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "experiments")


def load(path):
    if path is None or not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def find_cache(dws, obs_interval, suffix=""):
    core = f"l96_baselines_dws{dws}"
    if suffix:
        core += f"_{suffix}"
    core += f"_inf2.0_etkf_inf2.0_obsj2_int{obs_interval}"
    pattern = os.path.join(EXP_DIR, core + "*_fw.json")
    matches = sorted(glob.glob(pattern))
    return matches[-1] if matches else None


def fmt_row(method, case, rmse_a, rmse_b):
    delta = rmse_b - rmse_a
    rel = (delta / rmse_a * 100) if rmse_a else float("nan")
    return f"{method:<16} {case:<6} {rmse_a:>10.4f} {rmse_b:>10.4f} {delta:>+10.4f} {rel:>+7.1f}%"


def print_section(label_a, label_b, a, b, obs_label):
    if a is None or b is None:
        missing = []
        if a is None:
            missing.append(label_a)
        if b is None:
            missing.append(label_b)
        print(f"  SKIP: missing {' + '.join(missing)}")
        return

    methods = ["EnKF", "ETKF", "Strong-4DVar"]
    cases = [k for k in a.keys() if k in ("s0", "s1")]

    header = f"{'Method':<16} {'Case':<6} {label_a:>10} {label_b:>10} {'Delta':>10} {'Rel%':>8}"
    print(header)
    print("-" * len(header))

    for case in cases:
        for method in methods:
            a_rmse = a.get(case, {}).get(method, {}).get("mean", float("nan"))
            b_rmse = b.get(case, {}).get(method, {}).get("mean", float("nan"))
            print(fmt_row(method, case.upper(), a_rmse, b_rmse))
        print()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dws", type=int, default=500)
    parser.add_argument("--obs-interval", type=int, default=200, help="200=Obs15, 100=Obs30")
    args = parser.parse_args()

    obs_label = "Obs15" if args.obs_interval == 200 else "Obs30"
    s0b = load(find_cache(args.dws, args.obs_interval, suffix=""))
    s0c = load(find_cache(args.dws, args.obs_interval, suffix="s0c"))

    print(f"\n{'='*80}")
    print(f"  S0b vs S0c ({obs_label}) — h randomization effect")
    print(f"  S0b: {os.path.basename(find_cache(args.dws, args.obs_interval, suffix='')) or 'MISSING'}")
    print(f"  S0c: {os.path.basename(find_cache(args.dws, args.obs_interval, suffix='s0c')) or 'MISSING'}")
    print(f"{'='*80}")
    print_section("S0b", "S0c", s0b, s0c, obs_label)


if __name__ == "__main__":
    main()
