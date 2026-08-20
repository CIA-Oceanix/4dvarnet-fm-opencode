#!/usr/bin/env python3
"""Compare S0/S0b and S1/S1b DA baseline results (S0b/S1b = fast_weights randomized).

Usage:
    python reports/compare_s0_s0b.py [--dws 50] [--legacy PATH] [--fw PATH]
"""
import argparse, glob, json, os, sys

EXP_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "experiments")


def load(path):
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def find_cache(dws, fw=False):
    pattern = os.path.join(EXP_DIR, f"l96_baselines_dws{dws}_inf2.0_etkf_inf2.0_obsj2*.json")
    matches = sorted(glob.glob(pattern))
    if fw:
        fw_matches = [m for m in matches if "_fw." in m or m.endswith("_fw.json")]
        return fw_matches[-1] if fw_matches else None
    else:
        non_fw = [m for m in matches if "_fw." not in m and not m.endswith("_fw.json")]
        return non_fw[-1] if non_fw else None


def compare_tables(s0_path, s0b_path, methods, label_pair):
    s0 = load(s0_path)
    s0b = load(s0b_path)
    if s0 is None or s0b is None:
        missing = []
        if s0 is None: missing.append(f"legacy ({os.path.basename(s0_path)})")
        if s0b is None: missing.append(f"fw-rand ({os.path.basename(s0b_path)})")
        print(f"  SKIP {label_pair}: missing {' + '.join(missing)}")
        return

    print(f"\n{'='*80}")
    print(f"  {label_pair}: legacy vs fast_weights-randomized")
    print(f"  legacy:  {os.path.basename(s0_path)}")
    print(f"  fw-rand: {os.path.basename(s0b_path)}")
    print(f"{'='*80}")

    header = f"{'Method':<16} {'Case':<6} {'Legacy':>10} {'FW-rand':>10} {'Delta':>10} {'Rel%':>8}"
    print(header)
    print("-" * len(header))

    for method in methods:
        for case in ["s0", "s1"]:
            label = case.upper()
            leg = s0.get(case, {}).get(method, {})
            fw = s0b.get(case, {}).get(method, {})
            leg_rmse = leg.get("mean", float("nan"))
            fw_rmse = fw.get("mean", float("nan"))
            delta = fw_rmse - leg_rmse
            rel = (delta / leg_rmse * 100) if leg_rmse != 0 else float("nan")
            print(f"{method:<16} {label:<6} {leg_rmse:>10.4f} {fw_rmse:>10.4f} {delta:>+10.4f} {rel:>+7.1f}%")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dws", type=int, default=50)
    parser.add_argument("--legacy", type=str, default=None)
    parser.add_argument("--fw", type=str, default=None)
    args = parser.parse_args()

    methods = ["EnKF", "ETKF", "Strong-4DVar"]

    if args.legacy:
        s0_path = args.legacy
    else:
        s0_path = find_cache(args.dws, fw=False)
        if s0_path is None:
            print(f"No legacy cache found for dws={args.dws}")
            sys.exit(1)

    if args.fw:
        s0b_path = args.fw
    else:
        s0b_path = find_cache(args.dws, fw=True)
        if s0b_path is None:
            print(f"No fw-rand cache found for dws={args.dws}")
            sys.exit(1)

    compare_tables(s0_path, s0b_path, methods, "S0 (legacy) vs S0b (fw-randomized)")


if __name__ == "__main__":
    main()
