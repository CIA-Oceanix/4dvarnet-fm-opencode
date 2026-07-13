#!/usr/bin/env python3
"""Evaluate all baselines on rotating shallow water case study.

Usage
-----
::

    python evaluate_all_sw.py                        # defaults
    python evaluate_all_sw.py --Nx 32 --Ny 32        # smaller grid
    python evaluate_all_sw.py --batch-size 1          # sequential windows
    python evaluate_all_sw.py --da-window-steps 200   # shorter DA windows

Outputs
-------
* ``<output-dir>/sw_metrics.json``  — full results dict
* Formatted RMSE + EV table on stdout
* PASS / FAIL against EV performance targets
"""

import argparse
import json
import math
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data.shallow_water import ShallowWaterConfig
from evaluation.metrics import print_sw_metrics_table, validate_ev_targets
from evaluation.run_sw import run_sw_baselines


class _SafeEncoder(json.JSONEncoder):
    """JSON encoder that converts NaN/Inf to null for RFC 8259 compliance."""

    def default(self, obj):
        if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
            return None
        return super().default(obj)


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate SW baselines (S0 / S1 rotating shallow water)"
    )
    parser.add_argument("--device", default=None, help="torch device string")
    parser.add_argument("--num-test-windows", type=int, default=200)
    parser.add_argument("--da-window-steps", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=200,
                        help="Parallel windows (reduce for large grids)")
    parser.add_argument("--t-max", type=float, default=3.0)
    parser.add_argument("--enkf-inflation", type=float, default=2.0)
    parser.add_argument("--etkf-inflation", type=float, default=2.0)
    parser.add_argument("--output-dir", type=str, default="outputs/sw_baselines")
    parser.add_argument("--Nx", type=int, default=64)
    parser.add_argument("--Ny", type=int, default=64)
    parser.add_argument(
        "--methods", nargs="+",
        default=["Weak-4DVar", "Strong-4DVar", "EnKF", "ETKF"],
        choices=["Weak-4DVar", "Strong-4DVar", "EnKF", "ETKF"],
        help="DA methods to run (default: all four)",
    )
    args = parser.parse_args()

    # ---- device ----
    device = torch.device(
        args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    if torch.cuda.is_available():
        print(f"Device: {device} ({torch.cuda.get_device_name(0)})")
    else:
        print(f"Device: {device}")

    # ---- config ----
    config = ShallowWaterConfig(Nx=args.Nx, Ny=args.Ny)
    print(f"Config: Nx={config.Nx}  Ny={config.Ny}  state_dim={config.state_dim}")
    print(
        f"  da_window_steps={args.da_window_steps}  "
        f"num_test_windows={args.num_test_windows}"
    )
    print(
        f"  enkf_inflation={args.enkf_inflation}  "
        f"etkf_inflation={args.etkf_inflation}"
    )

    # ---- run baselines ----
    t0 = time.time()
    results = run_sw_baselines(
        config=config,
        num_test_windows=args.num_test_windows,
        da_window_steps=args.da_window_steps,
        batch_size=args.batch_size,
        t_max=args.t_max,
        enkf_inflation=args.enkf_inflation,
        etkf_inflation=args.etkf_inflation,
        output_dir=args.output_dir,
        methods=args.methods,
    )
    print(f"\nTotal wall-clock: {time.time() - t0:.1f}s")

    # ---- formatted tables ----
    for scenario in ("S0", "S1"):
        if scenario not in results:
            continue
        table_data: dict = {}
        for method_name, metrics in results[scenario].items():
            table_data[method_name] = metrics
        print_sw_metrics_table(table_data, scenario, args.Nx, args.Ny)

    # ---- EV target validation ----
    s0_targets = {"ocean": 0.95, "atmosphere": 0.95}
    s1_targets = {"ocean": 0.70, "atmosphere": 0.85}

    print("\n=== Performance Targets ===")
    for scenario, targets in [("S0", s0_targets), ("S1", s1_targets)]:
        if scenario not in results:
            continue
        for method_name, metrics in results[scenario].items():
            try:
                validation = validate_ev_targets(metrics, targets, scenario)
                for comp, info in validation.items():
                    status = "PASS" if info["passed"] else "FAIL"
                    print(
                        f"  {scenario}/{method_name}/{comp}: {status}  "
                        f"(EV={info['actual']:.3f}, target={info['target']})"
                    )
            except KeyError as exc:
                print(f"  {scenario}/{method_name}: skipped ({exc})")

    # ---- save JSON ----
    output_path = os.path.join(args.output_dir, "sw_metrics.json")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, cls=_SafeEncoder)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
