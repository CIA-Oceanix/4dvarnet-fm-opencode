#!/usr/bin/env python3
"""
Unified evaluation: run baselines + trained CFM models on CS1-CS4,
produce a comparison table (RMSE across X/Y/Z).
"""
import os, sys, json, argparse
import torch
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data.lorenz63 import Lorenz63Config, make_mixed_datasets
from evaluation.run import run_and_cache_baselines, _BASELINE_METHODS, _BASELINE_CASES, fmt_rmse
from evaluation.run import evaluate_baseline
from evaluation.baselines import Weak4DVar, Strong4DVar, EnKF, ETKF
from train import model_factory, evaluate_model, save_trajectories
from hydra import compose, initialize_config_dir

BASE = os.path.dirname(os.path.abspath(__file__))
EXP_DIR = os.path.join(BASE, "experiments")


def load_trained_model(exp_dir: str, device: torch.device):
    results_path = os.path.join(exp_dir, "results.json")
    if not os.path.exists(results_path):
        print(f"  No results.json in {exp_dir}, skipping")
        return None, None
    with open(results_path) as f:
        results = json.load(f)
    model_type = results.get("model_type", "tweedie")

    ckpt = os.path.join(exp_dir, "checkpoint_stage2.pt")
    if not os.path.exists(ckpt):
        ckpt = os.path.join(exp_dir, "checkpoints", "stage2.pt")
    if not os.path.exists(ckpt):
        print(f"  No checkpoint_stage2.pt in {exp_dir}, skipping")
        return None, None

    with initialize_config_dir(config_dir=os.path.join(BASE, "config"), version_base="1.3"):
        cfg = compose(config_name="lorenz63_default")
    model = model_factory(cfg, device)
    model.load_state_dict(torch.load(ckpt, map_location=device))
    model.to(device)
    model.eval()
    return model, model_type


def evaluate_cfm_model(model, model_type, datasets, device, exp_dir):
    test_keys = ["test_cs1", "test_cs2", "test_cs3", "test_cs4"]
    metrics = {}
    for key in test_keys:
        if key in datasets:
            m, s = evaluate_model(model, datasets[key], device, model_type)
            metrics[key] = (m, s)
            case = key.replace("test_", "")
            save_trajectories(model, datasets[key], device, model_type,
                              os.path.join(exp_dir, f"evaluate_all_trajs_{case}.npz"))
    return metrics


def run_baselines(datasets, device):
    print("\n── Running Baselines ──")
    results = run_and_cache_baselines(datasets, device, batch_size=200)
    return results


def build_table(baseline_results, cfm_results, exp_ids):
    rows = []
    for case_name, ds_key, _, _, label, _ in _BASELINE_CASES:
        row = {"Case": label}
        for method in _BASELINE_METHODS:
            bl = baseline_results.get(case_name, {}).get(method, {})
            row[f"{method}"] = bl.get("mean", float("nan"))
        for eid in exp_ids:
            m = cfm_results.get(eid, {}).get(ds_key)
            if m:
                row[f"CFM-{eid}"] = float(np.mean(m[0]))
        rows.append(row)
    return rows


def print_table(rows, headers):
    widths = {k: max(len(k), max(len(f"{r.get(k, ''):.4f}") if isinstance(r.get(k), (int,float)) else len(str(r.get(k, ''))) for r in rows)) for k in headers}
    line = " | ".join(f"{h:<{widths[h]}}" for h in headers)
    sep = "-+-".join("-" * widths[h] for h in headers)
    print(line)
    print(sep)
    for r in rows:
        vals = []
        for h in headers:
            v = r.get(h, "")
            if isinstance(v, float):
                vals.append(f"{v:<{widths[h]}.4f}")
            else:
                vals.append(f"{v:<{widths[h]}}")
        print(" | ".join(vals))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiments", nargs="*", default=[],
                        help="Experiment directories to evaluate")
    parser.add_argument("--device", default=None)
    parser.add_argument("--baselines-only", action="store_true",
                        help="Only run baselines, skip CFM evaluation")
    args = parser.parse_args()

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Device: {device}")

    base_cfg = Lorenz63Config(
        dt=0.01, T_max=3.0, obs_interval=20,
        R_var=0.5, B_var=2.0,
        num_windows=2000, window_spacing=2000,
        spinup_steps=10000, seed=42,
        sigma_true=10.0, rho_true=28.0, beta_true=2.6666666666666665,
        gamma=0.05, W_L_bar=0.0, c1=1.0, c2=0.1,
        sigma_0=0.08, sigma_L=0.20,
        tau_eta=5.0, sigma_eta=0.7071067811865476,
        param_bias=0.0, forcing_state_bias=0.0, forcing_coupling="linear",
    )
    datasets = make_mixed_datasets(base_cfg, num_test_windows=200,
                                   include_randparam_test=True, param_noise=0.2)

    baseline_results = run_baselines(datasets, device)

    cfm_results = {}
    exp_ids = [e for e in args.experiments if os.path.isdir(os.path.join(EXP_DIR, e))]

    if not args.baselines_only and not exp_ids:
        print("\nNo experiment directories provided/valid; run `python evaluate_all.py --experiments E1 ...`")
    elif not args.baselines_only:
        print("\n── Evaluating CFM Models ──")
        for eid in exp_ids:
            exp_dir = os.path.join(EXP_DIR, eid)
            print(f"  Loading {eid}...", end=" ", flush=True)
            model, model_type = load_trained_model(exp_dir, device)
            if model is None:
                continue
            print(f"done (type={model_type})")
            metrics = evaluate_cfm_model(model, model_type, datasets, device, exp_dir)
            cfm_results[eid] = metrics

    print("\n── Comparison Table ──")
    headers = ["Case"] + _BASELINE_METHODS + [f"CFM-{e}" for e in exp_ids]
    rows = build_table(baseline_results, cfm_results, exp_ids)
    print_table(rows, headers)

    combined = {"baselines": baseline_results, "cfm": {}}
    for eid, metrics in cfm_results.items():
        combined["cfm"][eid] = {k: fmt_rmse(*v) for k, v in metrics.items()}
    out_path = os.path.join(EXP_DIR, "evaluate_all.json")
    with open(out_path, "w") as f:
        json.dump(combined, f, indent=2)
    print(f"\nSaved comparison to {out_path}")


if __name__ == "__main__":
    main()
