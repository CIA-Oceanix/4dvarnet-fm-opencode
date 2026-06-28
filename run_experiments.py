#!/usr/bin/env python3
"""
Experiment orchestrator for 4DVarNet-FM.
Usage:
    python run_experiments.py                          # Run all 5 experiments
    python run_experiments.py --experiment A1          # Run specific
    python run_experiments.py --baselines-only         # Cache baselines only
"""
import os, sys, json, time, argparse, traceback, subprocess
import torch
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data.lorenz63 import Lorenz63Config, make_mixed_datasets
from data.dataloader import FlowMatchingDataset, ConcatFMDataset, collate_fm
from torch.utils.data import DataLoader
from models.solver import TweedieSolver
from training.stage1 import train_stage1
from training.stage2 import train_stage2
from evaluation.baselines import Weak4DVar, Strong4DVar, EnKF
from evaluation.metrics import rmse

BASE = os.path.dirname(os.path.abspath(__file__))
EXP_DIR = os.path.join(BASE, "experiments")
os.makedirs(EXP_DIR, exist_ok=True)

EXPERIMENTS = [
    {
        "id": "A1_baseline",
        "note": "Default config — establish baseline",
        "cfg": {
            "T_max": 3.0, "hidden_channels": [64, 128, 256],
            "epochs_stage1": 200, "epochs_stage2": 400,
            "lr": 1e-3, "use_energy": True, "batch_size": 32,
            "train_mix": "cs1+cs2", "skip_stage2": False,
        },
    },
    {
        "id": "C4_stage1_only",
        "note": "Gaussian estimator only — no non-Gaussian residual",
        "cfg": {
            "T_max": 3.0, "hidden_channels": [64, 128, 256],
            "epochs_stage1": 200, "epochs_stage2": 0,
            "lr": 1e-3, "use_energy": True, "batch_size": 32,
            "train_mix": "cs1+cs2", "skip_stage2": True,
        },
    },
    {
        "id": "D1_cs2_only",
        "note": "Train on CS2 data only — specialize for robustness",
        "cfg": {
            "T_max": 3.0, "hidden_channels": [64, 128, 256],
            "epochs_stage1": 200, "epochs_stage2": 400,
            "lr": 1e-3, "use_energy": True, "batch_size": 32,
            "train_mix": "cs2_only", "skip_stage2": False,
        },
    },
    {
        "id": "C1_longer_train",
        "note": "Slower LR, longer training — better convergence",
        "cfg": {
            "T_max": 3.0, "hidden_channels": [64, 128, 256],
            "epochs_stage1": 400, "epochs_stage2": 800,
            "lr": 3e-4, "use_energy": True, "batch_size": 32,
            "train_mix": "cs1+cs2", "skip_stage2": False,
        },
    },
    {
        "id": "B1_small_unet",
        "note": "Smaller U-Net [32,64,128] — less overfitting",
        "cfg": {
            "T_max": 3.0, "hidden_channels": [32, 64, 128],
            "epochs_stage1": 200, "epochs_stage2": 400,
            "lr": 1e-3, "use_energy": True, "batch_size": 32,
            "train_mix": "cs1+cs2", "skip_stage2": False,
        },
    },
]

# ── Dataloader factory ──────────────────────────────────────────

def make_experiment_dataloaders(datasets, batch_size=32, train_mix="cs1+cs2",
                                num_workers=4):
    kw = dict(batch_size=batch_size, collate_fn=collate_fm,
              num_workers=num_workers, pin_memory=True)
    source_map = {
        "cs1+cs2": [datasets["train_cs1"], datasets["train_cs2"]],
        "cs1_only": [datasets["train_cs1"]],
        "cs2_only": [datasets["train_cs2"]],
    }
    train_sources = source_map.get(train_mix, source_map["cs1+cs2"])
    return {
        "train": DataLoader(
            ConcatFMDataset(train_sources),
            shuffle=True, **kw,
        ),
        "val": DataLoader(
            ConcatFMDataset([datasets["val_cs1"], datasets["val_cs2"]]),
            shuffle=False, **kw,
        ),
    }

# ── Evaluation helpers ──────────────────────────────────────────

def evaluate_model(model, dataset, device, use_mean_estimator=False):
    rmse_list = []
    for i in range(len(dataset)):
        w = dataset[i]
        obs = w["obs"].unsqueeze(0).to(device)
        if use_mean_estimator:
            pred = model.estimate_mean(obs).detach().cpu().numpy()[0]
        else:
            pred = model(obs).detach().cpu().numpy()[0]
        truth = w["true_state"].numpy()
        rmse_list.append(rmse(pred, truth))
    all_rmse = np.stack(rmse_list, axis=0)
    return np.mean(all_rmse, axis=0), np.std(all_rmse, axis=0)

def evaluate_baseline(method, dataset, cfg, device, return_trajs=False, batch_size=1):
    sig, rho, bet = cfg.da_params
    rmse_list = []
    results_list = []

    use_corrupted = getattr(cfg, 'use_corrupted_forcing', True)
    force_key = "forcing_corrupted" if use_corrupted else "forcing_true"

    if batch_size > 1 and hasattr(method, 'assimilate_batch'):
        for i in range(0, len(dataset), batch_size):
            batch = [dataset[j] for j in range(i, min(i + batch_size, len(dataset)))]
            obs = torch.stack([w["obs"].to(device) for w in batch], dim=0)
            mask = torch.stack([w["obs_mask"].to(device) for w in batch], dim=0)
            truth = torch.stack([w["true_state"] for w in batch], dim=0)
            force = torch.stack([w[force_key].to(device) for w in batch], dim=0)

            results = method.assimilate_batch(obs, mask, force, truth, sigma=sig, rho=rho, beta=bet)
            for result in results:
                rmse_list.append(result.rmse)
                results_list.append(result)
    else:
        for i in range(len(dataset)):
            w = dataset[i]
            obs = w["obs"].to(device)
            mask = w["obs_mask"].to(device)
            truth = w["true_state"]
            force = w[force_key].to(device)
            result = method.assimilate(obs, mask, force, truth, sigma=sig, rho=rho, beta=bet)
            rmse_list.append(result.rmse)
            results_list.append(result)

    all_rmse = np.stack(rmse_list, axis=0)
    stats = (np.mean(all_rmse, axis=0), np.std(all_rmse, axis=0))
    if return_trajs:
        return stats, results_list
    return stats

def fmt_rmse(mean_arr, std_arr):
    return {
        "X": {"mean": float(mean_arr[0]), "std": float(std_arr[0])},
        "Y": {"mean": float(mean_arr[1]), "std": float(std_arr[1])},
        "Z": {"mean": float(mean_arr[2]), "std": float(std_arr[2])},
        "mean": float(np.mean(mean_arr)),
    }

# ── Baselines (uses pre-generated datasets) ─────────────────────

_BASELINE_METHODS = ["Weak-4DVar", "Strong-4DVar", "EnKF"]
_BASELINE_CASES = [("cs1", "test_cs1", 1, 0.0, "CS1"),
                   ("cs2", "test_cs2", 2, 0.15, "CS2")]


def _baseline_traj_path(case_name, method_name):
    key = f"{case_name}_{method_name.replace('-', '_').replace(' ', '_')}"
    return os.path.join(EXP_DIR, f"baselines_trajs_{key}.npz")


def run_and_cache_baselines(datasets, device, batch_size=1):
    cache_path = os.path.join(EXP_DIR, "baselines.json")

    # Load existing partial results if any
    partial = {}
    if os.path.exists(cache_path):
        with open(cache_path) as f:
            partial = json.load(f)
        print(f"  Found partial results ({cache_path}), resuming...")
    else:
        print("  Running baselines...")

    N = int(3.0 / 0.01)
    w4d = Weak4DVar(dt=0.01, da_window_steps=N, device=device)
    s4d = Strong4DVar(dt=0.01, da_window_steps=N, device=device)
    enkf = EnKF(dt=0.01, N_ensemble=30, device=device)
    method_map = {"Weak-4DVar": w4d, "Strong-4DVar": s4d, "EnKF": enkf}

    cfg_cs1 = Lorenz63Config(case=1, param_bias=0.0, T_max=3.0, seed=123)
    cfg_cs2 = Lorenz63Config(case=2, param_bias=0.15, forcing_state_bias=0.15,
                              forcing_coupling="quartic", T_max=3.0, seed=124)
    cfg_map = {"cs1": cfg_cs1, "cs2": cfg_cs2}

    if "config" not in partial:
        partial["config"] = {"T_max": 3.0, "da_window_steps": N}

    total_t0 = time.time()

    for case_name, ds_key, case_val, bias, label in _BASELINE_CASES:
        ds = datasets[ds_key]
        cfg = cfg_map[case_name]
        for name in _BASELINE_METHODS:
            if partial.get(case_name, {}).get(name) is not None:
                print(f"    {label}/{name:<15} already done, skipping")
                continue

            method = method_map[name]
            print(f"    {label}/{name:<15} ...", end=" ", flush=True)
            t1 = time.time()
            (m, s), bl_results = evaluate_baseline(method, ds, cfg, device, return_trajs=True, batch_size=batch_size)
            elapsed = time.time() - t1

            # Save RMSE to JSON
            if case_name not in partial:
                partial[case_name] = {}
            partial[case_name][name] = fmt_rmse(m, s)
            partial["total_time_seconds"] = time.time() - total_t0
            with open(cache_path, "w") as f:
                json.dump(partial, f, indent=2)

            # Save trajectories to per-(case,method) file
            trajs = np.stack([r.trajectory for r in bl_results], axis=0)
            truths = np.stack([ds[i]["true_state"].numpy() for i in range(len(ds))], axis=0)
            traj_data = {"trajectories": trajs, "truths": truths}
            if bl_results[0].ensemble_variance is not None:
                traj_data["ensemble_variance"] = np.stack(
                    [r.ensemble_variance for r in bl_results], axis=0)
            np.savez_compressed(_baseline_traj_path(case_name, name), **traj_data)

            print(f"X={m[0]:.4f} Y={m[1]:.4f} Z={m[2]:.4f}"
                  f"  mean={np.mean(m):.4f} [{elapsed:.1f}s]")

    # Combine per-method trajectory files into final .npz
    traj_path = os.path.join(EXP_DIR, "baselines_trajectories.npz")
    all_present = True
    for case_name, _, _, _, _ in _BASELINE_CASES:
        for name in _BASELINE_METHODS:
            if not os.path.exists(_baseline_traj_path(case_name, name)):
                all_present = False
                break

    if all_present:
        print("  Combining trajectories...")
        traj_arrays = {}
        for case_name, _, _, _, _ in _BASELINE_CASES:
            for name in _BASELINE_METHODS:
                src = _baseline_traj_path(case_name, name)
                data = np.load(src)
                prefix = f"{case_name}_{name.replace('-', '_').replace(' ', '_')}"
                for key in data.files:
                    traj_arrays[f"{prefix}_{key}"] = data[key]
                data.close()
                os.remove(src)
        np.savez_compressed(traj_path, **traj_arrays)
        print(f"    Saved: {traj_path}")
    else:
        print("    (incomplete — skipping trajectory combination)")

    # Regenerate synthesis report with latest baselines
    subprocess.run([sys.executable, "generate_report.py"], capture_output=True)
    return partial

# ── Single experiment ───────────────────────────────────────────

def run_single_experiment(exp, datasets, device, batch_size=256, num_workers=4):
    exp_id = exp["id"]
    cfg = dict(exp["cfg"])
    # Scale LR linearly with batch size (base config assumes batch_size=32)
    base_lr = cfg["lr"]
    cfg["lr"] = base_lr * (batch_size / 32.0)
    exp_dir = os.path.join(EXP_DIR, exp_id)
    os.makedirs(exp_dir, exist_ok=True)
    results_path = os.path.join(exp_dir, "results.json")

    if os.path.exists(results_path):
        print(f"  [{exp_id}] Results exist, skipping.")
        with open(results_path) as f:
            return json.load(f)

    print(f"\n{'=' * 60}")
    print(f"  [{exp_id}] {exp['note']}")
    print(f"{'=' * 60}")
    total_t0 = time.time()

    # Data loaders (creates from pre-generated datasets)
    print(f"  [1/5] Creating dataloaders (batch={batch_size}, workers={num_workers}, "
          f"train_mix={cfg['train_mix']})...")
    t0 = time.time()
    loaders = make_experiment_dataloaders(
        datasets, batch_size=batch_size,
        train_mix=cfg.get("train_mix", "cs1+cs2"),
        num_workers=num_workers,
    )
    print(f"    Done in {time.time()-t0:.1f}s")

    # Model
    print(f"  [2/5] Creating model (channels={cfg['hidden_channels']}, "
          f"energy={cfg['use_energy']}, lr={cfg['lr']:.1e})...")
    model = TweedieSolver(
        state_dim=3, hidden_channels=cfg["hidden_channels"],
        time_emb_dim=64, use_obs=True, use_energy=cfg["use_energy"],
        nu=1.0, K_inner=5, N_outer=10, dropout=0.1,
    ).to(device)

    # Stage 1
    print(f"  [3/5] Stage 1 ({cfg['epochs_stage1']} epochs, lr={cfg['lr']})...")
    t0 = time.time()
    orig_cwd = os.getcwd()
    os.chdir(exp_dir)
    try:
        model = train_stage1(model, loaders["train"], loaders["val"],
                             epochs=cfg["epochs_stage1"], lr=cfg["lr"], device=device)
        if os.path.exists("checkpoint_stage1.pt"):
            model.mean_estimator.load_state_dict(
                torch.load("checkpoint_stage1.pt", map_location=device))
    finally:
        os.chdir(orig_cwd)
    stage1_t = time.time() - t0
    print(f"    Done in {stage1_t:.1f}s")

    # Stage 2
    stage2_t = 0.0
    if not cfg.get("skip_stage2"):
        print(f"  [4/5] Stage 2 ({cfg['epochs_stage2']} epochs, lr={cfg['lr']})...")
        t0 = time.time()
        os.chdir(exp_dir)
        try:
            model = train_stage2(model, loaders["train"], loaders["val"],
                                 epochs=cfg["epochs_stage2"], lr=cfg["lr"], device=device)
            if os.path.exists("checkpoint_stage2.pt"):
                model.load_state_dict(
                    torch.load("checkpoint_stage2.pt", map_location=device))
        finally:
            os.chdir(orig_cwd)
        stage2_t = time.time() - t0
        print(f"    Done in {stage2_t:.1f}s")
    else:
        print(f"  [4/5] Skipped")

    # Evaluate
    print(f"  [5/5] Evaluating on test sets...")
    t0 = time.time()
    use_mean = cfg.get("skip_stage2", False)
    m1, s1 = evaluate_model(model, datasets["test_cs1"], device, use_mean_estimator=use_mean)
    m2, s2 = evaluate_model(model, datasets["test_cs2"], device, use_mean_estimator=use_mean)
    eval_t = time.time() - t0
    total_t = time.time() - total_t0

    deg = float(np.mean(m2) / (np.mean(m1) + 1e-10))
    result = {
        "experiment_id": exp_id,
        "note": exp["note"],
        "config": cfg,
        "total_time_seconds": total_t,
        "stage1_time_seconds": stage1_t,
        "stage2_time_seconds": stage2_t,
        "eval_time_seconds": eval_t,
        "fm_cs1": fmt_rmse(m1, s1),
        "fm_cs2": fmt_rmse(m2, s2),
        "fm_degradation": deg,
    }
    with open(results_path, "w") as f:
        json.dump(result, f, indent=2)

    print(f"\n  ── Results ─────────────────────────────────")
    print(f"  CS1: X={m1[0]:.4f} Y={m1[1]:.4f} Z={m1[2]:.4f}  mean={np.mean(m1):.4f}")
    print(f"  CS2: X={m2[0]:.4f} Y={m2[1]:.4f} Z={m2[2]:.4f}  mean={np.mean(m2):.4f}")
    print(f"  Degradation: {deg:.2f}x  |  Total: {total_t:.0f}s")
    # Regenerate synthesis report with latest results
    subprocess.run([sys.executable, "generate_report.py"], capture_output=True)
    return result

# ── Generate all datasets once ──────────────────────────────────

def generate_all_datasets(device):
    """Generate all datasets once. Caches to disk for reuse."""
    cache = os.path.join(EXP_DIR, "datasets.pt")
    if os.path.exists(cache):
        print("Loading cached datasets...")
        t0 = time.time()
        # Can't easily pickle Lorenz63Dataset due to tensor storage,
        # so we always regenerate. But the trajectory generation is the bottleneck.
        # Let's just regenerate.
        pass
    print("Generating all datasets (train/val/test, CS1+CS2)...")
    t0 = time.time()
    base_cfg = Lorenz63Config(T_max=3.0)
    datasets = make_mixed_datasets(base_cfg)
    elapsed = time.time() - t0
    print(f"  Done in {elapsed:.1f}s  "
          f"(train: {len(datasets['train_cs1'])+len(datasets['train_cs2'])}, "
          f"val: {len(datasets['val_cs1'])+len(datasets['val_cs2'])}, "
          f"test: {len(datasets['test_cs1'])+len(datasets['test_cs2'])})")
    return datasets, elapsed

# ── Main ────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", action="append", help="Run specific experiment(s)")
    parser.add_argument("--baselines-only", action="store_true", help="Cache baselines only")
    parser.add_argument("--regenerate-data", action="store_true",
                        help="Force data regeneration")
    parser.add_argument("--batch-size", type=int, default=32,
                        help="Batch size (default 32, LR scaled from config base)")
    parser.add_argument("--num-workers", type=int, default=4,
                        help="DataLoader workers (default 4)")
    parser.add_argument("--baseline-batch-size", type=int, default=1,
                        help="Batch size for baseline evaluation (default 1 = per-sample). "
                             "Use >1 (e.g., 64, 128) for GPU-efficient batched DA.")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dev_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    if device.type == "cuda":
        torch.set_float32_matmul_precision('high')
    print(f"Device: {device} ({dev_name})")
    print(f"Output: {EXP_DIR}")

    experiments = EXPERIMENTS
    if args.experiment:
        requested = set(args.experiment)
        experiments = [e for e in experiments if e["id"] in requested]
        missing = requested - {e["id"] for e in EXPERIMENTS}
        if missing:
            print(f"Warning: unknown experiments: {missing}")
        if not experiments:
            print("No matching experiments found.")
            return

    # Generate data once (used by baselines AND all experiments)
    data_gen_time = 0
    if os.path.exists(os.path.join(EXP_DIR, "datasets.pt")):
        print("Loading cached datasets...")
        datasets = torch.load(os.path.join(EXP_DIR, "datasets.pt"))
        print(f"  Loaded: train {len(datasets['train_cs1'])+len(datasets['train_cs2'])}, "
              f"val {len(datasets['val_cs1'])+len(datasets['val_cs2'])}, "
              f"test {len(datasets['test_cs1'])+len(datasets['test_cs2'])}")
    else:
        t0 = time.time()
        datasets = make_mixed_datasets(Lorenz63Config(T_max=3.0))
        data_gen_time = time.time() - t0
        print(f"Datasets generated in {data_gen_time:.1f}s")
        # Cache datasets
        torch.save(datasets, os.path.join(EXP_DIR, "datasets.pt"))

    # Baselines (cached separately)
    bl_path = os.path.join(EXP_DIR, "baselines.json")
    if os.path.exists(bl_path) and not args.baselines_only:
        with open(bl_path) as f:
            baselines = json.load(f)
        print(f"Baselines loaded from cache ({bl_path})")
    else:
        print()
        baselines = run_and_cache_baselines(datasets, device, batch_size=args.baseline_batch_size)

    if args.baselines_only:
        return

    # Run experiments
    print(f"\nExperiments to run: {[e['id'] for e in experiments]}")
    succeeded, failed = [], []
    for i, exp in enumerate(experiments):
        print(f"\n── [{i+1}/{len(experiments)}] ──", end="")
        try:
            r = run_single_experiment(exp, datasets, device,
                                      batch_size=args.batch_size,
                                      num_workers=args.num_workers)
            succeeded.append(r)
        except Exception as e:
            print(f"\n  ERROR in {exp['id']}: {e}")
            traceback.print_exc()
            failed.append(exp["id"])

    print(f"\n{'=' * 60}")
    print(f"  DONE: {len(succeeded)}/{len(experiments)} successful")
    if failed:
        print(f"  Failed: {failed}")
    print(f"\n  Run: python generate_report.py")

if __name__ == "__main__":
    main()
