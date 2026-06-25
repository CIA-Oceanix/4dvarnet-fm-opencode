import torch
import numpy as np
from data.lorenz63 import Lorenz63Config
from evaluation.baselines import Weak4DVar, Strong4DVar, EnKF
from evaluation.metrics import rmse, print_metrics_table
from models.solver import TweedieSolver
from training.stage1 import train_stage1
from training.stage2 import train_stage2


def run_baseline(
    method,
    dataset,
    cfg: Lorenz63Config,
    device: torch.device = torch.device("cpu"),
):
    sig, rho, bet = cfg.da_params
    w = dataset[0]
    obs = w["obs"]
    mask = w["obs_mask"]
    truth = w["true_state"]
    if cfg.use_corrupted_forcing:
        force = w["forcing_corrupted"]
    else:
        force = w["forcing_true"]

    return method.assimilate(obs, mask, force, truth, sigma=sig, rho=rho, beta=bet)


def run_baselines(
    dataset,
    cfg: Lorenz63Config,
    device: torch.device = torch.device("cpu"),
) -> dict:
    w4d = Weak4DVar(dt=cfg.dt, device=device)
    s4d = Strong4DVar(dt=cfg.dt, device=device)
    enkf = EnKF(dt=cfg.dt, device=device)

    return {
        "Weak-4DVar": run_baseline(w4d, dataset, cfg, device),
        "Strong-4DVar": run_baseline(s4d, dataset, cfg, device),
        "EnKF": run_baseline(enkf, dataset, cfg, device),
    }


def run_full_experiment(cfg: Lorenz63Config, device: torch.device = torch.device("cpu")):
    from data.lorenz63 import make_datasets

    print("Generating datasets...")
    datasets = make_datasets(cfg)

    print("\nRunning Case Study 1 (noise-free)...")
    cfg_cs1 = Lorenz63Config(case=1, param_bias=0.0, seed=cfg.seed)
    cs1_results = run_baselines(datasets["test_cs1"], cfg_cs1, device)

    print("\nRunning Case Study 2 (noisy)...")
    cfg_cs2 = Lorenz63Config(case=2, param_bias=cfg.param_bias, seed=cfg.seed)
    cs2_results = run_baselines(datasets["test_cs2"], cfg_cs2, device)

    print_metrics_table(cs1_results, "CASE STUDY 1: Noise-free forcings & parameters")
    print_metrics_table(cs2_results, "CASE STUDY 2: Noisy forcings & biased parameters")

    print(f"\n{'Method':<20} {'CS1 RMSE':<12} {'CS2 RMSE':<12} {'Degradation':<12}")
    print(f"{'-' * 56}")
    for name in cs1_results:
        r1 = np.mean(cs1_results[name].rmse)
        r2 = np.mean(cs2_results[name].rmse)
        deg = r2 / (r1 + 1e-10)
        print(f"{name:<20} {r1:<12.4f} {r2:<12.4f} {deg:<12.2f}x")

    return cs1_results, cs2_results
