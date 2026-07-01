#!/usr/bin/env python3
"""
4DVarNet-FM: Baseline evaluation with Hydra config management.
Usage:
    python eval_baselines.py                                         # defaults
    python eval_baselines.py baselines.da_window_steps=20            # override DWS
    python eval_baselines.py --config-name baselines/dws20           # preset
    python eval_baselines.py baselines.batch_size=128                # batch size
"""
import os
import sys
import time
import torch
import hydra
from omegaconf import DictConfig, OmegaConf

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data.lorenz63 import Lorenz63Config, make_mixed_datasets
from evaluation.run import run_and_cache_baselines, EXP_DIR


@hydra.main(config_path="config", config_name="lorenz63_default", version_base="1.3")
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dev_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    print(f"Device: {device} ({dev_name})")

    # Build Lorenz63Config from Hydra data config
    dc = cfg.data
    base_cfg = Lorenz63Config(
        dt=dc.dt, T_max=dc.T_max, obs_interval=dc.obs_interval,
        R_var=dc.R_var, B_var=dc.B_var,
        num_windows=dc.num_windows, window_spacing=dc.window_spacing,
        spinup_steps=dc.spinup_steps, seed=dc.get("seed", 42),
        param_bias=dc.get("param_bias", 0.0),
        forcing_state_bias=dc.get("forcing_state_bias", 0.0),
        forcing_coupling=dc.get("forcing_coupling", "linear"),
    )

    # Generate or load cached datasets
    datasets_cache = os.path.join(EXP_DIR, "datasets.pt")
    if os.path.exists(datasets_cache):
        t0 = time.time()
        datasets = torch.load(datasets_cache)
        print(f"Loaded cached datasets in {time.time()-t0:.1f}s")
    else:
        t0 = time.time()
        datasets = make_mixed_datasets(base_cfg, include_sparse_obs_test=True)
        print(f"Datasets generated in {time.time()-t0:.1f}s")
        torch.save(datasets, datasets_cache)
    total_train = len(datasets["train_cs1"]) + len(datasets["train_cs2"])
    total_val = len(datasets["val_cs1"]) + len(datasets["val_cs2"])
    total_test = sum(len(datasets[k]) for k in datasets if k.startswith("test_"))
    print(f"  train={total_train}, val={total_val}, test={total_test}")

    # Read baseline config
    bc = cfg.baselines
    weak_config = {
        "opt_steps": bc.weak4dvar.opt_steps,
        "lr": bc.weak4dvar.lr,
    } if "weak4dvar" in bc else {}
    strong_config = {
        "max_iter": bc.strong4dvar.max_iter,
        "lr": bc.strong4dvar.lr,
    } if "strong4dvar" in bc else {}
    enkf_config = {
        "N_ensemble": bc.N_ensemble,
        "inflation": bc.enkf.inflation,
    } if "enkf" in bc else {}
    etkf_config = {
        "N_ensemble": bc.N_ensemble,
        "inflation": bc.etkf.inflation,
    } if "etkf" in bc else {}

    # Run baselines
    run_and_cache_baselines(
        datasets, device,
        batch_size=bc.get("batch_size", 1),
        da_window_steps=bc.da_window_steps,
        weak_config=weak_config,
        strong_config=strong_config,
        enkf_config=enkf_config,
        etkf_config=etkf_config,
    )


if __name__ == "__main__":
    main()
