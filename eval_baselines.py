#!/usr/bin/env python3
"""
4DVarNet-FM: Baseline evaluation with Hydra config management.
Usage:
    python eval_baselines.py                                         # defaults
    python eval_baselines.py baselines.da_window_steps=20            # override DWS
    python eval_baselines.py baselines.batch_size=128                # batch size
"""
import os
import sys
import time
import torch
import hydra
from omegaconf import DictConfig, OmegaConf

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data.build import build_datasets
from evaluation.run import run_and_cache_baselines


@hydra.main(config_path="config", config_name="lorenz63", version_base="1.3")
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dev_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    print(f"Device: {device} ({dev_name})")

    # Build train/val/test_s0/test_s1 datasets, reusing the on-disk
    # dataset_cache/ (same cache generate_dataset.py warms up) keyed by a
    # hash of the data config -- so a config change (e.g. obs_interval)
    # regenerates automatically instead of silently reusing a stale dataset.
    t0 = time.time()
    datasets, test_keys, base_cfg, system, obs_var_indices = build_datasets(cfg)
    print(f"Datasets ready in {time.time()-t0:.1f}s")
    total_test = sum(len(datasets[k]) for k in test_keys if k in datasets)
    print(f"  train={len(datasets['train'])}, val={len(datasets['val'])}, test={total_test}")

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
        "N_ensemble": bc.enkf.N_ensemble,
        "inflation": bc.enkf.inflation,
    } if "enkf" in bc else {}
    etkf_config = {
        "N_ensemble": bc.etkf.N_ensemble,
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
