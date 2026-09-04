"""Shared dataset-construction logic for train.py and generate_dataset.py.

This is a straight extraction of train.py's data block so that both the
training entry point and the standalone cache-warmup script
(generate_dataset.py) build datasets identically and can never drift apart.
"""
import os
import logging

import numpy as np
import torch
from omegaconf import DictConfig

logger = logging.getLogger(__name__)


def build_datasets(cfg: DictConfig):
    """Build train/val/test (S0/S1) datasets for the system in cfg.data.

    Returns (datasets, test_keys, base_cfg, system, obs_var_indices).
    `obs_var_indices` is None for systems that don't use partial observation
    of sub-components (currently only lorenz96 does).
    """
    dc = cfg.data
    system = dc.get("system", "lorenz63")

    if system == "lorenz96":
        from data.lorenz96 import Lorenz96Config, make_l96_s0_s1_trainval
        NO = dc.get("NO", 8)
        J = dc.get("J", 4)
        obs_j = dc.get("obs_j", 2)
        obs_var_indices = None
        if obs_j < J:
            X_idx = list(range(NO))
            Y_idx = []
            for k in range(NO):
                for j in range(obs_j):
                    Y_idx.append(NO + k * J + j)
            obs_var_indices = tuple(X_idx + Y_idx)
        base_cfg = Lorenz96Config(
            case=dc.get("case", 1), dt=dc.dt, T_max=dc.T_max,
            obs_interval=dc.obs_interval, R_var=dc.R_var, B_var=dc.B_var,
            param_bias=dc.get("param_bias", 0.0),
            num_windows=dc.get("num_train_windows", 2000), window_spacing=dc.window_spacing,
            spinup_steps=dc.spinup_steps, seed=dc.get("seed", 42),
            NO=dc.get("NO", 8), J=dc.get("J", 4),
            h=dc.get("h", 1.0), hx=dc.get("hx", 1.0), eps=dc.get("eps", 0.1),
            F_true=dc.get("F_true", 8.0), F_da=dc.get("F_da", 8.0),
            gamma=dc.get("gamma", 0.05), W_L_bar=dc.get("W_L_bar", 0.0),
            c1=dc.get("c1", 1.0), c2=dc.get("c2", 0.1),
            sigma_0=dc.get("sigma_0", 0.08), sigma_L=dc.get("sigma_L", 0.20),
            tau_eta=dc.get("tau_eta", 5.0),
            sigma_eta=dc.get("sigma_eta", np.sqrt(0.5)),
            forcing_state_bias=dc.get("forcing_state_bias", 0.0),
            forcing_coupling=dc.get("forcing_coupling", "linear"),
            coupling_exponent_truth=dc.get("coupling_exponent_truth", 1.6),
            fast_weights=list(dc.get("fast_weights", [1.0, 1.0, 0.1, 0.1])),
            randomize=dict(dc.get("randomize", {})),
            obs_var_indices=obs_var_indices,
        )
        smoke_cached_data = dc.get("smoke_cached_data", None)
        if smoke_cached_data is not None:
            logger.info(f"Loading cached data from: {smoke_cached_data}")
            cached = torch.load(smoke_cached_data, weights_only=False)
            datasets = cached
            logger.info(f"  train: {len(cached['train'])} windows, val: {len(cached['val'])} windows")
            test_keys = ["test_s0", "test_s1"]
        else:
            test_cache_path = dc.get("test_cache", None)
            cached_test = None
            if test_cache_path and os.path.exists(test_cache_path):
                logger.info(f"Reusing cached test splits from {test_cache_path}")
                cached_full = torch.load(test_cache_path, weights_only=False)
                cached_test = {k: cached_full[k] for k in ("test_s0", "test_s1")
                               if k in cached_full}
            datasets = make_l96_s0_s1_trainval(
                base_cfg,
                num_train_windows=dc.get("num_train_windows", 1000),
                num_val_windows=dc.get("num_val_windows", 100),
                num_test_windows=dc.get("num_test_windows", 200),
                param_noise=dc.get("test_param_noise", 0.2),
                bias_range=(0.0, dc.get("bias_max", 0.2)),
                cached_datasets=cached_test,
                train_seed=dc.get("train_seed", None),
                val_seed=dc.get("val_seed", None),
                s0_seed=dc.get("s0_seed", None),
                s1_seed=dc.get("s1_seed", None),
                require_cache=dc.get("require_cache", False),
            )
            test_keys = ["test_s0", "test_s1"]
    else:
        from data.lorenz63 import Lorenz63Config, make_s0_s1_trainval
        obs_var_indices = None
        base_cfg = Lorenz63Config(
            dt=dc.dt, T_max=dc.T_max, obs_interval=dc.obs_interval,
            R_var=dc.R_var, B_var=dc.B_var,
            num_windows=dc.get("num_train_windows", 2000), window_spacing=dc.window_spacing,
            spinup_steps=dc.spinup_steps, seed=dc.get("seed", 42),
            sigma_true=dc.sigma_true, rho_true=dc.rho_true, beta_true=dc.beta_true,
            gamma=dc.gamma, W_L_bar=dc.W_L_bar, c1=dc.c1, c2=dc.c2,
            sigma_0=dc.sigma_0, sigma_L=dc.sigma_L,
            tau_eta=dc.tau_eta, sigma_eta=dc.sigma_eta,
            param_bias=dc.get("param_bias", 0.0),
            forcing_state_bias=dc.get("forcing_state_bias", 0.0),
            forcing_coupling=dc.get("forcing_coupling", "linear"),
            coupling_exponent_truth=dc.get("coupling_exponent_truth", 1.6),
        )
        smoke_cached_data = dc.get("smoke_cached_data", None)
        if smoke_cached_data is not None:
            logger.info(f"Loading cached data from: {smoke_cached_data}")
            cached = torch.load(smoke_cached_data, weights_only=False)
            datasets = cached
            logger.info(f"  train: {len(cached['train'])} windows, val: {len(cached['val'])} windows")
            test_keys = ["test_s0", "test_s1"]
        else:
            bias_max = dc.get("bias_max", 0.2)
            datasets = make_s0_s1_trainval(
                base_cfg,
                num_train_windows=dc.get("num_train_windows", 1000),
                num_val_windows=dc.get("num_val_windows", 100),
                num_test_windows=dc.get("num_test_windows", 200),
                param_noise=dc.get("test_param_noise", 0.2),
                bias_range=(0.0, bias_max),
                train_seed=dc.get("train_seed", None),
                val_seed=dc.get("val_seed", None),
                s0_seed=dc.get("s0_seed", None),
                s1_seed=dc.get("s1_seed", None),
                require_cache=dc.get("require_cache", False),
            )
            test_keys = ["test_s0", "test_s1"]

    return datasets, test_keys, base_cfg, system, obs_var_indices
