#!/usr/bin/env python3
"""
Builds the datasets for the Lorenz63 and Lorenz96 systems, and caches them in local disk 
Useful for comparing the performance of different models on the exact same dataset without having to regenerate it each time

Usage:
    python generate_dataset.py                                          # both lorenz63 and lorenz96 (default configs)
    python generate_dataset.py --config-name lorenz63                   # lorenz63 only
    python generate_dataset.py --config-name lorenz96                   # lorenz96 only
"""

import os
import sys

import hydra
from omegaconf import DictConfig, OmegaConf

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data.build import build_datasets

_DEFAULT_CONFIG_NAMES = ["lorenz63", "lorenz96"]


def _parse_config_name_and_overrides():
    """Split argv into an explicit --config-name (if any) and Hydra overrides.

    Returns (config_name_or_None, overrides).
    """
    args = sys.argv[1:]
    config_name = None
    overrides = []
    i = 0
    while i < len(args):
        a = args[i]
        if a in ("--config-name", "-cn"):
            config_name = args[i + 1]
            i += 2
        elif a.startswith("--config-name="):
            config_name = a.split("=", 1)[1]
            i += 1
        elif a.startswith("-cn="):
            config_name = a.split("=", 1)[1]
            i += 1
        else:
            overrides.append(a)
            i += 1
    return config_name, overrides


def build_and_report(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))

    datasets, test_keys, base_cfg, system, obs_var_indices = build_datasets(cfg)

    print(f"\n  ── Dataset sizes ({system}) ─────────────────────────────")
    for key, ds in datasets.items():
        print(f"  {key}: {len(ds)} windows")


def main():
    config_name, overrides = _parse_config_name_and_overrides()
    config_names = [config_name] if config_name else _DEFAULT_CONFIG_NAMES

    with hydra.initialize(config_path="config", version_base="1.3"):
        for name in config_names:
            print(f"\n{'=' * 60}\n  Building dataset for --config-name {name}\n{'=' * 60}")
            cfg = hydra.compose(config_name=name, overrides=overrides)
            build_and_report(cfg)


if __name__ == "__main__":
    main()
