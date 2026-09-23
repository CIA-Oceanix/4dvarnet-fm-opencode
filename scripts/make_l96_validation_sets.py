"""Validation windows for tuning evaluation-time settings (SDA guidance
weight, hybrid warm start, flow calibration) WITHOUT touching the test set.

New S0/S1 windows generated exactly like the P1 test splits
(``data.lorenz96.make_l96_s0_s1_trainval``'s test_s0 / test_s1 builders, same
config as ``evaluate_all_l96.py``'s P1 run) but with window seeds 1057 / 1067
instead of 123 / 131 -- residues mod 100 disjoint from train (42), val (99)
and test (23, 31), so no window can coincide. Two files:

- ``l96_valset_regular_w<N>.pt``: the regular 30-obs grid, as generated;
- ``l96_valset_rlayout_n10-100_k4-16_w<N>.pt``: the same windows re-observed
  with the canonical random observing system (``draw_layout``: stratified
  n_obs, step 0 observed, >=1 obs per DA window, 4-16 fast channels).

Each gets a per-field SHA-256 manifest like the canonical test set.

  python scripts/make_l96_validation_sets.py --n-windows 50
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from data.lorenz96 import (Lorenz96Config, RandomBiasLorenz96Dataset,  # noqa: E402
                           RandomParamLorenz96Dataset, _make_lorenz96_dynamics)
from eval_da_random_layout_l96 import _layout_seed, draw_layout  # noqa: E402
from evaluation.run_l96 import make_obs_j_indices  # noqa: E402
from scripts.make_l96_canonical_testset import fingerprint, sha256_file  # noqa: E402

SHARED = "/Odyssey/private/rfablet/Python/4dvarnet-fm-opencode/experiments"
RANDOMIZE = {k: {"randomized": True, "noise": 0.2, "biased": True, "bias": 0.1}
             for k in ("F", "c1", "hx", "eps", "fast_weights")}
RANDOMIZE["h"] = {"randomized": False, "noise": 0.2, "biased": False, "bias": 0.0}
SEEDS = {"test_s0": 1057, "test_s1": 1067}


def base_config() -> Lorenz96Config:
    return Lorenz96Config(
        dt=0.001, T_max=3.0, obs_interval=100, R_var=0.5, B_var=2.0,
        num_windows=2000, window_spacing=2000, spinup_steps=10000, seed=42,
        NO=8, J=4, h=1.0, hx=1.0, eps=0.1, F_true=8.0, F_da=8.0,
        gamma=0.05, W_L_bar=0.0, c1=1.0, c2=0.1, sigma_0=0.08, sigma_L=0.20,
        tau_eta=5.0, sigma_eta=np.sqrt(0.5), param_bias=0.0, forcing_state_bias=0.0,
        fast_weights=[1.0, 1.0, 0.1, 0.1], randomize=RANDOMIZE,
        obs_var_indices=make_obs_j_indices(8, 4, 2))


def build(n_windows: int) -> dict:
    cfg = base_config()
    dynamics = _make_lorenz96_dynamics(cfg)
    s0 = RandomParamLorenz96Dataset(
        Lorenz96Config(**{**cfg.__dict__, "seed": SEEDS["test_s0"], "num_windows": n_windows, "case": 1,
                          "param_bias": 0.0, "forcing_state_bias": 0.0}),
        dynamics=dynamics, fast_generation=False, param_noise=0.2)
    s1 = RandomBiasLorenz96Dataset(
        Lorenz96Config(**{**cfg.__dict__, "seed": SEEDS["test_s1"], "num_windows": n_windows, "case": 1,
                          "param_bias": 0.1, "forcing_state_bias": 0.1}),
        dynamics=dynamics, fast_generation=False, param_noise=0.2, bias_mode="fixed")
    return {"test_s0": s0, "test_s1": s1}


def save(data: dict, path: str, extra: dict) -> None:
    torch.save(data, path)
    n = len(data["test_s0"])
    manifest = {"testset": os.path.abspath(path), "sha256": sha256_file(path), "n_windows": n,
                "fields": {k[5:]: fingerprint([data[k][i] for i in range(n)]) for k in ("test_s0", "test_s1")},
                **extra}
    with open(path + ".manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"wrote {path}  sha256 {manifest['sha256'][:12]}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--n-windows", type=int, default=50)
    p.add_argument("--out-dir", default=SHARED)
    args = p.parse_args()

    data = build(args.n_windows)
    reg = os.path.join(args.out_dir, f"l96_valset_regular_w{args.n_windows}.pt")
    save(data, reg, {"obs": "regular 30-obs grid", "seeds": SEEDS})

    idx = list(make_obs_j_indices(8, 4, 2))
    n_range, k_range = (10, 100), (4, 16)
    for key in ("test_s0", "test_s1"):
        for i in range(args.n_windows):
            w = data[key][i]
            obs, mask = draw_layout(w["true_state"][:, idx].float(), n_range, k_range,
                                    _layout_seed(f"val_{key[5:]}", i, n_range, k_range, 0), 500)
            w["obs"] = obs.to(w["obs"].dtype)
            w["obs_mask"] = mask.to(w["obs_mask"].dtype)
    rl = os.path.join(args.out_dir, f"l96_valset_rlayout_n10-100_k4-16_w{args.n_windows}.pt")
    save(data, rl, {"obs": "random layout n10-100 k4-16 (draw_layout, step 0, >=1 obs/DA window)",
                    "seeds": SEEDS, "n_obs_range": list(n_range), "fast_range": list(k_range)})


if __name__ == "__main__":
    main()
