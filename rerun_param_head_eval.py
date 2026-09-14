#!/usr/bin/env python3
"""Regenerate the C1/C2/C3 param_head `results.json` param-RMSE tables.

The published per-parameter RMSE in these experiment `results.json` files was
computed by the train-script eval, whose true-param extraction read scalar
`true_w1..true_w4` keys. The cached L96 test windows store the fast weights
only as the `true_fast_weights` list, so all four fast-weight channels were
compared against a silent 0.0 (hence RMSE ~= the parameter's own magnitude).
``train.py`` now uses the list-aware ``_l96_true_param_vector`` helper; this
script re-runs each model's eval on the identical cached S0/S1 test set with
the fixed extraction and rewrites only ``param_rmse_s0`` / ``param_rmse_s1``,
leaving every other field of ``results.json`` untouched. No retraining.
"""
import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from evaluation.run_l96 import make_obs_j_indices
from train import evaluate_model, model_factory

EXPERIMENTS = [
    "C1_stateparam_head_s1",
    "C2_stateparam_head_state_true",
    "C3_param_head_true_deriv",
    "C4a_param_head_unet_true",
    "C4b_param_head_unet_l1b",
]

TEST_CACHE = "experiments/l96_datasets_obsj2_int100_nwin200.pt"
OBS_J = 2
NO, J = 8, 4
PARAM_NAMES = ["F", "c1", "hx", "eps", "w1", "w2", "w3", "w4"]


def build_model(exp: str, device):
    from hydra import compose, initialize_config_dir
    root = os.path.dirname(os.path.abspath(__file__))
    with initialize_config_dir(version_base=None, config_dir=os.path.join(root, "config")):
        cfg = compose(config_name="experiment/" + exp)
    model = model_factory(cfg, device)
    ckpt = torch.load(f"experiments/{exp}/checkpoints/stage1_best.ckpt",
                      map_location="cpu", weights_only=False)
    sd = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
    if all(k.startswith("model.") for k in sd):
        sd = {k[len("model."):]: v for k, v in sd.items()}
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing or unexpected:
        raise RuntimeError(f"{exp}: missing={missing}, unexpected={unexpected}")
    model.eval()
    return model


def run_eval(model, device, datasets):
    ovi = make_obs_j_indices(NO, J, OBS_J)
    out = {}
    for case in ("test_s0", "test_s1"):
        _, _, prmse = evaluate_model(
            model, datasets[case], device, "param_head",
            return_params=True, param_names=PARAM_NAMES,
            param_dim=8, obs_var_indices=ovi, use_biased_params=True,
        )
        out[case] = {nm: float(p) for nm, p in zip(PARAM_NAMES, prmse)}
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--caseless", action="store_true",
                        help="Print a compact table and skip rewriting")
    parser.add_argument("--exp", default=None, help="single experiment dir name")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(0)
    datasets = torch.load(TEST_CACHE, map_location="cpu", weights_only=False)

    exps = [args.exp] if args.exp else EXPERIMENTS
    for exp in exps:
        model = build_model(exp, device)
        res = run_eval(model, device, datasets)
        if args.caseless:
            print(f"== {exp} ==")
            for case, pr in res.items():
                print(f"  {case}: {np.round(list(pr.values()), 4)}")
            continue
        path = f"experiments/{exp}/results.json"
        with open(path) as f:
            orig = json.load(f)
        orig["param_rmse_s0"] = res["test_s0"]
        orig["param_rmse_s1"] = res["test_s1"]
        with open(path, "w") as f:
            json.dump(orig, f, indent=2)
        print(f"updated {path}")


if __name__ == "__main__":
    main()
