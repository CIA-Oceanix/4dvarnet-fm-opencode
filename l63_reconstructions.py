import json
import os

import numpy as np
import torch

from data.normalization import load_norm_stats
from eval_da_l63 import make_dataset, make_method, run_method
from eval_neural_l63 import NORM_STATS, checkpoints, run as run_neural
from evaluation.neural_inference import load_model

BASE = os.path.dirname(os.path.abspath(__file__))
TABLE_DIR = os.path.join(BASE, "experiments", "l63_da_table2")
COMMITTED_TABLE_DIR = os.path.join(BASE, "l63_results")
METHODS = ("ETKS", "D-Unet", "CFM", "SDA2", "D-Unet+SDA2")
TEST_SEED = {"s0": 123, "s1": 131}


def cache_path(case, obs_interval):
    return os.path.join(TABLE_DIR, f"reconstructions_obs{obs_interval}_{case}.npz")


def table_path(name):
    path = os.path.join(TABLE_DIR, name)
    return path if os.path.exists(path) else os.path.join(COMMITTED_TABLE_DIR, name)


def per_window_rmse(estimate, truth):
    return np.sqrt(((estimate - truth) ** 2).mean(axis=1)).mean(axis=-1)


def load_or_compute(case, obs_interval=10, n_members=30, num_windows=200, batch_size=50, seed=0):
    path = cache_path(case, obs_interval)
    if os.path.exists(path):
        return dict(np.load(path))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ds = make_dataset(case, TEST_SEED[case], num_windows, obs_interval)
    etks_table = json.load(open(table_path(f"table2_l63_da_obs{obs_interval}.json")))
    neural_table = json.load(open(table_path(f"table2_l63_neural_obs{obs_interval}.json")))
    out = {
        "truth": np.stack([ds[i]["true_state"].numpy() for i in range(len(ds))]),
        "obs": np.stack([ds[i]["obs"].numpy() for i in range(len(ds))]),
    }
    torch.manual_seed(seed)
    etks = make_method("ETKS", case, device, inflation=etks_table[case]["ETKS"]["hyperparameters"]["inflation"],
                       n_members=n_members)
    out["members_ETKS"] = run_method(etks, ds, case, device, batch_size)["members"].astype(np.float32)
    stats = load_norm_stats(NORM_STATS)
    prefix = "L63B" if obs_interval == 10 else f"L63B{obs_interval}"
    models = {k: load_model(c, cfg, device=device)[0] for k, (c, cfg) in checkpoints(prefix).items()}
    for method in METHODS[1:]:
        hp = neural_table[case][method]["hyperparameters"]
        res = run_neural(method, models, ds, device, stats, 1 if method == "D-Unet" else n_members,
                         batch_size, seed=seed, **hp)
        members = res["members"] if "members" in res else res["traj"][:, None]
        out[f"members_{method}"] = members.astype(np.float32)
    np.savez_compressed(path, **out)
    return out
