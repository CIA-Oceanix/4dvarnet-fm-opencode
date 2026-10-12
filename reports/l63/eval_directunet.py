import argparse
import json
import os
import sys

import numpy as np
import torch
from omegaconf import OmegaConf

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from generate_da_baselines_table import build_datasets  # noqa: E402
from train import _make_eval_batch, model_factory  # noqa: E402


def load_model(exp_dir: str, ckpt: str, device: torch.device):
    cfg = OmegaConf.load(os.path.join(exp_dir, "resolved_config.yaml"))
    model = model_factory(cfg, device)
    state = torch.load(os.path.join(exp_dir, "checkpoints", ckpt), map_location="cpu", weights_only=False)
    state = state.get("state_dict", state)
    state = {k.removeprefix("model.").replace("_orig_mod.", ""): v for k, v in state.items()}
    model.load_state_dict(state)
    return model.to(device).eval(), cfg


@torch.no_grad()
def score(model, cfg, ds, device: torch.device) -> dict[str, np.ndarray]:
    rmse, ratio = [], []
    for i in range(len(ds)):
        w = ds[i]
        batch = _make_eval_batch(w, device, param_names=("sigma", "rho", "beta", "c1"),
                                 param_dim=cfg.model.get("param_dim", 4), use_biased_params=False,
                                 obs_var_indices=None, with_true_forcing=False)
        pred = model(batch).detach().cpu().numpy()[0]
        truth = w["true_state"].numpy()
        rmse.append(float(np.sqrt(((pred - truth) ** 2).mean(0)).mean()))
        ratio.append(float(pred.var(0).mean() / truth.var(0).mean()))
    return {"rmse": np.array(rmse), "var_ratio": np.array(ratio)}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--name", default="L63B_directunet_monaiM")
    p.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    p.add_argument("--ckpt", default="stage1_best.ckpt")
    p.add_argument("--obs-interval", type=int, default=10)
    p.add_argument("--n-test", type=int, default=200)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cache = os.path.join(ROOT, "experiments", f"l63_testsets_obs{300 // args.obs_interval}_n{args.n_test}.pt")
    if os.path.exists(cache):
        ds = torch.load(cache, weights_only=False)
    else:
        ds = build_datasets(args.n_test, 1, args.obs_interval)
        torch.save({k: v for k, v in ds.items() if k.startswith("test")}, cache)
    out: dict[str, dict] = {}
    for case in ("s0", "s1"):
        per_seed = []
        for s in args.seeds:
            model, cfg = load_model(os.path.join(ROOT, "experiments", f"{args.name}_seed{s}"), args.ckpt, device)
            r = score(model, cfg, ds[f"test_{case}"], device)
            print(f"  seed {s} {case}: RMSE {r['rmse'].mean():.4f}", flush=True)
            per_seed.append(r)
        out[case] = {
            "rmse": np.mean([r["rmse"] for r in per_seed], 0).tolist(),
            "var_ratio": np.mean([r["var_ratio"] for r in per_seed], 0).tolist(),
            "seed_means": [float(r["rmse"].mean()) for r in per_seed],
        }
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs", f"{args.name}_eval.json")
    with open(path, "w") as f:
        json.dump(out, f)
    print("wrote", path)


if __name__ == "__main__":
    main()
