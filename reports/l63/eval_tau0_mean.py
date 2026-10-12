import argparse
import json
import os
import sys

import numpy as np
import torch

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from eval_directunet import load_model  # noqa: E402
from evaluation.psi_decomposition import psi_mean_estimate  # noqa: E402
from train import _make_eval_batch  # noqa: E402


@torch.no_grad()
def score(model, cfg, ds, device: torch.device, m_draws: int) -> dict[str, np.ndarray]:
    sigma = float(model.sigma_prior)
    rmse, mae, disp = [], [], []
    for i in range(len(ds)):
        w = ds[i]
        batch = _make_eval_batch(w, device, param_dim=cfg.model.get("param_dim", 0), obs_var_indices=None)
        x1 = batch.states.contiguous()
        m, spread_sq = psi_mean_estimate(model, batch, x1, sigma, m_draws)
        pred = m.float().cpu().numpy()[0]
        truth = w["true_state"].numpy()
        rmse.append(float(np.sqrt(((pred - truth) ** 2).mean(0)).mean()))
        mae.append(float(np.abs(pred - truth).mean()))
        disp.append(spread_sq / x1.numel())
    return {"rmse": np.array(rmse), "mae": np.array(mae), "disp": np.array(disp)}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--name", required=True)
    p.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    p.add_argument("--ckpt", default="stage1_best.ckpt")
    p.add_argument("--m-draws", type=int, default=30)
    p.add_argument("--testsets", default=os.path.join(ROOT, "experiments", "l63_testsets_obs30_n200.pt"))
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ds = torch.load(args.testsets, weights_only=False)
    out: dict[str, dict] = {}
    for case in ("s0", "s1"):
        per_seed = []
        for s in args.seeds:
            torch.manual_seed(s)
            model, cfg = load_model(os.path.join(ROOT, "experiments", f"{args.name}_seed{s}"), args.ckpt, device)
            r = score(model, cfg, ds[f"test_{case}"], device, args.m_draws)
            print(f"  seed {s} {case}: tau=0 mean RMSE {r['rmse'].mean():.4f}", flush=True)
            per_seed.append(r)
        out[case] = {k: np.mean([r[k] for r in per_seed], 0).tolist() for k in ("rmse", "mae", "disp")}
        out[case]["seed_means"] = [float(r["rmse"].mean()) for r in per_seed]
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs", f"{args.name}_tau0_mean.json")
    with open(path, "w") as f:
        json.dump(out, f)
    print("wrote", path)


if __name__ == "__main__":
    main()
