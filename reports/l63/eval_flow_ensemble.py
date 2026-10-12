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
from evaluation.metrics import crps  # noqa: E402
from train import _make_eval_batch  # noqa: E402


@torch.no_grad()
def score(model, cfg, ds, device: torch.device, n_members: int, n_outer: int, step_power: float) -> dict[str, np.ndarray]:
    rmse, crps_w, spread = [], [], []
    for i in range(len(ds)):
        w = ds[i]
        batch = _make_eval_batch(w, device, param_dim=cfg.model.get("param_dim", 0), obs_var_indices=None)
        ens = np.stack([
            model.sample(batch, N_outer=n_outer, step_power=step_power).float().cpu().numpy()[0]
            for _ in range(n_members)
        ])
        truth = w["true_state"].numpy()
        mean = ens.mean(0)
        rmse.append(float(np.sqrt(((mean - truth) ** 2).mean(0)).mean()))
        crps_w.append(float(np.mean(crps(ens, truth))))
        spread.append(float(np.sqrt(ens.var(0, ddof=1).mean(-1)).mean()))
    return {"rmse": np.array(rmse), "crps": np.array(crps_w), "spread": np.array(spread)}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--name", required=True)
    p.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    p.add_argument("--ckpt", default="stage1_best.ckpt")
    p.add_argument("--n-members", type=int, default=30)
    p.add_argument("--n-outer", type=int, default=20)
    p.add_argument("--step-power", type=float, default=0.5)
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
            r = score(model, cfg, ds[f"test_{case}"], device, args.n_members, args.n_outer, args.step_power)
            print(f"  seed {s} {case}: RMSE {r['rmse'].mean():.4f} CRPS {r['crps'].mean():.4f} "
                  f"spread/RMSE {r['spread'].mean() / r['rmse'].mean():.2f}", flush=True)
            per_seed.append(r)
        out[case] = {k: np.mean([r[k] for r in per_seed], 0).tolist() for k in ("rmse", "crps", "spread")}
        out[case]["seed_means"] = [float(r["rmse"].mean()) for r in per_seed]
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs", f"{args.name}_ens{args.n_members}_no{args.n_outer}.json")
    with open(path, "w") as f:
        json.dump(out, f)
    print("wrote", path)


if __name__ == "__main__":
    main()
