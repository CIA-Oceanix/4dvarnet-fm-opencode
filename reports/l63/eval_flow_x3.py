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
from eval_flow_ensemble import score  # noqa: E402
from models.vanilla_cfm import DEFAULT_STEP_POWER, flow_tau_grid  # noqa: E402


class VelocityAverage:
    def __init__(self, models: list) -> None:
        self.models = models
        self.sigma_prior = float(models[0].sigma_prior)

    def sample(self, batch, N_outer: int, step_power: float = DEFAULT_STEP_POWER) -> torch.Tensor:
        obs = batch.obs
        B, T, _ = obs.shape
        x = torch.randn_like(obs) * self.sigma_prior
        taus = flow_tau_grid(N_outer, step_power)
        for t0, t1 in zip(taus[:-1], taus[1:]):
            tau_step = torch.full((B,), t0, device=obs.device)
            mu = torch.stack([m.forward(x, batch, tau_step) for m in self.models]).mean(0)
            v = (mu - x) / (1.0 - tau_step.clamp(max=0.999).view(B, 1, 1).expand(-1, T, -1))
            x = x + (t1 - t0) * v
        return x


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
    loaded = [load_model(os.path.join(ROOT, "experiments", f"{args.name}_seed{s}"), args.ckpt, device) for s in args.seeds]
    ens = VelocityAverage([m for m, _ in loaded])
    cfg = loaded[0][1]
    out: dict[str, dict] = {}
    for case in ("s0", "s1"):
        torch.manual_seed(0)
        r = score(ens, cfg, ds[f"test_{case}"], device, args.n_members, args.n_outer, args.step_power)
        print(f"  x3 {case}: RMSE {r['rmse'].mean():.4f} CRPS {r['crps'].mean():.4f} "
              f"spread/RMSE {r['spread'].mean() / r['rmse'].mean():.2f}", flush=True)
        out[case] = {k: r[k].tolist() for k in ("rmse", "crps", "spread")}
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs", f"{args.name}_x3_ens{args.n_members}_no{args.n_outer}.json")
    with open(path, "w") as f:
        json.dump(out, f)
    print("wrote", path)


if __name__ == "__main__":
    main()
