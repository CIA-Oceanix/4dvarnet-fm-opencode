import argparse
import json
import os
import sys

import numpy as np
import torch

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data.dataloader import FlowMatchingBatch  # noqa: E402
from eval_directunet import load_model  # noqa: E402
from evaluation.metrics import crps  # noqa: E402
from evaluation.sda_sampler import sda_guided_sample  # noqa: E402
from generate_da_baselines_table import build_datasets  # noqa: E402
from train import _make_eval_batch  # noqa: E402

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
GW_GRID = [1.0, 2.0, 5.0, 10.0, 25.0, 50.0, 100.0]
HYBRID_GW_GRID = [0.5, 1.0, 2.0, 5.0, 10.0]
HYBRID_TAU0_GRID = [0.1, 0.2, 0.5]
R_VAR = 0.5


def load_sets(obs_interval: int, n_test: int, n_val: int) -> dict:
    path = os.path.join(ROOT, "experiments", f"l63_valsets_obs{300 // obs_interval}_n{n_val}.pt")
    if not os.path.exists(path):
        ds = build_datasets(1, n_val, obs_interval)
        torch.save({k: v for k, v in ds.items() if k.startswith("val")}, path)
    out = torch.load(path, weights_only=False)
    out.update(torch.load(os.path.join(ROOT, "experiments", f"l63_testsets_obs{300 // obs_interval}_n{n_test}.pt"),
                          weights_only=False))
    return out


def stack_batch(ds, idx: list[int], cfg, device: torch.device) -> tuple[FlowMatchingBatch, np.ndarray]:
    param_dim = cfg.model.get("param_dim", 0)
    bs = [_make_eval_batch(ds[i], device, param_dim=param_dim, obs_var_indices=None) for i in idx]
    cat = lambda name: torch.cat([getattr(b, name) for b in bs])
    batch = FlowMatchingBatch(
        cat("states"), cat("obs"), cat("obs_mask"), cat("forcing"),
        params=cat("params") if bs[0].params is not None else None,
        true_params=cat("true_params") if bs[0].true_params is not None else None,
    )
    truth = np.stack([ds[i]["true_state"].numpy() for i in idx])
    return batch, truth


def run(model, cfg, ds, device: torch.device, gw: float, n_members: int, n_outer: int,
        mean_model=None, tau0: float = 0.0, batch_size: int = 50) -> dict[str, np.ndarray]:
    rmse, crps_w, spread = [], [], []
    for i in range(0, len(ds), batch_size):
        idx = list(range(i, min(i + batch_size, len(ds))))
        batch, truth = stack_batch(ds, idx, cfg, device)
        mean_est = None
        if mean_model is not None:
            with torch.no_grad():
                mean_est = mean_model(batch)
        ens, _ = sda_guided_sample(model, batch, R_var=R_VAR, N_outer=n_outer, guidance_weight=gw,
                                   n_members=n_members, mean_estimate=mean_est, tau0=tau0)
        ens = ens.detach().float().cpu().numpy()
        for b in range(len(idx)):
            e = np.moveaxis(ens[b], -1, 0)
            mean = e.mean(0)
            rmse.append(float(np.sqrt(((mean - truth[b]) ** 2).mean(0)).mean()))
            crps_w.append(float(np.mean(crps(e, truth[b]))))
            spread.append(float(np.sqrt(e.var(0, ddof=1).mean(-1)).mean()))
    return {"rmse": np.array(rmse), "crps": np.array(crps_w), "spread": np.array(spread)}


def tune(model, cfg, ds, device, n_outer: int, mean_model=None) -> tuple[float, float, dict]:
    curve = {}
    grid = [(0.0, g) for g in GW_GRID] if mean_model is None else [(t, g) for t in HYBRID_TAU0_GRID for g in HYBRID_GW_GRID]
    for tau0, gw in grid:
        torch.manual_seed(0)
        r = run(model, cfg, ds, device, gw, 10, n_outer, mean_model=mean_model, tau0=tau0)
        curve[f"{tau0}/{gw}"] = float(r["rmse"].mean())
        print(f"    tune tau0={tau0} gw={gw}: val RMSE {r['rmse'].mean():.4f} CRPS {r['crps'].mean():.4f} "
              f"spread/RMSE {r['spread'].mean() / r['rmse'].mean():.2f}", flush=True)
    best = min(curve, key=lambda k: curve[k] if np.isfinite(curve[k]) else np.inf)
    tau0, gw = (float(x) for x in best.split("/"))
    return tau0, gw, curve


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--name", required=True)
    p.add_argument("--mean-name", default=None)
    p.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    p.add_argument("--ckpt", default="stage1_best.ckpt")
    p.add_argument("--n-members", type=int, default=30)
    p.add_argument("--n-outer", type=int, default=10)
    p.add_argument("--obs-interval", type=int, default=10)
    p.add_argument("--n-test", type=int, default=200)
    p.add_argument("--n-val", type=int, default=50)
    p.add_argument("--tag", default="")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sets = load_sets(args.obs_interval, args.n_test, args.n_val)
    exp = lambda n, s: os.path.join(ROOT, "experiments", f"{n}_seed{s}")

    model, cfg = load_model(exp(args.name, args.seeds[0]), args.ckpt, device)
    mean_model = load_model(exp(args.mean_name, args.seeds[0]), args.ckpt, device)[0] if args.mean_name else None
    tau0, gw, curve = tune(model, cfg, sets["val_s0"], device, args.n_outer, mean_model)
    print(f"selected tau0={tau0} gw={gw} (validation S0, seed {args.seeds[0]})", flush=True)

    out: dict = {"tuned": {"tau0": tau0, "gw": gw, "val_curve": curve}}
    for case in ("s0", "s1"):
        per_seed = []
        for s in args.seeds:
            torch.manual_seed(s)
            model, cfg = load_model(exp(args.name, s), args.ckpt, device)
            mm = load_model(exp(args.mean_name, s), args.ckpt, device)[0] if args.mean_name else None
            r = run(model, cfg, sets[f"test_{case}"], device, gw, args.n_members, args.n_outer, mean_model=mm, tau0=tau0)
            print(f"  seed {s} {case}: RMSE {r['rmse'].mean():.4f} CRPS {r['crps'].mean():.4f} "
                  f"spread/RMSE {r['spread'].mean() / r['rmse'].mean():.2f}", flush=True)
            per_seed.append(r)
        out[case] = {k: np.mean([r[k] for r in per_seed], 0).tolist() for k in ("rmse", "crps", "spread")}
        out[case]["seed_means"] = [float(r["rmse"].mean()) for r in per_seed]
    tag = args.tag or (f"{args.name}__from__{args.mean_name}" if args.mean_name else args.name)
    path = os.path.join(OUT_DIR, f"{tag}_guided_ens{args.n_members}_no{args.n_outer}.json")
    with open(path, "w") as f:
        json.dump(out, f)
    print("wrote", path)


if __name__ == "__main__":
    main()
