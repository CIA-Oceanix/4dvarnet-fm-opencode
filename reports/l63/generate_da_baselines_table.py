import argparse
import json
import os
import sys
import time
from typing import Any

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))

from data.lorenz63 import Lorenz63Config
from data.random_bias_dataset import RandomBiasLorenz63Dataset
from data.random_param_dataset import RandomParamLorenz63Dataset
from evaluation.baselines import ETKF, ETKS, EnKF, Strong4DVar, Weak4DVar
from models.lorenz63_dynamics import Lorenz63Dynamics

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
TRUTH_EXPONENT = 1.6
DA_EXPONENT = {"s0": 1.6, "s1": 1.0}
INFLATION_GRID = [1.0, 1.05, 1.1, 1.15, 1.2, 1.3, 1.5, 2.0, 2.5, 3.0, 4.0]
Q_GRID = [0.01, 0.03, 0.1, 0.3, 1.0]


def build_datasets(n_test: int, n_val: int, obs_interval: int) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for split, seed, n in (("val", 99, n_val), ("test", 123, n_test)):
        s0 = Lorenz63Config(case=1, param_bias=0.0, forcing_state_bias=0.0, seed=seed, num_windows=n, obs_interval=obs_interval)
        s1 = Lorenz63Config(case=1, param_bias=0.15, forcing_state_bias=0.1, seed=seed, num_windows=n, obs_interval=obs_interval)
        out[f"{split}_s0"] = RandomParamLorenz63Dataset(s0, param_noise=0.2)
        out[f"{split}_s1"] = RandomBiasLorenz63Dataset(s1, param_noise=0.2, bias_mode="fixed")
    return out


def make_method(name: str, case: str, device: torch.device, dws: int, n_ens: int, **kw: Any):
    dyn = Lorenz63Dynamics(dt=0.01, coupling_exponent=DA_EXPONENT[case])
    common = dict(dt=0.01, device=device, coupling_exponent=DA_EXPONENT[case], dynamics=dyn)
    if name == "ETKF":
        return ETKF(N_ensemble=n_ens, inflation=kw.get("inflation", 1.0), **common)
    if name in ("ETKS", "ETKS-100"):
        return ETKS(N_ensemble=100 if name == "ETKS-100" else n_ens, inflation=kw.get("inflation", 1.0), **common)
    if name == "EnKF":
        return EnKF(N_ensemble=n_ens, inflation=kw.get("inflation", 1.0), **common)
    if name == "Strong-4DVar":
        return Strong4DVar(da_window_steps=dws, max_iter=kw.get("max_iter", 40), lr=kw.get("lr", 0.1), **common)
    return Weak4DVar(
        da_window_steps=dws, Q_var=kw.get("Q_var", 0.05), opt_steps=kw.get("opt_steps", 150),
        lr=kw.get("lr", 0.02), **common)


def run_method(method: Any, ds: Any, device: torch.device, batch_size: int, case: str) -> dict[str, np.ndarray]:
    rmse, crps, spread = [], [], []
    for i in range(0, len(ds), batch_size):
        batch = [ds[j] for j in range(i, min(i + batch_size, len(ds)))]
        obs = torch.stack([w["obs"] for w in batch]).to(device)
        mask = torch.stack([w["obs_mask"] for w in batch]).to(device)
        truth = torch.stack([w["true_state"] for w in batch])
        force = torch.stack([w["forcing_corrupted" if case == "s1" else "forcing_true"] for w in batch]).to(device)
        kw = dict(
            sigma=torch.tensor([w["sigma"] for w in batch], device=device),
            rho=torch.tensor([w["rho"] for w in batch], device=device),
            beta=torch.tensor([w["beta"] for w in batch], device=device),
            c1=torch.tensor([w.get("c1", 1.0) for w in batch], device=device),
        )
        results = method.assimilate_batch(obs, mask, force, truth, **kw)
        for r in results:
            rmse.append(float(np.mean(r.rmse)))
            crps.append(float(np.mean(r.crps)) if r.crps is not None else np.nan)
            spread.append(float(np.mean(np.sqrt(r.ensemble_variance))) if r.ensemble_variance is not None else np.nan)
    return {"rmse": np.array(rmse), "crps": np.array(crps), "spread": np.array(spread)}


def tune(name: str, case: str, ds: Any, device: torch.device, args: argparse.Namespace) -> dict[str, float]:
    if name in ("ETKF", "EnKF", "ETKS", "ETKS-100"):
        key, grid = "inflation", INFLATION_GRID
    elif name == "Weak-4DVar":
        key, grid = "Q_var", Q_GRID
    else:
        return {}
    scores = {}
    for v in grid:
        m = make_method(name, case, device, args.dws, args.members, **{key: v})
        t0 = time.time()
        scores[v] = float(run_method(m, ds, device, args.batch_size, case)["rmse"].mean())
        print(f"    tune {name}/{case} {key}={v}: val RMSE {scores[v]:.4f} [{time.time() - t0:.0f}s]", flush=True)
    best = min(scores, key=lambda k: scores[k] if np.isfinite(scores[k]) else np.inf)
    return {key: best, "val_curve": scores}


def fmt(a: np.ndarray) -> str:
    return f"{a.mean():.3f} ± {a.std():.3f}"


def render(results: dict, n_test: int, n_val: int, members: int, dws: int, obs_interval: int) -> str:
    rows = []
    best = {c: min(results[m][c]["rmse"].mean() for m in results if m != "ETKS-100") for c in ("s0", "s1")}

    def cell(m: str, c: str) -> str:
        s = fmt(results[m][c]["rmse"])
        return f"**{s}**" if m != "ETKS-100" and results[m][c]["rmse"].mean() == best[c] else s

    for m, r in results.items():
        s0, s1 = r["s0"], r["s1"]
        par = {"ETKF": "λ {inflation_s0} / {inflation_s1}", "ETKS": "λ {inflation_s0} / {inflation_s1}", "ETKS-100": "λ {inflation_s0} / {inflation_s1}", "EnKF": "λ {inflation_s0} / {inflation_s1}",
               "Strong-4DVar": "—", "Weak-4DVar": "q {Q_var_s0} / {Q_var_s1}, 150 it"}[m]
        par = par.format(**r["tuned"])
        ens = m in ("ETKF", "EnKF", "ETKS", "ETKS-100")
        crps = f"{s0['crps'].mean():.3f} ({s1['crps'].mean():.3f})" if ens else "— (—)"
        spr = f"{s0['spread'].mean() / s0['rmse'].mean():.2f}" if ens else "—"
        ratio = s1["rmse"].mean() / s0["rmse"].mean()
        label = f"*ETKS, 100 members*" if m == "ETKS-100" else m
        nt = "ensemble size: 100 members (the others use 30), inflation re-selected on the validation windows" if m == "ETKS-100" else ""
        rows.append(f"| {label} | {par} | — | {cell(m, 's0')} | {cell(m, 's1')} | {ratio:.2f} | {crps} | {spr} | {nt} |")
    head = [
        "| scheme | params | seeds | S0 | S1 | S1/S0 | CRPS S0 (S1) | spread/RMSE (S0) | note |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    note = (
        f"Lorenz-63 with wind forcing, {n_test} test windows (seed 123, per-window parameters ±20%), 3 state channels, "
        f"{300 // obs_interval} observations per 300-step window (obs_interval {obs_interval}, R = 0.5), DA sub-windows of {dws} steps, {members} members. "
        "Every RMSE cell is the per-window RMSE (channel-averaged over time) on X, Y, Z, mean ± sd across windows. "
        f"Hyper-parameters are selected per case on {n_val} validation windows (seed 99)."
    )
    return "\n".join(["## 1. DA baselines", "", note, ""] + head + rows) + "\n"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--n-test", type=int, default=200)
    p.add_argument("--n-val", type=int, default=50)
    p.add_argument("--members", type=int, default=30)
    p.add_argument("--obs-interval", type=int, default=10)
    p.add_argument("--dws", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=50)
    p.add_argument("--methods", nargs="+", default=["ETKF", "EnKF", "Strong-4DVar", "Weak-4DVar"])
    p.add_argument("--merge", action="store_true")
    p.add_argument("--out", default=os.path.join(OUT_DIR, "l63_da_baselines_table.md"))
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    t0 = time.time()
    ds = build_datasets(args.n_test, args.n_val, args.obs_interval)
    print(f"datasets built in {time.time() - t0:.0f}s", flush=True)

    results: dict[str, dict] = {}
    prev = args.out.replace(".md", ".json")
    if args.merge and os.path.exists(prev):
        with open(prev) as f:
            for m, r in json.load(f).items():
                results[m] = {c: {k: np.array(v) for k, v in r[c].items()} for c in ("s0", "s1")} | {"tuned": r["tuned"]}
    for name in args.methods:
        results[name] = {"tuned": {}}
        for case in ("s0", "s1"):
            tuned = tune(name, case, ds[f"val_{case}"], device, args)
            for k, v in tuned.items():
                if k != "val_curve":
                    results[name]["tuned"][f"{k}_{case}"] = v
            kw = {k: v for k, v in tuned.items() if k != "val_curve"}
            t1 = time.time()
            m = make_method(name, case, device, args.dws, args.members, **kw)
            results[name][case] = run_method(m, ds[f"test_{case}"], device, args.batch_size, case)
            print(f"  {name}/{case} {kw}: test RMSE {fmt(results[name][case]['rmse'])} [{time.time() - t1:.0f}s]", flush=True)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        f.write(render(results, args.n_test, args.n_val, args.members, args.dws, args.obs_interval))
    raw = {m: {c: {k: v.tolist() for k, v in r[c].items()} for c in ("s0", "s1")} | {"tuned": r["tuned"]}
           for m, r in results.items()}
    with open(args.out.replace(".md", ".json"), "w") as f:
        json.dump(raw, f)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
