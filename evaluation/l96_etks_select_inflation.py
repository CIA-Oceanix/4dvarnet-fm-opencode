"""Select the ETKF and ETKS inflation per case from the validation sweep.

Reads the outputs of ``batch/run_l96_etks_val_tuning.sbatch`` (regular grid via
``evaluate_all_l96.py``, random layouts via ``eval_da_random_layout_l96.py``),
scores both layouts with the P1 per-window convention (``per_window_metrics``,
24 observed channels), and picks, per case and method, the lambda with the lowest
mean over the two layouts of CRPS relative to that layout's best CRPS.

    python evaluation/l96_etks_select_inflation.py [--out experiments/l96_etks_val_selection.json]
"""
import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from eval_da_obs_count_l96 import per_window_metrics  # noqa: E402
from evaluation.estimate_metrics import _groups_from_per_window  # noqa: E402
from evaluation.run_l96 import EXP_DIR, make_obs_j_indices  # noqa: E402

LAMBDAS = (1.0, 1.05, 1.1, 1.2, 1.3, 1.5, 2.0, 2.5, 3.0)
METHODS = ("ETKF", "ETKS")
CASES = ("s0", "s1")
VAL = "/Odyssey/private/rfablet/Python/4dvarnet-fm-opencode/experiments/l96_valset_regular_w50.pt"
ETKS_TAG = "etks-correct-Lfull"


def regular_path(lam: float) -> str:
    etkf = "" if lam == 1.0 else f"_etkf_inf{lam}"
    return os.path.join(EXP_DIR, f"l96_baselines_trajectories_dws500_val_{ETKS_TAG}-inf{lam}_infs0-1.5_s1-2.0"
                                 f"{etkf}_obsj2_int100_fw_dafw.npz")


def random_path(lam: float) -> str:
    tag = f"_rlayout_n10-100_k4-16_w50_d1_{ETKS_TAG}-inf{lam}_inf{lam}"
    return os.path.join(EXP_DIR, "l96_da_random_layout", f"summary{tag}.json")


def regular_scores(lam: float, truth: dict, idx: np.ndarray) -> dict:
    z = np.load(regular_path(lam))
    out = {}
    for case in CASES:
        for m in METHODS:
            k = f"{case}_{m}"
            met = per_window_metrics(z[f"{k}_trajectories"].astype(np.float64), truth[case],
                                     z[f"{k}_ensemble_variance"].astype(np.float64), idx)
            c = z[f"{k}_crps"].astype(np.float64)
            crps = _groups_from_per_window(c[..., idx] if c.shape[-1] > len(idx) else c)["all_obs"]
            rmse = met["rmse"]["all_obs"]
            out[(case, m)] = {"rmse": float(rmse.mean()), "crps": float(crps.mean()),
                              "spread_over_rmse": float(met["spread"]["all_obs"].mean() / rmse.mean())}
    return out


def random_scores(lam: float) -> dict:
    s = json.load(open(random_path(lam)))
    out = {}
    for case in CASES:
        for m in METHODS:
            c = s["cases"][case][m]
            rmse = c["rmse"]["all_obs"]["mean"]
            out[(case, m)] = {"rmse": rmse, "crps": c["crps"]["all_obs"]["mean"],
                              "spread_over_rmse": c["spread"]["all_obs"]["mean"] / rmse}
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", default=os.path.join(EXP_DIR, "l96_etks_val_selection.json"))
    args = p.parse_args()
    idx = np.array(make_obs_j_indices(8, 4, 2))
    ds = torch.load(VAL, weights_only=False)
    truth = {c: np.stack([w["true_state"].numpy()[:, idx] for w in ds[f"test_{c}"]]).astype(np.float64)
             for c in CASES}
    scores = {"regular": {lam: regular_scores(lam, truth, idx) for lam in LAMBDAS},
              "random": {lam: random_scores(lam) for lam in LAMBDAS}}
    result = {"lambdas": list(LAMBDAS), "criterion": "mean over layouts of CRPS / best CRPS of that layout",
              "selected": {}, "scores": {}}
    for case in CASES:
        for m in METHODS:
            rel = {lam: np.mean([scores[lay][lam][(case, m)]["crps"]
                                 / min(scores[lay][x][(case, m)]["crps"] for x in LAMBDAS)
                                 for lay in scores]) for lam in LAMBDAS}
            best = min(rel, key=rel.get)
            result["selected"][f"{case}_{m}"] = best
            print(f"\n{case} {m}: selected lambda {best}")
            print(f"  {'lambda':>6} {'reg RMSE':>9} {'reg CRPS':>9} {'reg spr/R':>9} "
                  f"{'rnd RMSE':>9} {'rnd CRPS':>9} {'rnd spr/R':>9} {'rel CRPS':>9}")
            for lam in LAMBDAS:
                r, q = scores["regular"][lam][(case, m)], scores["random"][lam][(case, m)]
                print(f"  {lam:>6} {r['rmse']:9.4f} {r['crps']:9.4f} {r['spread_over_rmse']:9.3f} "
                      f"{q['rmse']:9.4f} {q['crps']:9.4f} {q['spread_over_rmse']:9.3f} {rel[lam]:9.4f}")
                result["scores"][f"{case}_{m}_{lam}"] = {"regular": r, "random": q, "relative_crps": rel[lam]}
    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
