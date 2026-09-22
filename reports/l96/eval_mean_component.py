#!/usr/bin/env python3
"""The `Psi_mean` component on its own, as a deterministic estimator.

`FourDVarNetDecomposedCFM` computes `m_phi(y) = E[x1|y]` with no `x_tau`
dependence, so it is usable as a standalone point estimator -- one unrolled solve
per window, no ODE integration, no ensemble. This script scores exactly that,
which separates two things the full-model numbers conflate: whether a model's
deficit lives in its mean slot or in the tau-dependent part built on top of it.

For the `mu`-predicting models (`PredictStateCFM`, `FourDVarNetPredictStateCFM`)
the same quantity is `mu(x0, tau=0, y)` averaged over draws -- at `tau=0`, `x0` is
independent of `x1`, so `E[x1|x_tau,y] = E[x1|y]` exactly. Those models are
included for comparison, with the caveat that their estimate carries Monte-Carlo
noise where the decomposed model's is exact (`mu(0) = m` for every `x0`, by
construction).

CRPS for a deterministic forecast reduces exactly to the absolute error, so the
CRPS column here is MAE and is NOT comparable to the ensemble CRPS of the full
models -- an ensemble can only do better. RMSE and MSE are directly comparable.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation.estimate_metrics import (_groups_from_per_window,  # noqa: E402
                                         _mean_std, evaluate_estimates,
                                         per_window_deterministic_crps,
                                         per_window_rmse_ev)
from evaluation.neural_inference import BatchDict, load_model, prepare_dataset  # noqa: E402
from evaluation.psi_decomposition import psi_mean_estimate  # noqa: E402

DEFAULT_RUNS = [
    ("VanillaCFM S+", "experiments/A1_vanillacfm_monaiSplus_l96/checkpoints/stage1_best.ckpt", None),
    ("VanillaCFM M", "experiments/A1_vanillacfm_monaiM_l96/checkpoints/stage1_best.ckpt", None),
    ("VanillaCFM L", "experiments/A1_vanillacfm_monaiL_l96/checkpoints/stage1_best.ckpt", None),
    ("PredictStateCFM S+", "experiments/A2_predictstatecfm_monaiSplus_l96/checkpoints/stage1_best.ckpt", None),
    ("PredictStateCFM M", "experiments/A2_predictstatecfm_monaiM_l96/checkpoints/stage1_best.ckpt", None),
    ("PredictStateCFM L (lr3e-4)", "experiments/A2_predictstatecfm_monaiL_lr3e4_l96/checkpoints/stage1_best.ckpt", None),
    ("FDV1CFM S+", "experiments/M6a_fdv1cfm_tau_aware_monaiSplus_norm_l96/checkpoints/stage1_best.ckpt", None),
    ("FDV1CFM M", "experiments/C4_fdv1cfm_monaiM_noaug_l96/checkpoints/stage1_best.ckpt", None),
    ("FDV1CFM S+ (+aug)", "experiments/M6a_fdv1cfm_tau_aware_monaiSplus_norm_obsdens_l96/checkpoints/stage1_best.ckpt", None),
    ("FDV1CFM M (+aug)", "experiments/C4_fdv1cfm_monaiM_aug_l96/checkpoints/stage1_best.ckpt", None),
]


def full_model_rmse(exp_dir: str, case: str):
    """The full model's ens30 RMSE, under the SAME conventions, for comparison."""
    path = ROOT / exp_dir / "ens30_no10" / f"members_{case}.npz"
    if not path.exists():
        return None
    d = np.load(path)
    mean_traj = d["members"].mean(axis=-1)
    return {"rmse": per_window_rmse_ev(mean_traj, d["truth"])["rmse"],
            "rmse_pooled": evaluate_estimates(mean_traj, d["truth"])["groups"]}


def evaluate_mean(checkpoint, config, dataset, case, batch_size, m_draws, device):
    model, cfg = load_model(checkpoint, config, device=device)
    model.eval()
    sigma = float(model.sigma_prior)
    # Normalization is keyed off the CONFIG, exactly as eval_neural_l96.py does it.
    # Without this the model -- trained on z-scored obs -- is fed RAW obs, and its
    # normalized-space predictions are then scored against raw truth. Both halves
    # are wrong and the symptom is an RMSE near 1.5-1.9 (the normalized-space
    # magnitude) instead of ~0.35, which is how this was caught.
    norm_stats = None
    if cfg.data.get("normalize", False):
        from data.normalization import load_norm_stats
        from evaluation.archive import resolve_norm_stats
        norm_stats = load_norm_stats(str(resolve_norm_stats(cfg, required=True)))
    _, dataloaders, obs_var_indices = prepare_dataset(
        cfg, dataset, batch_size=batch_size, norm_stats=norm_stats)
    idx = torch.as_tensor(obs_var_indices, dtype=torch.long).to(device)

    preds, truths = [], []
    disp_sq = 0.0
    n_tot = 0
    with torch.no_grad():
        for bd in dataloaders[case]:
            batch = BatchDict({k: (v.to(device) if torch.is_tensor(v) else v)
                               for k, v in bd.items()})
            x1 = batch.true_state[..., idx].contiguous()
            m, spread_sq = psi_mean_estimate(model, batch, x1, sigma, m_draws)
            mn = m.float().cpu().numpy()
            tn = x1.float().cpu().numpy()
            if norm_stats is not None:
                # ONLY the prediction. make_collate_eval normalizes `obs` and
                # leaves `true_state` RAW always (see its docstring), so the
                # truth needs no conversion -- denormalizing it too would scale
                # an already-physical tensor a second time.
                from data.normalization import denormalize
                mn = denormalize(mn, norm_stats)
            preds.append(mn)
            truths.append(tn)
            disp_sq += spread_sq
            n_tot += x1.numel()
    # Across-draw dispersion of m itself: 0 means m is a deterministic function
    # of y. NOT implied by the decomposed architecture -- mu(0)=m holds for every
    # x0, but a frozen mean trained with init_state_var>0 (FDV1-Stier) is itself
    # stochastic, so its m varies per call and Psi_NG(0)=0 holds only in
    # expectation.
    return (np.concatenate(preds, 0), np.concatenate(truths, 0), type(model).__name__,
            float(np.sqrt(disp_sq / n_tot)))


def pathlib_parent(ckpt: str) -> Path:
    """experiments/<run>/checkpoints/x.ckpt -> experiments/<run>"""
    return Path(ckpt).parent.parent


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="experiments/l96_datasets_obsj2_int100_nwin200.pt")
    p.add_argument("--case", default="s0", choices=["s0", "s1"])
    p.add_argument("--group", default="all_obs", choices=["slow", "obs_fast", "all_obs"])
    p.add_argument("--batch-size", type=int, default=25)
    p.add_argument("--m-draws", type=int, default=4)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output", default="reports/l96/outputs/m4_mean_component.md")
    p.add_argument("--json-output", default="reports/l96/outputs/m4_mean_component.json")
    args = p.parse_args()

    device = torch.device(args.device)
    exp_dirs = {label: str(pathlib_parent(ckpt)) for label, ckpt, _ in DEFAULT_RUNS}
    rows = []
    for label, ckpt, cfg_name in DEFAULT_RUNS:
        if not (ROOT / ckpt).exists():
            print(f"MISSING (skipped): {ckpt}")
            continue
        torch.manual_seed(args.seed)
        pred, truth, cls, disp = evaluate_mean(str(ROOT / ckpt), cfg_name, args.dataset,
                                               args.case, args.batch_size, args.m_draws, device)
        sq = ((pred - truth) ** 2).mean(axis=1)                    # (W, D)
        mse = {k: _mean_std(v) for k, v in _groups_from_per_window(sq).items()}
        rmse = per_window_rmse_ev(pred, truth)["rmse"]
        pooled = evaluate_estimates(pred, truth)["groups"]
        mae = per_window_deterministic_crps(pred, truth)
        g = args.group
        full = full_model_rmse(exp_dirs[label], args.case) if label in exp_dirs else None
        rows.append(dict(label=label, model=cls, n_windows=int(pred.shape[0]),
                         draw_dispersion=disp,
                         mse=mse[g], rmse=rmse[g], rmse_pooled=float(pooled[g]),
                         crps_mae=mae[g],
                         full_rmse=full["rmse"][g]["mean"] if full else None,
                         full_rmse_pooled=float(full["rmse_pooled"][g]) if full else None,
                         all_groups=dict(mse=mse, rmse=rmse, crps_mae=mae)))
        print(f"  {label}: mean-alone RMSE {rmse[g]['mean']:.4f} ± {rmse[g]['std']:.4f}"
              f"  (draw dispersion {disp:.4f})", flush=True)

    if not rows:
        print("nothing evaluated")
        return

    def fmt(m):
        return f"{m['mean']:.4f} ± {m['std']:.4f}"

    header = (f"# `Psi_mean` alone vs the full CFM — {args.case.upper()}, group `{args.group}`\n\n"
              f"{rows[0]['n_windows']} windows, mean ± std across windows. Same conventions "
              "as `m4_per_window_metrics.md`: RMSE is the repo's `mean_d sqrt(mse_d)`, "
              "`pooled` is `evaluate_estimates`'s.\n\n"
              "CRPS of a point forecast is the absolute error, so that column is MAE and is "
              "NOT comparable to the full models' ensemble CRPS. `full RMSE` is the ens30 "
              "ensemble-mean RMSE of the same run, for the mean-vs-full comparison; "
              "`draws` is the across-draw dispersion of `m` itself (0 = deterministic in y).\n\n"
              "| model | RMSE | pooled | MSE | CRPS (=MAE) | full RMSE | mean−full | draws |\n"
              "|---|---|---|---|---|---|---|---|\n")
    def delta(r):
        if r["full_rmse"] is None:
            return "n/a", "n/a"
        d = 100 * (r["rmse"]["mean"] - r["full_rmse"]) / r["full_rmse"]
        return f"{r['full_rmse']:.4f}", f"{d:+.1f}%"
    body = ""
    for r in rows:
        f_rmse, d = delta(r)
        body += (f"| {r['label']} | {fmt(r['rmse'])} | {r['rmse_pooled']:.4f} | {fmt(r['mse'])} | "
                 f"{fmt(r['crps_mae'])} | {f_rmse} | {d} | {r['draw_dispersion']:.4f} |\n")
    print("\n" + header + body)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(header + body)
    Path(args.json_output).write_text(json.dumps(
        dict(case=args.case, group=args.group, m_draws=args.m_draws, rows=rows), indent=2))
    print(f"wrote {out} and {args.json_output}")


if __name__ == "__main__":
    main()
