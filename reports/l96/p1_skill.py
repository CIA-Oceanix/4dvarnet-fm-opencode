"""Pooled skill scores of the P1 paper tables (``tab:s0`` / ``tab:s1``), regular test set.

Per channel of the 24 observed ones, pooled over the 200 windows and all time steps, then averaged
over channels:
- EV = 1 - MSE / Var(truth), the variance of the case's own test truth;
- CRPSS = 1 - CRPS / CRPS_clim, CRPS_clim the CRPS of the climatological (test-truth marginal)
  distribution, half the Gini mean difference of 200 000 truth values drawn with seed 0;
- for point estimates (DirectUNet, the 4D-Vars) the MAE skill 1 - MAE / CRPS_clim in the CRPSS
  column (the CRPS of a point forecast is its MAE).
Learned rows are averaged over their finished seeds. Ported from the scratch script that produced
the 1D numbers of the paper (``l96_skill.py``); every U-Net row is the 2D backbone.
"""
from __future__ import annotations

import numpy as np

import _inputs

EX = _inputs.root("benchmark_extended", "here")
P1 = _inputs.root("p1_benchmark", "here") / "da_current_2026-09-29"
IDX = list(range(8)) + [8 + k * 4 + j for k in range(8) for j in range(2)]
DA = (("ETKF", P1 / ("l96_baselines_trajectories_dws500_s0c_test_etks-correct-Lfull-infs0-1.15_s1-2.5_infs0-1.2_s1-3.0"
                     "_etkf_infs0-1.15_s1-2.5_obsj2_int100_fw_dafw.npz"), "ETKF", False),
      ("EnKF", EX / "da_inflation_2026-09-28" / ("l96_baselines_trajectories_dws500_s0c_crps_infs0-1.2_s1-3.0_etkf_infs0-1.15"
                                                 "_s1-2.5_obsj2_int100_fw_dafw.npz"), "EnKF", False),
      ("ETKS", P1 / ("l96_baselines_trajectories_dws500_s0c_test_etks-correct-Lfull-infs0-1.15_s1-2.5_infs0-1.2_s1-3.0"
                     "_etkf_infs0-1.15_s1-2.5_obsj2_int100_fw_dafw.npz"), "ETKS", False),
      ("Strong-4DVar", P1 / "l96_baselines_trajectories_dws500_s0c_inf2.0_etkf_inf2.0_obsj2_int100_fw_dafw.npz",
       "Strong_4DVar", True),
      ("Weak-4DVar", EX / "weak4dvar_2026-10-04" / ("l96_baselines_trajectories_dws500_s0c_test_weak4dvar-qs0-0.03_s1-0.3-it40"
                                                     "_obsj2_int100_fw_dafw.npz"), "Weak_4DVar", True))
SEEDS = (1, 2, 3)
REGE = EX / "eval_regular"
LEARNED = (("D-Unet (DirectUNet-2D-M)", [EX / f"L96B_directunet2d_monaiM_seed{s}" / "ens1_no1" for s in SEEDS], True),
           ("CFM (VanillaCFM-2D-M)", [EX / f"L96B_vanillacfm_monai2dM_ep1200_seed{s}" / "ens30_no20" for s in SEEDS], False),
           ("PredictStateCFM-2D-M", [EX / f"L96B_predictstatecfm_monai2dM_ep1200_seed{s}" / "ens30_no20" for s in SEEDS], False),
           ("SDA1-2D-M", [REGE / f"L96B_sda1_monai2dM_ep1200_seed{s}" / "ens30_gw25" for s in SEEDS], False),
           ("SDA2-2D-M", [REGE / f"L96B_sda2_monai2dM_ep1200_seed{s}" / "ens30_gw25" for s in SEEDS], False),
           ("SDA3-fix-2D-M", [REGE / f"L96B_sda3fix_monai2dM_ep1200_seed{s}" / "ens30_gw25" for s in SEEDS], False),
           ("D-Unet+SDA (DirectUNet-2D-M -> SDA3-fix-2D-M)",
            [REGE / f"hybrid_DU2dMs{s}_L96B_sda3fix_monai2dM_ep1200_seed1" / "tau0.1_gw1" for s in SEEDS], False),
           ("DirectUNet-2D-M -> SDA1-2D-M",
            [REGE / f"hybrid_DU2dMs{s}_L96B_sda1_monai2dM_ep1200_seed1" / "tau0.1_gw1" for s in SEEDS], False),
           ("DirectUNet-2D-M -> SDA3-fix-M (1D prior, interim)",
            [REGE / f"hybrid_DU2dMs{s}_A3_sda3fix_monaiM_ep1200_l96" / "tau0.1_gw1" for s in SEEDS], False))


def _half_gini(x: np.ndarray) -> float:
    s = np.sort(x)
    n = s.size
    return float(np.sum((2 * np.arange(1, n + 1) - n - 1) * s) / n ** 2)


def _sel(a: np.ndarray) -> np.ndarray:
    return a[..., IDX] if a.shape[-1] == 40 else a


def _truth(case: str) -> np.ndarray:
    return np.load(EX / "L96B_directunet2d_monaiM_seed1" / "ens1_no1" / f"estimates_{case}.npz")["truth"].astype(np.float64)


def scores(case: str) -> list[dict]:
    truth = _truth(case)
    flat = truth.reshape(-1, truth.shape[-1])
    rng = np.random.default_rng(0)
    var_c = flat.var(0)
    crps_c = np.array([_half_gini(rng.choice(flat[:, j], 200000, replace=False)) for j in range(flat.shape[1])])

    def ev(traj: np.ndarray) -> float:
        return float(np.mean(1 - ((traj - truth) ** 2).mean((0, 1)) / var_c))

    def crpss(crps_wj: np.ndarray) -> float:
        return float(np.mean(1 - crps_wj.mean(0) / crps_c))

    def mae_skill(traj: np.ndarray) -> float:
        return float(np.mean(1 - np.abs(traj - truth).mean((0, 1)) / crps_c))

    rows = []
    for label, path, key, point in DA:
        z = np.load(path)
        t = _sel(z[f"{case}_{key}_trajectories"]).astype(np.float64)
        rows.append({"label": label, "family": "DA", "seeds": "—", "ev": ev(t),
                     "skill": mae_skill(t) if point else crpss(_sel(z[f"{case}_{key}_crps"])), "mae": point})
    for label, dirs, point in LEARNED:
        ok = [d for d in dirs if (d / f"estimates_{case}.npz").exists() and (point or (d / f"scores_{case}.npz").exists())]
        e, c = [], []
        for d in ok:
            t = np.load(d / f"estimates_{case}.npz")["trajectories"].astype(np.float64)
            e.append(ev(t))
            c.append(mae_skill(t) if point else crpss(np.load(d / f"scores_{case}.npz")["crps"]))
        rows.append({"label": label, "family": "learned", "seeds": f"{len(ok)}/{len(dirs)}",
                     "ev": float(np.mean(e)) if e else None, "skill": float(np.mean(c)) if c else None, "mae": point})
    return rows


def section(heading: str = "##") -> list[str]:
    S = {c: scores(c) for c in ("s0", "s1")}
    A = [f"{heading} Pooled skill scores (paper tables `tab:s0` / `tab:s1`, regular test set)\n",
         "Per channel, pooled over the 200 windows and all time steps, then averaged over the 24 observed channels: "
         "**EV** = 1 − MSE / Var(truth); **CRPSS** = 1 − CRPS / CRPS_clim against the climatological (test-truth marginal) "
         "CRPS; point estimates show the MAE skill 1 − MAE / CRPS_clim instead (in brackets). Learned rows: 2D backbone, "
         "mean over the finished seeds (`seeds`); `pending` = not trained yet. Code: `reports/l96/p1_skill.py`. Per column, "
         "best **bold**, second *italic*.\n",
         "| scheme | seeds | EV S0 | CRPSS S0 | EV S1 | CRPSS S1 |", "|---|---|---|---|---|---|"]
    marks = {}
    for c in ("s0", "s1"):
        for k in ("ev", "skill"):
            vals = sorted({r[k] for r in S[c] if r[k] is not None and not (k == "skill" and r["mae"])}, reverse=True)
            marks[(c, k)] = {v: m for v, m in zip(vals[:2], ("**", "*"))}

    def cell(r: dict, c: str, k: str) -> str:
        v = r[k]
        if v is None:
            return "pending"
        if k == "skill" and r["mae"]:
            return f"({v:.3f})"
        m = marks[(c, k)].get(v, "")
        return f"{m}{v:.3f}{m}"
    for r0, r1 in zip(S["s0"], S["s1"]):
        A.append(f"| {r0['label']} | {r0['seeds']} | {cell(r0, 's0', 'ev')} | {cell(r0, 's0', 'skill')} | "
                 f"{cell(r1, 's1', 'ev')} | {cell(r1, 's1', 'skill')} |")
    A.append("")
    return A


if __name__ == "__main__":
    print("\n".join(section()))

