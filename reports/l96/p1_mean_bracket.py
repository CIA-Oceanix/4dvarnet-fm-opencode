"""Mean part of the flow-matching score of the P1 L96 ensembles (paper section 5.3).

For tau > 0 the Gaussian flow-matching score of a scheme mixes the error of its mean with the
characterisation of its distribution. The score of the best Gaussian with the scheme's mean (its
"mean part") lies between two computable bounds, per case, scheme and tau:

- lower: the Gaussian score with the variance set to the realised squared error of the mean,
  mean(e^2 / (1 + alpha_tau e^2)) in standardised units (from the stored ensemble means);
- upper: the score of the scalar-calibrated Gaussian, FMS_tau(c*) (committed FMS summaries).

Rows and seeds as the tau-resolved figures of the paper: ETKF and ETKS (one DA run), CFM, SDA and
D-Unet+SDA (1D backbone, 3 seeds averaged). Writes reports/l96/outputs/fm_score/p1_mean_bracket.json.
Caveat: the stored S0 DA means differ from those behind the committed DA scores by up to about 3 %
in MSE (the S1 and learned rows match exactly), so the S0 DA bounds carry that uncertainty.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import _inputs  # noqa: E402
from reports.l96.probe_fm_score import norm_stats  # noqa: E402

HERE = Path(__file__).resolve().parent
CUR = HERE / "outputs" / "fm_score" / "current"
OUT = HERE / "outputs" / "fm_score" / "p1_mean_bracket.json"
EX = _inputs.root("benchmark_extended", "here")
REG = EX / "eval_regular"
DA_NPZ = _inputs.root("p1_benchmark", "here") / "da_current_2026-09-29" / (
    "l96_baselines_trajectories_dws500_s0c_test_etks-correct-Lfull-infs0-1.15_s1-2.5_infs0-1.2_s1-3.0"
    "_etkf_infs0-1.15_s1-2.5_obsj2_int100_fw_dafw.npz")
DA_TAG = ("da_reg_l96_baselines_trajs_dws500_s0c_test_etks-correct-Lfull-infs0-1.15_s1-2.5_infs0-1.2_s1-3.0"
          "_etkf_infs0-1.15_s1-2.5_obsj2_int100_fw_dafw")
TAUS = (0.0, 0.25, 0.5, 0.75)
IDX = list(range(8)) + [8 + k * 4 + j for k in range(8) for j in range(2)]
SEEDS = (1, 2, 3)
ROWS = {
    "ETKF": [(f"{DA_TAG}_{{c}}_ETKF_members.json", ("da", "ETKF"))],
    "ETKS": [(f"{DA_TAG}_{{c}}_ETKS_members.json", ("da", "ETKS"))],
    "CFM": [(f"vanillacfm_ep1200_seed{s}_reg_{{c}}.json", EX / f"L96B_vanillacfm_monaiM_ep1200_seed{s}" / "ens30_no20")
            for s in SEEDS],
    "SDA": [(f"A3_sda2_monaiM_ep1200_l96_seed{s}_reg_{{c}}.json", REG / f"A3_sda2_monaiM_ep1200_l96_seed{s}" / "ens30_gw25")
            for s in SEEDS],
    "D-Unet+SDA": [(f"hybrid_DU1200s{s}_sda3fix1200s1_reg_{{c}}.json",
                    REG / f"hybrid_DU1200s{s}_A3_sda3fix_monaiM_ep1200_l96" / "tau0.1_gw2") for s in SEEDS],
}


def alpha(tau: float) -> float:
    return (tau / (1 - tau)) ** 2


def lower_bound(err2: np.ndarray) -> list[float]:
    return [float((err2 / (1 + alpha(t) * err2)).mean()) for t in TAUS]


def _sel(a: np.ndarray) -> np.ndarray:
    return a[..., IDX] if a.shape[-1] == 40 else a


def _mean(src, case: str) -> np.ndarray:
    if isinstance(src, tuple):
        return _sel(np.load(DA_NPZ)[f"{case}_{src[1]}_trajectories"]).astype(np.float64)
    return np.load(src / f"estimates_{case}.npz")["trajectories"].astype(np.float64)


def main() -> None:
    _, sd = norm_stats()
    sd = _sel(sd)
    res = {}
    for case in ("s0", "s1"):
        truth = np.load(EX / "L96B_directunet_monaiM_ep1200_seed1" / "ens1_no1" / f"estimates_{case}.npz")["truth"]
        truth = truth.astype(np.float64)
        res[case] = {}
        for name, items in ROWS.items():
            fms, upper, lower = [], [], []
            for js, src in items:
                d = json.loads((CUR / js.format(c=case)).read_text())
                fms.append([d["gauss"][str(t)]["all"] for t in TAUS])
                upper.append([d["gauss"]["0.0"]["all"]] + [d["scale"]["all"][str(t)]["at_c_star"] for t in TAUS[1:]])
                lower.append(lower_bound(((_mean(src, case) - truth) / sd) ** 2))
            res[case][name] = {"fms": np.mean(fms, 0).tolist(), "upper": np.mean(upper, 0).tolist(),
                               "lower": np.mean(lower, 0).tolist(), "n_seeds": len(items)}
    OUT.write_text(json.dumps({"taus": list(TAUS), "cases": res}, indent=1) + "\n")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
