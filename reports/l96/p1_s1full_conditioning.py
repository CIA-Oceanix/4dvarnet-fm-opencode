"""L96 S1-full rows and value of the conditioning information (P1 paper, Sects. 4 and 5.3).

S1-full: the S1 DA model keeps the full 40-variable model (biased parameters and corrupted forcing,
no truncation), on the 200 regular P1 test windows, at the hyper-parameters selected on the
validation sweep (``batch/run_l96_s1_full_model_val.sbatch``; ETKF / ETKS 1.3, EnKF 1.5, Weak-4DVar
q 0.03). Scores as ``p1_skill.py`` (EV, CRPSS or the MAE skill for point estimates) plus the pooled
spread--skill ratio in standardised units, sqrt(E v / E e^2), for the ensembles.

Conditioning: the S0 test windows with nominal parameters (``no_theta``), zero forcing
(``no_forcing``) or both (``neither``) given to the schemes (``scripts/make_l96_conditioning_variants.py``;
same truth and observations as the P1 S0 windows). ``full`` is the S0 benchmark row. DA: ETKF, EnKF,
ETKS and Strong-4DVar on the full model at the S0 settings; 2D SDA2 / SDA3-fix priors, seed 1. EV as
``p1_skill.py``. The 1D SDA2 / SDA3-fix rows (seed 1) only kept their eval JSON, so they report the
eval script's EV (``metrics.<case>.ev.groups.all_obs``), against the same quantity of their S0 row.

Inputs: the ``p1_benchmark/here`` bundle (``s1full_2026-11-09/``, ``conditioning_2026-11-09/``) and
``benchmark_extended/here`` (S0 rows). Writes ``reports/l96/outputs/p1_s1full_conditioning.json``.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
sys.path.insert(0, str(HERE))
import _inputs  # noqa: E402
import p1_skill  # noqa: E402
from reports.l96.probe_fm_score import norm_stats  # noqa: E402

P1H = _inputs.root("p1_benchmark", "here")
EX = _inputs.root("benchmark_extended", "here")
S1F = P1H / "s1full_2026-11-09"
CON = P1H / "conditioning_2026-11-09"
OUT = HERE / "outputs" / "p1_s1full_conditioning.json"
VARIANTS = ("no_theta", "no_forcing", "neither")
ENSEMBLES = ("ETKF", "EnKF", "ETKS")
SDA2D = {"SDA2-2D-M": "L96B_sda2_monai2dM_ep1200_seed1", "SDA3-fix-2D-M": "L96B_sda3fix_monai2dM_ep1200_seed1"}
SDA1D = {"SDA2-M (1D)": "A3_sda2_monaiM_ep1200_l96_seed1", "SDA3-fix-M (1D)": "A3_sda3fix_monaiM_ep1200_l96_seed1"}


class Skill:
    """The pooled scores of ``p1_skill.py`` against one case's test truth."""

    def __init__(self, case: str) -> None:
        self.truth = p1_skill._truth(case)
        flat = self.truth.reshape(-1, self.truth.shape[-1])
        rng = np.random.default_rng(0)
        self.var_c = flat.var(0)
        self.crps_c = np.array([p1_skill._half_gini(rng.choice(flat[:, j], 200000, replace=False))
                                for j in range(flat.shape[1])])
        self.sd = p1_skill._sel(norm_stats()[1])

    def ev(self, traj: np.ndarray) -> float:
        return float(np.mean(1 - ((traj - self.truth) ** 2).mean((0, 1)) / self.var_c))

    def crpss(self, crps_wj: np.ndarray) -> float:
        return float(np.mean(1 - crps_wj.mean(0) / self.crps_c))

    def mae_skill(self, traj: np.ndarray) -> float:
        return float(np.mean(1 - np.abs(traj - self.truth).mean((0, 1)) / self.crps_c))

    def ss(self, traj: np.ndarray, var: np.ndarray) -> float:
        return float(np.sqrt((var / self.sd ** 2).mean() / (((traj - self.truth) / self.sd) ** 2).mean()))


def _one(pattern: str, root: Path) -> Path:
    hits = sorted(root.glob(pattern))
    assert len(hits) == 1, (pattern, hits)
    return hits[0]


def _da_row(sk: Skill, path: Path, point: bool) -> dict:
    z = np.load(path)
    t = p1_skill._sel(z["trajectories"]).astype(np.float64)
    if point:
        return {"ev": sk.ev(t), "mae_skill": sk.mae_skill(t)}
    return {"ev": sk.ev(t), "crpss": sk.crpss(p1_skill._sel(z["crps"])),
            "ss": sk.ss(t, p1_skill._sel(z["ensemble_variance"]).astype(np.float64))}


def s1_full() -> dict:
    sk = Skill("s1")
    rows = {m: _da_row(sk, _one(f"l96_baselines_trajs_dws500_s1full_test*_s1_{m}.npz", S1F), False) for m in ENSEMBLES}
    rows["Strong-4DVar"] = _da_row(sk, _one("l96_baselines_trajs_dws500_s1full_test*_s1_Strong_4DVar.npz", S1F), True)
    w = p1_skill._sel(np.load(_one("*s1full*weak4dvar*.npz", S1F))["s1_Weak_4DVar_trajectories"]).astype(np.float64)
    rows["Weak-4DVar"] = {"ev": sk.ev(w), "mae_skill": sk.mae_skill(w)}
    return rows


def conditioning() -> dict:
    sk = Skill("s0")
    full = {r["label"]: r["ev"] for r in p1_skill.scores("s0")}
    out = {}
    for m in ENSEMBLES + ("Strong-4DVar",):
        key = m.replace("-", "_")
        out[m] = {"full": full[m]} | {v: sk.ev(p1_skill._sel(np.load(_one(
            f"l96_baselines_trajs_dws500_cond_{v}_*_s1_{key}.npz", CON / "da"))["trajectories"]).astype(np.float64))
            for v in VARIANTS}
    for label, name in SDA2D.items():
        ref = np.load(EX / "eval_regular" / name / "ens30_gw25" / "estimates_s0.npz")["trajectories"]
        out[label] = {"full": sk.ev(ref.astype(np.float64))} | {
            v: sk.ev(np.load(CON / "sda2d" / name / v / "estimates_s1.npz")["trajectories"].astype(np.float64))
            for v in VARIANTS}
    for label, name in SDA1D.items():
        ref = json.loads((EX / "eval_regular" / name / "ens30_gw25" / "sda_eval.json").read_text())
        out[label] = {"full": ref["metrics"]["s0"]["ev"]["groups"]["all_obs"], "ev_definition": "eval script"} | {
            v: json.loads((CON / "sda1d" / f"{name}_{v}.json").read_text())["metrics"]["s1"]["ev"]["groups"]["all_obs"]
            for v in VARIANTS}
    return out


def main() -> None:
    res = {"s1_full": s1_full(), "conditioning": conditioning()}
    OUT.write_text(json.dumps(res, indent=1) + "\n")
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
