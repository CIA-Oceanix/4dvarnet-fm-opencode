"""Distributional score FMS_tau section of the P1 report.

Reads the per-window FM-score arrays of the current benchmark rows from the
``p1_benchmark`` input bundle (``here/fm_score_current``, linked from the runs of
``docs/plans/analysis/l96_p1_fm_score_bootstrap.md``) and renders the tables of
``summarise_fm_score_current.build``: FMS at tau = 0 / 0.25 / 0.5 / 0.75 in
physical units with 95% window-bootstrap intervals, calibration, paired
differences against PredictStateCFM-M, and the non-Gaussian check; then a
reading generated from those numbers and the caveats.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import _inputs  # noqa: E402
from summarise_fm_score_current import build  # noqa: E402

INPUTS = _inputs.root("p1_benchmark", "here") / "fm_score_current"
COLS = (("reg", "s0"), ("reg", "s1"), ("rand", "s0"), ("rand", "s1"))
HYBRID = "Hybrid DirectUNet-M → SDA3-fix-M"


def _loss(table: dict, name: str) -> str:
    return " / ".join(f"{table[(name, lay, case)]['closs0.75'][0]:.0f}" if (name, lay, case) in table else "—"
                      for lay, case in COLS)


def section(heading: str = "##") -> list[str]:
    L, t = build(INPUTS, heading)
    best75 = {c: min((v["fms0.75"][0], n) for (n, lay, case), v in t.items()
                     if (lay, case) == c and "members" not in n)[1] for c in COLS}
    L += ["**Reading** (columns regular S0 / regular S1 / random S0 / random S1):", "",
          "- **Accuracy and distributional skill rank differently.** The hybrid has the lowest FMS at τ = 0 (its MSE) "
          "in every column, but the best FMS at τ = 0.75 is " + " / ".join(best75[c] for c in COLS) + "; the paired "
          "table tests the crossover window by window. Its sampler is over-confident (calibration loss at τ = 0.75: "
          + _loss(t, HYBRID) + "%).",
          "- **Model error shows up as calibration loss for the DA schemes only.** Calibration loss at τ = 0.75 (%): "
          f"ETKF {_loss(t, 'ETKF (λ 1.15 / 2.5)')}, ETKS {_loss(t, 'ETKS (λ 1.15 / 2.5)')}; PredictStateCFM-M "
          f"{_loss(t, 'PredictStateCFM-M')}, SDA3-fix-M {_loss(t, 'SDA3-fix-M')}, SDA2-M {_loss(t, 'SDA2-M')} "
          "(SDA2's falls under model error: conditioning on the biased parameters widens its prior).",
          "- **Aggregate spread/skill is not calibration.** A pooled spread/skill near 1 can coexist with a "
          "calibration loss at τ = 0.75 (random S1: ETKF spread/skill "
          f"{t[('ETKF (λ 1.15 / 2.5)', 'rand', 's1')]['spread_skill'][0]:.2f}, loss "
          f"{t[('ETKF (λ 1.15 / 2.5)', 'rand', 's1')]['closs0.75'][0]:.0f}%): large τ weights the elements where "
          "the spread is small relative to the error, so a scalar inflation can match the total error but not where "
          "it occurs.", "",
          "**Caveats.**", "",
          "- DA rows are the benchmark files in the Gaussian form; their member re-runs (non-Gaussian table) are a "
          "fresh, unseeded ensemble realisation: two realisations of the ETKS differ by ~2% in RMSE at S0 (agree within "
          "0.2% at S1). This realisation noise is not in the window bootstrap.",
          "- EnKF and Strong-4DVar have no stored ensemble (Gaussian / point only); the random-layout Strong-4DVar "
          "trajectories were not kept.",
          "- The window bootstrap does not include seed variability (seeds are averaged per window); seed ranges are "
          "in `docs/results/l96_p1_fm_score.md`.", ""]
    return L
