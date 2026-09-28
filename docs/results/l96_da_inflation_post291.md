# L96 ETKF / EnKF inflation after the square-root fix (#291)

**Status:** RESULTS (2026-09-28). Validation-window sweeps that retune the ETKF and EnKF inflation after #291; basis of `L96_ETKF_INFLATION = {"s0": 1.15, "s1": 2.5}` and `L96_ENKF_INFLATION = {"s0": 1.2, "s1": 3.0}`.

**Numbers:** `reports/l96/outputs/da_inflation_sweep/` (one JSON per method × layout × inflation; produced by `scripts/sweep_l96_etkf_inflation.py`).

## Why

#291 made the ETKF square root keep the anomaly directions a thin SVD dropped
when there are fewer observations than members (24 vs 30). Rerun at the old
per-case default (S0 1.5, S1 2.0; `batch/run_l96_da_post291.sbatch`, same
command as `batch/run_l96_da_crps.sbatch`), the fixed ETKF was worse at S0
(regular 0.687 → 0.708, random layout 0.798 → 0.860) and equal or better at
S1 (1.478 → 1.479, 1.418 → 1.375), while the EnKF (untouched by #291)
reproduced its stored RMSE within 0.1%. The S0 default had been tuned against
the bug's spread loss.

## Protocol

- 50 validation windows per case (`l96_valset_regular_w50.pt`,
  `l96_valset_rlayout_n10-100_k4-16_w50.pt`; seeds disjoint from
  train/val/test), so the test set is not touched.
- DA settings copied from the benchmark runs: regular grid as
  `evaluate_all_l96.py` (dws 500, per-window fast weights), random layout as
  `eval_da_random_layout_l96.py` (fast-ring initial fill, R 0.5).
- `scripts/sweep_l96_etkf_inflation.py`, `batch/run_l96_etkf_inflation_sweep.sbatch`; per-cell JSON in `reports/l96/outputs/da_inflation_sweep/`
  (SLURM 56438 + extension 56491), master after #291.
- Cells: RMSE (mean over windows and channels of per-channel RMSE, the report
  convention) / analysis-ensemble CRPS / pooled spread-over-RMSE
  (sqrt(E var / E squared error), calibrated = 0.984 for 30 members). Bold =
  RMSE-optimal per column.

## ETKF

| λ | regular S0 | regular S1 | random S0 | random S1 |
|---|---|---|---|---|
| 1 | 0.694 / 0.345 / 0.47 | 1.943 / 1.597 / 0.07 | 0.743 / 0.364 / 0.52 | 1.932 / 1.579 / 0.07 |
| 1.05 | — | — | 0.687 / 0.311 / 0.68 | 1.848 / 1.482 / 0.08 |
| 1.1 | 0.660 / 0.301 / 0.63 | 1.851 / 1.493 / 0.08 | **0.674 / 0.302 / 0.80** | 1.775 / 1.375 / 0.09 |
| 1.15 | 0.635 / 0.283 / 0.75 | 1.808 / 1.433 / 0.09 | 0.698 / 0.315 / 0.88 | 1.723 / 1.285 / 0.11 |
| 1.2 | **0.633 / 0.280 / 0.84** | 1.768 / 1.373 / 0.10 | 0.739 / 0.342 / 0.93 | 1.682 / 1.209 / 0.13 |
| 1.25 | 0.633 / 0.285 / 0.91 | 1.736 / 1.318 / 0.11 | — | — |
| 1.3 | 0.658 / 0.301 / 0.96 | 1.707 / 1.266 / 0.13 | 0.771 / 0.369 / 1.07 | 1.618 / 1.089 / 0.17 |
| 1.4 | 0.702 / 0.332 / 1.02 | 1.664 / 1.178 / 0.15 | 0.833 / 0.412 / 1.15 | 1.567 / 1.002 / 0.23 |
| 1.5 | 0.727 / 0.353 / 1.10 | 1.633 / 1.109 / 0.18 | 0.881 / 0.449 / 1.25 | 1.527 / 0.939 / 0.28 |
| 1.7 | 0.794 / 0.404 / 1.21 | 1.579 / 0.999 / 0.24 | 0.999 / 0.547 / 1.85 | 1.472 / 0.860 / 0.39 |
| 2 | 0.877 / 0.470 / 1.34 | 1.516 / 0.880 / 0.36 | 1.191 / 0.739 / 2.69 | 1.434 / 0.808 / 0.57 |
| 2.25 | — | — | 1.353 / 0.941 / 3.18 | **1.427 / 0.797 / 0.72** |
| 2.5 | 0.980 / 0.558 / 1.52 | 1.453 / 0.785 / 0.58 | 1.587 / 1.240 / 3.62 | 1.435 / 0.801 / 0.86 |
| 3 | 1.074 / 0.643 / 1.66 | 1.416 / 0.761 / 0.79 | 2.029 / 1.862 / 4.05 | 1.496 / 0.839 / 1.11 |
| 3.5 | 1.152 / 0.723 / 1.82 | 1.398 / 0.768 / 0.99 | — | — |
| 4 | 1.230 / 0.808 / 1.98 | **1.391 / 0.792 / 1.17** | — | — |
| 5 | 1.393 / 1.004 / 2.33 | 1.420 / 0.864 / 1.48 | — | — |

| case | RMSE-optimal λ, regular / random | chosen (one value per case) | RMSE cost vs the per-layout optimum, regular / random |
|---|---|---|---|
| S0 | 1.2 / 1.1 | **1.15** | +0.4% / +3.6% |
| S1 | 4 / 2.25 | **2.5** | +4.4% / +0.6% |

- **S0**: the fixed ETKF wants 1.1–1.2, not 1.5 (RMSE −13% regular, −21% random at 1.15), with pooled spread/RMSE
  close to 1 at the optimum; RMSE and CRPS optima agree.
- **S1**: the layouts disagree — the regular grid wants 3.5–4.0 (and its
  RMSE-optimal point is over-dispersed, 1.17: heavy inflation lowers RMSE by
  trusting the observations more, not by representing the model error), the
  random layout 2.0–2.5. Even at its best the S1 filter is over-confident on
  the random layout (≤ 0.86).
- **Choice** (one value per case for both layouts, decided 2026-09-28):
  S0 1.15, S1 2.5 — at most +3.6% (S0 random) / +4.5% (S1 regular) over the
  per-layout optima.

## EnKF

Same protocol (`--method EnKF`, SLURM 56511). The EnKF is untouched by #291;
its old default (S0 1.5, S1 2.0) was simply over-inflated at S0, as the pooled
calibration already showed (1.13 regular, 1.24 random).

| λ | regular S0 | regular S1 | random S0 | random S1 |
|---|---|---|---|---|
| 1 | 0.850 / 0.452 / 0.38 | 1.947 / 1.602 / 0.07 | 0.892 / 0.470 / 0.46 | 1.939 / 1.587 / 0.07 |
| 1.05 | 0.761 / 0.379 / 0.50 | 1.913 / 1.566 / 0.07 | 0.766 / 0.364 / 0.61 | 1.863 / 1.500 / 0.08 |
| 1.1 | 0.700 / 0.331 / 0.63 | 1.863 / 1.507 / 0.08 | **0.721 / 0.325 / 0.78** | 1.788 / 1.394 / 0.09 |
| 1.15 | 0.657 / 0.300 / 0.76 | 1.828 / 1.459 / 0.09 | 0.742 / 0.334 / 0.85 | 1.741 / 1.313 / 0.11 |
| 1.2 | **0.642 / 0.286 / 0.86** | 1.784 / 1.395 / 0.10 | 0.732 / 0.333 / 0.95 | 1.706 / 1.246 / 0.12 |
| 1.3 | 0.663 / 0.298 / 0.98 | 1.728 / 1.300 / 0.12 | 0.777 / 0.363 / 1.07 | 1.648 / 1.136 / 0.16 |
| 1.5 | 0.728 / 0.345 / 1.13 | 1.658 / 1.153 / 0.17 | 0.875 / 0.433 / 1.24 | 1.574 / 0.995 / 0.24 |
| 1.7 | 0.780 / 0.387 / 1.23 | 1.609 / 1.045 / 0.22 | 0.984 / 0.522 / 1.75 | 1.528 / 0.912 / 0.34 |
| 2 | 0.853 / 0.447 / 1.36 | 1.550 / 0.929 / 0.32 | 1.168 / 0.720 / 2.73 | 1.484 / 0.845 / 0.48 |
| 2.5 | 0.961 / 0.541 / 1.53 | 1.492 / 0.828 / 0.51 | 1.580 / 1.227 / 3.61 | **1.460 / 0.814 / 0.75** |
| 3 | 1.048 / 0.626 / 1.69 | 1.459 / 0.788 / 0.71 | 2.024 / 1.849 / 4.04 | 1.491 / 0.836 / 1.00 |
| 3.5 | 1.132 / 0.712 / 1.84 | 1.434 / 0.783 / 0.91 | 2.344 / 2.354 / 4.24 | 1.574 / 0.894 / 1.19 |
| 4 | 1.222 / 0.804 / 1.99 | **1.424 / 0.803 / 1.10** | 2.567 / 2.729 / 4.34 | 1.708 / 0.983 / 1.29 |

| case | RMSE-optimal λ, regular / random | chosen | RMSE cost vs the per-layout optimum, regular / random |
|---|---|---|---|
| S0 | 1.2 / 1.1 | **1.2** | +0.0% / +1.5% |
| S1 | 4 / 2.5 | **3.0** | +2.4% / +2.1% |

**Selection rule, both filters:** one value per case for both observing
systems, minimising the worse of the two layouts' RMSE losses relative to their
own optimum (decided 2026-09-28). It gives ETKF 1.15 / 2.5 and EnKF 1.2 / 3.0;
at S0 the EnKF RMSE at the new default is 12% (regular) / 16% (random) lower than at the old 1.5.

## Caveats

- 50 windows, unseeded ensembles: differences below ~1% between neighbouring
  λ are within noise (e.g. regular S0 1.15–1.25).
- No benchmark row has been rerun at the new defaults yet; every ETKF/EnKF
  number in the L96 reports is still at inflation 1.5 / 2.0 (and, for the ETKF,
  the pre-#291 filter).
