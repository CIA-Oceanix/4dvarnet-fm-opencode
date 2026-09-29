# L96 benchmark under the benchmark default -- regular and random observing systems

Two-scale L96, 24D observed space (8 slow + 16 fast), 200 shared test windows, S0 (true parameters) and S1 (biased DA parameters / corrupted forcing). Every cell is the mean over the 200 windows of the per-window RMSE (physical units); `±` is the sd across training seeds where a row has several.

## Protocol

- **Training (benchmark default, `config/l96_benchmark_default.yaml`)**: P1 recipe (monai, `normalize`, cosine annealing, 400 epochs, lr 1e-3, clip 10, batch 16; the default budget was raised to 1200 epochs on 2026-09-26 -- this report keeps the 400-epoch runs, the 1200-epoch ones are in `l96_benchmark_extended.md`) with a random observing system redrawn every batch with fresh noise -- 10-300 stratified obs times (step 0 always observed), 4-16 observed fast channels per window (subset per obs time), slow channels always. Validation windows re-observed once with a fixed seed from the same distribution (checkpoint = `stage1_best` by that val loss). 3 seeds.
- **P1 fixed obs**: the P1 checkpoints -- regular 30-obs grid, noise frozen per window across epochs.
- **Flow sampling (VanillaCFM, PredictStateCFM)**: 30 members, 20 early-fine Euler steps tau_k = 1-(1-k/20)^0.5 (the models' default grid since 2026-09-24; earlier renders of this report used 10 uniform steps, `ens30_no10`). See `docs/results/l96_cfm_tau_consistency_l96b.md`.
- **SDA**: the P1 checkpoints unchanged -- the prior and its validation loss never see observations, so the obs protocol does not apply to training. Guided sampling: 30 members, 10 steps, guidance weight 20 (tuned on the regular grid), r_var 0.5, with the NaN-channel guidance fix (#244). Training budget 400 epochs, the same as every learned row here, so this report is budget-matched; the SDA priors have **not** been retrained at the 1200-epoch default (2026-09-26), and the extended report's SDA rows are still 400-epoch checkpoints.
- **DA**: ETKF / EnKF (30 members, inflation S0 1.5 / S1 2.0), Strong-4DVar; per-window `fast_weights` in the forward model; DA window 500.
- **Regular test set**: the P1 cache, 30 regular obs times, all 24 channels.
- **Random test set** (canonical, `l96_testset_rlayout_n10-100_k4-16_w200_d1.pt`, sha256 `48688eebf8f4...`): the same 200 windows re-observed with the exact layouts the DA baselines assimilated (`eval_da_random_layout_l96.py`): n_obs uniform in 10-100 at stratified times with step 0 observed and at least one obs per DA window, 4-16 observed fast channels per window. Truth, forcings and parameters are bitwise the P1 cache's.
- **Consistency**: before rendering, every learned result was checked to name its test set and to carry that set's truth window for window, and both DA runs to match the canonical layouts (`scripts/check_l96_testset_consistency.py`). All passed.
- **Metrics**: flows are scored on the 30-member ensemble mean (RMSE) plus ensemble CRPS and spread/RMSE; DirectUNet is a single pass; DA stores no members, so it has no CRPS.

![S0 RMSE, regular vs random test set](l96_benchmark_default.png)

## Findings

1. **Learned schemes under the benchmark default beat every DA baseline on S0, on both test sets**: DirectUNet-M 0.380 vs ETKF 0.685 (regular), 0.493 vs Strong-4DVar 0.742 (random).
2. **Model error (S1)**: every learned scheme is flat (S1/S0 0.98-1.00), while the DA baselines degrade 2.04-2.16x -- their forward model carries the bias.
3. **Family ranking under the benchmark default**: DirectUNet-M 0.380 ± 0.004 < PredictStateCFM-M 0.403 ± 0.005 < VanillaCFM-M 0.430 ± 0.006 (regular S0; the same order on the random set). The P1 ranking, where the flows led, does not survive fair training: P1's DirectUNet was overfitting frozen per-window obs noise.
4. **P1 fixed-obs checkpoints do not transfer**: best on the regular grid for the flows, but DirectUNet-M x3.3, PredictStateCFM-M x2.7, VanillaCFM-M x2.9 on the random set, against x1.24-1.30 for the benchmark-default models.
5. **SDA needs no retraining to be robust** (random/regular ~1.2): best on the random set SDA1-M 0.614, between the benchmark-default models and DA on RMSE, far ahead of DA under model error, and M is the right size (L is no better).
6. **DA is the least sensitive to the observing system** (random/regular ETKF 1.16, EnKF 1.19, Strong-4DVar 1.06): its weakness is model error, not the observing system.

## RMSE

| group | scheme | seeds | regular S0 | regular S1 | random S0 | random S1 | random/regular (S0) | S1/S0 (regular) |
|---|---|---|---|---|---|---|---|---|
| DA | ETKF | — | 0.685 | 1.479 | 0.796 | 1.417 | 1.16 | 2.16 |
| DA | EnKF | — | 0.711 | 1.514 | 0.850 | 1.428 | 1.19 | 2.13 |
| DA | Strong-4DVar | — | 0.703 | 1.436 | 0.742 | 1.444 | 1.06 | 2.04 |
| Benchmark default | DirectUNet-M | 3 | 0.380 ± 0.004 | 0.379 ± 0.004 | 0.493 ± 0.002 | 0.490 ± 0.003 | 1.30 | 1.00 |
| Benchmark default | PredictStateCFM-M | 3 | 0.403 ± 0.005 | 0.400 ± 0.006 | 0.509 ± 0.006 | 0.513 ± 0.005 | 1.26 | 0.99 |
| Benchmark default | VanillaCFM-M | 3 | 0.430 ± 0.006 | 0.427 ± 0.005 | 0.535 ± 0.001 | 0.544 ± 0.003 | 1.24 | 0.99 |
| P1 fixed obs | DirectUNet-M | 1 | 0.470 | 0.471 | 1.547 | 1.513 | 3.29 | 1.00 |
| P1 fixed obs | PredictStateCFM-M | 1 | 0.354 | 0.350 | 0.968 | 0.949 | 2.74 | 0.99 |
| P1 fixed obs | VanillaCFM-M | 1 | 0.341 | 0.336 | 0.998 | 0.986 | 2.92 | 0.98 |
| SDA (obs-free training) | SDA1-S+ | 1 | 0.626 | 0.625 | 0.720 | 0.725 | 1.15 | 1.00 |
| SDA (obs-free training) | SDA1-M | 1 | 0.506 | 0.505 | 0.614 | 0.625 | 1.21 | 1.00 |
| SDA (obs-free training) | SDA1-L | 1 | 0.534 | 0.533 | 0.636 | 0.648 | 1.19 | 1.00 |
| SDA (obs-free training) | SDA2-M | 1 | 0.513 | 0.523 | 0.615 | 0.636 | 1.20 | 1.02 |
| SDA (obs-free training) | SDA3-M | 1 | 0.523 | 0.529 | 0.625 | 0.643 | 1.20 | 1.01 |

## Probabilistic scores (flows and SDA, S0)

| group | scheme | CRPS regular | CRPS random | spread/RMSE regular | spread/RMSE random |
|---|---|---|---|---|---|
| Benchmark default | PredictStateCFM-M | 0.183 ± 0.002 | 0.228 ± 0.003 | 0.696 | 0.677 |
| Benchmark default | VanillaCFM-M | 0.196 ± 0.005 | 0.239 ± 0.000 | 0.790 | 0.751 |
| P1 fixed obs | PredictStateCFM-M | 0.166 | 0.550 | 0.487 | 0.303 |
| P1 fixed obs | VanillaCFM-M | 0.153 | 0.578 | 0.614 | 0.312 |
| SDA (obs-free training) | SDA1-S+ | 0.318 | 0.369 | 0.520 | 0.456 |
| SDA (obs-free training) | SDA1-M | 0.258 | 0.315 | 0.416 | 0.385 |
| SDA (obs-free training) | SDA1-L | 0.273 | 0.327 | 0.399 | 0.384 |
| SDA (obs-free training) | SDA2-M | 0.242 | 0.290 | 0.549 | 0.519 |
| SDA (obs-free training) | SDA3-M | 0.250 | 0.298 | 0.533 | 0.510 |

## Deterministic variance ratio (S0, predicted/true variance)

| group | scheme | regular | random |
|---|---|---|---|
| Benchmark default | DirectUNet-M | 0.950 | 0.879 |
| P1 fixed obs | DirectUNet-M | 0.978 | 0.820 |

## Per-window RMSE / EV / CRPS summary (current benchmark)

Common to `l96_benchmark_extended.md`, `l96_benchmark_default.md` and `p1_l96_benchmark.md`, and generated from the `benchmark_extended` inputs by `reports/l96/per_window_summary.py`. The DA rows use the #295 inflation (#298), the ETKS is the #299 row, and DirectUNet / CFM are at 1200 epochs. The SDA priors are the 400-epoch checkpoints (not retrained at 1200). The hybrid is shown for reference.

Per window, on the 24 observed channels of the 200 P1 test windows: **RMSE** is the per-channel RMSE over time; **EV** is per-channel 1 - SSE/SST over time, with SST about the window's own mean, so it is lower than the pooled EV of the benchmark JSONs; **CRPS** is taken on the analysis ensemble (DA) or the members (flows, SDA). Each is averaged over channels. Seeds are averaged per window; cells are mean ± sd over the 200 windows. Per column, the best value is in **bold** and the second best in *italics*. Deterministic schemes have no CRPS. The random-layout Strong-4DVar trajectories were not kept, so that cell has no EV.

**RMSE** (lower is better)

| family | scheme | seeds | regular S0 | regular S1 | random S0 | random S1 |
|---|---|---|---|---|---|---|
| DA | ETKF (inflation 1.15 / 2.5) | — | 0.610 ± 0.148 | 1.409 ± 0.230 | 0.679 ± 0.243 | 1.407 ± 0.308 |
| DA | EnKF (inflation 1.2 / 3.0) | — | 0.641 ± 0.143 | 1.416 ± 0.229 | 0.706 ± 0.236 | 1.484 ± 0.370 |
| DA | ETKS (inflation 1.15 / 2.5) | — | 0.497 ± 0.164 | 1.338 ± 0.224 | 0.572 ± 0.258 | 1.347 ± 0.298 |
| DA | Strong-4DVar | — | 0.703 ± 0.199 | 1.436 ± 0.232 | 0.742 ± 0.310 | 1.444 ± 0.246 |
| SDA (gw 25, 400 ep) | SDA1-M | 3 | 0.501 ± 0.083 | 0.500 ± 0.083 | 0.612 ± 0.233 | 0.624 ± 0.229 |
| SDA (gw 25, 400 ep) | SDA2-M | 3 | 0.500 ± 0.086 | 0.509 ± 0.089 | 0.605 ± 0.230 | 0.627 ± 0.226 |
| SDA (gw 25, 400 ep) | SDA3-fix-M | 3 | 0.501 ± 0.083 | 0.501 ± 0.084 | 0.610 ± 0.234 | 0.619 ± 0.228 |
| DirectUNet (1200 ep) | DirectUNet-M | 3 | *0.340* ± 0.062 | 0.339 ± 0.063 | 0.450 ± 0.315 | *0.444* ± 0.269 |
| CFM (1200 ep) | PredictStateCFM-M | 3 | 0.341 ± 0.078 | *0.338* ± 0.079 | *0.443* ± 0.260 | 0.447 ± 0.233 |
| CFM (1200 ep) | VanillaCFM-M | 3 | 0.351 ± 0.079 | 0.349 ± 0.080 | 0.462 ± 0.277 | 0.468 ± 0.250 |
| *reference* | *Hybrid DirectUNet-M(1200 ep) -> SDA3-fix-M* | 3 | **0.312** ± 0.060 | **0.307** ± 0.061 | **0.388** ± 0.225 | **0.385** ± 0.203 |

**EV** (higher is better)

| family | scheme | seeds | regular S0 | regular S1 | random S0 | random S1 |
|---|---|---|---|---|---|---|
| DA | ETKF (inflation 1.15 / 2.5) | — | 0.821 ± 0.067 | 0.265 ± 0.076 | 0.766 ± 0.143 | 0.152 ± 0.303 |
| DA | EnKF (inflation 1.2 / 3.0) | — | 0.802 ± 0.070 | 0.258 ± 0.076 | 0.754 ± 0.135 | 0.028 ± 0.564 |
| DA | ETKS (inflation 1.15 / 2.5) | — | 0.872 ± 0.077 | 0.350 ± 0.084 | 0.818 ± 0.143 | 0.222 ± 0.259 |
| DA | Strong-4DVar | — | 0.745 ± 0.124 | 0.229 ± 0.072 | — | — |
| SDA (gw 25, 400 ep) | SDA1-M | 3 | 0.893 ± 0.013 | 0.893 ± 0.012 | 0.823 ± 0.118 | 0.813 ± 0.119 |
| SDA (gw 25, 400 ep) | SDA2-M | 3 | 0.893 ± 0.013 | 0.888 ± 0.012 | 0.827 ± 0.115 | 0.811 ± 0.116 |
| SDA (gw 25, 400 ep) | SDA3-fix-M | 3 | 0.893 ± 0.013 | 0.892 ± 0.012 | 0.824 ± 0.118 | 0.816 ± 0.117 |
| DirectUNet (1200 ep) | DirectUNet-M | 3 | *0.950* ± 0.011 | *0.950* ± 0.011 | 0.889 ± 0.142 | *0.892* ± 0.124 |
| CFM (1200 ep) | PredictStateCFM-M | 3 | 0.947 ± 0.016 | 0.947 ± 0.016 | *0.890* ± 0.116 | 0.889 ± 0.108 |
| CFM (1200 ep) | VanillaCFM-M | 3 | 0.944 ± 0.016 | 0.944 ± 0.016 | 0.882 ± 0.123 | 0.879 ± 0.116 |
| *reference* | *Hybrid DirectUNet-M(1200 ep) -> SDA3-fix-M* | 3 | **0.957** ± 0.011 | **0.958** ± 0.011 | **0.916** ± 0.095 | **0.916** ± 0.090 |

**CRPS** (lower is better)

| family | scheme | seeds | regular S0 | regular S1 | random S0 | random S1 |
|---|---|---|---|---|---|---|
| DA | ETKF (inflation 1.15 / 2.5) | — | 0.270 ± 0.071 | 0.758 ± 0.132 | 0.306 ± 0.111 | 0.786 ± 0.173 |
| DA | EnKF (inflation 1.2 / 3.0) | — | 0.287 ± 0.069 | 0.763 ± 0.124 | 0.320 ± 0.109 | 0.837 ± 0.186 |
| DA | ETKS (inflation 1.15 / 2.5) | — | 0.238 ± 0.085 | 0.771 ± 0.142 | 0.269 ± 0.122 | 0.744 ± 0.179 |
| DA | Strong-4DVar | — | — | — | — | — |
| SDA (gw 25, 400 ep) | SDA1-M | 3 | 0.255 ± 0.037 | 0.254 ± 0.036 | 0.312 ± 0.108 | 0.320 ± 0.107 |
| SDA (gw 25, 400 ep) | SDA2-M | 3 | 0.239 ± 0.031 | 0.240 ± 0.032 | 0.287 ± 0.087 | 0.292 ± 0.082 |
| SDA (gw 25, 400 ep) | SDA3-fix-M | 3 | 0.249 ± 0.034 | 0.247 ± 0.034 | 0.303 ± 0.101 | 0.305 ± 0.095 |
| DirectUNet (1200 ep) | DirectUNet-M | 3 | — | — | — | — |
| CFM (1200 ep) | PredictStateCFM-M | 3 | *0.150* ± 0.033 | *0.148* ± 0.034 | *0.194* ± 0.115 | *0.196* ± 0.102 |
| CFM (1200 ep) | VanillaCFM-M | 3 | 0.152 ± 0.033 | 0.151 ± 0.034 | 0.202 ± 0.129 | 0.205 ± 0.115 |
| *reference* | *Hybrid DirectUNet-M(1200 ep) -> SDA3-fix-M* | 3 | **0.147** ± 0.026 | **0.144** ± 0.027 | **0.184** ± 0.111 | **0.180** ± 0.097 |


## Caveats

- P1 and SDA rows are single runs; benchmark-default rows are 3 seeds.
- The SDA guidance weight (20) was tuned on the regular grid, not re-tuned for the random set.
- The SDA rows are superseded by `l96_benchmark_extended.md`: validation-tuned guidance weight 25, 3 seeds, and SDA3-fix in place of SDA3 (whose bias conditioning was never active in training). Those rows keep the 400-epoch SDA priors, so neither report compares SDA with 1200-epoch DirectUNet/CFM at matched budget.
- The random test set is one draw of one observing-system distribution (10-100 obs, 4-16 fast channels); rankings between learned families depend on the regime (on a sparser 30-obs / 8-fast set VanillaCFM-M led DirectUNet-M).
- DA rows reuse existing runs on the identical inputs: regular S0 ETKF/EnKF at inflation 1.5, regular S1 and Strong-4DVar from the corrected inflation-2.0 run; random-set rows from the random-layout DA runs (#243).
