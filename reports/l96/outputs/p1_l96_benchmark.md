# P1 L96 benchmark — DA baselines, deterministic, flow-matching and SDA

Two-scale L96, `obs_interval=100`, `obs_j=2` (24D observed space), 200 shared cached test windows. Group `all_obs`; every cell is **mean ± std across windows**, in **physical units**. **S0** = true parameters; **S1** = ±20% parameter perturbation with ±10% bias (the DA forward model uses the biased `*_da` values), so `S1/S0` is a robustness-to-model-error ratio: 1.0 means untouched.

**One recipe for every learned row**: monai backbone, `data.normalize: true`, cosine annealing, 400 epochs, `batch_size: 16`, lr 1e-3, grad clip 10.0, **no obs-density augmentation**. Deviations are called out per row. Unrolled/4DVarNet schemes are out of P1 scope.

**Status (2026-09-27): superseded as the L96 benchmark by `l96_benchmark_extended.md`; kept as the P1-protocol record.** Every learned row here, SDA included, is trained for 400 epochs, so the families are budget-matched *within this report*. The benchmark default has since moved DirectUNet and the CFMs to 1200 epochs, and the SDA priors have since been retrained at 1200 epochs too: section 4 adds those rows (seed 1, the validation-tuned guidance weight 25), flagged because they sit outside this report's 400-epoch budget match; they do not enter the cross-family bests. The 400-epoch SDA rows use guidance weight 20 (validation-tuned: 25). **The DA baselines (section 1) are the current benchmark rows (updated 2026-09-29)**, on the same 200 regular-grid windows as every learned row: DA fast weights, the fixed ETKF square root (#291), the per-method inflation retuned on validation windows (#295: ETKF 1.15 / 2.5, EnKF 1.2 / 3.0), and the ETKS smoother (#299). They replace the P1-protocol DA rows (inflation 2.0, no DA fast weights, so S0 was not a perfect model).

## Protocol

| family | protocol | columns |
|---|---|---|
| DA baselines | per-window assimilation (dws=500, DA fast weights); ETKS = the ETKF plus a full-window smoother | RMSE; analysis-ensemble CRPS and spread for ETKF/EnKF/ETKS; MAE (the CRPS of a point forecast) for Strong-4DVar |
| Deterministic | single forward pass | RMSE/MAE; `var ratio` = predicted/true variance (1.0 calibrated, ~0.35 collapsed) |
| Flow matching | `ens30_no20` (30 members, 20 early-fine steps; `ens30_no10` before 2026-09-24) | ensemble-mean RMSE; proper ensemble CRPS; `spread/RMSE` (1.0 calibrated) |
| SDA | guided `ens30`, `gw=20`, `r_var=0.5` | as above. **Must** use `eval_sda_l96.py`: SDA1 is an unconditional prior and the unguided sampler returns climatological spread |
| tau=0 mean | `mu(x0, 0, y)` over 30 draws | the flow's implied posterior mean as a point estimator; MAE (= CRPS of a point forecast), NOT comparable to ensemble CRPS |

A single draw from a generative model is a strictly worse estimator than its ensemble mean, so the two never share a column.


## 1. DA baselines

| scheme | params | S0 RMSE | S1 RMSE | S1/S0 | S0 CRPS (MAE for 4D-Var) | S1 CRPS (MAE for 4D-Var) | sp/RMSE (S0) | note |
|---|---|---|---|---|---|---|---|---|
| ETKF | λ 1.15 / 2.5 | 0.6102 ± 0.1480 | 1.4093 ± 0.2300 | 2.310 | 0.2704 | 0.7580 | 0.607 |  |
| EnKF | λ 1.2 / 3.0 | 0.6414 ± 0.1434 | 1.4159 ± 0.2287 | 2.207 | 0.2866 | 0.7634 | 0.647 |  |
| **ETKS** | λ 1.15 / 2.5 | **0.4969 ± 0.1644** | 1.3380 ± 0.2237 | 2.692 | 0.2382 | 0.7706 | 0.412 |  |
| Strong-4DVar | — | 0.7028 ± 0.1991 | 1.4362 ± 0.2325 | 2.044 | 0.4464 | 0.9879 | — |  |

Sources (report bundle `da_current_2026-09-29/`): `l96_baselines_trajectories_dws500_s0c_crps_infs0-1.2_s1-3.0_etkf_infs0-1.15_s1-2.5_obsj2_int100_fw_dafw.npz`, `l96_baselines_trajectories_dws500_s0c_inf2.0_etkf_inf2.0_obsj2_int100_fw_dafw.npz`, `l96_baselines_trajectories_dws500_s0c_test_etks-correct-Lfull-infs0-1.15_s1-2.5_infs0-1.2_s1-3.0_etkf_infs0-1.15_s1-2.5_obsj2_int100_fw_dafw.npz`. S0 trajectories are stored in the full 40D state and indexed to the 24D observed subspace; S1 is already reduced. Rows and protocol: `l96_benchmark_extended.md`.


## 2. Deterministic point estimators

| scheme | params | S0 RMSE | S1 RMSE | S1/S0 | S0 MAE | S1 MAE | var ratio | note |
|---|---|---|---|---|---|---|---|---|
| DirectUNet-S+ | 1.48 M | 0.4899 ± 0.0773 | 0.4894 ± 0.0751 | 0.999 | 0.3130 | 0.3138 | 1.000 |  |
| **DirectUNet-M** | 5.89 M | **0.4699 ± 0.0731** | 0.4706 ± 0.0716 | 1.002 | 0.2947 | 0.2948 | 0.978 |  |
| DirectUNet-L | 23.5 M | 0.8579 ± 0.1277 | 0.8590 ± 0.1296 | 1.001 | 0.5777 | 0.5792 | 0.724 | single unreliable run — an identical config gave 0.4705 on another seed |

### The flows' tau=0 mean components, as deterministic estimators (S0)

| scheme | RMSE | MAE | draw dispersion |
|---|---|---|---|
| PredictStateCFM-L (tau=0) | 0.4072 ± 0.0658 | 0.2555 | 0.0497 |
| PredictStateCFM-M (tau=0) | 0.4081 ± 0.0669 | 0.2582 | 0.0546 |
| VanillaCFM-M (tau=0) | 0.4101 ± 0.0689 | 0.2637 | 0.0919 |
| VanillaCFM-L (tau=0) | 0.4183 ± 0.0672 | 0.2654 | 0.0739 |
| PredictStateCFM-S+ (tau=0) | 0.4481 ± 0.0740 | 0.2922 | 0.0732 |
| VanillaCFM-S+ (tau=0) | 0.4726 ± 0.0849 | 0.3244 | 0.1227 |

`draw dispersion` is the across-draw scatter of `m` itself; 0 would mean fully deterministic in `y`. These rows cost 30 model calls against DirectUNet's 1.


## 3. Flow matching

| scheme | params | S0 RMSE | S1 RMSE | S1/S0 | S0 CRPS | S1 CRPS | sp/RMSE (S0) | note |
|---|---|---|---|---|---|---|---|---|
| VanillaCFM-S+ | 1.48 M | 0.4096 ± 0.0917 | 0.4070 ± 0.0870 | 0.994 | 0.2020 ± 0.0495 | 0.2023 ± 0.0469 | 0.639 |  |
| **VanillaCFM-M** | 5.89 M | **0.3412 ± 0.0734** | 0.3358 ± 0.0727 | 0.984 | 0.1527 ± 0.0320 | 0.1509 ± 0.0318 | 0.614 |  |
| VanillaCFM-L | 23.5 M | 0.3522 ± 0.0678 | 0.3488 ± 0.0649 | 0.990 | 0.1578 ± 0.0294 | 0.1565 ± 0.0281 | 0.595 |  |
| PredictStateCFM-S+ | 1.48 M | 0.4079 ± 0.0793 | 0.4073 ± 0.0808 | 0.998 | 0.1986 ± 0.0361 | 0.1994 ± 0.0382 | 0.485 |  |
| PredictStateCFM-M | 5.89 M | 0.3536 ± 0.0716 | 0.3504 ± 0.0734 | 0.991 | 0.1656 ± 0.0320 | 0.1650 ± 0.0334 | 0.487 |  |
| PredictStateCFM-L(lr3e-4) | 23.5 M | 0.3459 ± 0.0702 | 0.3412 ± 0.0708 | 0.987 | 0.1605 ± 0.0312 | 0.1589 ± 0.0320 | 0.470 | lr 3e-4 — the standard 1e-3 diverged at this tier |
| PredictStateCFM-L(lr5e-4) | 23.5 M | 0.3641 ± 0.0685 | 0.3594 ± 0.0662 | 0.987 | 0.1664 ± 0.0299 | 0.1650 ± 0.0298 | 0.488 | lr 5e-4 — the standard 1e-3 diverged at this tier |

## 4. SDA (score-based prior + guidance)

| scheme | params | S0 RMSE | S1 RMSE | S1/S0 | S0 CRPS | S1 CRPS | sp/RMSE (S0) | note |
|---|---|---|---|---|---|---|---|---|
| SDA1-S+ | 1.48 M | 0.6260 ± 0.1132 | 0.6253 ± 0.1118 | 0.999 | 0.3178 ± 0.0525 | 0.3174 ± 0.0518 | 0.520 |  |
| **SDA1-M** | 5.89 M | **0.5063 ± 0.0912** | 0.5049 ± 0.0906 | 0.997 | 0.2579 ± 0.0415 | 0.2577 ± 0.0407 | 0.416 |  |
| SDA1-L | 23.5 M | 0.5342 ± 0.0951 | 0.5332 ± 0.0948 | 0.998 | 0.2732 ± 0.0454 | 0.2727 ± 0.0446 | 0.399 |  |
| SDA2-M | 5.89 M | 0.5125 ± 0.0966 | 0.5226 ± 0.1005 | 1.020 | 0.2423 ± 0.0349 | 0.2431 ± 0.0362 | 0.549 |  |
| SDA3-M | 5.89 M | 0.5228 ± 0.1002 | 0.5295 ± 0.1035 | 1.013 | 0.2504 ± 0.0372 | 0.2474 ± 0.0373 | 0.533 | bias never active in training (DA params = truth): identical to SDA2 by construction |
| SDA3-fix-M | 5.89 M | 0.4984 ± 0.0930 | 0.5065 ± 0.0964 | 1.016 | 0.2469 ± 0.0397 | 0.2503 ± 0.0412 | 0.454 | SDA3 with the noisy-DA-bias conditioning active (added 2026-09-26) |
| SDA1-M, 1200 ep | 5.89 M | 0.4577 ± 0.0816 | 0.4564 ± 0.0812 | 0.997 | 0.2321 ± 0.0365 | 0.2309 ± 0.0353 | 0.408 | 1200-epoch prior, gw 25 (validation-tuned) -- outside this report's 400-epoch budget match |
| SDA2-M, 1200 ep | 5.89 M | 0.4506 ± 0.0820 | 0.4612 ± 0.0844 | 1.023 | 0.2139 ± 0.0302 | 0.2171 ± 0.0300 | 0.532 | 1200-epoch prior, gw 25 (validation-tuned) -- outside this report's 400-epoch budget match |
| SDA3-fix-M, 1200 ep | 5.89 M | 0.4564 ± 0.0861 | 0.4586 ± 0.0856 | 1.005 | 0.2207 ± 0.0362 | 0.2216 ± 0.0350 | 0.480 | 1200-epoch prior, gw 25 (validation-tuned) -- outside this report's 400-epoch budget match |

## Cross-family reading

Best S0 of each family: **flow matching 0.3412**, deterministic 0.4699, SDA 0.5063, DA baselines 0.4969 — flow matching is 27% better than the deterministic baseline and 33% better than SDA, at matched tier, parameter count, schedule and data.

- **The S1/S0 ratio separates the two worlds.** Every learned scheme is essentially flat under model error (ratio ~1.00) because it never uses a forward model; the DA baselines degrade by 2.0-2.7x, since their forward operator carries the bias. That makes the S1 column the strongest argument for the learned schemes, and it is a structural difference rather than a tuning one.
- **M is the right tier for every learned family.** S+ -> M is a large gain everywhere; M -> L gains nothing and is actively unreliable (2 of 5 L-tier runs failed to train).
- **The CFM parameterization is irrelevant.** VanillaCFM (velocity target) and PredictStateCFM (endpoint target) are statistically identical at S+ (paired t = 0.7, p = 0.48) despite a 3x gap in training val_loss — val_loss is not comparable across objectives.
- **SDA's params conditioning buys nothing here**: SDA2/SDA3 never beat SDA1-M, and with the biased DA params at S1 (fixed 2026-09-25; S1 previously fed them the true params) both lose 1-2%, since both were trained with DA params equal to the true ones. The guidance weight is worth far more (8% from tuning alone); an SDA3 trained on noisy DA params is robust at S1 at the tuned gw 25 (`l96_benchmark_extended.md`) -- at this report's gw 20 its SDA3-fix-M row still loses 1.6%, and on the random observing system it degrades about as much as the unconditional SDA1 (1.5% vs 2.0% at gw 25). At 1200 epochs (flagged rows) every SDA prior gains 7-9% at S0 and the pattern holds: SDA2 still loses at S1, SDA3-fix does not.
- **Every flow's tau=0 mean beats DirectUNet as a point estimator**, so the advantage is not only about sampling.

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

- DA baselines receive the same per-window parameters as truth generation (S0) or their biased `*_da` counterparts (S1); the learned schemes see observations only. This is what makes the comparison apples-to-apples, and also why their S1 behaviour differs so much.
- DA CRPS is the per-window analysis-ensemble CRPS stored by the runs (30 members), comparable to the generative families' ensemble CRPS; Strong-4DVar's column is its MAE.
- SDA is the only learned family with a tuned inference hyper-parameter (`gw`).
- SDA budget: the unflagged SDA rows are 400-epoch priors, like every other row here; the flagged `1200 ep` rows are the benchmark-budget priors (seed 1; 3-seed rows in `l96_benchmark_extended.md`) and are not budget-matched with this report's other families.
- The two PredictStateCFM-L rows use a non-standard lr, forced by an optimization failure at 1e-3 (val_loss jumped 6x at epoch 10 and never recovered).
- L-tier numbers are single runs with large measured seed sensitivity (DirectUNet-L: 0.4705 vs 0.8579 on two seeds of one config). Treat any single L cell as indicative.
- Checkpoint-selection noise on this family was measured at 15-21%; smaller differences need the paired within-window tests, not these point estimates.

## Reconstruction examples

Best / median / worst windows, ranked by Strong-4DVar per-window RMSE (the same convention as the consolidated benchmark, so window choices are comparable across reports). Rows are Truth / Obs / one best scheme per subcategory; columns are the state and |error| maps for the slow X (8D) and fast Y (16D) blocks. Generative rows are plotted as their **ensemble mean** — the estimator their RMSE column scores.

| case | rank | window | 4DVar win-RMSE | ETKS | Strong-4DVar | DirectUNet-M | VanillaCFM-M | SDA1-M |
|---|---|---|---|---|---|---|---|---|
| S0 | best | 146 | 0.352 | 0.437 | 0.352 | 0.391 | 0.277 | 0.405 |
| S0 | median | 2 | 0.773 | 1.137 | 0.773 | 0.632 | 0.517 | 0.727 |
| S0 | worst | 56 | 1.497 | 0.724 | 1.497 | 0.651 | 0.580 | 0.629 |
| S1 | best | 35 | 0.975 | 0.882 | 0.975 | 0.408 | 0.249 | 0.363 |
| S1 | median | 10 | 1.474 | 1.341 | 1.474 | 0.525 | 0.332 | 0.496 |
| S1 | worst | 75 | 1.990 | 1.778 | 1.990 | 0.645 | 0.497 | 0.774 |

**Observing system.** `obs_mask` is a single 1-D mask over time shared by every dimension, so slow and fast are observed at exactly the same instants: **30 observed timesteps per window** (`obs_interval=100` over T=3000). What differs between them is spatial, not temporal — all 8 slow X are observed, but only 16 of the 32 fast Y (`obs_j=2` of `J=4`), which is why the fast block is 16 rows.

Each observation is drawn as a 11-column band rather than a single 1/3000 column, which would be ~0.15 px and mostly vanish under rasterization. That is display-only and inflates the Obs row's |error| column (labelled 'obs noise') by ~1.7%; the undistorted values are S0/best 0.7003, S0/median 0.7113, S0/worst 0.6836, S1/best 0.7013, S1/median 0.7116, S1/worst 0.7327.

![S0 best](figures/p1_l96_hovm_s0_best.png)
![S0 median](figures/p1_l96_hovm_s0_median.png)
![S0 worst](figures/p1_l96_hovm_s0_worst.png)
![S1 best](figures/p1_l96_hovm_s1_best.png)
![S1 median](figures/p1_l96_hovm_s1_median.png)
![S1 worst](figures/p1_l96_hovm_s1_worst.png)

