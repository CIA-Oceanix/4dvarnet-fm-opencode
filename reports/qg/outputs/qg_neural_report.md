# QG Case Study: Benchmarked Schemes Overview

Two-layer quasi-geostrophic (QG) Phillips-channel case study -- the four DA baselines (ETKF, EnKF, Strong-4DVar, Weak-4DVar) plus five neural DirectUNet variants (Q1 obs-only, Q3 oracle forcing+param conditioning, Q4 noisy-trained conditioning, Q3-noise0.05, Q5 noisy conditioning + IC), on both the S0 (no model error) reference case and the S1 (combined model-error) case.

## 1. Benchmarked schemes

| scheme | type | key config | description |
|---|---|---|---|
| ETKF | Ensemble DA (deterministic square-root) | N=80, inflation=1.0, loc_radius=6.0, etkf_ridge=1.0 | Ensemble Transform Kalman Filter -- deterministic ensemble-square-root analysis update, sequentially cycled over the assimilation window. No stochastic observation perturbation. `etkf_ridge=1.0` (Kalman-gain transform-matrix regularization) is the default as of 2026-09-12 -- see `qg_da_report.md`'s sensitivity-analysis section. |
| EnKF | Ensemble DA (stochastic, perturbed-obs) | N=80, inflation=1.0, loc_radius=6.0 | Perturbed-observation Ensemble Kalman Filter -- each ensemble member assimilates an independently perturbed observation. Same hyperparameters as ETKF for a controlled comparison. |
| Strong-4DVar | Variational DA (deterministic, perfect-model) | window=60 steps, LBFGS, max_iter=60, b_var_scale=1.0 | 4D-Var assuming the DA dynamical model is exact over the assimilation window (strong constraint) -- optimizes only the initial condition. |
| Weak-4DVar | Variational DA (deterministic, weak-constraint) | window=60 steps, LBFGS, max_iter=60, b_var_scale=1.0, q_var_scale=0.1 | 4D-Var with an added per-step model-error control term (weak constraint) -- can partially compensate for a biased/mismatched dynamical model, at the cost of a larger control space. |
| Q1 (DirectUNet) | Neural (deterministic, single-pass, supervised) | MONAI circular 2D U-Net, hidden=[64,128,256] (M tier), obs-only, cosine LR, 200 epochs | Direct single forward-pass estimator mapping observations to the full state (no iterative assimilation cycle, no dynamical model at inference time). Circular-padded Conv2d over the doubly-periodic (ny,nx) grid; trained via supervised regression on a combined psi + weighted PV-q loss. |
| Q3 (oracle cond.) | Neural (deterministic, single-pass, oracle-conditioned) | Q1 arch + true wind_curl field + true [U1,rd,rek] params (cond_mode="true"), lag=1.0d/noise=0.01 training | Q1 plus exact (unjittered) forcing/parameter conditioning, mirroring the L96 SDA CFM oracle-conditioning study. Trained and tested at train_qg_neural.py's own lag=1.0d/noise=0.01 default. |
| Q4 (noisy cond.) | Neural (deterministic, single-pass, robustly-conditioned) | Q1 arch + resampled-severity corrupted forcing/params (cond_mode="noisy", noisy_max=1.5), lag=1.0d/noise=0.01 training | Q1 plus forcing/parameter conditioning where every training draw samples a fresh random corruption severity in [0, noisy_max] (mirroring the L96 SDA3 CFM study), instead of Q3's always-exact conditioning -- intended to be more robust to conditioning-input error at inference time. |
| Q3-noise0.05 | Neural (deterministic, single-pass, oracle-conditioned) | Same as Q3, retrained at lag=5.0d/noise=0.05 (the DA reference case's own values), gradient_clip_val=1.0 | Q3 retrained at the DA baselines' own obs-noise level instead of train_qg_neural.py's easier 0.01 default, for a genuinely matched comparison. `gradient_clip_val=1.0` (was the 10.0 default) is required at this noise level -- the default clip caused a reproducible training collapse, confirmed via a seed ablation to be independent of random seed; see PLAN.md's 2026-09-12/13 collapse-ablation notes. |
| Q5 (noisy+IC) | Neural (deterministic, single-pass, robustly-conditioned + IC) | Q4-style noisy cond. (noisy_max=2.0, s1_param_bias=s1_amp_bias=0.1) plus the raw initial-condition snapshot as a 4th input; lag=5.0d/noise=0.05 training, gradient_clip_val=1.0 | Adds the same background information a DA method's own init_state gives it for free -- the raw IC (inverted to psi, broadcast identically across the window, a third conditioning class distinct from the per-day forcing field and the per-window param vector). Trained at the DA reference case's own lag/noise so the IC is actually informative (a lag=1.0d snapshot is barely decorrelated). |

DA-method reference-case detail (full lag/noise sensitivity analysis, per-layer breakdown, reconstruction figures) is in `qg_da_report.md` (`generate_qg_da_report.py`) -- not repeated here.

## 2. Summary metrics (S0 + S1, N=100, lag=5.0d/noise_frac=0.05/s1_param_bias=s1_amp_bias=0.1)

Pooled EV (higher is better) on ψ (streamfunction, both layers) and PV-q (both layers). All 9 rows evaluated at the identical config above.

| scheme | S0 ψ EV | S0 PV-q EV | S1 ψ EV | S1 PV-q EV |
|---|---|---|---|---|
| EnKF | 0.9474 | **0.4812** | 0.8957 | 0.3308 |
| ETKF | 0.9570 | *0.4760* | 0.9261 | *0.3570* |
| Weak-4DVar | *0.9660* | -0.0345 | **0.9468** | -0.5008 |
| Strong-4DVar | **0.9714** | -0.1257 | 0.9314 | -0.8573 |
| Q1 (DirectUNet, obs-only) † | 0.8971 | 0.1605 | 0.8971 | 0.1605 |
| Q3 (oracle forcing+param cond.) † | 0.9049 | 0.1771 | 0.8909 | 0.1411 |
| Q4 (noisy-trained forcing+param cond.) † | 0.9032 | 0.1710 | 0.9010 | 0.1643 |
| Q3-noise0.05 (oracle cond., matched noise) | 0.9106 | 0.1892 | 0.8931 | 0.1459 |
| Q5 (noisy cond. + IC, matched lag/noise/bias) | 0.9347 | 0.3867 | *0.9335* | **0.3850** |

(Best per column **bolded**, second-best *italicized*, ranked across all 9 rows.)

† **Not apples-to-apples**: trained at lag=1.0d/noise_frac=0.01 (train_qg_neural.py's easier defaults) and only re-evaluated -- not retrained -- at this lag=5.0d/noise=0.05 config, so it's being tested outside its training distribution. Q3-noise0.05 and Q5 (no marker) were trained at this exact config and are a genuinely fair comparison.

