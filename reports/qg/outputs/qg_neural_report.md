# QG Case Study: Benchmarked Schemes Overview

Two-layer quasi-geostrophic (QG) Phillips-channel case study -- the four DA baselines (ETKF, EnKF, Strong-4DVar, Weak-4DVar) plus four neural DirectUNet variants on the T-merged-into-channels backbone (Q1 obs-only, Q2 oracle forcing+param conditioning, Q3 noisy-trained conditioning, Q4 noisy conditioning + IC), on both the S0 (no model error) reference case and the S1 (combined model-error) case.

## 1. Benchmarked schemes

| scheme | type | key config | description |
|---|---|---|---|
| ETKF | Ensemble DA (deterministic square-root) | N=80, inflation=1.0, loc_radius=2.0, etkf_ridge=0.1 | Ensemble Transform Kalman Filter -- deterministic ensemble-square-root analysis update, sequentially cycled over the assimilation window. No stochastic observation perturbation. `loc_radius=2.0`/`etkf_ridge=0.1` is the default as of 2026-09-15 (was loc_radius=6.0/etkf_ridge=1.0) -- see `qg_da_report.md`'s sensitivity-analysis section. |
| EnKF | Ensemble DA (stochastic, perturbed-obs) | N=80, inflation=1.0, loc_radius=2.0 | Perturbed-observation Ensemble Kalman Filter -- each ensemble member assimilates an independently perturbed observation. Same hyperparameters as ETKF for a controlled comparison. |
| Strong-4DVar | Variational DA (deterministic, perfect-model) | window=60 steps, LBFGS, max_iter=60, b_var_scale=1.0 | 4D-Var assuming the DA dynamical model is exact over the assimilation window (strong constraint) -- optimizes only the initial condition. |
| Weak-4DVar | Variational DA (deterministic, weak-constraint) | window=60 steps, LBFGS, max_iter=60, b_var_scale=1.0, q_var_scale=0.1 | 4D-Var with an added per-step model-error control term (weak constraint) -- can partially compensate for a biased/mismatched dynamical model, at the cost of a larger control space. |
| Q1 (DirectUNet, T-channels) | Neural (deterministic, single-pass, supervised) | MonaiDirectUNetQGChannelTime, hidden=[64,128,256] (M tier), obs-only, lag=5.0d/noise=0.05 training, gradient_clip_val=1.0, 200 epochs | Direct single forward-pass estimator mapping observations to the full state (no iterative assimilation cycle, no dynamical model at inference time). Merges the T (days) axis into the *channel* dimension (`models.monai_unet_qg2d.MonaiUNet2DQGSolver`) so the 2D circular-conv backbone's ordinary channel mixing can relate one day to another, unlike the older batch-folding backbone (every day processed fully independently). Trained via supervised regression on a combined psi + weighted PV-q loss. |
| Q2 (oracle cond.) | Neural (deterministic, single-pass, oracle-conditioned) | Q1 arch + true wind_curl field + true [U1,rd,rek] params (cond_mode="true"), lag=5.0d/noise=0.05 training | Q1 plus exact (unjittered) forcing/parameter conditioning, mirroring the L96 SDA CFM oracle-conditioning study. |
| Q3 (noisy cond.) | Neural (deterministic, single-pass, robustly-conditioned) | Q1 arch + resampled-severity corrupted forcing/params (cond_mode="noisy", noisy_max=1.5), lag=5.0d/noise=0.05 training | Q1 plus forcing/parameter conditioning where every training draw samples a fresh random corruption severity in [0, noisy_max] (mirroring the L96 SDA3 CFM study), instead of Q2's always-exact conditioning -- intended to be more robust to conditioning-input error at inference time. |
| Q4 (noisy cond. + IC) | Neural (deterministic, single-pass, robustly-conditioned + IC) | Q3-style noisy cond. (noisy_max=2.0, s1_param_bias=s1_amp_bias=0.1) plus the raw initial-condition snapshot as a 4th input; lag=5.0d/noise=0.05 training, gradient_clip_val=1.0 | Adds the same background information a DA method's own init_state gives it for free -- the raw IC (inverted to psi, broadcast identically across the window, a third conditioning class distinct from the per-day forcing field and the per-window param vector). |

DA-method reference-case detail (full lag/noise sensitivity analysis, per-layer breakdown, reconstruction figures) is in `qg_da_report.md` (`generate_qg_da_report.py`) -- not repeated here.

## 2. Summary metrics (S0 + S1, N=100, lag=5.0d/noise_frac=0.05/s1_param_bias=s1_amp_bias=0.1)

Pooled EV (higher is better) on ψ (streamfunction, both layers) and PV-q (both layers). All 8 rows evaluated at the identical config above.

| scheme | S0 ψ EV | S0 PV-q EV | S1 ψ EV | S1 PV-q EV |
|---|---|---|---|---|
| EnKF | 0.9584 | 0.5000 | 0.9410 | 0.4126 |
| ETKF | 0.9593 | 0.4982 | 0.9424 | 0.4116 |
| Weak-4DVar | 0.9660 | -0.0345 | 0.9468 | -0.5008 |
| Strong-4DVar | 0.9714 | -0.1257 | 0.9314 | -0.8573 |
| Q1 (DirectUNet, T-channels, obs-only) | 0.9805 | *0.6238* | *0.9805* | *0.6238* |
| Q2 (oracle forcing+param cond.) | **0.9827** | 0.6093 | 0.9490 | 0.5571 |
| Q3 (noisy-trained forcing+param cond.) | 0.9798 | 0.5978 | 0.9763 | 0.5959 |
| Q4 (noisy cond. + IC) | *0.9824* | **0.6745** | **0.9808** | **0.6744** |

(Best per column **bolded**, second-best *italicized*, ranked across all 8 rows.)

All four neural rows (Q1-Q4) were trained directly at this exact lag=5.0d/noise=0.05/s1_param_bias=s1_amp_bias=0.1 config -- a genuinely fair, matched-distribution comparison against the DA baselines for every row, no per-row caveat needed. (An earlier version of this table covered an older batch-folding-backbone DirectUNet family where three of five rows were only re-evaluated, not retrained, at this config and needed a † marker -- see PLAN.md's 2026-09-14 "T-channels bench refresh" section for the full history.)

