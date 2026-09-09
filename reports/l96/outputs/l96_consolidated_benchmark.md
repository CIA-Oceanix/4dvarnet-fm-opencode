# L96 Consolidated Benchmark — DA baselines vs neural models

Setup: two-scale L96, Obs30 (`obs_interval=100`, `obs_j=2` → 24D observed space), dws=500, 200 shared cached test windows; S1 = ±20% params + ±10% bias (DA forward model uses biased `*_da`).

RMSE/EV are recomputed from the stored trajectory arrays via `evaluation/estimate_metrics.py`; ES for DA ensemble methods (EnKF/ETKF) and L3 (ens30×10) are proper ensemble scores (N=30, MAE − 0.5·pairwise spread) read from cached run outputs; ES for deterministic methods is the N=1 per-dim MAE proxy. **bold** marks the best value per column.

## Benchmarked schemes

| ID | Type | Description |
|---|---|---|
| Strong-4DVar | Variational | Strong-constraint 4D-Var over the dws=500 window (`B_var=2.0`, `R_var=0.5`, `max_iter=10`, `lr=0.2`, autodiff minimization); assimilates the full window trajectory. |
| EnKF | Ensemble KF | Stochastic ensemble Kalman filter, `N_ens=30`, inflation=2.0, no localization; sequential observation updates. |
| ETKF | Ensemble KF | Deterministic ensemble square-root filter, `N_ens=30`, inflation=2.0, no localization. |
| L1b | Neural (DirectUNet) | Single-pass regression obs → state, hidden [64,128,256]; obs-only conditioning; 200 epochs. |
| L2b | Neural (CFM, τ=0) | Conditional flow matching trained at τ=0 only; sampled with a single Euler step (deterministic, conditional-mean-like); hidden [64,128,256]; 400 epochs. |
| L3 | Neural (CFM, multi-τ) | Standard multi-τ CFM training; evaluated as a 30-member ensemble with 10 Euler steps (`ens30×10`, N=30, deterministic τ-schedule 0→1, fresh x₀ per member); hidden [64,128,256]; 400 epochs. |
| L4 | Neural (DirectUNet) | As L1b with small backbone [32,64,128]. |
| L5 | Neural (CFM, τ=0) | As L2b with small backbone [32,64,128]. |
| L6 | Neural (CFM, τ=0) | As L2b plus corrupted-forcing conditioning (`cond_extra_dim=1`); tests the robustness value of forcing input. |
| V2 | Neural (TweedieCFM) | Two-stage Tweedie CFM: stage-1 MeanEstimatorCell (obs → mean), stage-2 residual velocity UNet; hidden [64,128,256]; 100+400 epochs; multi-τ, **K_inner=1 (kinner1 variant)**; evaluated as a 30-member ensemble with 10 Euler steps (`ens30×10`, N=30); N_outer=10. The V2 row reports the **K_inner=1 ablation** (see the dedicated `l96_tweediecfm_benchmark.md` for the full V2 family). |
| V3 | Neural (PredictStateCFM) | Single-stage CFM predicting the final-state mean μ = E[x₁|x_τ,y]; hidden [64,128,256]; 400 epochs; evaluated as a 30-member ensemble with 10 Euler steps (`ens30×10`, N=30); N_outer=10. |
| SDA1 | Neural (SDA prior + DPS guidance) | Unconditional flow-matching prior p(x₁) -- no obs, params, or forcing conditioning at all, trained on the same S0/S1 mix (`forcing_state_bias=0.1` in train/val, same as every other L-series/V2/V3 config); hidden [64,128,256]; 400 epochs. State estimated at inference time only, via DPS/Pi-GDM-style observation guidance (normalized-gradient step on the Tweedie posterior-mean estimate, `evaluation/sda_sampler.py`) with N_outer=10 Euler steps, guidance_weight=40 (picked by an S0 RMSE sweep over {0.3..400}), R_var=0.5 (matches `data.R_var`); evaluated as a 30-member ensemble (fresh x₀ per member, `ens30×10`, N=30), same convention as L3/V2/V3. |
| SDA2-mixed | Neural (SDA prior, params+forcing cond. + DPS guidance) | As SDA1 but the prior is additionally conditioned on the per-window physical params (F, c1, hx, eps, w1-w4) and the corrupted forcing signal (`ConditionalPriorCFM`, `models/sda.py`) -- obs is still never a network input, only the guidance term at inference conditions on it. Trained on the identical S0/S1 mix as SDA1 (`forcing_state_bias=0.1`); hidden [64,128,256]; 400 epochs; guidance_weight=40, N_outer=10, R_var=0.5; evaluated as a 30-member ensemble (`ens30×10`, N=30). |
| SDA2-nominal | Neural (SDA prior, params+forcing cond., nominal-only train) | Identical architecture/inference to SDA2-mixed but trained with `forcing_state_bias=0.0` (genuinely nominal-only train/val -- never sees the S1-level forcing corruption at training time, unlike every other row in this table); hidden [64,128,256]; 400 epochs; guidance_weight=40, N_outer=10, R_var=0.5; evaluated as a 30-member ensemble (`ens30×10`, N=30). Tests whether the amortized S1/S0 resilience seen elsewhere in this table survives when training-time exposure to model error is removed entirely. |
| FDV1 | Neural (4DVarNet-style unrolled solver) | Unrolled solver: the update at each of N_outer=10 iterations is the output of a weight-tied UNet1D fed `concat(state, obs)` (`update_input='obs+state'`, no gradient/cost term at all -- see `models/fourdvarnet.py::FourDVarNetSolver`), `x_{k+1} = x_k - (1/N_outer)*UNet(x_k, obs)`, zero-initialized; hidden [64,128,256]; 400 epochs; loss = final-iteration MSE only. Fully deterministic (no ensemble, no randomness anywhere) -- evaluated as a single pass (N=1), same convention as Strong-4DVar/L1b/L2b. Design taxonomy (`update_input` string) traced to CIA-Oceanix/4dvarnet-global-mapping's `ronan_devs` branch (`GradSolver_withStep`); gradient-conditioned modes (`grad-only`/`grad+state`/`subgrad+state`) reserved for a future FDV2. |
| FDV1CFM | Neural (4DVarNet-CFM, PredictStateCFM + FDV1 backbone) | V3 (`PredictStateCFM`) CFM parameterization -- predicts μ = E[x1|x_τ,y] at a randomly-sampled outer flow-time τ, trained via MSE(μ,x1), sampled by forward ODE integration `x += dt*(μ-x)/(1-τ)` over N_outer=10 steps -- but μ is computed by FDV1's own K_inner=5-step weight-tied unrolled `obs+state` refinement (`models/fourdvarnet.py::FourDVarNetPredictStateCFM`), started from the current x_τ, instead of a single UNet1D forward pass as plain V3 uses. Total NFE per sample = N_outer×K_inner = 50 (5x V3's 10, 5x FDV1's 10). hidden [64,128,256]; 400 epochs, single random τ per training batch (cheaper to train than FDV1 itself, which backprops through its full 10-step unroll every batch). A rare (~1-in-several-thousand ens30 samples) divergence of the inner unroll on out-of-distribution x_τ is guarded with a `clip_range=50.0` clamp after each inner step (same convention as this codebase's L96/QG dynamics integrators) -- inactive for in-distribution trajectories (|x|<10). Evaluated as a 30-member ensemble with 10 Euler steps (`ens30×10`, N=30). |
| FDV1+SDA1 | Neural (FDV1 mean + SDA1 warm-started guidance) | No retraining: FDV1's frozen point estimate warm-starts SDA1's guided sampling trajectory (`evaluation/sda_sampler.py`'s `mean_estimate`/`tau0`, a "SDEdit"-style warm start -- `x_τ0 = (1-τ0)·noise + τ0·FDV1_estimate`, Euler-integrated only from τ0 to 1) instead of starting from pure noise; `guided_obs_cost`/the Tweedie x_hat_1 machinery are unchanged. Hyperparameters (`tau0=0.7`, `guidance_weight=2`) picked by an S0-only grid sweep over `tau0∈{0,0.3,0.5,0.7,0.8}×guidance_weight∈{0,1,2,5,10,40,100}` -- any guidance stronger than ~2 actively hurts once warm-started (the DPS step size calibrated for pure-noise starts is too aggressive here); evaluated as a 30-member ensemble (`ens30×10`, N=30). |
| FDV1+SDA2 | Neural (FDV1 mean + SDA2-nominal warm-started guidance) | As FDV1+SDA1 but warm-starting SDA2-nominal (params+forcing-conditioned prior) instead of SDA1; `tau0=0.5`, `guidance_weight=2` (own S0-only grid sweep -- SDA2's conditioning makes more remaining Euler steps useful than SDA1's fully-unconditional prior, hence the lower `tau0`); evaluated as a 30-member ensemble (`ens30×10`, N=30). **New best neural scheme in this table on RMSE/EV** (FDV1CFM above still has the best ES). |
| FDV1+FDV1CFM | Neural (FDV1 mean + FDV1-CFM warm-started sampling) | No retraining: FDV1's frozen point estimate warm-starts FDV1-CFM's own sampling trajectory via the same `mean_estimate`/`tau0` SDEdit-style mechanism as the FDV1+SDA hybrids (`models/fourdvarnet.py::FourDVarNetPredictStateCFM.sample`) -- `x_τ0 = (1-τ0)·noise + τ0·FDV1_estimate`, Euler-integrated only from τ0 to 1. A literal τ0=0 warm start (recentering the τ=0 noise on FDV1's estimate, still running the full N_outer steps) was tried first and made things *worse* (single-sample RMSE 0.897 vs. 0.555 unwarm-started): CFM training pairs τ=0 with near-zero-magnitude noise only, so injecting a real-state-scale mean there is an out-of-training-distribution (τ, |x_τ|) combination that confuses the first refinement step, and that confusion compounds since no steps are skipped. `tau0=0.7` (picked by an S0-only sweep over `tau0∈{0,0.2,...,0.9}`, single-sample RMSE, plateauing over `tau0∈[0.6,0.8]`) avoids this the same way the FDV1+SDA hybrids do. Evaluated as a 30-member ensemble (`ens30×10`, N=30); no clamp activations observed in this evaluation (unlike FDV1CFM alone) since the shorter, better-anchored trajectory has much less room to diverge. |
| DirectUNet-S+(monai,cos) | Neural (DirectUNet, monai backbone) | MonaiUNet1D backbone (FiLM-conditioned ResBlocks vs UNet1D's additive-only conditioning; `models/monai_unet_adapter.py`) swapped into L1b's single-pass regression, plus per-channel z-score normalization (`data.normalize=true`); S+ capacity tier (hidden [32,64,128], 1.48M params), cosine-annealed LR; 200 epochs. |
| DirectUNet-M(monai,flat) | Neural (DirectUNet, monai backbone) | As above, M capacity tier (hidden [64,128,256], 5.89M params), flat LR; 200 epochs. |
| DirectUNet-M(monai,cos) | Neural (DirectUNet, monai backbone) | As above, M tier, cosine-annealed LR (near-identical to the flat-LR M result -- cosine's gain is tier-dependent, see the L tier below). |
| DirectUNet-L(monai,cos) | Neural (DirectUNet, monai backbone) | As above, L capacity tier (hidden [128,256,512], 23.5M params), cosine-annealed LR. Flat LR at this tier was badly unstable (RMSE 0.77-0.89 across two seeds, worse than S+/M); cosine annealing fixed it entirely, making L the best DirectUNet tier overall. Its DA-benchmark eval needs a reduced `--batch-size` (16, not the 200 default) -- the wider bottleneck OOMs a single-batch pass at 200 (26.8GiB attention-adjacent allocation on a 44GB GPU). |
| CFM-M(monai,flat) | Neural (CFM, τ=0, monai backbone) | MonaiUNet1D backbone swapped into L2b's τ=0 CFM, plus per-channel normalization; M tier, flat LR; 400 epochs. Best single monai-backbone result of the non-hybrid schemes. |
| CFM-M(monai,cos) | Neural (CFM, τ=0, monai backbone) | As above, cosine-annealed LR (near-identical to flat -- a wash for this tier). |
| CFM-S+(monai,cos) | Neural (CFM, τ=0, monai backbone) | As above, S+ capacity tier (hidden [32,64,128]), cosine-annealed LR. |
| SDA1(monai) | Neural (SDA prior + DPS guidance, monai backbone) | MonaiUNet1D backbone + per-channel normalization swapped into SDA1's unconditional prior; M tier; 400 epochs. Same pure-noise-start guidance convention as the non-monai SDA1 above (guidance_weight=40, N_outer=10, R_var=0.5; R_var's value is provably inert here -- the DPS step normalizes its own gradient by its norm, which exactly cancels any positive R_var scale factor -- verified empirically, R_var=0.5 vs 50.0 give trajectories differing only by float32 noise). |
| SDA2(monai) | Neural (SDA prior, params+forcing cond. + DPS guidance, monai backbone) | As SDA1-monai but params+forcing-conditioned, mirroring SDA2-mixed above. |
| SDA3(monai) | Neural (SDA prior, noisy-params cond. + DPS guidance, monai backbone) | As SDA2-monai but conditioned on a per-window noisy params estimate (a fresh random 0-1.5x fraction of the true-to-DA bias, `data.noisy_da_bias`/`noisy_da_max`) instead of the true params -- tests robustness to imperfect conditioning. Best plain (non-hybrid) SDA result. |
| DirectUNet+SDA1 | Neural (DirectUNet-M mean + SDA1-monai warm-started guidance) | Same SDEdit-style warm start as the FDV1+SDA hybrids (`evaluation/sda_sampler.py`'s `mean_estimate`/`tau0`), but warm-starting from DirectUNet-M(monai)'s frozen point estimate instead of FDV1's -- i.e. SDA is used to sample the *anomaly* around DirectUNet's reconstruction rather than starting from pure noise. `tau0=0.3`, `guidance_weight=2.0` (S0-only grid sweep over `tau0∈{0,0.3,0.5,0.7,0.8}×guidance_weight∈{0,1,2,5,10,40,100}`, all three SDA variants converged on this same point). |
| DirectUNet+SDA2 | Neural (DirectUNet-M mean + SDA2-monai warm-started guidance) | As above, warm-starting SDA2-monai (params+forcing-conditioned) instead of SDA1-monai. |
| DirectUNet+SDA3 | Neural (DirectUNet-M mean + SDA3-monai warm-started guidance) | As above, warm-starting SDA3-monai (noisy-params conditioned) instead of SDA1-monai. Best RMSE/EV among the DirectUNet-warm-started hybrids -- but see FDV1+SDA3-monai below, which beats it comfortably by swapping in a better mean-estimate model. |
| FDV1(monai) | Neural (4DVarNet-style unrolled solver, monai backbone) | MonaiUNet1D backbone (`unet_backbone=monai`, `models/fourdvarnet.py::FourDVarNetSolver`) swapped into FDV1's `obs+state` unrolled solver (`x_{k+1} = x_k - (1/N_outer)*UNet(x_k, obs)`, N_outer=10, zero-initialized, deterministic, no ensemble); M tier; 400 epochs. Rebased its branch onto `origin/master` first to pick up gradient checkpointing (PR #172), which fixes an earlier CUDA-OOM this exact backbone+solver combination hit at production scale. **Note:** its first eval attempt gave a nonsensical RMSE 3.47/EV -3.75 -- traced to `eval_neural_l96.py`'s `--n-outer` CLI default (1), which is correct for tau0-only CFM but silently starves this solver's zero-init refinement of the N_outer=10 iterations it needs; fixed by passing `--n-outer 10` explicitly, recovering a sane, in fact excellent, result. |
| FDV1+SDA3(monai) | Neural (FDV1-monai mean + SDA3-monai warm-started guidance) | Same SDEdit-style warm start as the DirectUNet+SDA hybrids above, but swapping in FDV1-monai (the unrolled solver) as the mean-estimate model instead of DirectUNet-M -- SDA3-monai is used to sample the anomaly around FDV1's point estimate. `tau0=0.3`, `guidance_weight=2.0` (own S0-only sweep over `tau0∈{0.1..0.5}×guidance_weight∈{0.5,1,2,5}`, converged on the exact same point as the DirectUNet+SDA hybrids' independent sweep). One real wrinkle specific to this combination: FDV1-monai was trained WITHOUT `data.normalize=true` (unlike DirectUNet-M/SDA3, which share that normalized space) -- feeding it normalized obs (as SDA expects) silently produced a garbage mean estimate that poisoned the whole hybrid (confirmed: RMSE 1.36/EV 0.35) before this was caught; fixed by preparing a SEPARATE raw-obs dataloader for FDV1's own `.sample()` call and normalizing its output before handing it to SDA as the warm start. **Best scheme in this table overall.** |
| FDV1+SDA1(monai) | Neural (FDV1-monai mean + SDA1-monai warm-started guidance) | As FDV1+SDA3-monai but warm-starting SDA1-monai (unconditional prior) instead of SDA3-monai, for coherence with the DirectUNet+SDA1/2/3 family above. Same `tau0=0.3`/`guidance_weight=2.0` (not re-swept per SDA variant -- three independent sweeps this session all converged on this exact point regardless of SDA variant or mean model). |
| FDV1+SDA2(monai) | Neural (FDV1-monai mean + SDA2-monai warm-started guidance) | As above, warm-starting SDA2-monai (true-params conditioned) instead of SDA1-monai. |

Shared setup: all L-series neural models are trained and evaluated on the identical DA-parity benchmark (all-5 params ±20% randomized per window; S1 adds a ±10% bias; models operate in the 24D observed subspace with obs-only inputs unless noted). DA baselines receive the same per-window parameters as the truth generation (S0) or their biased `*_da` counterparts (S1), which is what makes the DA-vs-neural comparison apples-to-apples.

## RMSE (pooled, lower is better)

### RMSE by variable group

| Method | S0 all | S0 slow | S0 fast | S1 all | S1 slow | S1 fast | S1/S0 |
|---|---|---|---|---|---|---|---|
| ETKF | 0.8883 | 0.4749 | 1.0950 | 1.4998 | 1.2394 | 1.6299 | 1.688 |
| EnKF | 0.9131 | 0.4974 | 1.1210 | 1.5381 | 1.2490 | 1.6826 | 1.684 |
| Strong-4DVar | 0.8116 | 0.4683 | 0.9833 | 1.4617 | 1.0760 | 1.6546 | 1.801 |
| L1b |   —   |   —   |   —   |   —   |   —   |   —   | n/a |
| L2b |   —   |   —   |   —   |   —   |   —   |   —   | n/a |
| L3 |   —   |   —   |   —   |   —   |   —   |   —   | n/a |
| L4 |   —   |   —   |   —   |   —   |   —   |   —   | n/a |
| L5 |   —   |   —   |   —   |   —   |   —   |   —   | n/a |
| L6 |   —   |   —   |   —   |   —   |   —   |   —   | n/a |
| V2 |   —   |   —   |   —   |   —   |   —   |   —   | n/a |
| V3 |   —   |   —   |   —   |   —   |   —   |   —   | n/a |
| SDA1 |   —   |   —   |   —   |   —   |   —   |   —   | n/a |
| SDA2-mixed |   —   |   —   |   —   |   —   |   —   |   —   | n/a |
| SDA2-nominal |   —   |   —   |   —   |   —   |   —   |   —   | n/a |
| FDV1 |   —   |   —   |   —   |   —   |   —   |   —   | n/a |
| FDV1CFM |   —   |   —   |   —   |   —   |   —   |   —   | n/a |
| FDV1+SDA1 |   —   |   —   |   —   |   —   |   —   |   —   | n/a |
| FDV1+SDA2 |   —   |   —   |   —   |   —   |   —   |   —   | n/a |
| FDV1+FDV1CFM |   —   |   —   |   —   |   —   |   —   |   —   | n/a |
| DirectUNet-S+(monai,cos) | 0.5152 | 0.2974 | 0.6240 | 0.5155 | 0.2977 | 0.6244 | 1.001 |
| DirectUNet-M(monai,flat) | 0.5011 | 0.2671 | 0.6182 | 0.5027 | 0.2656 | 0.6213 | 1.003 |
| DirectUNet-M(monai,cos) | 0.5064 | 0.2630 | 0.6280 | 0.5072 | 0.2635 | 0.6291 | 1.002 |
| DirectUNet-L(monai,cos) | 0.4865 | 0.2576 | 0.6009 | 0.4867 | 0.2553 | 0.6023 | 1.000 |
| CFM-M(monai,flat) | 0.4810 | 0.2594 | 0.5918 | 0.4784 | 0.2564 | 0.5894 | 0.995 |
| CFM-M(monai,cos) | 0.4865 | 0.2611 | 0.5991 | 0.4876 | 0.2587 | 0.6021 | 1.002 |
| CFM-S+(monai,cos) | 0.5325 | 0.3544 | 0.6216 | 0.5335 | 0.3535 | 0.6235 | 1.002 |
| SDA1(monai) | 0.5532 | 0.3073 | 0.6762 | 0.5521 | 0.3054 | 0.6755 | 0.998 |
| SDA2(monai) | 0.5588 | 0.2853 | 0.6955 | 0.5610 | 0.2807 | 0.7011 | 1.004 |
| SDA3(monai) | 0.5365 | 0.2887 | 0.6603 | 0.5355 | 0.2850 | 0.6607 | 0.998 |
| DirectUNet+SDA1 | 0.4274 | 0.1989 | 0.5416 | 0.4252 | 0.1967 | 0.5394 | 0.995 |
| DirectUNet+SDA2 | 0.4288 | 0.1942 | 0.5461 | 0.4268 | 0.1912 | 0.5446 | 0.995 |
| DirectUNet+SDA3 | 0.4204 | 0.1849 | 0.5382 | 0.4183 | 0.1824 | 0.5362 | 0.995 |
| FDV1(monai) | 0.4275 | 0.2125 | 0.5350 | 0.4235 | 0.2100 | 0.5302 | 0.991 |
| FDV1+SDA3(monai) | **0.3783** | **0.1611** | **0.4870** | **0.3736** | **0.1578** | **0.4815** | 0.987 |
| FDV1+SDA1(monai) | 0.3852 | 0.1770 | 0.4892 | 0.3801 | 0.1737 | 0.4833 | 0.987 |
| FDV1+SDA2(monai) | 0.3845 | 0.1688 | 0.4924 | 0.3800 | 0.1650 | 0.4875 | 0.988 |

Note on conventions: the DA metric cache stores the **mean of per-window RMSEs** (evaluation/run_l96.py), while this table uses the **pooled** convention (`sqrt(mean sq err)` over all windows/timesteps) for every method — the same convention as the neural evaluation. Pooled RMSE is ≤ mean-of-window RMSE, so DA values here are slightly lower (more favorable) than in the legacy cache; both orderings agree.

## Explained Variance (higher is better)

### EV by variable group

| Method | S0 all | S0 slow | S0 fast | S1 all | S1 slow | S1 fast |
|---|---|---|---|---|---|---|
| ETKF | 0.6915 | 0.9395 | 0.5676 | 0.2149 | 0.5772 | 0.0338 |
| EnKF | 0.6758 | 0.9337 | 0.5469 | 0.1717 | 0.5706 | -0.0278 |
| Strong-4DVar | 0.7478 | 0.9412 | 0.6510 | 0.2330 | 0.6813 | 0.0089 |
| L1b |   —   |   —   |   —   |   —   |   —   |   —   |
| L2b |   —   |   —   |   —   |   —   |   —   |   —   |
| L3 |   —   |   —   |   —   |   —   |   —   |   —   |
| L4 |   —   |   —   |   —   |   —   |   —   |   —   |
| L5 |   —   |   —   |   —   |   —   |   —   |   —   |
| L6 |   —   |   —   |   —   |   —   |   —   |   —   |
| V2 |   —   |   —   |   —   |   —   |   —   |   —   |
| V3 |   —   |   —   |   —   |   —   |   —   |   —   |
| SDA1 |   —   |   —   |   —   |   —   |   —   |   —   |
| SDA2-mixed |   —   |   —   |   —   |   —   |   —   |   —   |
| SDA2-nominal |   —   |   —   |   —   |   —   |   —   |   —   |
| FDV1 |   —   |   —   |   —   |   —   |   —   |   —   |
| FDV1CFM |   —   |   —   |   —   |   —   |   —   |   —   |
| FDV1+SDA1 |   —   |   —   |   —   |   —   |   —   |   —   |
| FDV1+SDA2 |   —   |   —   |   —   |   —   |   —   |   —   |
| FDV1+FDV1CFM |   —   |   —   |   —   |   —   |   —   |   —   |
| DirectUNet-S+(monai,cos) | 0.8985 | 0.9763 | 0.8595 | 0.8977 | 0.9756 | 0.8588 |
| DirectUNet-M(monai,flat) | 0.9017 | 0.9809 | 0.8621 | 0.9002 | 0.9806 | 0.8601 |
| DirectUNet-M(monai,cos) | 0.8989 | 0.9814 | 0.8576 | 0.8980 | 0.9809 | 0.8565 |
| DirectUNet-L(monai,cos) | 0.9072 | 0.9822 | 0.8698 | 0.9064 | 0.9821 | 0.8686 |
| CFM-M(monai,flat) | 0.9098 | 0.9820 | 0.8737 | 0.9101 | 0.9819 | 0.8742 |
| CFM-M(monai,cos) | 0.9076 | 0.9817 | 0.8705 | 0.9063 | 0.9816 | 0.8687 |
| CFM-S+(monai,cos) | 0.8959 | 0.9663 | 0.8607 | 0.8947 | 0.9656 | 0.8592 |
| SDA1(monai) | 0.8815 | 0.9747 | 0.8350 | 0.8811 | 0.9743 | 0.8345 |
| SDA2(monai) | 0.8764 | 0.9781 | 0.8255 | 0.8741 | 0.9782 | 0.8220 |
| SDA3(monai) | 0.8876 | 0.9775 | 0.8427 | 0.8871 | 0.9775 | 0.8419 |
| DirectUNet+SDA1 | 0.9259 | 0.9894 | 0.8941 | 0.9261 | 0.9893 | 0.8945 |
| DirectUNet+SDA2 | 0.9249 | 0.9899 | 0.8924 | 0.9250 | 0.9899 | 0.8925 |
| DirectUNet+SDA3 | 0.9273 | 0.9908 | 0.8955 | 0.9275 | 0.9908 | 0.8958 |
| FDV1(monai) | 0.9271 | 0.9879 | 0.8967 | 0.9280 | 0.9879 | 0.8981 |
| FDV1+SDA3(monai) | **0.9406** | **0.9930** | **0.9144** | **0.9417** | **0.9931** | **0.9160** |
| FDV1+SDA1(monai) | 0.9396 | 0.9916 | 0.9137 | 0.9408 | 0.9917 | 0.9154 |
| FDV1+SDA2(monai) | 0.9391 | 0.9924 | 0.9125 | 0.9401 | 0.9925 | 0.9139 |

## Energy Score (lower is better)

### ES by variable group

| Method | S0 all | S0 slow | S0 fast | S1 all | S1 slow | S1 fast |
|---|---|---|---|---|---|---|
| ETKF | 0.4548 | 0.2857 | 0.5393 | 0.8511 | 0.7626 | 0.8953 |
| EnKF | 0.4599 | 0.2915 | 0.5440 | 0.9018 | 0.7942 | 0.9556 |
| Strong-4DVar | 0.4850* | 0.3551* | 0.5499* | 0.9892* | 0.8197* | 1.0740* |
| L1b |   —  * |   —  * |   —  * |   —  * |   —  * |   —  * |
| L2b |   —  * |   —  * |   —  * |   —  * |   —  * |   —  * |
| L3 |   —   |   —   |   —   |   —   |   —   |   —   |
| L4 |   —  * |   —  * |   —  * |   —  * |   —  * |   —  * |
| L5 |   —  * |   —  * |   —  * |   —  * |   —  * |   —  * |
| L6 |   —  * |   —  * |   —  * |   —  * |   —  * |   —  * |
| V2 |   —   |   —   |   —   |   —   |   —   |   —   |
| V3 |   —   |   —   |   —   |   —   |   —   |   —   |
| SDA1 |   —   |   —   |   —   |   —   |   —   |   —   |
| SDA2-mixed |   —   |   —   |   —   |   —   |   —   |   —   |
| SDA2-nominal |   —   |   —   |   —   |   —   |   —   |   —   |
| FDV1 |   —  * |   —  * |   —  * |   —  * |   —  * |   —  * |
| FDV1CFM |   —   |   —   |   —   |   —   |   —   |   —   |
| FDV1+SDA1 |   —   |   —   |   —   |   —   |   —   |   —   |
| FDV1+SDA2 |   —   |   —   |   —   |   —   |   —   |   —   |
| FDV1+FDV1CFM |   —   |   —   |   —   |   —   |   —   |   —   |
| DirectUNet-S+(monai,cos) | 0.3278* | 0.2232* | 0.3801* | 0.3290* | 0.2233* | 0.3819* |
| DirectUNet-M(monai,flat) | 0.3064* | 0.1938* | 0.3628* | 0.3082* | 0.1924* | 0.3661* |
| DirectUNet-M(monai,cos) | 0.3080* | 0.1903* | 0.3669* | 0.3101* | 0.1905* | 0.3700* |
| DirectUNet-L(monai,cos) | 0.2937* | 0.1860* | 0.3475* | 0.2944* | 0.1836* | 0.3498* |
| CFM-M(monai,flat) | 0.2943 | 0.1881 | 0.3475 | 0.2937 | 0.1854 | 0.3479 |
| CFM-M(monai,cos) | 0.2959 | 0.1870 | 0.3504 | 0.2972 | 0.1854 | 0.3531 |
| CFM-S+(monai,cos) | 0.3427 | 0.2677 | 0.3802 | 0.3438 | 0.2669 | 0.3823 |
| SDA1(monai) | 0.3624 | 0.2244 | 0.4314 | 0.3624 | 0.2212 | 0.4330 |
| SDA2(monai) | 0.3621 | 0.2032 | 0.4415 | 0.3631 | 0.1991 | 0.4451 |
| SDA3(monai) | 0.3441 | 0.1957 | 0.4183 | 0.3439 | 0.1925 | 0.4195 |
| DirectUNet+SDA1 | 0.2558 | 0.1477 | 0.3099 | 0.2556 | 0.1457 | 0.3105 |
| DirectUNet+SDA2 | 0.2544 | 0.1415 | 0.3108 | 0.2545 | 0.1392 | 0.3122 |
| DirectUNet+SDA3 | 0.2472 | 0.1341 | 0.3037 | 0.2474 | 0.1323 | 0.3050 |
| FDV1(monai) | 0.2721* | 0.1610* | 0.3276* | 0.2710* | 0.1588* | 0.3271* |
| FDV1+SDA3(monai) | **0.2204** | **0.1178** | **0.2717** | **0.2188** | **0.1146** | **0.2709** |
| FDV1+SDA1(monai) | 0.2282 | 0.1328 | 0.2760 | 0.2263 | 0.1293 | 0.2748 |
| FDV1+SDA2(monai) | 0.2262 | 0.1239 | 0.2774 | 0.2247 | 0.1205 | 0.2768 |

`*` = ES from a one-member ensemble (N=1, deterministic; ES = per-dim MAE). Unmarked = proper ensemble ES (N=30, MAE − 0.5·pairwise spread). EnKF/ETKF ES are read from the bug-fixed DA cache; L3 ES from the ens30×10 run; Strong-4DVar and other neural models are deterministic (N=1).

## Per-trajectory detail: mean +/- std across the 200 test windows

Every table above pools all windows/timesteps into one number per method. This section instead computes RMSE/EV/CRPS **per window** (pooled over that window's own timesteps only) and reports the mean +/- std of that per-window distribution across the ~200 test windows -- i.e. how much reconstruction quality varies window-to-window, not just its average. Scoped to the DA baselines plus this session's monai-backbone schemes (not every historical row, to keep this bounded); the pooled tables above already cover everything. Note the RMSE means here are systematically a bit lower than the pooled RMSE above -- mean(sqrt(x)) <= sqrt(mean(x)) (Jensen's inequality), not a discrepancy.

### RMSE per window (mean +/- std, lower is better)

| Method | S0 all | S0 slow | S0 fast | S1 all | S1 slow | S1 fast |
|---|---|---|---|---|---|---|
| ETKF | 0.866±0.145 | 0.468±0.043 | 1.065±0.202 | 1.475±0.241 | 1.220±0.197 | 1.602±0.266 |
| EnKF | 0.894±0.135 | 0.489±0.046 | 1.097±0.187 | 1.512±0.248 | 1.230±0.197 | 1.654±0.278 |
| Strong-4DVar | 0.738±0.191 | 0.454±0.066 | 0.881±0.261 | 1.437±0.233 | 1.052±0.194 | 1.629±0.256 |
| DirectUNet-S+(monai,cos) | 0.501±0.076 | 0.292±0.042 | 0.605±0.100 | 0.501±0.077 | 0.291±0.047 | 0.606±0.100 |
| DirectUNet-M(monai,flat) | 0.486±0.075 | 0.262±0.038 | 0.598±0.102 | 0.488±0.073 | 0.260±0.039 | 0.601±0.101 |
| DirectUNet-M(monai,cos) | 0.491±0.073 | 0.258±0.037 | 0.607±0.099 | 0.493±0.069 | 0.258±0.037 | 0.610±0.094 |
| DirectUNet-L(monai,cos) | 0.470±0.073 | 0.253±0.037 | 0.579±0.099 | 0.472±0.071 | 0.250±0.038 | 0.583±0.095 |
| CFM-M(monai,flat) | 0.466±0.072 | 0.254±0.038 | 0.571±0.098 | 0.464±0.071 | 0.251±0.038 | 0.570±0.096 |
| CFM-M(monai,cos) | 0.471±0.073 | 0.256±0.038 | 0.578±0.099 | 0.473±0.074 | 0.253±0.038 | 0.582±0.101 |
| CFM-S+(monai,cos) | 0.511±0.088 | 0.332±0.078 | 0.600±0.107 | 0.512±0.085 | 0.331±0.077 | 0.603±0.104 |
| SDA1(monai) | 0.541±0.074 | 0.304±0.029 | 0.660±0.107 | 0.541±0.072 | 0.302±0.028 | 0.661±0.105 |
| SDA2(monai) | 0.546±0.076 | 0.281±0.030 | 0.679±0.114 | 0.548±0.077 | 0.277±0.029 | 0.684±0.117 |
| SDA3(monai) | 0.524±0.071 | 0.284±0.029 | 0.644±0.107 | 0.523±0.069 | 0.280±0.032 | 0.645±0.106 |
| DirectUNet+SDA1 | 0.410±0.074 | 0.194±0.034 | 0.518±0.103 | 0.408±0.073 | 0.191±0.036 | 0.516±0.100 |
| DirectUNet+SDA2 | 0.411±0.074 | 0.189±0.033 | 0.522±0.102 | 0.410±0.072 | 0.186±0.035 | 0.522±0.100 |
| DirectUNet+SDA3 | 0.403±0.074 | 0.180±0.033 | 0.514±0.103 | 0.401±0.073 | 0.177±0.034 | 0.513±0.100 |
| FDV1(monai) | 0.410±0.078 | 0.207±0.037 | 0.511±0.104 | 0.407±0.074 | 0.204±0.039 | 0.509±0.099 |
| FDV1+SDA3(monai) | 0.358±0.075 | 0.155±0.034 | 0.460±0.101 | 0.355±0.072 | 0.151±0.035 | 0.457±0.097 |
| FDV1+SDA1(monai) | 0.365±0.074 | 0.172±0.033 | 0.462±0.101 | 0.362±0.071 | 0.167±0.037 | 0.459±0.097 |
| FDV1+SDA2(monai) | 0.365±0.074 | 0.163±0.034 | 0.466±0.101 | 0.362±0.072 | 0.159±0.036 | 0.463±0.096 |

### EV per window (mean +/- std, higher is better)

| Method | S0 all | S0 slow | S0 fast | S1 all | S1 slow | S1 fast |
|---|---|---|---|---|---|---|
| ETKF | 0.692±0.113 | 0.940±0.011 | 0.568±0.166 | 0.215±0.252 | 0.577±0.133 | 0.034±0.315 |
| EnKF | 0.676±0.107 | 0.934±0.013 | 0.547±0.156 | 0.172±0.269 | 0.571±0.134 | -0.028±0.341 |
| Strong-4DVar | 0.748±0.141 | 0.941±0.019 | 0.651±0.205 | 0.233±0.241 | 0.681±0.114 | 0.009±0.307 |
| DirectUNet-S+(monai,cos) | 0.898±0.034 | 0.976±0.007 | 0.860±0.048 | 0.898±0.034 | 0.976±0.009 | 0.859±0.048 |
| DirectUNet-M(monai,flat) | 0.902±0.033 | 0.981±0.006 | 0.862±0.048 | 0.900±0.033 | 0.981±0.007 | 0.860±0.048 |
| DirectUNet-M(monai,cos) | 0.899±0.032 | 0.981±0.006 | 0.858±0.047 | 0.898±0.031 | 0.981±0.006 | 0.857±0.045 |
| DirectUNet-L(monai,cos) | 0.907±0.032 | 0.982±0.006 | 0.870±0.046 | 0.906±0.030 | 0.982±0.006 | 0.869±0.044 |
| CFM-M(monai,flat) | 0.910±0.030 | 0.982±0.006 | 0.874±0.044 | 0.910±0.030 | 0.982±0.006 | 0.874±0.044 |
| CFM-M(monai,cos) | 0.908±0.031 | 0.982±0.006 | 0.871±0.046 | 0.906±0.033 | 0.982±0.006 | 0.869±0.047 |
| CFM-S+(monai,cos) | 0.896±0.038 | 0.966±0.018 | 0.861±0.052 | 0.895±0.038 | 0.966±0.019 | 0.859±0.051 |
| SDA1(monai) | 0.882±0.036 | 0.975±0.005 | 0.835±0.054 | 0.881±0.035 | 0.974±0.005 | 0.835±0.053 |
| SDA2(monai) | 0.876±0.039 | 0.978±0.005 | 0.826±0.059 | 0.874±0.040 | 0.978±0.005 | 0.822±0.060 |
| SDA3(monai) | 0.888±0.035 | 0.978±0.005 | 0.843±0.053 | 0.887±0.034 | 0.978±0.006 | 0.842±0.051 |
| DirectUNet+SDA1 | 0.926±0.029 | 0.989±0.004 | 0.894±0.043 | 0.926±0.029 | 0.989±0.005 | 0.894±0.042 |
| DirectUNet+SDA2 | 0.925±0.030 | 0.990±0.004 | 0.892±0.043 | 0.925±0.029 | 0.990±0.005 | 0.892±0.043 |
| DirectUNet+SDA3 | 0.927±0.029 | 0.991±0.004 | 0.895±0.043 | 0.927±0.029 | 0.991±0.004 | 0.896±0.042 |
| FDV1(monai) | 0.927±0.030 | 0.988±0.005 | 0.897±0.044 | 0.928±0.028 | 0.988±0.006 | 0.898±0.041 |
| FDV1+SDA3(monai) | 0.941±0.027 | 0.993±0.004 | 0.914±0.040 | 0.942±0.026 | 0.993±0.004 | 0.916±0.037 |
| FDV1+SDA1(monai) | 0.940±0.027 | 0.992±0.004 | 0.914±0.040 | 0.941±0.026 | 0.992±0.004 | 0.915±0.037 |
| FDV1+SDA2(monai) | 0.939±0.027 | 0.992±0.004 | 0.913±0.040 | 0.940±0.026 | 0.993±0.004 | 0.914±0.038 |

### CRPS per window (mean +/- std, lower is better)

| Method | S0 all | S0 slow | S0 fast | S1 all | S1 slow | S1 fast |
|---|---|---|---|---|---|---|
| ETKF | 0.612±0.094* | 0.369±0.036* | 0.734±0.128* | 1.032±0.186* | 0.946±0.160* | 1.075±0.203* |
| EnKF | 0.637±0.089* | 0.390±0.040* | 0.760±0.120* | 1.084±0.198* | 0.974±0.161* | 1.139±0.221* |
| Strong-4DVar | 0.485±0.110* | 0.355±0.048* | 0.550±0.145* | 0.989±0.174* | 0.820±0.149* | 1.074±0.189* |
| DirectUNet-S+(monai,cos) | 0.328±0.048* | 0.223±0.035* | 0.380±0.059* | 0.329±0.049* | 0.223±0.040* | 0.382±0.059* |
| DirectUNet-M(monai,flat) | 0.306±0.044* | 0.194±0.030* | 0.363±0.058* | 0.308±0.044* | 0.192±0.031* | 0.366±0.057* |
| DirectUNet-M(monai,cos) | 0.308±0.043* | 0.190±0.030* | 0.367±0.056* | 0.310±0.041* | 0.191±0.029* | 0.370±0.053* |
| DirectUNet-L(monai,cos) | 0.294±0.042* | 0.186±0.029* | 0.347±0.055* | 0.294±0.041* | 0.184±0.029* | 0.350±0.053* |
| CFM-M(monai,flat) | 0.277±0.041 | 0.172±0.029 | 0.330±0.055 | 0.277±0.040 | 0.170±0.029 | 0.330±0.053 |
| CFM-M(monai,cos) | 0.279±0.041 | 0.171±0.028 | 0.333±0.055 | 0.280±0.042 | 0.169±0.029 | 0.335±0.055 |
| CFM-S+(monai,cos) | 0.310±0.057 | 0.236±0.070 | 0.346±0.062 | 0.310±0.056 | 0.235±0.070 | 0.348±0.060 |
| SDA1(monai) | 0.271±0.032 | 0.168±0.016 | 0.323±0.047 | 0.272±0.031 | 0.166±0.017 | 0.324±0.046 |
| SDA2(monai) | 0.264±0.027 | 0.150±0.016 | 0.321±0.042 | 0.265±0.028 | 0.147±0.017 | 0.323±0.043 |
| SDA3(monai) | 0.254±0.025 | 0.145±0.016 | 0.308±0.039 | 0.254±0.025 | 0.143±0.018 | 0.309±0.039 |
| DirectUNet+SDA1 | 0.216±0.038 | 0.120±0.024 | 0.263±0.051 | 0.215±0.038 | 0.119±0.026 | 0.264±0.049 |
| DirectUNet+SDA2 | 0.213±0.036 | 0.114±0.023 | 0.262±0.049 | 0.213±0.036 | 0.112±0.025 | 0.263±0.047 |
| DirectUNet+SDA3 | 0.208±0.036 | 0.109±0.023 | 0.257±0.048 | 0.208±0.036 | 0.107±0.024 | 0.258±0.047 |
| FDV1(monai) | 0.272±0.046* | 0.161±0.029* | 0.328±0.060* | 0.271±0.046* | 0.159±0.031* | 0.327±0.058* |
| FDV1+SDA3(monai) | 0.182±0.036 | 0.094±0.024 | 0.227±0.047 | 0.181±0.036 | 0.091±0.025 | 0.226±0.046 |
| FDV1+SDA1(monai) | 0.189±0.037 | 0.106±0.025 | 0.231±0.049 | 0.188±0.037 | 0.103±0.027 | 0.230±0.048 |
| FDV1+SDA2(monai) | 0.186±0.036 | 0.098±0.024 | 0.230±0.047 | 0.185±0.036 | 0.095±0.026 | 0.230±0.046 |

`*` = CRPS from a one-member reconstruction (N=1, deterministic; CRPS = per-dim MAE, the N=1 special case of the ensemble formula). Unmarked = proper ensemble CRPS (per-dimension Energy Score, N=30, MAE − 0.5·pairwise member distance) from the stored `members_*.npz`.

## Consistency checks

- DA cached metrics vs recomputed-from-npz (42 values): max |Δ| = 2.16e-04 → PASS
- Neural stored truth vs dataset true_state[:, obs_var_indices]: max |Δ| = 0.00e+00 → PASS

## Reconstruction examples (Hovmöller)

Windows ranked by per-window pooled 24D RMSE of Strong-4DVar (best DA scheme); each figure shows rows = Truth/methods and columns = state / |error| maps for the slow X (8D) and fast Y (16D) blocks. State colors share one scale per figure; error maps share one scale across all rows/methods (99.5th-percentile cap, noted on the colorbar). Dotted vertical lines on the truth row mark observation times.

| Case | Rank | Window | 4DVar win-RMSE | Strong-4DVar | DirectUNet-L(monai,cos) | CFM-M(monai,flat) | SDA3(monai) | DirectUNet+SDA3 | FDV1(monai) | FDV1+SDA3(monai) |
|---|---|---|---|---|---|---|---|---|---|---|
| S0 | worst | 58 | 1.432 | 1.432 | 0.543 | 0.524 | 0.603 | 0.446 | 0.444 | 0.384 |
| S0 | median | 187 | 0.794 | 0.794 | 0.498 | 0.474 | 0.580 | 0.433 | 0.408 | 0.359 |
| S0 | best | 155 | 0.407 | 0.407 | 0.396 | 0.405 | 0.411 | 0.303 | 0.305 | 0.253 |
| S1 | worst | 75 | 1.991 | 1.991 | 0.657 | 0.601 | 0.756 | 0.617 | 0.596 | 0.537 |
| S1 | median | 198 | 1.482 | 1.482 | 0.475 | 0.461 | 0.503 | 0.422 | 0.404 | 0.351 |
| S1 | best | 35 | 0.977 | 0.977 | 0.374 | 0.379 | 0.385 | 0.309 | 0.332 | 0.285 |

![s0-worst](figs/l96_hovm_s0_worst.png)

![s0-median](figs/l96_hovm_s0_median.png)

![s0-best](figs/l96_hovm_s0_best.png)

![s1-worst](figs/l96_hovm_s1_worst.png)

![s1-median](figs/l96_hovm_s1_median.png)

![s1-best](figs/l96_hovm_s1_best.png)
