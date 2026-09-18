# L96 Consolidated Benchmark — DA baselines vs neural models

Setup: two-scale L96, Obs30 (`obs_interval=100`, `obs_j=2` → 24D observed space), dws=500, 200 shared cached test windows; S1 = ±20% params + ±10% bias (DA forward model uses biased `*_da`).

RMSE/EV are recomputed from the stored trajectory arrays via `evaluation/estimate_metrics.py`; ES for DA ensemble methods (EnKF/ETKF) and L3 (ens30×10) are proper ensemble scores (N=30, MAE − 0.5·pairwise spread) read from cached run outputs; ES for deterministic methods is the N=1 per-dim MAE proxy. **bold** marks the best value per column.

## Benchmarked schemes

| ID | Type | Description |
|---|---|---|
| Strong-4DVar | Variational | Strong-constraint 4D-Var over the dws=500 window (`B_var=2.0`, `R_var=0.5`, `max_iter=10`, `lr=0.2`, autodiff minimization); assimilates the full window trajectory. |
| EnKF | Ensemble KF | Stochastic ensemble Kalman filter, `N_ens=30`, inflation=2.0, no localization; sequential observation updates. |
| ETKF | Ensemble KF | Deterministic ensemble square-root filter, `N_ens=30`, inflation=2.0, no localization. |
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
| FDV2(monai) | Neural (4DVarNet-style unrolled solver, gradient-conditioned, monai backbone) | MonaiUNet1D backbone swapped into FDV2's `update_input='grad+state'` solver (each of N_outer=10 iterations builds a real variational cost `prior_cost(state) + obs_weight*obs_cost(state, obs)` and feeds the UNet its true `torch.autograd.grad(..., create_graph=True)`), with a **fixed** (non-trainable) `prior_weight=1.0` held for all 400 epochs at a constant `lr=0.001` -- the "fixedw" variant. **Diverged to NaN at epoch 190/400** (val_loss NaN first, train_loss still normal mid-epoch -- a sudden collapse, not gradual drift); root-caused to `models/fourdvarnet.py::_normalize_channels`, which only floors the `grad+state` cost-gradient channel's RMS norm (`clamp_min(1e-8)`) with no matching ceiling, unlike the state branch `x` (clamped every iteration via `clip_range=50.0`) -- as the model converges and the raw gradient's RMS shrinks toward that floor, dividing by it can blow the normalized channel up unbounded. The one config difference from the FDV2/UNet1D run that stayed stable (job 52305): there `prior_weight` was trainable and was actively shrinking (0.99->0.76 over 60 epochs), damping the gradient magnitude away from this regime; fixed at 1.0 here, nothing does. Job killed rather than left to burn its remaining epochs post-NaN. **No valid trained result -- unavailable below.** Fix for a retry: trainable `prior_weight` (as in the stable UNet1D recipe) and/or an upper clamp on the normalized gradient channel symmetric to the state branch's `clip_range=50.0`. |
| FDV1+SDA3(monai) | Neural (FDV1-monai mean + SDA3-monai warm-started guidance) | Same SDEdit-style warm start as the DirectUNet+SDA hybrids above, but swapping in FDV1-monai (the unrolled solver) as the mean-estimate model instead of DirectUNet-M -- SDA3-monai is used to sample the anomaly around FDV1's point estimate. `tau0=0.3`, `guidance_weight=2.0` (own S0-only sweep over `tau0∈{0.1..0.5}×guidance_weight∈{0.5,1,2,5}`, converged on the exact same point as the DirectUNet+SDA hybrids' independent sweep). One real wrinkle specific to this combination: FDV1-monai was trained WITHOUT `data.normalize=true` (unlike DirectUNet-M/SDA3, which share that normalized space) -- feeding it normalized obs (as SDA expects) silently produced a garbage mean estimate that poisoned the whole hybrid (confirmed: RMSE 1.36/EV 0.35) before this was caught; fixed by preparing a SEPARATE raw-obs dataloader for FDV1's own `.sample()` call and normalizing its output before handing it to SDA as the warm start. **Best scheme in this table overall.** |
| FDV1+SDA1(monai) | Neural (FDV1-monai mean + SDA1-monai warm-started guidance) | As FDV1+SDA3-monai but warm-starting SDA1-monai (unconditional prior) instead of SDA3-monai, for coherence with the DirectUNet+SDA1/2/3 family above. Same `tau0=0.3`/`guidance_weight=2.0` (not re-swept per SDA variant -- three independent sweeps this session all converged on this exact point regardless of SDA variant or mean model). |
| FDV1+SDA2(monai) | Neural (FDV1-monai mean + SDA2-monai warm-started guidance) | As above, warm-starting SDA2-monai (true-params conditioned) instead of SDA1-monai. |
| DirectUNet-M(monai,cos,obsdensity) | Neural (DirectUNet, monai backbone, obs-density-augmented) | As DirectUNet-M(monai,cos) but trained with `data.obs_density_augment=true` (`data/obs_density.py::sample_training_density_mask`): at every (window, obs-time), 40% chance of full fast-Y density, else a uniformly-random `keep_k∈{0,...,15}` of the 16 fast-Y channels kept, redrawn every batch. Closes the OOD gap found in the fast-Y observation-density generalization sweep (PR #180): DirectUNet/CFM read `obs` only via `nan_to_num`, no mask channel, so an unseen-at-training partial-channel dropout pattern reads as a spurious near-zero observation. An L-tier attempt at this same augmentation collapsed toward the fast-Y conditional mean even at full density (variance ratio ~35%); M-tier avoids it entirely (variance ratio ~99%). Evaluated here at full canonical density only (N=1, single pass) -- see `l96_obs_density_augmented_training.md` for the dedicated reduced-density sweep. |
| CFM-M(monai,flat,obsdensity) | Neural (CFM, τ=0, monai backbone, obs-density-augmented) | As CFM-M(monai,flat) but with the identical `obs_density_augment` training augmentation described above -- unlike the DirectUNet-L attempt, CFM tolerated it cleanly at M-tier with no collapse (variance ratio ~96%). Evaluated here at full canonical density only (N=1, single pass); see `l96_obs_density_augmented_training.md` for the reduced-density sweep. |
| DirectUNet(aug)+SDA3 | Neural (DirectUNet-M(monai,cos,obsdensity) mean + SDA3-monai warm-started guidance) | Same SDEdit-style warm start as DirectUNet+SDA3 above, but swapping in the `obs_density_augment`-trained DirectUNet-M mean estimate instead of the non-augmented one -- same `tau0=0.3`/`guidance_weight=2.0`. Third-best full-density RMSE/EV in this whole table (0.389 S0 -- behind FDV1-Stier+SDA3(monai)'s 0.359 and FDV1+SDA3-monai's 0.379, and comfortably ahead of non-augmented DirectUNet+SDA3's 0.420) **and** the best absolute worst-case RMSE at zero fast-Y density among everything tested in that sweep (1.065, edging out plain SDA3's 1.082 -- see `l96_obs_density_augmented_training.md`; neither FDV1+SDA3 nor FDV1-Stier+SDA3 were part of that reduced-density sweep, so they aren't ruled out as contenders there); its full-density ES/CRPS here are on the N=1 MAE-proxy convention, not the proper ensemble score its non-augmented sibling rows get (see the note above table). |
| FDV1-Stier(monai) | Neural (4DVarNet-style unrolled solver, monai backbone, S-tier) | As FDV1(monai) above (`update_input=obs+state`, `models/fourdvarnet.py::FourDVarNetSolver`, N_outer=10, monai backbone) but at a smaller "S" capacity tier (`hidden_channels=[32,64,128]`, `monai_num_res_blocks=1`, 1,055,544 params -- vs the M tier's `hidden_channels=[64,128,256]`, `monai_num_res_blocks=2`, 5,889,048 params every other FDV1/FDV2 row in this table uses), a random (not zero) initial condition (`init_state_var=0.1`, i.e. `x_0 ~ N(0, 0.1)`), and a small auxiliary prior-consistency loss term during training (`aux_var_cost_weight=0.01`, `prior_cost(x_final)+prior_cost(states)`, no `prior_weight`/`obs_cost` mixed in -- a real bug in an earlier version of this loss term, since fixed, is documented in `CHANGELOG.d/2026-09-12-fdv-aux-prior-loss-drop-prior-weight-obs-cost.md`); 400 epochs, cosine-annealed LR. Genuinely better than the M-tier FDV1(monai) baseline above on every RMSE/EV metric despite being ~5.6x smaller in parameter count. Evaluated as a single pass (N=1), same convention as FDV1(monai). |
| subgrad+state-Stier(monai) | Neural (4DVarNet-style unrolled solver, cheap proxy-gradient-conditioned, monai backbone, S-tier) | FDV2's `update_input=subgrad+state` solver (`models/fourdvarnet.py::FourDVarNetSolver`) -- a cheap two-residual proxy gradient (the observation residual `obs-x` plus the prior-autoencoder residual `x-Phi(x)`, fed to the UNet alongside the state, with **no** `torch.autograd.grad` call at all, unlike FDV2(monai)'s real `grad+state` cost gradient above) -- on the same S-tier MonaiUNet1D as `FDV1-Stier(monai)` (`hidden_channels=[32,64,128]`, `monai_num_res_blocks=1`, 1,055,544 main-solver params, vs the M tier's `hidden_channels=[64,128,256]`, `monai_num_res_blocks=2`, 5,889,048 params every earlier FDV2 attempt in this table's history used), plus the same `init_state_var=0.1` (random initial condition) and `aux_var_cost_weight=0.01` (small auxiliary prior-consistency loss, `prior_cost(x_final)+prior_cost(states)`, no `prior_weight`/`obs_cost` mixed in) recipe as `FDV1-Stier(monai)`; 400 epochs, cosine-annealed LR. **Best standalone (non-hybrid) neural mean-estimate model found in this entire investigation** -- better than `FDV1-Stier(monai)` itself, and dramatically better than every earlier `subgrad+state`/`grad+state` attempt at the M tier, which all showed a severe fast-Y reconstruction collapse (variance ratio ~0.5-0.6); this run shows no such collapse at all (fast-Y variance ratio ~1.01-1.02, essentially perfectly calibrated). Evaluated as a single pass (N=1), same convention as FDV1-Stier(monai). |
| FDV1-Stier+SDA3(monai) | Neural (FDV1-Stier-monai mean + SDA3-monai warm-started guidance) | Same SDEdit-style warm start as FDV1+SDA3(monai) above (`evaluation/sda_sampler.py`'s `mean_estimate`/`tau0`), but swapping in FDV1-Stier(monai) (above) as the mean-estimate model instead of the M-tier FDV1(monai) -- same `tau0=0.3`, `guidance_weight=2.0`, `r_var=0.5`, ens30 (`n_members=30`, `n_outer=10`); not re-swept, reusing the established convention. Same raw-obs handling as FDV1+SDA3(monai) above (FDV1-Stier was also trained without `data.normalize=true`; `--no-mean-normalized`, separate raw dataloader, `eval_sda_mean_hybrid_l96.py`'s existing mechanism, unchanged). Was the best scheme in this table overall (previous-best 0.3587/0.3583 S0/S1 pooled RMSE, beating FDV1+SDA3(monai)'s 0.3787/0.3740) -- see `subgrad+state-Stier+SDA3(monai)` below, which now beats it on every RMSE/EV/ES metric by swapping in an even better (and cheaper) mean-estimate model. One real bug hit and fixed while producing this row: running the hybrid eval from the `4dvarnet-fm-fdv-monai` worktree (needed because that's where the `SDA3_monai_cond_noisy_l96_norm` checkpoint, `l96_norm_stats_obsj2.pt`, and the cached test dataset live) silently built the WRONG UNet tier for the mean-estimate model -- that worktree's `train.py`/`models/fourdvarnet.py` predates this session's `monai_num_res_blocks` feature (a `FourDVarNetSolver` param needed alongside `hidden_channels` to actually get an S tier vs M/S+), so it silently defaulted to `monai_num_res_blocks=2`, building an S+-tier-shaped model (1,482,264 params) instead of true S-tier (1,055,544), partially failing to load the checkpoint (shape mismatches on 8 `unet.*` keys, silently skipped) and producing garbage output (RMSE ~1.8, negative EV) on the first attempt. Fixed by running the exact same command from `4dvarnet-fm-obs-density-gen` instead (which has the feature), with absolute paths for the SDA3 checkpoint/config/dataset/norm-stats pointing into `4dvarnet-fm-fdv-monai`'s `experiments/` dir -- `model_factory` then correctly built the true 1,055,544-param S-tier model matching the checkpoint exactly (verified: zero `unet.*` shape-mismatch warnings, and a direct `load_model(...)` unit check confirming `sum(p.numel() for p in model.unet.parameters()) == 1055544`). |
| subgrad+state-Stier+SDA3(monai) | Neural (subgrad+state-Stier-monai mean + SDA3-monai warm-started guidance) | Same SDEdit-style warm start as FDV1-Stier+SDA3(monai) above (`evaluation/sda_sampler.py`'s `mean_estimate`/`tau0`), but swapping in `subgrad+state-Stier(monai)` (above) as the mean-estimate model instead of `FDV1-Stier(monai)` -- same `tau0=0.3`, `guidance_weight=2.0`, `r_var=0.5`, ens30 (`n_members=30`, `n_outer=10`); not re-swept, reusing the established convention. Same raw-obs handling as `FDV1-Stier+SDA3(monai)` above (this checkpoint's config also has no `data.normalize` block; `--no-mean-normalized`, separate raw dataloader, `eval_sda_mean_hybrid_l96.py`'s existing mechanism, unchanged). **New best scheme in this entire table, on every RMSE/EV/ES metric** -- beats `FDV1-Stier+SDA3(monai)`'s previous-best 0.3587/0.3583 S0/S1 pooled RMSE with 0.3410/0.3402. |

Shared setup: all L-series neural models are trained and evaluated on the identical DA-parity benchmark (all-5 params ±20% randomized per window; S1 adds a ±10% bias; models operate in the 24D observed subspace with obs-only inputs unless noted). DA baselines receive the same per-window parameters as the truth generation (S0) or their biased `*_da` counterparts (S1), which is what makes the DA-vs-neural comparison apples-to-apples.

## RMSE (pooled, lower is better)

### RMSE by variable group

| Method | S0 all | S0 slow | S0 fast | S1 all | S1 slow | S1 fast | S1/S0 |
|---|---|---|---|---|---|---|---|
| ETKF | 0.8883 | 0.4749 | 1.0950 | 1.4998 | 1.2394 | 1.6299 | 1.688 |
| EnKF | 0.9131 | 0.4974 | 1.1210 | 1.5381 | 1.2490 | 1.6826 | 1.684 |
| Strong-4DVar | 0.8116 | 0.4683 | 0.9833 | 1.4617 | 1.0760 | 1.6546 | 1.801 |
| DirectUNet-S+(monai,cos) | 0.5152 | 0.2974 | 0.6240 | 0.5155 | 0.2977 | 0.6244 | 1.001 |
| DirectUNet-M(monai,flat) | 0.5010 | 0.2720 | 0.6155 | 0.5020 | 0.2701 | 0.6180 | 1.002 |
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
| FDV1(monai) | 0.4251 | 0.2111 | 0.5320 | 0.4218 | 0.2093 | 0.5280 | 0.992 |
| FDV2(monai) |   —   |   —   |   —   |   —   |   —   |   —   | n/a |
| FDV1+SDA3(monai) | 0.3787 | 0.1619 | 0.4871 | 0.3740 | 0.1580 | 0.4820 | 0.988 |
| FDV1+SDA1(monai) | 0.3852 | 0.1770 | 0.4892 | 0.3801 | 0.1737 | 0.4833 | 0.987 |
| FDV1+SDA2(monai) | 0.3845 | 0.1688 | 0.4924 | 0.3800 | 0.1650 | 0.4875 | 0.988 |
| DirectUNet-M(monai,cos,obsdensity) | 0.4728 | 0.2696 | 0.5744 | 0.4759 | 0.2716 | 0.5781 | 1.007 |
| CFM-M(monai,flat,obsdensity) | 0.4656 | 0.2632 | 0.5667 | 0.4658 | 0.2629 | 0.5673 | 1.001 |
| DirectUNet(aug)+SDA3 | 0.3890 | 0.1818 | 0.4927 | 0.3895 | 0.1805 | 0.4941 | 1.001 |
| FDV1-Stier(monai) | 0.4012 | 0.2042 | 0.4997 | 0.3994 | 0.2021 | 0.4981 | 0.996 |
| subgrad+state-Stier(monai) | 0.3729 | 0.1821 | 0.4682 | 0.3730 | 0.1801 | 0.4694 | 1.000 |
| FDV1-Stier+SDA3(monai) | 0.3588 | 0.1558 | 0.4602 | 0.3584 | 0.1524 | 0.4614 | 0.999 |
| subgrad+state-Stier+SDA3(monai) | **0.3411** | **0.1454** | **0.4390** | **0.3403** | **0.1438** | **0.4385** | 0.997 |

Note on conventions: the DA metric cache stores the **mean of per-window RMSEs** (evaluation/run_l96.py), while this table uses the **pooled** convention (`sqrt(mean sq err)` over all windows/timesteps) for every method — the same convention as the neural evaluation. Pooled RMSE is ≤ mean-of-window RMSE, so DA values here are slightly lower (more favorable) than in the legacy cache; both orderings agree.

## Explained Variance (higher is better)

### EV by variable group

| Method | S0 all | S0 slow | S0 fast | S1 all | S1 slow | S1 fast |
|---|---|---|---|---|---|---|
| ETKF | 0.6915 | 0.9395 | 0.5676 | 0.2149 | 0.5772 | 0.0338 |
| EnKF | 0.6758 | 0.9337 | 0.5469 | 0.1717 | 0.5706 | -0.0278 |
| Strong-4DVar | 0.7478 | 0.9412 | 0.6510 | 0.2330 | 0.6813 | 0.0089 |
| DirectUNet-S+(monai,cos) | 0.8985 | 0.9763 | 0.8595 | 0.8977 | 0.9756 | 0.8588 |
| DirectUNet-M(monai,flat) | 0.9023 | 0.9802 | 0.8633 | 0.9010 | 0.9799 | 0.8616 |
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
| FDV1(monai) | 0.9279 | 0.9880 | 0.8979 | 0.9287 | 0.9879 | 0.8990 |
| FDV2(monai) |   —   |   —   |   —   |   —   |   —   |   —   |
| FDV1+SDA3(monai) | 0.9406 | 0.9930 | 0.9144 | 0.9416 | 0.9931 | 0.9159 |
| FDV1+SDA1(monai) | 0.9396 | 0.9916 | 0.9137 | 0.9408 | 0.9917 | 0.9154 |
| FDV1+SDA2(monai) | 0.9391 | 0.9924 | 0.9125 | 0.9401 | 0.9925 | 0.9139 |
| DirectUNet-M(monai,cos,obsdensity) | 0.9142 | 0.9805 | 0.8810 | 0.9126 | 0.9797 | 0.8790 |
| CFM-M(monai,flat,obsdensity) | 0.9166 | 0.9814 | 0.8841 | 0.9160 | 0.9810 | 0.8835 |
| DirectUNet(aug)+SDA3 | 0.9387 | 0.9911 | 0.9125 | 0.9381 | 0.9910 | 0.9116 |
| FDV1-Stier(monai) | 0.9362 | 0.9888 | 0.9100 | 0.9363 | 0.9888 | 0.9101 |
| subgrad+state-Stier(monai) | 0.9443 | 0.9911 | 0.9209 | 0.9438 | 0.9911 | 0.9202 |
| FDV1-Stier+SDA3(monai) | 0.9469 | 0.9935 | 0.9236 | 0.9465 | 0.9936 | 0.9229 |
| subgrad+state-Stier+SDA3(monai) | **0.9518** | **0.9943** | **0.9305** | **0.9517** | **0.9943** | **0.9304** |

## Energy Score (lower is better)

### ES by variable group

| Method | S0 all | S0 slow | S0 fast | S1 all | S1 slow | S1 fast |
|---|---|---|---|---|---|---|
| ETKF | 0.4548 | 0.2857 | 0.5393 | 0.8511 | 0.7626 | 0.8953 |
| EnKF | 0.4599 | 0.2915 | 0.5440 | 0.9018 | 0.7942 | 0.9556 |
| Strong-4DVar | 0.4850* | 0.3551* | 0.5499* | 0.9892* | 0.8197* | 1.0740* |
| DirectUNet-S+(monai,cos) | 0.3278* | 0.2232* | 0.3801* | 0.3290* | 0.2233* | 0.3819* |
| DirectUNet-M(monai,flat) | 0.3082* | 0.1975* | 0.3635* | 0.3094* | 0.1952* | 0.3664* |
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
| FDV1(monai) | 0.2668* | 0.1582* | 0.3211* | 0.2667* | 0.1575* | 0.3214* |
| FDV2(monai) |   —   |   —   |   —   |   —   |   —   |   —   |
| FDV1+SDA3(monai) | 0.2200 | 0.1177 | 0.2712 | 0.2187 | 0.1150 | 0.2706 |
| FDV1+SDA1(monai) | 0.2282 | 0.1328 | 0.2760 | 0.2263 | 0.1293 | 0.2748 |
| FDV1+SDA2(monai) | 0.2262 | 0.1239 | 0.2774 | 0.2247 | 0.1205 | 0.2768 |
| DirectUNet-M(monai,cos,obsdensity) | 0.3026* | 0.2001* | 0.3538* | 0.3054* | 0.2014* | 0.3574* |
| CFM-M(monai,flat,obsdensity) | 0.2929* | 0.1946* | 0.3420* | 0.2934* | 0.1941* | 0.3431* |
| DirectUNet(aug)+SDA3 | 0.2334* | 0.1330* | 0.2836* | 0.2345* | 0.1322* | 0.2856* |
| FDV1-Stier(monai) | 0.2464* | 0.1540* | 0.2925* | 0.2451* | 0.1512* | 0.2921* |
| subgrad+state-Stier(monai) | 0.2226* | 0.1361* | 0.2659* | 0.2223* | 0.1338* | 0.2665* |
| FDV1-Stier+SDA3(monai) | 0.1708 | 0.0901 | 0.2112 | 0.1704 | 0.0875 | 0.2119 |
| subgrad+state-Stier+SDA3(monai) | **0.1596** | **0.0828** | **0.1979** | **0.1592** | **0.0812** | **0.1982** |

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
| DirectUNet-M(monai,flat) | 0.486±0.073 | 0.267±0.038 | 0.596±0.099 | 0.488±0.071 | 0.265±0.040 | 0.599±0.096 |
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
| FDV1(monai) | 0.407±0.079 | 0.205±0.039 | 0.508±0.106 | 0.405±0.074 | 0.204±0.037 | 0.506±0.099 |
| FDV2(monai) |   —   |   —   |   —   |   —   |   —   |   —   |
| FDV1+SDA3(monai) | 0.358±0.077 | 0.156±0.034 | 0.460±0.105 | 0.355±0.073 | 0.152±0.033 | 0.457±0.099 |
| FDV1+SDA1(monai) | 0.365±0.074 | 0.172±0.033 | 0.462±0.101 | 0.362±0.071 | 0.167±0.037 | 0.459±0.097 |
| FDV1+SDA2(monai) | 0.365±0.074 | 0.163±0.034 | 0.466±0.101 | 0.362±0.072 | 0.159±0.036 | 0.463±0.096 |
| DirectUNet-M(monai,cos,obsdensity) | 0.459±0.071 | 0.265±0.037 | 0.556±0.095 | 0.462±0.072 | 0.266±0.039 | 0.560±0.097 |
| CFM-M(monai,flat,obsdensity) | 0.450±0.075 | 0.258±0.037 | 0.546±0.101 | 0.450±0.077 | 0.258±0.038 | 0.546±0.104 |
| DirectUNet(aug)+SDA3 | 0.372±0.070 | 0.177±0.032 | 0.470±0.095 | 0.373±0.071 | 0.175±0.033 | 0.472±0.096 |
| FDV1-Stier(monai) | 0.381±0.078 | 0.198±0.040 | 0.473±0.104 | 0.379±0.080 | 0.195±0.043 | 0.471±0.106 |
| subgrad+state-Stier(monai) | 0.352±0.077 | 0.175±0.040 | 0.441±0.101 | 0.351±0.080 | 0.172±0.042 | 0.441±0.105 |
| FDV1-Stier+SDA3(monai) | 0.338±0.074 | 0.150±0.035 | 0.432±0.100 | 0.337±0.077 | 0.146±0.034 | 0.432±0.105 |
| subgrad+state-Stier+SDA3(monai) | 0.320±0.074 | 0.139±0.035 | 0.410±0.099 | 0.318±0.076 | 0.136±0.037 | 0.409±0.100 |

### EV per window (mean +/- std, higher is better)

| Method | S0 all | S0 slow | S0 fast | S1 all | S1 slow | S1 fast |
|---|---|---|---|---|---|---|
| ETKF | 0.692±0.113 | 0.940±0.011 | 0.568±0.166 | 0.215±0.252 | 0.577±0.133 | 0.034±0.315 |
| EnKF | 0.676±0.107 | 0.934±0.013 | 0.547±0.156 | 0.172±0.269 | 0.571±0.134 | -0.028±0.341 |
| Strong-4DVar | 0.748±0.141 | 0.941±0.019 | 0.651±0.205 | 0.233±0.241 | 0.681±0.114 | 0.009±0.307 |
| DirectUNet-S+(monai,cos) | 0.898±0.034 | 0.976±0.007 | 0.860±0.048 | 0.898±0.034 | 0.976±0.009 | 0.859±0.048 |
| DirectUNet-M(monai,flat) | 0.902±0.032 | 0.980±0.006 | 0.863±0.047 | 0.901±0.031 | 0.980±0.007 | 0.862±0.046 |
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
| FDV1(monai) | 0.928±0.031 | 0.988±0.005 | 0.898±0.044 | 0.929±0.029 | 0.988±0.005 | 0.899±0.041 |
| FDV2(monai) |   —   |   —   |   —   |   —   |   —   |   —   |
| FDV1+SDA3(monai) | 0.941±0.028 | 0.993±0.004 | 0.914±0.041 | 0.942±0.026 | 0.993±0.004 | 0.916±0.038 |
| FDV1+SDA1(monai) | 0.940±0.027 | 0.992±0.004 | 0.914±0.040 | 0.941±0.026 | 0.992±0.004 | 0.915±0.037 |
| FDV1+SDA2(monai) | 0.939±0.027 | 0.992±0.004 | 0.913±0.040 | 0.940±0.026 | 0.993±0.004 | 0.914±0.038 |
| DirectUNet-M(monai,cos,obsdensity) | 0.914±0.029 | 0.980±0.006 | 0.881±0.042 | 0.913±0.030 | 0.980±0.007 | 0.879±0.044 |
| CFM-M(monai,flat,obsdensity) | 0.917±0.031 | 0.981±0.006 | 0.884±0.045 | 0.916±0.032 | 0.981±0.006 | 0.883±0.046 |
| DirectUNet(aug)+SDA3 | 0.939±0.025 | 0.991±0.004 | 0.912±0.037 | 0.938±0.026 | 0.991±0.004 | 0.912±0.038 |
| FDV1-Stier(monai) | 0.936±0.029 | 0.989±0.005 | 0.910±0.042 | 0.936±0.030 | 0.989±0.006 | 0.910±0.044 |
| subgrad+state-Stier(monai) | 0.944±0.027 | 0.991±0.005 | 0.921±0.039 | 0.944±0.028 | 0.991±0.005 | 0.920±0.041 |
| FDV1-Stier+SDA3(monai) | 0.947±0.026 | 0.993±0.004 | 0.924±0.038 | 0.946±0.028 | 0.994±0.004 | 0.923±0.041 |
| subgrad+state-Stier+SDA3(monai) | 0.952±0.025 | 0.994±0.004 | 0.930±0.036 | 0.952±0.025 | 0.994±0.004 | 0.930±0.037 |

### CRPS per window (mean +/- std, lower is better)

| Method | S0 all | S0 slow | S0 fast | S1 all | S1 slow | S1 fast |
|---|---|---|---|---|---|---|
| ETKF | 0.612±0.094* | 0.369±0.036* | 0.734±0.128* | 1.032±0.186* | 0.946±0.160* | 1.075±0.203* |
| EnKF | 0.637±0.089* | 0.390±0.040* | 0.760±0.120* | 1.084±0.198* | 0.974±0.161* | 1.139±0.221* |
| Strong-4DVar | 0.485±0.110* | 0.355±0.048* | 0.550±0.145* | 0.989±0.174* | 0.820±0.149* | 1.074±0.189* |
| DirectUNet-S+(monai,cos) | 0.328±0.048* | 0.223±0.035* | 0.380±0.059* | 0.329±0.049* | 0.223±0.040* | 0.382±0.059* |
| DirectUNet-M(monai,flat) | 0.308±0.043* | 0.197±0.030* | 0.363±0.057* | 0.309±0.042* | 0.195±0.031* | 0.366±0.055* |
| DirectUNet-M(monai,cos) | 0.308±0.043* | 0.190±0.030* | 0.367±0.056* | 0.310±0.041* | 0.191±0.029* | 0.370±0.053* |
| DirectUNet-L(monai,cos) | 0.294±0.042* | 0.186±0.029* | 0.347±0.055* | 0.294±0.041* | 0.184±0.029* | 0.350±0.053* |
| CFM-M(monai,flat) | 0.294±0.042* | 0.188±0.029* | 0.347±0.056* | 0.294±0.041* | 0.185±0.030* | 0.348±0.053* |
| CFM-M(monai,cos) | 0.296±0.042* | 0.187±0.029* | 0.350±0.056* | 0.297±0.043* | 0.185±0.030* | 0.353±0.056* |
| CFM-S+(monai,cos) | 0.343±0.059* | 0.268±0.073* | 0.380±0.064* | 0.344±0.058* | 0.267±0.073* | 0.382±0.062* |
| SDA1(monai) | 0.362±0.043* | 0.224±0.020* | 0.431±0.063* | 0.362±0.041* | 0.221±0.019* | 0.433±0.061* |
| SDA2(monai) | 0.362±0.043* | 0.203±0.021* | 0.442±0.066* | 0.363±0.044* | 0.199±0.022* | 0.445±0.067* |
| SDA3(monai) | 0.344±0.039* | 0.196±0.020* | 0.418±0.060* | 0.344±0.038* | 0.193±0.023* | 0.420±0.059* |
| DirectUNet+SDA1 | 0.256±0.042* | 0.148±0.027* | 0.310±0.056* | 0.256±0.041* | 0.146±0.029* | 0.311±0.054* |
| DirectUNet+SDA2 | 0.254±0.040* | 0.142±0.026* | 0.311±0.054* | 0.255±0.039* | 0.139±0.028* | 0.312±0.052* |
| DirectUNet+SDA3 | 0.247±0.040* | 0.134±0.025* | 0.304±0.053* | 0.247±0.040* | 0.132±0.027* | 0.305±0.052* |
| FDV1(monai) | 0.267±0.047* | 0.158±0.031* | 0.321±0.060* | 0.267±0.044* | 0.158±0.030* | 0.321±0.056* |
| FDV2(monai) |   —   |   —   |   —   |   —   |   —   |   —   |
| FDV1+SDA3(monai) | 0.220±0.040* | 0.118±0.027* | 0.271±0.052* | 0.219±0.039* | 0.115±0.027* | 0.271±0.049* |
| FDV1+SDA1(monai) | 0.228±0.041* | 0.133±0.028* | 0.276±0.053* | 0.226±0.041* | 0.129±0.030* | 0.275±0.052* |
| FDV1+SDA2(monai) | 0.226±0.040* | 0.124±0.027* | 0.277±0.052* | 0.225±0.040* | 0.121±0.028* | 0.277±0.050* |
| DirectUNet-M(monai,cos,obsdensity) | 0.303±0.043* | 0.200±0.030* | 0.354±0.056* | 0.305±0.044* | 0.201±0.032* | 0.357±0.056* |
| CFM-M(monai,flat,obsdensity) | 0.293±0.043* | 0.195±0.029* | 0.342±0.056* | 0.293±0.044* | 0.194±0.029* | 0.343±0.057* |
| DirectUNet(aug)+SDA3 | 0.233±0.038* | 0.133±0.025* | 0.284±0.049* | 0.234±0.038* | 0.132±0.026* | 0.286±0.050* |
| FDV1-Stier(monai) | 0.246±0.045* | 0.154±0.032* | 0.293±0.056* | 0.245±0.047* | 0.151±0.036* | 0.292±0.058* |
| subgrad+state-Stier(monai) | 0.223±0.043* | 0.136±0.032* | 0.266±0.053* | 0.222±0.046* | 0.134±0.033* | 0.267±0.057* |
| FDV1-Stier+SDA3(monai) | 0.171±0.035 | 0.090±0.025 | 0.211±0.045 | 0.170±0.037 | 0.088±0.025 | 0.212±0.047 |
| subgrad+state-Stier+SDA3(monai) | 0.160±0.035 | 0.083±0.025 | 0.198±0.044 | 0.159±0.037 | 0.081±0.026 | 0.198±0.046 |

`*` = CRPS from a one-member reconstruction (N=1, deterministic; CRPS = per-dim MAE, the N=1 special case of the ensemble formula). Unmarked = proper ensemble CRPS (per-dimension Energy Score, N=30, MAE − 0.5·pairwise member distance) from the stored `members_*.npz`.

## Consistency checks

- DA cached metrics vs recomputed-from-npz (42 values): max |Δ| = 2.16e-04 → PASS
- Neural stored truth vs dataset true_state[:, obs_var_indices]: max |Δ| = 0.00e+00 → PASS

## Reconstruction examples (Hovmöller)

Windows ranked by per-window pooled 24D RMSE of Strong-4DVar (best DA scheme); each figure shows rows = Truth/methods and columns = state / |error| maps for the slow X (8D) and fast Y (16D) blocks. State colors share one scale per figure; error maps share one scale across all rows/methods (99.5th-percentile cap, noted on the colorbar). Dotted vertical lines on the truth row mark observation times.

| Case | Rank | Window | 4DVar win-RMSE | Strong-4DVar | DirectUNet-L(monai,cos) | CFM-M(monai,flat) | SDA3(monai) | DirectUNet+SDA3 | FDV1(monai) | FDV1+SDA3(monai) |
|---|---|---|---|---|---|---|---|---|---|---|
| S0 | worst | 58 | 1.432 | 1.432 | 0.543 | 0.524 | 0.603 | 0.446 | 0.512 | 0.469 |
| S0 | median | 187 | 0.794 | 0.794 | 0.498 | 0.474 | 0.580 | 0.433 | 0.400 | 0.360 |
| S0 | best | 155 | 0.407 | 0.407 | 0.396 | 0.405 | 0.411 | 0.303 | 0.381 | 0.323 |
| S1 | worst | 75 | 1.991 | 1.991 | 0.657 | 0.601 | 0.756 | 0.617 | 0.607 | 0.550 |
| S1 | median | 198 | 1.482 | 1.482 | 0.475 | 0.461 | 0.503 | 0.422 | 0.421 | 0.366 |
| S1 | best | 35 | 0.977 | 0.977 | 0.374 | 0.379 | 0.385 | 0.309 | 0.357 | 0.307 |

![s0-worst](figs/l96_hovm_s0_worst.png)

![s0-median](figs/l96_hovm_s0_median.png)

![s0-best](figs/l96_hovm_s0_best.png)

![s1-worst](figs/l96_hovm_s1_worst.png)

![s1-median](figs/l96_hovm_s1_median.png)

![s1-best](figs/l96_hovm_s1_best.png)
