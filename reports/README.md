# Results Overview

10 Observations per Window

[beta τ: Beta-Tuned Timestep Diffusion Model](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/00328.pdf) paper

[logit-normal τ: Scaling Rectified Flow Transformers for High-Resolution Image Synthesis](https://arxiv.org/pdf/2403.03206) paper

[score-based: Rozet & Louppe, "Score-based Data Assimilation" (NeurIPS 2023)](https://proceedings.neurips.cc/paper_files/paper/2023/hash/7f7fa581cc8a1970a4332920cdf87395-Abstract-Conference.html), as formalized in Sec. 3.4 / Eq. (12)-(14) of `docs/preprint_4dvarnet_fm_2025.pdf` — samples p(x1|y) by guiding a purely **unconditional** flow-matching prior (`models/sda.py::UnconditionalPriorCFM`, trained via `config/models/sda_prior.yaml`, no obs/forcing/params conditioning at all) with a DPS-style normalized-gradient observation-cost correction at each Euler step (`evaluation/sda_sampler.py::sda_guided_sample`, run via `eval_sda_l63.py`) — the same model class and sampler the L96 SDA benchmark uses, instead of training a model that sees y directly. `guidance_weight=20` (picked by an S0 sweep over {0,1,3,10,20,40,80,150}, same procedure as the L96 SDA1 row, different optimal magnitude); row below regenerated against this implementation and supersedes the earlier VanillaCFM-based-prior + different-guidance-rule numbers.

**SDA1/SDA2/SDA3/hybrid rows (Monai section), mirroring the L96 SDA family's naming:** `SDA1 (Monai)` is the same unconditional guided prior as the plain-section `Score-based CFM (unconditional prior)` row above, with the MONAI backbone swapped in (`config/models/monai_sda_prior.yaml`) -- otherwise identical guidance setup. `SDA2 (Monai)` conditions the prior on the window's true params + forcing (`models/sda.py::ConditionalPriorCFM`, `config/models/monai_sda_prior_cond.yaml`); `SDA3 (Monai)` instead conditions on a noisy params estimate (`data.noisy_da_bias`/`noisy_da_max=1.5`, `config/models/monai_sda_prior_cond_noisy.yaml`). `DirectUNet+SDA1/2/3 (Monai)` are SDEdit-style warm-started hybrids -- DirectUNet (Monai)'s frozen point estimate seeds the guided sampling trajectory at `tau0=0.3` instead of pure noise (`evaluation/sda_sampler.py`'s `mean_estimate`/`tau0`, `eval_sda_l63.py --config-name models/monai_sda_prior[_cond[_noisy]] +hybrid.mean_model=monai_direct_unet`), same convention as the L96 `DirectUNet+SDA1/2/3` rows; `tau0=0.3` taken from the L96 hybrids' swept value, not independently re-swept for L63. All five use `guidance_weight=20`, `N_outer=10`, N_ensemble=50, matching the plain-section SDA1 row above.

**FDV1/FDV2/subgrad+state rows (Monai section), mirroring the L96 FDV family's naming:** unrolled 4DVarNet-style solvers (`models/fourdvarnet.py::FourDVarNetSolver`, N_outer=10, zero-initialized, deterministic -- evaluated as a single pass, N=1, same convention as the L96 FDV1/FDV2 rows), distinguished by what's fed to the weight-tied UNet at each iteration: `FDV1 (Monai)` uses `update_input='obs+state'` (`config/models/monai_fdv_obsstate.yaml`, no gradient/cost term); `FDV2 (Monai)` uses `update_input='grad+state'` (`config/models/monai_fdv_grad.yaml`, a real variational-cost gradient via `torch.autograd.grad`); `subgrad+state (Monai)` uses the cheaper proxy-gradient variant (`config/models/monai_fdv_subgrad.yaml`, obs-residual + prior-autoencoder-residual, no autograd call). `FDV1/FDV2/subgrad+state+SDA1/2/3 (Monai)` are the corresponding SDEdit-style warm-started hybrids (same `tau0=0.3`/`guidance_weight=20` convention as the `DirectUNet+SDA*` rows above, `eval_sda_l63.py +hybrid.mean_model=monai_fdv_{obsstate,grad,subgrad}`), evaluated as N=50 ensembles. **Status as of 2026-09-22:** all three mean models were relaunched with `epochs: 3000` (up from an earlier 400-epoch cap that ended at max_epochs rather than early-stopping) -- `FDV1 (Monai)`'s s0 case finished (early-stopped at epoch 1595/3000), s1 is still training; `FDV2 (Monai)` and `subgrad+state (Monai)` were just (re)launched and have no results yet. `XXX` marks every cell not yet available; this table will be updated as each run/hybrid finishes.

## State estimation 

| Model | N_ens | s0 RMSE | s0 R² | s0 CRPS | s1 RMSE | s1 R² | s1 CRPS |
|---|---:|---:|---:|---:|---:|---:|---:|
| Weak-4DVar | 1 | 0.878 | 0.972 | 0.601 | 2.172 | 0.910 | 1.666 |
| Strong-4DVar | 1 | 0.998 | 0.955 | 0.673 | 2.620 | 0.858 | 2.019 |
| EnKF | 50 | 1.229 | 0.976 | 0.679 | 2.739 | 0.881 | 1.834 |
| ETKF | 50 | 1.212 | 0.976 | 0.678 | 2.762 | 0.881 | 1.895 |
| DirectUNet | 1 | 0.598 | 0.995 | 0.441 | 0.582 | 0.995 | 0.438 |
| TweedieCFM (K=5) | 50 | 0.561 | 0.995 | 0.301 | 0.598 | 0.994 | 0.317 |
| TweedieCFM (K=1) | 50 | 0.568 | 0.995 | 0.305 | 0.562 | 0.995 | 0.297 |
| VanillaCFM | 50 | 0.549 | 0.996 | 0.289 | 0.558 | 0.996 | 0.306 |
| VanillaCFM (τ=0 only) | 50 | 0.604 | 0.995 | 0.418 | 0.580 | 0.995 | 0.404 |
| VanillaCFM (logit-normal τ) | 50 | 0.608 | 0.994 | 0.304 | 0.564 | 0.995 | 0.299 |
| VanillaCFM (beta τ) | 50 | 0.588 | 0.995 | 0.320 | 0.545 | 0.996 | 0.302 |
| VanillaCFM (beta τ, unconditional) | 50 | 4.047 | 0.760 | 2.380 | 1.732 | 0.936 | 0.931 |
| Score-based CFM (unconditional prior) | 50 | 1.958 | 0.941 | 0.874 | 1.647 | 0.958 | 0.761 |

## State estimation (Monai backbone)

Same models/configs as above, swapping the custom `UNet1D` (`models/unet.py`) for the
MONAI `UNet1D`/`MonaiUNet1D` backbone (`models/monai_unet_adapter.py`,
`config/models/monai_*.yaml`, `monai_*` `model_type`s) — otherwise identical
architecture size, training config, and S0/S1 protocol.

| Model | N_ens | s0 RMSE | s0 R² | s0 CRPS | s1 RMSE | s1 R² | s1 CRPS |
|---|---:|---:|---:|---:|---:|---:|---:|
| Weak-4DVar | 1 | 0.878 | 0.972 | 0.601 | 2.172 | 0.910 | 1.666 |
| Strong-4DVar | 1 | 0.998 | 0.955 | 0.673 | 2.620 | 0.858 | 2.019 |
| EnKF | 50 | 1.229 | 0.976 | 0.679 | 2.739 | 0.881 | 1.834 |
| ETKF | 50 | 1.212 | 0.976 | 0.678 | 2.762 | 0.881 | 1.895 |
| DirectUNet (Monai)                         | 1  | 0.629 | 0.994 | 0.469 | 0.610 | 0.995 | 0.463 |
| TweedieCFM (K=5) (Monai)                   | 50 | 0.607 | 0.994 | 0.334 | 0.656 | 0.994 | 0.430 |
| TweedieCFM (K=1) (Monai)                   | 50 | 0.681 | 0.993 | 0.468 | 0.695 | 0.993 | 0.460 |
| VanillaCFM (Monai)                         | 50 | 0.573 | 0.995 | 0.312 | 0.563 | 0.995 | 0.312 |
| VanillaCFM (τ=0 only) (Monai)              | 50 | 0.623 | 0.994 | 0.433 | 0.587 | 0.995 | 0.415 |
| VanillaCFM (logit-normal τ) (Monai)        | 50 | 0.774 | 0.989 | 0.364 | 0.689 | 0.993 | 0.348 |
| VanillaCFM (beta τ) (Monai)                | 50 | 0.631 | 0.994 | 0.346 | 0.606 | 0.995 | 0.348 |
| VanillaCFM (beta τ, unconditional) (Monai) | 50 | 3.347 | 0.827 | 1.986 | 1.070 | 0.982 | 0.602 |
| SDA1 (Monai) | 50 | 1.546 | 0.963 | 0.695 | 1.481 | 0.965 | 0.687 |
| SDA2 (Monai) | 50 | 0.892 | 0.988 | 0.480 | 0.772 | 0.991 | 0.458 |
| SDA3 (Monai) | 50 | 1.018 | 0.983 | 0.517 | 0.859 | 0.990 | 0.489 |
| DirectUNet+SDA1 (Monai) | 50 | 0.791 | 0.991 | 0.486 | 0.777 | 0.992 | 0.475 |
| DirectUNet+SDA2 (Monai) | 50 | 0.759 | 0.992 | 0.466 | 0.748 | 0.992 | 0.472 |
| DirectUNet+SDA3 (Monai) | 50 | 0.754 | 0.992 | 0.455 | 0.762 | 0.992 | 0.468 |
| FDV1 (Monai) | 1 | 0.648 | 0.994 | 0.487 | XXX | XXX | XXX |
| FDV2 (Monai) | 1 | XXX | XXX | XXX | XXX | XXX | XXX |
| subgrad+state (Monai) | 1 | XXX | XXX | XXX | XXX | XXX | XXX |
| FDV1+SDA1 (Monai) | 50 | XXX | XXX | XXX | XXX | XXX | XXX |
| FDV1+SDA2 (Monai) | 50 | XXX | XXX | XXX | XXX | XXX | XXX |
| FDV1+SDA3 (Monai) | 50 | XXX | XXX | XXX | XXX | XXX | XXX |
| FDV2+SDA1 (Monai) | 50 | XXX | XXX | XXX | XXX | XXX | XXX |
| FDV2+SDA2 (Monai) | 50 | XXX | XXX | XXX | XXX | XXX | XXX |
| FDV2+SDA3 (Monai) | 50 | XXX | XXX | XXX | XXX | XXX | XXX |
| subgrad+state+SDA1 (Monai) | 50 | XXX | XXX | XXX | XXX | XXX | XXX |
| subgrad+state+SDA2 (Monai) | 50 | XXX | XXX | XXX | XXX | XXX | XXX |
| subgrad+state+SDA3 (Monai) | 50 | XXX | XXX | XXX | XXX | XXX | XXX |

