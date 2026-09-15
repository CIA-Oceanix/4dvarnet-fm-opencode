# Results Overview

10 Observations per Window

[beta τ: Beta-Tuned Timestep Diffusion Model](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/00328.pdf) paper

[logit-normal τ: Scaling Rectified Flow Transformers for High-Resolution Image Synthesis](https://arxiv.org/pdf/2403.03206) paper

[score-based: Rozet & Louppe, "Score-based Data Assimilation" (NeurIPS 2023)](https://proceedings.neurips.cc/paper_files/paper/2023/hash/7f7fa581cc8a1970a4332920cdf87395-Abstract-Conference.html), as formalized in Sec. 3.4 / Eq. (12)-(14) of `docs/preprint_4dvarnet_fm_2025.pdf` — samples p(x1|y) by guiding a purely **unconditional** flow-matching prior (`models/sda.py::UnconditionalPriorCFM`, trained via `config/models/sda_prior.yaml`, no obs/forcing/params conditioning at all) with a DPS-style normalized-gradient observation-cost correction at each Euler step (`evaluation/sda_sampler.py::sda_guided_sample`, run via `eval_sda_l63.py`) — the same model class and sampler the L96 SDA benchmark uses, instead of training a model that sees y directly. `guidance_weight=20` (picked by an S0 sweep over {0,1,3,10,20,40,80,150}, same procedure as the L96 SDA1 row, different optimal magnitude); row below regenerated against this implementation and supersedes the earlier VanillaCFM-based-prior + different-guidance-rule numbers.

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
| DirectUNet (Monai) | 1 | 0.629 | 0.994 | 0.469 | 0.610 | 0.995 | 0.463 |
| TweedieCFM (K=5) (Monai) | 50 | 0.607 | 0.994 | 0.334 | 0.656 | 0.994 | 0.430 |
| TweedieCFM (K=1) (Monai) | 50 | 0.681 | 0.993 | 0.468 | 0.695 | 0.993 | 0.460 |
| VanillaCFM (Monai) | 50 | 0.573 | 0.995 | 0.312 | 0.563 | 0.995 | 0.312 |
| VanillaCFM (τ=0 only) (Monai) | 50 | 0.623 | 0.994 | 0.433 | 0.587 | 0.995 | 0.415 |
| VanillaCFM (logit-normal τ) (Monai) | 50 | 0.774 | 0.989 | 0.364 | 0.689 | 0.993 | 0.348 |
| VanillaCFM (beta τ) (Monai) | 50 | 0.631 | 0.994 | 0.346 | 0.606 | 0.995 | 0.348 |
| VanillaCFM (beta τ, unconditional) (Monai) | 50 | 3.347 | 0.827 | 1.986 | 1.070 | 0.982 | 0.602 |

