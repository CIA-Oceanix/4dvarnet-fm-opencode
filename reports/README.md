# Results Overview

10 Observations per Window

[beta τ: Beta-Tuned Timestep Diffusion Model](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/00328.pdf) paper

[logit-normal τ: Scaling Rectified Flow Transformers for High-Resolution Image Synthesis](https://arxiv.org/pdf/2403.03206) paper

## State estimation 

| Model | N_ens | s0 RMSE | s0 R² | s0 CRPS | s1 RMSE | s1 R² | s1 CRPS |
|---|---:|---:|---:|---:|---:|---:|---:|
| Weak-4DVar | 1 | 0.874 | 0.972 | 0.598 | 2.171 | 0.910 | 1.666 |
| Strong-4DVar | 1 | 0.927 | 0.962 | 0.640 | 2.630 | 0.858 | 2.020 |
| EnKF | 50 | 1.286 | 0.973 | 0.706 | 2.741 | 0.882 | 1.842 |
| ETKF | 50 | 1.236 | 0.976 | 0.693 | 2.776 | 0.879 | 1.895 |
| DirectUNet | 1 | 0.598 | 0.995 | 0.441 | 0.582 | 0.995 | 0.438 |
| TweedieCFM (K=5) | 50 | 0.561 | 0.995 | 0.301 | 0.598 | 0.994 | 0.317 |
| TweedieCFM (K=1) | 50 | 0.568 | 0.995 | 0.305 | 0.562 | 0.995 | 0.297 |
| VanillaCFM | 50 | 0.549 | 0.996 | 0.289 | 0.558 | 0.996 | 0.306 |
| VanillaCFM (τ=0 only) | 50 | 0.604 | 0.995 | 0.418 | 0.580 | 0.995 | 0.404 |
| VanillaCFM (logit-normal τ) | 50 | 0.608 | 0.994 | 0.304 | 0.564 | 0.995 | 0.299 |
| VanillaCFM (beta τ) | 50 | 0.588 | 0.995 | 0.320 | 0.545 | 0.996 | 0.302 |
| VanillaCFM (beta τ, unconditional) | 50 | 4.047 | 0.760 | 2.380 | 1.732 | 0.936 | 0.931 |

## Joint state + parameter estimation

Lorenz-63 parameters (σ, ρ, β, c1)

| Model | N_ens | s0 RMSE | s0 R² | s0 σ RMSE | s0 ρ RMSE | s0 β RMSE | s0 c1 RMSE | s1 RMSE | s1 R² | s1 σ RMSE | s1 ρ RMSE | s1 β RMSE | s1 c1 RMSE |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Joint-Weak-4DVar | – | 2.076 | 0.910 | 27.197 | 4.045 | 0.987 | 0.000 | 2.067 | 0.902 | 24.127 | 3.587 | 0.966 | 0.150 |
| Joint-Strong-4DVar | – | 2.840 | 0.807 | 33.196 | 8.127 | 3.221 | 0.000 | 2.753 | 0.832 | 15.466 | 7.979 | 1.002 | 0.150 |
| Joint-EnKF | – | 2.184 | 0.930 | 4.807 | 2.439 | 0.729 | 1.039 | 2.249 | 0.927 | 4.199 | 2.161 | 0.618 | 0.940 |
| Joint-ETKF | – | 1.280 | 0.975 | 0.788 | 1.003 | 0.163 | 0.035 | 1.928 | 0.943 | 2.125 | 1.535 | 0.324 | 0.158 |
| Joint-DirectUNet | 1 | 0.908 | 0.988 | 1.088 | 0.655 | 0.185 | 0.018 | 0.803 | 0.991 | 1.049 | 0.833 | 0.259 | 0.027 |
| Joint-VanillaCFM | 50 | 0.790 | 0.991 | 2.525 | 5.898 | 0.599 | 0.291 | 0.769 | 0.991 | 1.300 | 3.032 | 0.397 | 0.186 |
| Joint-TweedieCFM | 50 | 0.702 | 0.993 | 1.472 | 2.466 | 0.288 | 0.196 | 0.693 | 0.993 | 1.577 | 3.893 | 0.360 | 0.284 |
