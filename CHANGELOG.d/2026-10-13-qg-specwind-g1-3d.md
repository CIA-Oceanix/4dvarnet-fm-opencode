## 2026-10-13: QG gyrostat neural — 3D DirectUNet (G1-3D), size-matched to G1-M

**Summary:** New model type `direct_unet_3d` (`MonaiDirectUNet3DQG`): the G1 single-pass estimator on MONAI's `DiffusionModelUNet` with `spatial_dims=3` over (time, ny, nx) — obs and obs mask of both layers as 4 input channels, 2-layer psi out, periodic in space and zero-padded in time, weights shared across days, any window length. Config `G1_direct_unet_3d_specwind_b16_w15_kpow.yaml`: 4 levels [32, 64, 128, 128], 2 residual blocks, 16.8 M parameters (2D G1-M: 14.5 M), trained like the 2D G1-M 15-day + K^-1.5 row except a peak LR of 3e-4.
**Files modified:**
- `models/monai_unet_qg2d.py` — `MonaiDirectUNet3DQG`
- `train_qg_neural.py` — `direct_unet_3d` in `build_model`, `DIRECT_TYPES`, `TIERED_TYPES`, CLI choices; `decoder_output_module` now finds the last decoder block of the 3D backbones too (the activation monitor was blind for the 3D SDA prior)
- `config/experiment/G1_direct_unet_3d_specwind_b16_w15_kpow.yaml` — new
- `tests/test_qg_direct_unet_3d.py` — new; `tests/test_qg_sda_prior_3d.py` — weights randomized before the periodicity check: MONAI zero-initializes the U-Net's output conv, so on a fresh network that check compared 0 with 0
**Rationale:** candidates benchmarked at batch 16 on an RTX 8000 (fp32): 3D [32,64,128] 10.0 M 1.22 s/step; [40,80,160] 15.6 M 2.03 s; [32,64,128]x3 blocks 13.0 M 1.55 s; [32,64,128,128] 16.8 M 1.17 s (chosen: closest size at the lowest cost, one more level of temporal reach); 2D G1-M 0.15 s.
**Verification:** `pytest tests/test_qg_direct_unet_3d.py tests/test_qg_sda_prior_3d.py` — 16 passed (periodicity and obs-mask dependence now checked on non-zero outputs); ruff clean; run launched on the cluster.
