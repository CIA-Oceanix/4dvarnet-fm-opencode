## 2026-10-16: G3-3D SDA prior at G1-3D's widths (capacity-matched)

**Summary:** New config `config/experiment/G3_sda_prior_3d_specwind_b16_w15_matched.yaml`: the 3D SDA prior with hidden_channels [32, 64, 128, 128] (4 levels, 16.83 M) instead of [32, 64, 128] (3 levels, 10.0 M), so it matches G1-3D (`G1_direct_unet_3d_specwind_b16_w15_kpow`, 16.83 M) up to inputs and conditioning. Training protocol unchanged.
**Files modified:** `config/experiment/G3_sda_prior_3d_specwind_b16_w15_matched.yaml` — new
**Rationale:** S/M/L tiers must be comparable across DirectUNet, CFM and SDA prior; the published-SDA and warm-start SDA rows (QG S0 0.881 / 0.891) used a prior with ~60% of G1-3D's parameters and one level less.
**Verification:** model built from the config: 16.83 M parameters (G1-3D 16.83 M); `pytest -m "not slow" tests/test_qg_direct_unet_3d.py`. Training: job 61856 (+ resume 61857).
