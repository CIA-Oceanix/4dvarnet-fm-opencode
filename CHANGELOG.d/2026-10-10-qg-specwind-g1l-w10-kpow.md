## 2026-10-10: QG gyrostat neural — G1-L with a 10-day network window and the K^-1.5 density draw

**Summary:** New config `G1L_direct_unet_tchannels_specwind_b16_w10_kpow.yaml`: `G1L_direct_unet_tchannels_specwind_b16_w15_kpow.yaml` with `train_window_days: 10` (nothing else changes; the 30-day test windows are estimated from 10-day sub-windows 5 days apart).
**Files modified:** `config/experiment/G1L_direct_unet_tchannels_specwind_b16_w10_kpow.yaml` — new
**Rationale:** the 15-day + K^-1.5 G1-L reached S0 0.856 / S1 0.816 (above strong 4D-Var 0.840 and EnKS N=320 0.841, paired CIs excluding 0); the 15-day window gained +0.024 over 30 days at every density on G1-M, so a shorter window is the next test. A second seed of the 15-day run is launched alongside (SEED=1, existing config).
**Verification:** config-only (keys covered by tests/test_qg_specwind_kpow_window.py); diff against the 15-day config is the window length; runs launched on the cluster.
