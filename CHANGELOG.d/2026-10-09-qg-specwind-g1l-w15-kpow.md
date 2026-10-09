## 2026-10-09: QG gyrostat neural — G1-L with the 15-day window and the K^-1.5 density draw

**Summary:** New config `G1L_direct_unet_tchannels_specwind_b16_w15_kpow.yaml`: `G1L_direct_unet_tchannels_specwind_b16.yaml` plus `train_window_days: 15`, `window_stride: 5` and `cols_per_day_power: 1.5` (nothing else changes).
**Files modified:** `config/experiment/G1L_direct_unet_tchannels_specwind_b16_w15_kpow.yaml` — new
**Rationale:** on G1-M the two changes together gave S0 0.846 / S1 0.810 (+0.033 over the uniform 30-day G1-M, level with or above 4D-Var 0.840 and EnKS N=320 0.841); G1-L has been ~+0.02 above G1-M in every comparison so far.
**Verification:** config-only (keys covered by tests/test_qg_specwind_kpow_window.py); diff against the G1-L b16 config is the three keys; run launched on the cluster.
