## 2026-10-12: QG gyrostat neural — G1-L with a 20-day network window and the K^-1.5 density draw

**Summary:** New config `G1L_direct_unet_tchannels_specwind_b16_w20_kpow.yaml`: `G1L_direct_unet_tchannels_specwind_b16_w15_kpow.yaml` with `train_window_days: 20` (nothing else changes; the 30-day test windows are estimated from 20-day sub-windows 5 days apart).
**Files modified:** `config/experiment/G1L_direct_unet_tchannels_specwind_b16_w20_kpow.yaml` — new
**Rationale:** the 15-day + K^-1.5 G1-L scored S0 0.856/0.857 over two seeds (above strong 4D-Var and EnKS N=320); the 10-day window lost 0.022 at K = 3. The 20-day run checks the other side of the window optimum.
**Verification:** config-only (keys covered by tests/test_qg_specwind_kpow_window.py); diff against the 15-day config is the window length; run launched on the cluster.
