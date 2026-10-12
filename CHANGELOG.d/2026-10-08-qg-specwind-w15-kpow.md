## 2026-10-08: QG gyrostat neural — G1-M with both the 15-day window and the K^-1.5 density draw

**Summary:** New config `G1_direct_unet_tchannels_specwind_b16_w15_kpow.yaml`: the 15-day network window of `..._b16_w15.yaml` combined with the `cols_per_day_power: 1.5` density draw of `..._b16_kpow.yaml`, trained from scratch on the batch-16 protocol.
**Files modified:** `config/experiment/G1_direct_unet_tchannels_specwind_b16_w15_kpow.yaml` — new
**Rationale:** each change alone put G1-M ~0.055 ahead of the uniform 30-day G1-M in val loss at epoch ~590; this run tests whether the two gains add up.
**Verification:** config-only (keys already covered by tests/test_qg_specwind_kpow_window.py); run launched as job on the cluster.
