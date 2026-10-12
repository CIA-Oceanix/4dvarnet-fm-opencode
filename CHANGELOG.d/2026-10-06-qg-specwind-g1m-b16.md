## 2026-10-06: QG gyrostat neural — G1-M on the batch-16 protocol

**Summary:** New config `G1_direct_unet_tchannels_specwind_b16.yaml`: the M-tier DirectUNet (G1, [64, 128, 256]) on exactly the G1-L batch-16 budget (batch 16, 1200 epochs, peak LR 5e-4 with a 40-epoch warm-up then cosine, regeneration every 30 epochs, 10 workers).
**Files modified:** `config/experiment/G1_direct_unet_tchannels_specwind_b16.yaml`; `tests/test_qg_neural.py` (the config equals G1-L b16 apart from the channels).
**Rationale:** G1-L b16 at epoch 966/1200 scored 0.828 at S0 / 0.796 at S1 (EnKS 0.812 / S1-tuned 0.665), against 0.792 / 0.759 for G1-L at batch 2 x 200 epochs. The M row on the same protocol gives the capacity comparison where it matters.
**Verification:** `pytest tests/test_qg_neural.py -k "b16 or large_batch or tier"`; run launched (G1_direct_unet_tchannels_specwind_b16_s0).
