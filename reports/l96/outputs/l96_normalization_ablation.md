# L96 Per-Channel Normalization Ablation

**Question:** no model or data-pipeline code applies normalization to L96 states/observations anywhere -- raw physical units flow through the whole pipeline unchanged. The shared `UNet1D` backbone's `LayerNorm` normalizes jointly across channels *per timestep*, not per-channel, so it doesn't compensate. This ablation tests whether adding real per-channel z-score normalization changes performance, for the two simplest/most standard neural baselines: **DirectUNet** and **VanillaCFM** -- plus a third leg, **MonaiDirectUNet**, that swaps DirectUNet's `UNet1D` backbone for MONAI's `DiffusionModelUNet` (same data/hyperparameters otherwise) to test whether DirectUNet's collapse under normalization is fixable with a better-conditioned backbone rather than by retuning DirectUNet itself.

**Normalization scheme:** per-channel z-score over the 24D observed subspace (8 slow + 16 fast), `x_norm[d] = (x_raw[d] - mean[d]) / std[d]`. Stats computed once from a freshly-generated L96 train split (1000 windows) and shared across every model/phase (`experiments/l96_norm_stats_obsj2.pt`, see `precompute_l96_norm_stats.py`). Both `states` and `obs` use the same stats.

**Phases:**
- **Original** -- existing benchmark checkpoint, no normalization (baseline).
- **Phase 1** -- same frozen checkpoint, inference-only normalize-input/denormalize-output. Sanity check: expected to look *worse* than Original (the frozen weights were tuned for raw-scale obs, so normalized-scale obs is an out-of-distribution input). If Phase 1 looks unchanged from Original, that's a sign the wiring isn't actually being applied, not a genuine finding.
- **Phase 2** -- retrained from scratch with `data.normalize: true`. The actual research question: does normalization help, hurt, or wash out once the model is trained on normalized-scale data end to end?

**FDV1 deferred:** FDV1 (`models/fourdvarnet.py`, `eval_fdv1_l96.py`) is not yet merged to `master` (still on `feature/l96-fdv1-sda-hybrid`), so this ablation covers DirectUNet + VanillaCFM only; FDV1's leg is a follow-up once that branch merges.

---

## L1b_direct_unet_s0s1

DirectUNet, single-pass obs -> state regression. Hidden [64,128,256], 200 epochs.

### S0

| Phase | RMSE (all) | RMSE (slow) | RMSE (fast) | EV (all) | ES (all) | S1/S0 |
|---|---|---|---|---|---|---|
| Original | 0.6219 | 0.4018 | 0.7319 | 0.8568 | 0.3914 | 1.0056 |
| Phase 1 | 3.0023 | 4.5662 | 2.2203 | -2.0674 | 2.7420 | 1.0015 |
| Phase 2 | 1.7296 | 1.8842 | 1.6522 | 0.0266 | 1.3680 | 0.9947 |

### S1

| Phase | RMSE (all) | RMSE (slow) | RMSE (fast) | EV (all) | ES (all) | S1/S0 |
|---|---|---|---|---|---|---|
| Original | 0.6254 | 0.4005 | 0.7379 | 0.8539 | 0.3949 | 1.0056 |
| Phase 1 | 3.0067 | 4.5825 | 2.2188 | -2.1348 | 2.7450 | 1.0015 |
| Phase 2 | 1.7204 | 1.8617 | 1.6498 | 0.0251 | 1.3575 | 0.9947 |

---

## L1b_monai_unet_s0s1

MonaiUNet1D-backed analogue of L1b: models.direct_unet.DirectUNet -> models.monai_unet_adapter.MonaiDirectUNet (wraps MONAI's DiffusionModelUNet, FiLM-style timestep conditioning + real ResBlocks, vs UNet1D's additive-only conditioning). Identical data setup, normalization, and hyperparameters as L1b_direct_unet_s0s1_norm (single-pass obs -> state regression, hidden [64,128,256], 200 epochs, lr=0.001, gradient_clip_val=10.0) -- only the backbone differs. No unnormalized baseline was trained for this model (testing normalization-robustness specifically was the point), so Original/Phase 1 are --.

### S0

| Phase | RMSE (all) | RMSE (slow) | RMSE (fast) | EV (all) | ES (all) | S1/S0 |
|---|---|---|---|---|---|---|
| Original | -- | -- | -- | -- | -- | -- |
| Phase 1 | -- | -- | -- | -- | -- | -- |
| Phase 2 | 0.5392 | 0.2781 | 0.6697 | 0.8851 | 0.3274 | 1.0025 |

### S1

| Phase | RMSE (all) | RMSE (slow) | RMSE (fast) | EV (all) | ES (all) | S1/S0 |
|---|---|---|---|---|---|---|
| Original | -- | -- | -- | -- | -- | -- |
| Phase 1 | -- | -- | -- | -- | -- | -- |
| Phase 2 | 0.5405 | 0.2768 | 0.6724 | 0.8836 | 0.3290 | 1.0025 |

---

## L3_vanilla_cfm_s0s1

VanillaCFM, standard multi-tau conditional flow matching, ens30 x 10 Euler steps. Hidden [64,128,256], 400 epochs.

### S0

| Phase | RMSE (all) | RMSE (slow) | RMSE (fast) | EV (all) | ES (all) | S1/S0 |
|---|---|---|---|---|---|---|
| Original | 0.5643 | 0.3325 | 0.6803 | 0.8788 | 0.3578 | 1.0034 |
| Phase 1 | 3.0554 | 4.1526 | 2.5067 | -2.0674 | 2.9020 | 1.0010 |
| Phase 2 | 0.5516 | 0.3218 | 0.6665 | 0.8839 | 0.3478 | 1.0009 |

### S1

| Phase | RMSE (all) | RMSE (slow) | RMSE (fast) | EV (all) | ES (all) | S1/S0 |
|---|---|---|---|---|---|---|
| Original | 0.5662 | 0.3345 | 0.6821 | 0.8773 | 0.3598 | 1.0034 |
| Phase 1 | 3.0584 | 4.1649 | 2.5051 | -2.1235 | 2.9057 | 1.0010 |
| Phase 2 | 0.5521 | 0.3223 | 0.6670 | 0.8831 | 0.3504 | 1.0009 |

---

## Findings

Phase 1 confirms the wiring works both ways: normalizing a frozen checkpoint's input badly degrades it (RMSE ~5x worse, EV strongly negative for both models), as expected for feeding an out-of-distribution input scale into weights tuned for the other scale.

Phase 2 (retrained from scratch, identical hyperparameters otherwise) gives **opposite outcomes for the two models**:

- **L1b (DirectUNet): normalization breaks training.** `train_loss` flat-lined at ~1.000 (the no-skill MSE for unit-variance normalized targets) from epoch ~9 through epoch 199 -- the model collapsed to predicting the per-channel mean and never recovered. The resulting checkpoint scores RMSE 1.73/1.72 (S0/S1) vs the Original's 0.62/0.63 -- a ~2.8x degradation -- and EV collapses from 0.86 to ~0.03 (essentially no skill). This was checked against a mid-training false-alarm risk (verified: the no-op regression tests pass, S0/S1 are handled by identical code paths, and NaN-masked/unobserved timesteps pass through normalize() as NaN unchanged, so the masking mechanism is unaffected) -- it is a real, reproducible training failure with these exact hyperparameters (`lr=0.001`, `gradient_clip_val=10.0`), not a bug in the normalize/denormalize wiring.
- **L3 (VanillaCFM): normalization helps slightly.** Training converged normally (loss 0.71 -> 0.046 over 400 epochs). The retrained checkpoint scores RMSE 0.552/0.552 (S0/S1) vs the Original's 0.564/0.566 -- a small but consistent improvement, with EV up from 0.879/0.877 to 0.884/0.883.

**Takeaway:** normalization is not a free win here -- it is architecture-sensitive. VanillaCFM (which already conditions on a time/noise-scale signal and has to learn a well-behaved velocity field regardless of input scale) tolerates and mildly benefits from normalization. DirectUNet (a single-pass regression with no such structure) failed to train at all under the exact hyperparameters tuned for the raw-scale version -- normalization changed its loss landscape enough to need re-tuning (e.g. learning rate) that was deliberately out of scope for this controlled ablation. Whether DirectUNet can be recovered with a lower learning rate is an open follow-up, not answered by this ablation.

- **MonaiDirectUNet (L1b + MONAI's DiffusionModelUNet backbone): normalization not only doesn't collapse it, it outperforms L1b's own unnormalized baseline.** Same data setup and hyperparameters as L1b's Phase 2 (lr=0.001, gradient_clip_val=10.0, 200 epochs) -- the only change is swapping `models.unet.UNet1D` for MONAI's `DiffusionModelUNet` (proper timestep-embedding MLP + FiLM-style scale-shift conditioning + real ResBlocks, vs UNet1D's additive-only conditioning; see the architecture gap analysis this model was built to test, `/homes/rfablet/.claude/plans/monai-diffunet-prototype.md`). Training loss descended smoothly the whole run (0.75 -> 0.017 train, val plateauing ~0.114-0.117 from epoch ~50) with no flatlining. Final RMSE 0.539/0.541 (S0/S1) -- *better* than L1b's own unnormalized Original (0.62/0.63) and ~3.2x better than L1b's collapsed normalized Phase 2 (1.73/1.72). EV 0.885/0.884, on par with VanillaCFM's normalized result (0.884/0.883) despite being architecturally a single-pass deterministic regressor like DirectUNet, not a flow-matching model with an inherent noise-scale signal. This is consistent with the hypothesis that DirectUNet's collapse under normalization was a backbone-conditioning limitation, not something fundamental to single-pass regression: a backbone with real scale-aware conditioning tolerates the normalized input distribution just as well as (here, slightly better than) a model that conditions on an explicit noise/time scale. Caveat: MonaiDirectUNet has 3.11x more parameters than DirectUNet at matching `hidden_channels` (5,889,048 vs 1,896,600 at [64,128,256]), so this is not a controlled like-for-like capacity comparison -- it shows the normalization failure is avoidable with a richer backbone, not that UNet1D's conditioning scheme specifically is the sole cause.

---

