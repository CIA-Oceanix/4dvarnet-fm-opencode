## 2026-09-10: L96 — add DirectUNet(aug)+SDA3 hybrid row to the consolidated benchmark

**Summary:** `l96_obs_density_augmented_training.md` already evaluated the
`obs_density_augment`-trained DirectUNet-M mean warm-starting SDA3's guided
sampling (`DirectUNet(aug)+SDA3`, full-density RMSE 0.389 S0 -- second only
to `FDV1+SDA3(monai)`'s 0.379, and well ahead of the non-augmented
`DirectUNet+SDA3`'s 0.420), but that row was never carried over to
`l96_consolidated_benchmark.md`, unlike the plain `DirectUNet-M(aug)`/
`CFM-M(aug)` rows added in PR #188. Added it to all 7 tables (scheme
description, pooled RMSE/EV/ES, per-window RMSE/EV/CRPS) at the same
position as `DirectUNet+SDA3` (its non-augmented sibling), computed directly
from `experiments/l96_obs_density_directunet_aug_sda3_hybrid/estimates_directunet_sda3_{s0,s1}_keep16.npz`
via `evaluation/estimate_metrics.py`, same convention as every other
directly-computed row in this table.
**Files modified:** `reports/l96/outputs/l96_consolidated_benchmark.md` --
new row in every table; expanded the top-of-file note to cover this third
directly-computed row and flag one real nuance: unlike its non-augmented
sibling hybrids (`DirectUNet+SDA3`/`FDV1+SDA3`), this row's stored `.npz`
only kept the per-repeat ensemble **mean** trajectory (the
`eval_obs_density_l96.py` sweep convention), not the raw 30 members, so its
ES/CRPS here fall back to the N=1 MAE-proxy convention (marked `*`) rather
than a true ensemble spread score -- RMSE/EV are unaffected (mean-based
either way).
**Rationale:** User noticed the gap directly ("it's not included in the
consolidated benchmark") -- the underlying evaluation already existed and
this is a strong result (second-best full-density RMSE/EV in the whole
table), so it belonged in the consolidated view rather than only in the
dedicated obs-density report. No new checkpoint archival needed: both
constituent checkpoints (`L1b_monai_unet_s0s1_norm_obsdensity`,
`SDA3_monai_cond_noisy_l96_norm`) are already in the canonical
`experiments/l96/` location from the 2026-09-10 checkpoint consolidation.
**Verification:** numbers cross-checked against `evaluation/estimate_metrics.py`'s
`evaluate_estimates`/`per_window_rmse_ev`/`per_window_deterministic_crps`
run directly against the stored npz (S0 pooled RMSE 0.3889, matching the
0.389 already reported in `l96_obs_density_augmented_training.md`); no code
changed, markdown-only.
