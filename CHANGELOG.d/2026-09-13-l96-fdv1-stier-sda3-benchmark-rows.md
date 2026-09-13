## 2026-09-13: L96 — add FDV1-Stier(monai) and FDV1-Stier+SDA3(monai) rows to the consolidated benchmark

**Summary:** Added two new rows to `l96_consolidated_benchmark.md`:
`FDV1-Stier(monai)`, a standalone FDV1 mean-estimate model at a smaller "S"
capacity tier (`hidden_channels=[32,64,128]`, `monai_num_res_blocks=1`,
1,055,544 params vs the M tier's 5,889,048) with a random initial condition
(`init_state_var=0.1`) and a small auxiliary prior-consistency loss term
(`aux_var_cost_weight=0.01`), which beats the M-tier `FDV1(monai)` baseline
on every RMSE/EV metric despite being ~5.6x smaller; and
`FDV1-Stier+SDA3(monai)`, the same SDEdit-style SDA3 warm-start hybrid as
`FDV1+SDA3(monai)` but swapping in `FDV1-Stier(monai)` as the mean estimate,
which becomes the **new best scheme in the entire table** on every
RMSE/EV/ES metric (pooled RMSE 0.3587/0.3583 S0/S1, vs `FDV1+SDA3(monai)`'s
previous-best 0.3787/0.3740).
**Files modified:** `reports/l96/outputs/l96_consolidated_benchmark.md` --
new row in the methods table and all 6 numeric tables (pooled RMSE/EV/ES,
per-window RMSE/EV/CRPS) at the same position as `FDV1(monai)`/
`FDV1+SDA3(monai)` respectively; moved the bold "best value per column"
marking from `FDV1+SDA3(monai)` to `FDV1-Stier+SDA3(monai)` in the three
pooled tables (it now wins every column in all three); softened
`FDV1+SDA3(monai)`'s "best scheme in this table overall" description and
updated `DirectUNet(aug)+SDA3`'s ranking language (now third-best, not
second-best, on full-density RMSE/EV); extended the top-of-file
explanatory note with a new adjacent paragraph covering these two rows'
provenance and a cross-worktree gotcha (see below).
**Rationale:** Both numbers were already computed and verified this session
(directly via `evaluation/estimate_metrics.py` against saved
`estimates_{s0,s1}.npz`/`members_{s0,s1}.npz`, same convention as every
other directly-computed row in this table -- not via this script's full
regeneration path, which needs DA-baseline/joint-comparison caches only
available in the training worktree) and represent a genuine new best
result (smaller AND better mean model; hybrid beats the prior best on
every metric), so they belong in the consolidated view. Producing the
hybrid row surfaced a real bug worth documenting: running the eval from
the `4dvarnet-fm-fdv-monai` worktree (needed for the SDA3
checkpoint/norm-stats/cached test dataset) silently built an S+-tier
model (1,482,264 params) instead of true S-tier (1,055,544), because that
worktree's `train.py`/`models/fourdvarnet.py` predates this session's
`monai_num_res_blocks` feature and silently defaulted it to 2 --
partially failing to load the checkpoint (8 `unet.*` shape mismatches,
silently skipped) and producing garbage output (RMSE ~1.8, negative EV).
Fixed by running the identical command from `4dvarnet-fm-obs-density-gen`
instead (which has the feature), with absolute paths into the other
worktree's `experiments/` dir for the SDA3 checkpoint/config/dataset/
norm-stats.
**Verification:** numbers cross-checked against `evaluation/estimate_metrics.py`'s
`evaluate_estimates`/`per_window_rmse_ev`/`per_window_deterministic_crps`/
`per_window_ensemble_crps`/`evaluate_ensemble_estimates` run directly
against the stored npz files (pooled RMSE/EV/ES matched to 4 decimal
places against numbers already reported earlier this session; the hybrid
row's true 1,055,544-param S-tier build was independently confirmed via
`sum(p.numel() for p in model.unet.parameters()) == 1055544` with zero
`unet.*` shape-mismatch warnings); no code changed, markdown-only.
