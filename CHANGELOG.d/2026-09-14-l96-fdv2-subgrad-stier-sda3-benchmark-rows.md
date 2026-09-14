## 2026-09-14: L96 — add subgrad+state-Stier(monai) and subgrad+state-Stier+SDA3(monai) rows to the consolidated benchmark

**Summary:** Added two new rows to `l96_consolidated_benchmark.md`:
`subgrad+state-Stier(monai)`, a standalone FDV2 mean-estimate model using the
cheap `update_input=subgrad+state` proxy-gradient solver (observation
residual `obs-x` plus prior-autoencoder residual `x-Phi(x)`, no
`torch.autograd.grad` call) at the same S-tier MonaiUNet1D
(`hidden_channels=[32,64,128]`, `monai_num_res_blocks=1`, 1,055,544 params)
and `init_state_var=0.1`/`aux_var_cost_weight=0.01` recipe as
`FDV1-Stier(monai)`, which becomes the **best standalone (non-hybrid) neural
mean-estimate model found in this entire investigation** -- better than
`FDV1-Stier(monai)` itself, and free of the severe fast-Y collapse
(variance ratio ~0.5-0.6) every earlier `subgrad+state`/`grad+state` attempt
at the M tier showed; and `subgrad+state-Stier+SDA3(monai)`, the same
SDEdit-style SDA3 warm-start hybrid as `FDV1-Stier+SDA3(monai)` but swapping
in `subgrad+state-Stier(monai)` as the mean estimate, which becomes the
**new best scheme in the entire table** on every RMSE/EV/ES metric (pooled
RMSE 0.3410/0.3402 S0/S1, vs `FDV1-Stier+SDA3(monai)`'s previous-best
0.3587/0.3583).
**Files modified:** `reports/l96/outputs/l96_consolidated_benchmark.md` --
new row in the methods table and all 6 numeric tables (pooled RMSE/EV/ES,
per-window RMSE/EV/CRPS) at the same position as `FDV1-Stier(monai)`/
`FDV1-Stier+SDA3(monai)` respectively; moved the bold "best value per
column" marking from `FDV1-Stier+SDA3(monai)` to
`subgrad+state-Stier+SDA3(monai)` in the three pooled tables (it now wins
every column in all three); softened `FDV1-Stier+SDA3(monai)`'s "new best
scheme" description to note it has since been beaten; extended the
top-of-file explanatory note with a new adjacent paragraph covering these
two rows' identical direct-npz-metrics provenance.
**Rationale:** Both numbers were already computed and verified this session
(directly via `evaluation/estimate_metrics.py` against saved
`estimates_{s0,s1}.npz`/`members_{s0,s1}.npz`, same convention as every
other directly-computed row in this table -- not via this script's full
regeneration path, which needs DA-baseline/joint-comparison caches only
available in the training worktree) and represent a genuine new best
result (a cheaper solver AND a better mean model; the hybrid beats the
prior best on every metric), so they belong in the consolidated view.
**Verification:** numbers cross-checked against `evaluation/estimate_metrics.py`'s
`evaluate_estimates`/`per_window_rmse_ev`/`per_window_deterministic_crps`/
`per_window_ensemble_crps` run directly against the stored npz files
(pooled RMSE/EV matched to 6 decimal places against numbers already
reported earlier this session for both rows; row 2's npz files were
independently confirmed to be the `subgrad+state` hybrid's own data, not a
stale copy of `FDV1-Stier+SDA3`'s, by matching the recomputed pooled RMSE
against the expected 0.341023/0.340167 S0/S1 values before trusting any
other number from them); no code changed, markdown-only.
