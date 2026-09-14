## 2026-09-14: L96 — add grad+state-Stier-priorresidual(N5/tbptt2x5)(monai) rows to the consolidated benchmark

**Summary:** Added two new rows to `l96_consolidated_benchmark.md`:
`grad+state-Stier-priorresidual-N5(monai)` and
`grad+state-Stier-priorresidual-tbptt2x5(monai)`, both the REAL
`update_input=grad+state` gradient-conditioned FDV2 solver
(`torch.autograd.grad(var_cost, x, create_graph=True)`, unlike
`subgrad+state`'s cheap proxy) made trainable at S-tier via two new
`FourDVarNetSolver` params from this session's investigation --
`prior_residual` (`Phi(x) = x + prior_unet(x, tau)`, a permanent identity
anchor) and `prior_dropout` (decouples `prior_unet`'s dropout from the main
solver unet's, avoiding extra mask noise in the `create_graph=True`
double-backward computation). `N5` uses a single `N_outer=5` block; `tbptt2x5`
restores the full `N_outer=10` depth split into `tbptt_n_blocks=2` truncated-
BPTT blocks, isolating the double-backward-chain-length effect from the
iteration-count effect. Neither is a new best (both worse than
`subgrad+state-Stier(monai)`/`FDV1-Stier(monai)` on every metric) --
included as a documented negative/informative result: the fixes eliminate
the outright divergence/NaN that killed every earlier non-`prior_residual`
`grad+state` attempt at this tier, and the fuller-unroll `tbptt2x5` variant
narrows the gap to the cheap proxy (~4-5% lower RMSE than `N5`), but neither
closes it (~10-15% worse pooled RMSE than `subgrad+state-Stier(monai)`).

**Files modified:** `reports/l96/outputs/l96_consolidated_benchmark.md` --
new rows in the methods table and all 6 numeric tables (pooled RMSE/EV/ES,
per-window RMSE/EV/CRPS), inserted right after `subgrad+state-Stier(monai)`
and before `FDV2(monai)` in each; extended the top-of-file explanatory note
with a new paragraph covering these two rows' identical direct-npz-metrics
provenance and their negative-result framing.

**Rationale:** Both numbers were already computed this session via
`eval_fdv1_l96.py` (full S0/S1 pass against
`experiments/FDV2_grad_state_monai_l96_Stier_priorresidual_{N5,tbptt2x5}/
checkpoints/stage1_best.ckpt`) and represent a genuinely useful completion
of the `prior_residual`/`prior_dropout`/Jacobian-decomposition investigation
narrative already documented on this branch -- the consolidated benchmark
is the natural place to record where the real-gradient solver lands
relative to `subgrad+state`'s proxy once the divergence bug is fixed.

**Verification:** Pooled RMSE/EV/ES recomputed directly via
`evaluation/estimate_metrics.py`'s `evaluate_estimates` against the stored
`estimates_{s0,s1}.npz` files, matched to 6 decimal places against the
numbers already reported earlier this session for both rows (N5:
0.428564/0.423037 S0/S1; tbptt2x5: 0.409801/0.407637 S0/S1) before trusting
`per_window_rmse_ev`/`per_window_deterministic_crps`'s per-window mean±std
numbers used in the other three tables; no code changed, markdown-only.
