## 2026-09-12: L96 — build prior_unet whenever aux_var_cost_weight>0, not only for _PRIOR_MODES

**Summary:** `FourDVarNetSolver.__init__` now builds `self.prior_unet` when
`update_input in _PRIOR_MODES OR aux_var_cost_weight > 0` (was: only the
former). This lets `update_input="obs+state"` (FDV1's own configuration)
train with FDV2's auxiliary `prior_cost(x_final)+prior_cost(states)` loss
term while `_build_update_input`'s tensor construction for that mode stays
completely untouched (`cat([x, obs_clean])` only -- `prior_unet` is never
referenced there). New experiment config
`config/experiment/FDV1_obsstate_monai_l96_auxpriorcost.yaml` (+ matching
sbatch script) runs this ablation.
**Files modified:** `models/fourdvarnet.py` (`FourDVarNetSolver.__init__`,
`compute_loss` docstring); `tests/test_fourdvarnet.py` (renamed
`test_aux_var_cost_weight_skipped_without_prior_unet` ->
`test_aux_var_cost_weight_skipped_when_zero` since it's no longer true that
`obs+state` always skips the aux term; added
`test_obs_state_builds_prior_unet_when_aux_weight_positive` and
`test_obs_state_update_input_unaffected_by_aux_weight`); new
`config/experiment/FDV1_obsstate_monai_l96_auxpriorcost.yaml`,
`batch/run_l96_fdv1_obsstate_monai_auxpriorcost_train.sbatch`.
**Rationale:** None of the three FDV2 retrains under the two 2026-09-11
fixes (normalization dilution, prior_weight/obs_cost loss bug) resolved
their fast-Y reconstruction collapse (fast-Y variance ratio 0.48-0.53 vs
FDV1's healthy 0.926) -- both killed once this was confirmed by eval,
grad+state at epoch ~200+/400 (prior_weight had stabilized ~0.5-0.6, not
collapsing to 0 anymore, but val_loss still far from FDV1's), subgrad
equal-capacity at epoch ~370+/400 (val_loss ~0.91, still far from FDV1's
0.205). Measuring the aux loss term's actual contribution to those three
runs' total loss (0.2%-9.5% across configs) showed no correlation with
collapse severity, weakening (not yet ruling out) the hypothesis that the
prior-consistency term itself is what suppresses fast-Y content. This
ablation tests that hypothesis directly on FDV1's own architecture, with
the update-input construction difference removed from the comparison
entirely.
**Verification:** `pytest tests/test_fourdvarnet.py tests/test_fourdvarnet_monai.py`
-- 98 passed. `ruff check` clean on the touched .py files. Manual smoke
test: built the model from the new config via `model_factory`, confirmed
`prior_unet is not None`, ran one `compute_loss`+`backward()` step, confirmed
gradients reach `prior_unet`'s parameters.
