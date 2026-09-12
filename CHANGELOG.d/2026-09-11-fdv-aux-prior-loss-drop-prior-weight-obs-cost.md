## 2026-09-11: L96 — FDV2 auxiliary loss must not let prior_weight cheat the loss

**Summary:** `FourDVarNetSolver.compute_loss`'s auxiliary term (added when
`prior_unet` exists and `aux_var_cost_weight>0`) previously computed the
*full* per-iteration `var_cost` formula outer-loop-side:
`prior_weight*prior_cost(z) + obs_cost(z)` for `z in {x_final, states}`.
Changed to a pure prior-consistency term:
`prior_cost(x_final) + prior_cost(states)`, i.e.
`||x_final - prior_unet(x_final)||^2 + ||states - prior_unet(states)||^2`,
with no `prior_weight` multiplier and no `obs_cost` term.
**Files modified:** `models/fourdvarnet.py` (`FourDVarNetSolver.compute_loss`);
`tests/test_fourdvarnet.py` (new
`test_aux_var_cost_weight_is_pure_prior_cost_no_obs_or_prior_weight`
regression test, exact-equality reconstruction of the corrected formula).
**Rationale:** A *trainable* `prior_weight` inside this auxiliary term could
shrink the loss simply by driving itself to 0 -- the `obs_cost` half is
computed directly on `x_final`/`states` and isn't gated by `prior_weight`,
so zeroing `prior_weight` erases the entire prior-consistency term for free,
with no actual improvement in prior quality required. Confirmed empirically
on the live `grad+state` retrain (job 53104, soft-clip fix): `prior_weight`
collapsed from ~0.97 at epoch 0 to exactly 0.0 by epoch ~120
(0.766@15 -> 0.146@30 -> 0.0033@60 -> 2.7e-10@90 -> 0.0@120+), with
`train_loss` degrading (1.23 -> 1.54) past that point rather than continuing
to improve -- i.e. the model was actively getting worse once the free-rider
optimum was found. User identified the fix directly: the auxiliary term
should be pure `||x-prior_unet(x)||^2`, and `prior_weight` "shouldn't be
involved in the training loss. That's likely why it goes to 0." `prior_weight`
retains its only legitimate role inside the per-iteration solver update
(`_build_update_input`/`_solver_iteration`'s own `var_cost`, used to compute
the actual gradient step during the unroll) -- unchanged.
**Verification:** `pytest tests/test_fourdvarnet.py tests/test_fourdvarnet_monai.py
tests/test_lightning_module.py tests/test_l96_normalization_configs.py` --
155 passed (154 pre-existing + 1 new). `ruff check` clean. This changes the
training objective for every FDV2 config with `aux_var_cost_weight>0`
(`grad+state`, and both `subgrad+state` variants, since `prior_unet` exists
there too, even though `subgrad+state` never trains `prior_weight` itself --
its `obs_cost` half of the old formula was still present and is now
removed). All three currently-considered FDV2 jobs (grad+state job 53104,
still running; subgrad+state equal-capacity job 53077, killed before this
fix; subgrad+state priorS+tbptt job 53078, completed 400 epochs but
evaluated at still-collapsed fast-Y variance ratio 0.53) were trained under
the OLD buggy formula and should be considered non-final pending a retrain
under this fix.
