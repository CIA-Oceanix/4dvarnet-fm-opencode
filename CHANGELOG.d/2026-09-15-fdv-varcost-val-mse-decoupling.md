## 2026-09-15: var_cost loss -- decouple val_loss from training objective, log train_mse_proxy

**Summary:** Correction to `loss_type="var_cost"` (2026-09-15, same day):
validation/eval now ALWAYS uses the supervised `F.mse_loss(x_final, states)`
criterion, regardless of `loss_type` -- previously eval also used the
var_cost, making `val_loss`/`stage1_best.ckpt` selection incomparable in
scale/meaning to every other (MSE-trained) config. Training also now
stashes the supervised MSE it deliberately does NOT train on, logged as a
new `train_mse_proxy` metric alongside the real `train_loss` (var_cost),
so training curves show directly how well the self-supervised var_cost
proxy tracks the true MSE objective epoch-by-epoch.
**Files modified:**
- `models/fourdvarnet.py` -- `compute_loss`: the `loss_type=="var_cost"`
  branch now only fires when `self.training` (validation/eval falls
  through to the ordinary `F.mse_loss(x_final, states)` unconditionally);
  when it fires, also sets `self._last_train_mse_proxy =
  F.mse_loss(x_final, batch.states).detach()` (a pure monitoring value,
  never entering the backward graph). `__init__` initializes
  `self._last_train_mse_proxy = None`.
- `training/lightning_module.py` -- `training_step` logs `train_mse_proxy`
  when `getattr(self.model, "_last_train_mse_proxy", None) is not None`,
  mirroring the existing `obs_weight`/`prior_weight` optional-logging
  pattern.
- `tests/test_fourdvarnet.py` -- rewrote `TestVarCostTrainingLoss`'s
  affected tests: `test_training_loss_ignores_batch_states` (moved to
  TRAINING mode, since eval no longer uses var_cost at all),
  `test_eval_loss_is_always_mse_regardless_of_loss_type` (new: eval-mode
  loss matches `F.mse_loss` exactly and DOES change when states are
  corrupted, unlike training mode), `test_train_mse_proxy_stashed_and_matches_mse`
  (new), `test_mse_mode_never_sets_train_mse_proxy` (new).
- `config/experiment/FDV2_subgrad_trueprior_monai_l96_varcost.yaml` --
  header note documenting the correction.
**Rationale:** User asked to keep val_loss on the same MSE-based
criterion as every other (MSE-trained) config, so checkpoint selection and
cross-config comparison stay meaningful, and to log the MSE alongside the
var_cost training loss specifically to assess how good a proxy the
self-supervised objective is for the true supervised one. The first
attempt at this job (job 53629, killed before completing) used var_cost
for both training AND validation, which this corrects.
**Verification (a genuinely NEW, real bug caught while writing tests, unrelated to this feature):**
found that `models/unet.py`'s `UNet1D.__init__` never passes its own
`dropout` constructor argument to the `Down`/`Up` blocks it builds (only
`self.bottleneck` receives it) -- so `dropout` has silently never
controlled those blocks' actual dropout rate, project-wide, only
`ConvBlock`'s own hardcoded default. Confirmed empirically: `dropout=0.0`
did not make two consecutive forward passes deterministic in train mode;
inspecting `named_modules()` showed `unet.downs.*.drop`/`unet.ups.*.drop`
at `p=0.1` regardless of the constructor's `dropout` value, while
`unet.bottleneck.drop` correctly tracked it. NOT fixed here (out of scope
for this task) -- flagged to the user; new tests work around it by
re-seeding before each forward call instead of relying on `dropout=0.0`.
`pytest tests/test_fourdvarnet.py tests/test_lightning_module.py -q` (153
passed, `fdv-monai-proto` env); `ruff check models/fourdvarnet.py
tests/test_fourdvarnet.py training/lightning_module.py train.py` clean.
