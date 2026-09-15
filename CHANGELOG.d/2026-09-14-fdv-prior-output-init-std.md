## 2026-09-14: FourDVarNetSolver prior_output_init_std -- fixes a permanent dead zone under prior_residual

**Summary:** Added `prior_output_init_std` (default `0.0` -> MONAI's own
`zero_module()` init unchanged, backward-compatible) to `FourDVarNetSolver`
and `MonaiUNet1D`: overrides `prior_unet`'s final output conv's exact-zero
init with `N(0, prior_output_init_std)` instead.

**Files modified:**
- `models/monai_unet_adapter.py` -- `MonaiUNet1D.__init__` gains
  `output_init_std=0.0`; when `>0`, reinitializes
  `self.backbone.out[-1]`'s conv weight (bias left at 0) after construction.
- `models/fourdvarnet.py` -- `_build_backbone_unet` gains
  `monai_output_init_std=0.0`, forwarded only for `unet_backbone="monai"`
  (ignored for `unet1d`). `FourDVarNetSolver.__init__` gains
  `prior_output_init_std=0.0`, threaded ONLY into `self.prior_unet`'s
  construction (never `self.unet`'s).
- `conf/schema.py` -- `FourDVarNetConfig.prior_output_init_std: float = 0.0`.
- `train.py` -- `model_factory` threads `fdv.get("prior_output_init_std", 0.0)`.
- `tests/test_fourdvarnet_monai.py` -- new `TestPriorOutputInitStd` class (4
  tests): default preserves exact-zero init, nonzero value gives the
  expected empirical std and a genuinely nonzero `prior_unet` output,
  nonzero value gives `prior_cost` a real (nonzero) gradient at that layer,
  `unet1d` backbone (no `zero_module` convention) ignores the flag safely.

**Rationale:** A checkpoint diagnostic on job 53509
(`FDV2_grad_state_monai_l96_Stier_priorresidual_tbptt2x5`, 300+ epochs
trained under `prior_residual=true`) found `prior_unet`'s final output
conv's weight still bit-for-bit `0.0` -- i.e. `f(x)` (the raw correction
`prior_unet` outputs) identically zero for every input, the entire time.
Root cause, confirmed by direct derivation: under `prior_residual=true`,
`Phi(x) = x + f(x)`, so `prior_cost(x) = ||x - Phi(x)||^2` collapses
algebraically to `||f(x)||^2` -- a pure self-penalty on the network's own
output, with an exact critical point at `f(x)=0`. Since `f(x)` and its
Jacobian `J=d(f)/dx` both start at exactly `0` (MONAI's `zero_module()`),
EVERY gradient path that could move the final layer's weight away from `0`
is provably also `0` there: the aux `prior_cost` loss
(`d(prior_cost)/d(f(x)) = 2f(x) = 0`), the per-iteration
`g_prior = 2*J(x)^T@f(x)` signal (both its terms are proportional to `f(x)`
or to `J ∝ W_out`, both zero), and the deep double-backward path from the
outer supervised loss through `g_prior`'s own weight-dependence (same
vanishing argument, one level up the chain rule). A permanent, structural
fixed point -- not a slow-training issue, and not fixable by more epochs.
This does NOT affect `subgrad+state` (verified directly against
`subgrad+state-Stier(monai)`'s own checkpoint: final conv weight norm
5.30, clearly trained) -- that mode's default `prior_residual=false` means
`prior_cost` never collapses this way, and its own `g_prior=x-Phi(x)` proxy
is LINEAR in `Phi(x)` (constant slope `-1`), never structurally vanishing
regardless of `Phi(x)`'s magnitude.

`prior_output_init_std=0.1` breaks the fixed point: `f(x)` is then
small-but-nonzero from step 1, so `d(prior_cost)/d(f(x)) = 2f(x)` is
genuinely nonzero -- the same mechanism that makes `zero_module`'s
zero-init safe for an ordinary, externally-targeted diffusion loss (whose
target never algebraically cancels with the network's own prediction).

New configs test this fix on both `grad+state` (where the collapse was
found) and `subgrad+state` (an ablation -- that mode was never broken, so
this tests whether `prior_residual`'s Jacobian-anchor benefit helps
regardless):
`config/experiment/FDV2_grad_state_monai_l96_Stier_priorresidual_nonzeroinit.yaml`,
`config/experiment/FDV2_subgrad_state_monai_l96_Stier_priorresidual_nonzeroinit.yaml`,
and their matching `batch/run_l96_fdv2_*_priorresidual_nonzeroinit_train.sbatch`
launchers.

**Verification:** `pytest tests/test_fourdvarnet_monai.py -q` (29 passed,
monai env) and `pytest tests/test_fourdvarnet.py tests/test_lightning_module.py -q`
(113 passed, default env) -- both clean.
