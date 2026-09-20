## 2026-09-13: L96 — new FDV2 update_input: gradsplit+state

**Summary:** New `FourDVarNetSolver`/`FourDVarNetPredictStateCFM` `update_input`
mode `"gradsplit+state"` -- the real-autograd counterpart to `"subgrad+state"`.
Same two-residual-plus-state input shape (`g_obs`, `g_prior`, `x`), but
`g_obs`/`g_prior` are each a TRUE `torch.autograd.grad` of their own cost
term (`obs_weight*obs_cost(x, obs)` / `prior_weight*prior_cost(x)`) taken
SEPARATELY, instead of `"grad-only"`/`"grad+state"`'s single combined
gradient of the summed `var_cost`, and instead of `"subgrad+state"`'s cheap
hand-written proxy residuals (`obs-x`, `x-Phi(x)`). Added to
`_AUTOGRAD_MODES` (needs the same requires_grad/re-leafing handling as
`grad-only`/`grad+state`) and `_PRIOR_MODES`. New experiment config
`FDV2_gradsplit_state_monai_l96_Stier.yaml` (S-tier, `init_state_var=0.1`,
`aux_var_cost_weight=0.01` -- the combination already found to help
`grad+state`/`subgrad+state` at this tier).
**Files modified:** `models/fourdvarnet.py` (`_IMPLEMENTED_UPDATE_INPUTS`,
`_AUTOGRAD_MODES`, `_PRIOR_MODES`, `_UPDATE_INPUT_CHANNEL_MULTIPLIER`,
`_build_update_input` new branch + docstring, class docstring);
`tests/test_fourdvarnet.py` (`_GRAD_MODES`/`_ALL_UPDATE_INPUT_MODES`
tuples, new `test_build_update_input_gradsplit_prior_channel_is_normalized`/
`test_build_update_input_gradsplit_obs_channel_is_not_normalized`,
extended `test_double_backward_create_graph_survives_checkpoint`'s mode
list); `tests/test_fourdvarnet_monai.py` (new
`TestMonaiBackboneForwardFDV2GradsplitState`); new
`config/experiment/FDV2_gradsplit_state_monai_l96_Stier.yaml`,
`batch/run_l96_fdv2_gradsplit_state_monai_Stier_train.sbatch`.
**Rationale:** User asked for a variant with `subgrad+state`'s two-channel
input structure (giving the UNet the obs term and the prior term as
independent signals, rather than one summed gradient) but using genuine
`torch.autograd.grad` for each term instead of `subgrad+state`'s cheap
proxy formulas. `g_prior` is normalized/soft-clipped like
`grad-only`/`grad+state`'s combined gradient (a real gradient of the dense,
unbounded `prior_cost` term); `g_obs` is deliberately NOT normalized, since
`obs_cost`'s gradient w.r.t. `x` is architecturally masked to exactly zero
outside observation times -- identically sparse to `subgrad+state`'s own
`g_obs` proxy, inheriting the same normalization-dilution vulnerability
(2026-09-11 fix) regardless of being a true gradient rather than a
hand-written one. Whether a channel needs normalization is decided by its
sparsity/boundedness, not by whether it came from autograd or a proxy
formula.
**Verification:** `pytest tests/test_fourdvarnet.py tests/test_fourdvarnet_monai.py`
-- 112 passed (108 pre-existing + 4 new). `ruff check` clean (pre-existing
E402 in `test_fourdvarnet_monai.py`'s `importorskip` pattern, unrelated).
Manual smoke tests: direct `FourDVarNetSolver(update_input="gradsplit+state")`
construction/forward/backward, and the full `model_factory`-built config
(confirmed `unet`/`prior_unet` param counts, `prior_weight` trainable,
gradients reach both submodules).
