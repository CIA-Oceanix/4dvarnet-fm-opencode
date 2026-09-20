## 2026-09-14: FourDVarNetSolver prior_residual diagnostic knob

**Summary:** Added `prior_residual` (default `False`, backward-compatible) to
`FourDVarNetSolver`: when `True`, the trainable prior operator becomes
`Phi(x) = x + prior_unet(x, tau)` instead of the bare backbone output,
threaded through `_prior_ae`/`_prior_cost`/`_build_update_input`/
`_solver_iteration` and both call sites in `compute_loss`'s auxiliary
prior-consistency term.

**Files modified:**
- `models/fourdvarnet.py` — `_prior_ae`/`_prior_cost` gain `residual=False`;
  `_build_update_input`/`_solver_iteration`/`FourDVarNetSolver.__init__` gain
  `prior_residual=False`, forwarded everywhere `_prior_ae`/`_prior_cost` are
  called. `FourDVarNetPredictStateCFM` deliberately untouched (its own
  `_solver_iteration` call omits the new trailing arg, staying at the
  `False` default).
- `conf/schema.py` — `FourDVarNetConfig.prior_residual: bool = False`.
- `train.py` — `model_factory` threads `fdv.get("prior_residual", False)`.
- `tests/test_fourdvarnet.py` — new `TestPriorResidual` class (7 tests):
  `_prior_ae`/`_prior_cost` residual math, `_build_update_input` threading
  (subgrad+state's `g_prior` sign flips from `x - raw` to `-raw`), default
  omitted reproduces existing behavior exactly, forward+backward+aux-loss
  integration.
- `config/experiment/FDV2_gradsplit_state_monai_l96_Stier_priorresidual.yaml`,
  `batch/run_l96_fdv2_gradsplit_state_monai_Stier_priorresidual_train.sbatch`
  — new diagnostic training config/launcher (S-tier gradsplit+state +
  `prior_residual: true`, otherwise identical to the existing
  `FDV2_gradsplit_state_monai_l96_Stier` config).

**Rationale:** A Jacobian decomposition of the plateaued
`FDV2_gradsplit_state_monai_l96_Stier` job (53447)'s trained `prior_unet`
found the real `prior_cost` gradient's Jacobian term (`-2*J^T@r`,
`J=d(Phi)/dx`) dominating the useful residual term (`2*r`, subgrad+state's
own proxy direction) by 11-53x in RMS across three `x` regimes, and nearly
orthogonal to it (cos~0.04-0.13). Checked directly against the installed
`monai==1.6.0` wheel: MonaiUNet1D's backbone already uses the same
`zero_module()` zero-init convention as OpenAI's guided-diffusion UNet
(ResnetBlock's final conv, the whole network's output conv) — so the gap
isn't missing zero-init. But this project's `Phi(x) = prior_unet(x, tau)`
(`_prior_ae`, unchanged) uses the raw backbone output with no outer
residual/skip wrapper: zero-init only anchors `Phi` near identity AT
INITIALIZATION (the output layer's weight literally being zero), with
nothing architecturally preventing its Jacobian from drifting arbitrarily
far from identity once training moves those weights away from zero — unlike
a residual/denoising-style prior wrapper (`Phi(x) = x - CNN(x)`, the likely
convention behind `ocean4dvarnet`/`4dvarnet-global-mapping`'s `ronan_devs`
branch, which trains a real-gradient 4DVarNet solver successfully with an
OpenAI guided-diffusion UNet backbone), whose identity anchor holds
permanently, by construction, regardless of the inner network's weights.
`prior_residual=True` bakes that same permanent anchor into this project's
MonaiUNet1D-backed prior, as a diagnostic to test whether it resolves the
gradsplit+state training plateau.

**Verification:** `pytest tests/test_fourdvarnet.py -q` — new
`TestPriorResidual` tests pass alongside the full existing suite.
