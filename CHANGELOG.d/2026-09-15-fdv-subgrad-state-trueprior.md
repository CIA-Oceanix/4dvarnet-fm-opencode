## 2026-09-15: FourDVarNetSolver subgrad+state+trueprior (true-ODE prior, full physical state)

**Summary:** New `update_input="subgrad+state+trueprior"` mode for
`FourDVarNetSolver` replaces the learned `prior_unet` with the actual known
L96 ODE (`Lorenz96Dynamics`) as the prior term, to test whether a "true"
physical prior can match/beat DA baselines under the same unrolled-solver
framework. The prior residual uses a time-shifted, dynamical-consistency
formula `x[t+1] - Phi_ode(x[t])` (weak-constraint-4DVar style), not the
pointwise `x - Phi(x)` used by learned priors, since the true ODE is a
genuine evolution operator rather than an autoencoder-like map. Because
`Lorenz96Dynamics` needs the full 40-dim physical state (not the 24-dim
observed subspace `FourDVarNetSolver` normally works in), the solver's
`state_dim` is expanded to the full state for this mode, with a new
`obs_var_indices`/`_embed_obs_to_full_state` bridge that scatters the
sparse observed channels into the full-dim working space (NaN + a genuinely
per-channel mask, AND-ed with the existing per-time mask) at unobserved
channel positions.
**Files modified:**
- `models/fourdvarnet.py` — `_true_ode_prior_residual`, `_embed_obs_to_full_state`,
  `_FULL_STATE_UPDATE_INPUTS`, new `"subgrad+state+trueprior"` branch in
  `_build_update_input`, `FourDVarNetSolver.__init__`/`_unrolled_blocks` wiring
  (`obs_var_indices`, `true_dynamics_dt`, `true_dynamics_NO/J/h` ctor args;
  `self.true_dynamics = Lorenz96Dynamics(...)`).
- `data/dataloader.py` — `FlowMatchingBatch`/`FlowMatchingDataset` gain
  `true_forcing`/`with_true_forcing` (the true, uncorrupted per-window forcing
  needed by the ODE prior); `collate_fm` refactored into `_collate_fm_impl`
  with an explicit `with_true_forcing` flag (producer/consumer must agree
  explicitly, since `with_params`'s variable-length suffix already makes
  pure tuple-length inference ambiguous); `collate_fm`/`make_collate_fm`
  stay backward compatible by default.
- `train.py` — `make_l96_dataloaders` threads `with_true_forcing`; the
  `fourdvarnet` `model_factory` branch derives `obs_var_indices`/
  `true_dynamics_dt` from `cfg.data` when `fdv.full_state_target` is set.
- `conf/schema.py` — `FourDVarNetConfig.full_state_target: bool = False`.
- `tests/test_fourdvarnet.py` — `TestTrueOdePriorResidual` (shape/zero-pad at
  t=0, matches a per-timestep reference loop, gradient never detached),
  `TestEmbedObsToFullState` (scatter + per-channel/temporal mask AND),
  `TestFourDVarNetSolverTrueprior` (ctor validation, channel multiplier,
  forward shape/finiteness, gradients reach `unet`, `Lorenz96Dynamics`
  contributes zero trainable parameters, checkpoint/no-checkpoint parity).
**Rationale:** User wants to assess whether a true physical prior (not a
learned one) can reach DA-baseline performance under the 4DVarNet-FM
unrolled-solver framework, as a scientific control on how much benefit comes
from a *learned* differentiable prior vs. the differentiable-prior mechanism
itself. Starting with the "perfect-model" training variant (Phi always reads
true params/forcing) for the S0 (true params/forcing at eval) config first,
per explicit user instruction.
**Verification:** `pytest tests/test_fourdvarnet.py -q` (124 passed, full
regression, `fdv` env); `pytest tests/test_fourdvarnet_monai.py -q -m "not
slow"` (29 passed, `fdv-monai-proto` env); `pytest tests/test_lightning_module.py
tests/test_l96_normalization.py tests/test_training.py
tests/test_lorenz96_training.py tests/test_param_head.py -q -m "not slow"`
(all passed/skipped as expected, `fdv` env); `ruff check
models/fourdvarnet.py conf/schema.py train.py tests/test_fourdvarnet.py
data/dataloader.py` clean.
