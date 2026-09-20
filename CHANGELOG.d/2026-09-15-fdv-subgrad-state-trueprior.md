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
  `_make_eval_batch`/`evaluate_model`/`save_trajectories` gain a
  `with_true_forcing` flag so the end-of-training eval/trajectory-dump path
  also populates `true_forcing`/`true_params` on the eval batch (previously
  only the training dataloader did) -- without it, `model.sample(batch)`
  crashed with `subgrad+state+trueprior`'s own assertion
  (`prior_ode_forcing/prior_ode_params/true_dynamics` all required) the
  moment training finished, since `_make_eval_batch`'s old `param_dim==0`
  short-circuit returned a batch with no params/forcing at all. Caught by a
  1-epoch/20-window smoke test of the new config, not by the unit tests
  (which exercise `_unrolled_blocks` directly with a hand-built batch, never
  through this separate eval-time code path).
- `config/experiment/FDV2_subgrad_state_monai_l96_trueprior_perfectmodel.yaml`,
  `batch/run_l96_fdv2_subgrad_state_monai_trueprior_perfectmodel_train.sbatch`
  — the "perfect-model" S0 training config/launcher (state_dim=40,
  S-tier MonaiUNet1D, 400 epochs, Phi always reads
  `batch.true_forcing`/`batch.true_params`).
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
tests/test_lorenz96_training.py tests/test_param_head.py
tests/test_joint_estimation_l96_neural.py -q -m "not slow"` (all
passed/skipped as expected, `fdv` env); `ruff check models/fourdvarnet.py
conf/schema.py train.py tests/test_fourdvarnet.py data/dataloader.py`
clean; end-to-end smoke test of the new experiment config (20 train / 10
val / 10 test windows, 1 epoch, `fdv-monai-proto` env) -- first attempt
crashed at eval with the `_make_eval_batch` bug above, second attempt
(after the fix) ran stage1 training through to S0/S1 eval and wrote finite
per-channel RMSE for both.
