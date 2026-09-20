## 2026-09-20: Split the true-ODE prior layer out of PR #202

**Summary:** Third and final code extraction from PR #202, after `L96Weak4DVar`
(#229) and the gradsplit/prior-knob layer (#230). This lands the **true-ODE
prior**: `update_input="subgrad+state+trueprior"` and `"subgrad+trueprior"`,
which replace the learned `prior_unet` with the actual known L96 ODE
(`Lorenz96Dynamics`) as the solver's prior term. Only `loss_type="var_cost"`
now remains unsplit.

**Files modified:** `models/fourdvarnet.py` (`_true_ode_prior_residual`,
`_embed_obs_to_full_state`, `_FULL_STATE_UPDATE_INPUTS`, the two new
`update_input` branches, and the `obs_var_indices`/`true_dynamics_*`
constructor wiring); `data/dataloader.py` (`true_forcing`/`with_true_forcing`
on `FlowMatchingBatch`/`FlowMatchingDataset`, `collate_fm` refactored into
`_collate_fm_impl` behind an explicit flag); `train.py`,
`evaluation/neural_inference.py`, `eval_neural_l96.py`, `conf/schema.py`;
two experiment configs + batch scripts; `tests/test_fourdvarnet.py`,
`tests/test_neural_inference.py`.

**Why the prior residual differs from every other mode.** The learned-prior
modes use the pointwise autoencoder residual `x - Phi(x)`. The true ODE is a
genuine *evolution* operator, not an autoencoder-like map, so this uses the
time-shifted dynamical-consistency residual `x[t+1] - Phi_ode(x[t])` --
weak-constraint-4D-Var's own form. `Lorenz96Dynamics` also needs the full 40-dim
physical state rather than the 24-dim observed subspace the solver normally
works in, which is what `_embed_obs_to_full_state` and the
`_FULL_STATE_UPDATE_INPUTS` machinery exist for.

**Includes the `coupling_exponent` correctness fix (`7fff006`).**
`self.true_dynamics = Lorenz96Dynamics(...)` never passed `coupling_exponent`,
silently taking that class's generic default of `1.0`, while every ground-truth
trajectory is generated at `coupling_exponent_truth=1.6`. So the "true" ODE
prior in **every trueprior run to date (jobs 53564, 53595, 53681) was not the
dynamics that generated the data.** The constructor now defaults to `1.6`.
**Consequence for the science: job 53564's S0 RMSE 0.459 stands as a valid
measurement of what was run, but must not be cited as a measurement of the
literally-true-ODE prior.** That re-run has not been done.

**Two merge conflicts resolved, one of them semantic and worth recording:**

1. `models/fourdvarnet.py::forward`'s observation-mask handling. The incoming
   commit unconditionally does `batch.obs_mask.unsqueeze(-1)`, because when it
   was written L96's `(B, T)` per-timestep mask was the only case. Master has
   since gained QG's genuine per-cell `(B, T, D)` mask and a *conditional*
   unsqueeze to match. Taking the incoming version verbatim would have made
   QG's 3-D mask 4-D and silently treated every unobserved cell's zero-fill as
   a real `obs=0` measurement. Resolved as a union: master's conditional
   unsqueeze is kept and the trueprior full-state embedding branches off it.
   `pytest -k qg` was run specifically to cover this.
2. `FourDVarNetSolver.__init__`'s signature: `7fff006` was authored after the
   `var_cost` work, so its context carried `loss_type`/`var_cost_Q_var`. Only
   `true_dynamics_coupling_exponent=1.6` was taken; verified zero `loss_type` /
   `var_cost_Q_var` / `_var_cost_training_loss` references remain.

**One lint fix.** `evaluation/neural_inference.py:207-211` carried five
pre-existing `F541`s (a five-fragment implicit-concatenation string where every
fragment is `f`-prefixed but none has a placeholder). They were invisible until
now because CI lints changed files only and this layer is the first to touch
that file. Dropping the `f` prefixes is behaviour-identical.

**Not run.** No training run is included. Both configs are the "perfect-model"
S0 recipe; the results that exist on disk (jobs 53564/53595/53681) predate the
`coupling_exponent` fix above.

**Verification:** `pytest tests/test_fourdvarnet.py tests/test_fourdvarnet_monai.py
tests/test_neural_inference.py tests/test_training.py
tests/test_l96_normalization.py tests/test_lorenz96_training.py
tests/test_param_head.py -q -m "not slow"` (fdv-monai-proto) -- **282 passed, 1
deselected** (the last four cover `data/dataloader.py`'s consumers, since
`collate_fm` was refactored; there is no `tests/test_dataloader.py`).
`pytest tests/ -q -m "not slow" -k qg` -- **241 passed, 749 deselected**, run
specifically to cover the observation-mask conflict resolution above. Lint
reproduced with CI's exact command (`git diff --name-only --diff-filter=ACMR
origin/master HEAD -- '*.py' | xargs ruff check`) -- **All checks passed**.
