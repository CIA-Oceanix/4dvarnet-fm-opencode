## 2026-09-15: FourDVarNetSolver loss_type="var_cost" -- weak-constraint-4DVar self-supervised training

**Summary:** New `loss_type="var_cost"` for `FourDVarNetSolver` (only
allowed for the `_FULL_STATE_UPDATE_INPUTS` modes, i.e.
`subgrad+state+trueprior`/`subgrad+trueprior`): replaces the default
MSE-against-ground-truth training objective with a genuinely
self-supervised, weak-constraint-4DVar-style variational cost --
`obs_cost/R_var + prior_cost/var_cost_Q_var`, evaluated on the solver's own
output, with no direct ground-truth supervision at all.
**Files modified:**
- `models/fourdvarnet.py` -- new module-level `_var_cost_training_loss`
  (`obs_cost` via the existing `_masked_obs_cost`; `prior_cost` via the
  existing `_true_ode_prior_residual`, i.e. the SAME dynamical-consistency
  residual already used as the update-input's own `g_prior` channel);
  `FourDVarNetSolver.__init__` gains `loss_type` ("mse" default,
  backward-compatible, or "var_cost") and `var_cost_Q_var` (default 0.05),
  validated to require a `_FULL_STATE_UPDATE_INPUTS` mode; `compute_loss`
  branches on `self.loss_type`, applying the var_cost function (instead of
  `F.mse_loss`) to every tbptt block during training and to `x_final` only
  at eval, preserving the existing deep-supervision/eval-only structure.
- `train.py` -- `model_factory`'s `fourdvarnet` branch threads
  `loss_type`/`var_cost_Q_var` from `cfg.model.fdv`.
- `tests/test_fourdvarnet.py` -- `TestVarCostTrainingLoss` (rejected for
  non-full-state modes, unknown `loss_type` raises, the helper matches a
  hand-computed value, `compute_loss` is finite with gradients reaching
  `unet`, the var_cost path provably ignores `batch.states` -- corrupting
  it with NaN doesn't change the loss -- while the default "mse" mode is
  unaffected and still depends on `batch.states`).
- `config/experiment/FDV2_subgrad_trueprior_monai_l96_varcost.yaml`,
  `batch/run_l96_fdv2_subgrad_trueprior_monai_varcost_train.sbatch` -- same
  architecture as the sibling `subgrad+trueprior` "perfect-model" config
  (job 53595), only the loss differs.
**Rationale:** User asked for a `subgrad+trueprior` variant trained on a
4DVar cost (obs + prior terms) instead of direct MSE supervision, and to
derive the weighting from the existing Weak-4DVar baseline
(`evaluation/baselines.py::Weak4DVar`) rather than guessing a fresh ratio.
Weak4DVar's own cost is `J_total = 0.5*J_b + 0.5*J_o + 0.5*J_q` with
`J_o = sum(((H(x)-obs)*mask)^2)/R_var` and `J_q = sum(q^2)/Q_var` where
`q[t] = x[t] - dynamics.step(x[t-1])` -- exactly the same quantity this
model's own `g_prior`/`prior_cost` measures, just accumulated over the
whole window here instead of optimized as a free per-step control
variable. Reused its exact class defaults, `R_var=0.5`/`Q_var=0.05` (also
matching `data.R_var=0.5`/`data.B_var=2.0`, already this project's standing
convention) -- a 10x tighter trust in the known dynamics than in the raw
observations per unit squared error, rather than inventing an unjustified
new ratio. No `B_var`-analog background term was added: unlike
Weak4DVar's free per-window `x0` control variable, this solver has no
comparable background state to anchor against.
**Verification:** `pytest tests/test_fourdvarnet.py -q -k VarCost` (12
selected/passed); `pytest tests/test_fourdvarnet.py -q` (full regression,
135 passed, `fdv-monai-proto` env); `ruff check models/fourdvarnet.py
tests/test_fourdvarnet.py train.py` clean; end-to-end smoke test (20
train/10 val/10 test windows, 1 epoch) ran cleanly through training (loss
computed via the new var_cost path, finite: train_loss=3.83, val_loss=3.49
-- a different scale than the MSE loss's, as expected) and eval (unaffected
by `loss_type`, which only changes `compute_loss`, not `forward`/`sample`).
