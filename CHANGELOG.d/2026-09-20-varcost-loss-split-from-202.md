## 2026-09-20: Split `loss_type="var_cost"` out of PR #202 — the split is complete

**Summary:** Fourth and final extraction from PR #202, after `L96Weak4DVar`
(#229), the gradsplit/prior-knob layer (#230) and the true-ODE prior (#231).
Adds `loss_type="var_cost"` to `FourDVarNetSolver`: a genuinely
**self-supervised, weak-constraint-4D-Var training objective**
(`obs_cost/R_var + prior_cost/var_cost_Q_var`, evaluated on the solver's own
output) that replaces the default MSE-against-ground-truth, with **no direct
ground-truth supervision at all**.

Allowed only for the `_FULL_STATE_UPDATE_INPUTS` modes, which is why it had to
follow #231: its `prior_cost` is the same `_true_ode_prior_residual` that mode's
own `g_prior` channel uses.

**With this, all 22 commits of #202 are accounted for and that PR can be
closed.** It had grown to 64 files / +5161-83 across four independent lines of
work on a branch that was `DIRTY` and had never once run CI; it is now four
merged PRs, each with a green pytest gate.

**Files modified:** `models/fourdvarnet.py` (module-level
`_var_cost_training_loss`; `loss_type`/`var_cost_Q_var` constructor args,
validated to require a full-state mode; `compute_loss` branches on
`self.loss_type`, applying var_cost per tbptt block in training and to
`x_final` only at eval, preserving the existing deep-supervision structure);
`train.py` (threads both from `cfg.model.fdv`);
`training/lightning_module.py` (logs `train_mse_proxy`);
`tests/test_fourdvarnet.py` (`TestVarCostTrainingLoss`); one experiment config
+ batch script.

**Why `train_mse_proxy` exists.** Under `var_cost` the val_loss is no longer
comparable to any MSE-trained run, so `compute_loss` also stashes the supervised
MSE it did **not** train on, purely for logging. That makes it possible to watch,
epoch by epoch, how well the self-supervised objective actually being optimized
tracks the true MSE criterion. The `LitModel` hook is guarded by
`getattr(self.model, "_last_train_mse_proxy", None) is not None`, so it is a
no-op for every other model and case study.

**Why this matters beyond the code.** This is the arm that `R4` of
`docs/scoping/da_paper_structural_hypotheses.md` asks for. R4 is the circularity
objection — an amortized estimator needs a simulator to generate training data,
and if that simulator is the truth model the comparison has handed it the truth
— and the doc currently concedes it is "not fully answerable". A solver trained
with no ground truth at all is a direct, if partial, answer. The test suite
pins the property that makes the claim checkable: corrupting `batch.states`
with NaN provably does not change the loss.

**One merge conflict, resolved as a union.** `FourDVarNetSolver.__init__`'s
signature — `b3ae19d` predates `7fff006`, so its context lacked
`true_dynamics_coupling_exponent`, which #231 has since put on master. Both kept.
This is the exact mirror of the conflict #231 resolved in the other direction.

**Not run.** No training run is included. The one config
(`FDV2_subgrad_trueprior_monai_l96_varcost.yaml`) shares #231's architecture and
differs only in the loss, so like every trueprior config it is subject to the
`coupling_exponent` caveat recorded there: **jobs 53564/53595/53681 predate that
fix and are not measurements of the literally-true ODE prior.**

**Verification:** `pytest tests/test_fourdvarnet.py tests/test_fourdvarnet_monai.py
tests/test_training.py tests/test_lorenz96_training.py
tests/test_neural_inference.py -q -m "not slow"` (fdv-monai-proto) — **263
passed, 1 deselected**. Lint reproduced with CI's exact command
(`git diff --name-only --diff-filter=ACMR origin/master HEAD -- '*.py' | xargs
ruff check`) — **All checks passed**. No separate QG regression run this time:
unlike #230's `qg_*` union and #231's observation-mask resolution, nothing here
touches shared case-study code paths — the one shared-file change is the guarded
no-op log above — and CI's full-tree pytest covers it.
