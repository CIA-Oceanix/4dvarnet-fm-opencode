## 2026-09-15: Dedicated eval-pipeline support for subgrad+state+trueprior

**Summary:** Extended the shared `evaluation/neural_inference.py` inference
pipeline (`prepare_dataset`/`_run_case_inference`/`run_inference`,
consumed by `eval_neural_l96.py`) so `FourDVarNetSolver`'s
`"subgrad+state+trueprior"` mode gets scored with the exact same pooled
RMSE/EV convention as every other consolidated-benchmark row, restricted
to the same 24-dim observed subspace, rather than a separate ad-hoc
script.
**Files modified:**
- `evaluation/neural_inference.py` -- `prepare_dataset` gains
  `needs_true_forcing` (selects `collate_joint_eval`, a strict superset of
  `collate_eval` that also carries `forcing_true`/`params`/`true_params`);
  `_run_case_inference`/`run_inference` gain `trueprior_phi_source`
  (`"true"`/`"biased"`, split per S0/S1 case) which feeds the window's true
  or DA-biased forcing/params to the ODE prior at inference, and slice the
  model's full-40D prediction down to `model.obs_var_indices` (24D) right
  after `model.sample()` so the existing truth-subsampling logic engages
  exactly as it does for every other model -- all changes gated behind
  `model.obs_var_indices is not None`, `None` for every other config, so
  this is a provable no-op elsewhere.
- `eval_neural_l96.py` -- auto-detects the mode from the loaded model,
  cross-checks its `obs_var_indices` against `prepare_dataset`'s
  independently-derived one (asserts they match), and adds a
  `--trueprior-phi-source-s1 {true,biased}` CLI flag for the S0-true/
  S1-noisy stress-test variant (not yet run against any checkpoint).
- `tests/test_neural_inference.py` -- `TestTruepriorInference` (prediction/
  truth correctly sliced to `obs_var_indices`, true vs. biased Phi source
  gives different predictions, a no-`obs_var_indices` model is unaffected
  by the flag, `run_inference` applies the source per-case, unknown source
  raises).
**Rationale:** Needed to answer whether the true-ODE-prior scheme actually
matches/beats the DA baselines under an apples-to-apples metric -- the
generic pipeline's existing truth-subsampling only triggers when the
prediction is already smaller than the cached truth, which silently scored
this model's full 40D output (16 unobserved fast-Y dims included) instead
of the 24D observed subspace every other row uses.
**Verification:** `pytest tests/test_neural_inference.py -q -k Trueprior`
(5 passed); `pytest tests/test_neural_inference.py tests/test_sda_sampler.py
tests/test_config_persistence.py -q -m "not slow"` (all passed, `fdv-monai-proto`
env, no regressions); `ruff check evaluation/neural_inference.py
eval_neural_l96.py tests/test_neural_inference.py` clean (5 pre-existing
unrelated F541 findings at `evaluation/neural_inference.py:205-213`, outside
this diff, left untouched). End-to-end run against job 53564's checkpoint
(`experiments/FDV2_subgrad_state_monai_l96_trueprior_perfectmodel/checkpoints/stage1_best.ckpt`):
S0 RMSE 0.4590 (EV 0.9250), S1 RMSE 0.4563 (EV 0.9256) -- beats all three
DA baselines (ETKF 0.888/EnKF 0.913/Strong-4DVar 0.812 pooled S0 RMSE) but
behind every comparable learned-prior scheme (best: `subgrad+state-Stier(monai)`
at 0.373).
