## 2026-09-13: adopt gradient_clip_val=1.0 for Q3-noise0.05/Q5, add Q5 to cross-scenario eval

**Summary:** Confirmed via two ablations that Q3-noise0.05's epoch-19-20
training collapse was caused by the default `gradient_clip_val=10.0`, not
random seed ("bad luck"): a same-config rerun with a different explicit
seed collapsed too (just later, epoch 22-24, and stayed permanently
frozen), while a rerun with `--gradient-clip-val 1.0` trained cleanly
through all 200 epochs (S0 psi EV=0.911/q EV=0.189). Re-ran Q5 at the
tighter clip too for consistency (same `obs_noise_std_frac=0.05`); Q5 never
collapsed at either clip value and the two runs are within noise of each
other (psi EV 0.935 vs 0.934, q EV 0.372 vs 0.385). Adopted
`gradient_clip_val: 1.0` as both configs' standing default. Also added Q5
to the DA-matched cross-scenario eval: it substantially closes the PV-q gap
left by Q1/Q3/Q4 (0.376 vs their ~0.16-0.18), within reach of ETKF (0.405).

**Files modified:**
- `train_qg_neural.py` -- `--gradient-clip-val` now defaults to `None` and
  falls back to the experiment YAML's `training.gradient_clip_val` (then
  10.0), mirroring the existing `obs_noise_std_frac`/`init_lag_days`
  YAML-fallback pattern -- avoids the same "CLI hardcoded default silently
  overrides YAML" bug class already fixed once this session.
- `config/experiment/Q3_direct_unet_s0_oracle_cond_noise05.yaml`,
  `Q5_direct_unet_s1_noisy_ic_cond.yaml` -- added
  `training.gradient_clip_val: 1.0`.
- `batch/run_qg_q3_noise05_train.sbatch`, `run_qg_q5_train.sbatch` --
  comments updated to record the resolution; new
  `batch/run_qg_q5_gradclip1_train.sbatch` (the Q5 rerun).
- `eval_qg_neural_s0_s1.py` -- added a `"Q5"` scheme entry (`cond_mode=
  "scenario"`, `include_ic=True`, `ic_dim=2`) and a `"Q3-noise0.05"` entry,
  both pointing at their validated `_gradclip1` checkpoints; `_load_model`/
  `_eval_one` now thread `ic_dim`/`include_ic`.

**Rationale:** A validated, reproducible training recipe (not tied to a
lucky random seed) is required before this config becomes a standing
benchmark entry; gradient clipping is the actionable, low-risk fix with no
observed downside on the one scheme (Q5) that didn't even need it.

**Verification:** `pytest tests/test_qg_neural.py -m "not slow"` (fdv env)
-- 34 passed, 1 skipped. `ruff check train_qg_neural.py
eval_qg_neural_s0_s1.py` clean. `eval_qg_neural_s0_s1.py --schemes Q1 Q3 Q4
Q3-noise0.05 Q5 --lag-days 5.0 --noise-frac 0.05 --s1-param-bias 0.1
--s1-amp-bias 0.1` run end-to-end against the gradclip1 checkpoints.
