## 2026-09-17: L96Weak4DVar -- port QG4DVar's whitened-control + gradient-safety design to L96

**Summary:** New `L96Weak4DVar` class in `evaluation/baselines.py`, mirroring
`evaluation/run_qg_baselines.py::QG4DVar`'s design -- deliberately not a
modification of this module's own `Weak4DVar`/`Strong4DVar` (used by QG's
non-QG4DVar paths and other case studies; keeping this a new, separate class
avoids destabilizing shared code).
**Files modified:**
- `evaluation/baselines.py` -- `L96Weak4DVar`: whitened background control
  (`x0 = xb + L*w`, `L = b_var_scale*sigma`, `sigma = xb.std()`, cost
  `0.5*sum(w^2)`); whitened weak-mode model-error control (`q_t =
  dynamics.step(q_{t-1}) + Lq*u_t`, `Lq = q_var_scale*sigma`, cost
  `0.5*sum(u[1:]^2)`); Adam path sanitizes (`nan_to_num`) then clamps
  (`grad_clip`) every gradient; both Adam and LBFGS paths reset the
  controls to zero (falling back to the pure background/dynamics forecast)
  the instant the loss or any parameter goes non-finite, instead of letting
  a NaN cascade through every later window via `current_bg`. Uses this
  module's own `Lorenz96Dynamics`/`ObsOperator`/`_init_bg_from_obs`/
  `_interp_observations`/`_safe_ref` (unchanged) rather than QG's
  spectral-inversion machinery. `mode="strong"` (no free `q`) is also
  supported for parity/comparison.
- `tests/test_baselines_l96weak4dvar.py` -- 12 new tests: init defaults,
  `_forward_strong`/`_forward_weak` shape+finiteness (and that `u=0` weak
  mode reduces exactly to the strong-mode forecast), `_reset_nan` zeroes
  params, Adam/LBFGS both reset on a NaN loss, Adam sanitizes+clips a
  deliberately huge/NaN gradient, `assimilate` in both modes (shape,
  finiteness, `rmse` shape), and a deliberately adversarial-observation +
  aggressive-lr stress test confirming the safety net keeps the trajectory
  finite where the plain `Weak4DVar` class would diverge.
**Rationale:** Multi-day investigation (2026-09-15/16/17) found this
module's plain `Weak4DVar` diverges catastrophically on L96's two-scale
chaotic dynamics as soon as a free per-step model-error control is
introduced over a real (`da_window_steps=500`) window -- Adam
(`opt_steps=150, lr=0.02`, this project's own established default) gives
RMSE~12-26; LBFGS with `Strong4DVar`'s own *validated* settings
(`max_iter=10, lr=0.2` -- confirmed by reproducing that class's ~0.81-1.0
RMSE with a from-scratch reimplementation) diverges to outright NaN on the
very first outer step. The SAME settings work fine for the strong-constraint
(`x0`-only) case in both classes -- isolating the failure to the
raw-physical-unit, unclipped, unrecoverable weak-constraint parameterization
itself, not the L96 dynamics or "weak-constraint DA is impossible on chaotic
systems" in general. `QG4DVar` (used in production for QG) never hits this
because it already has the three mechanisms ported here. User asked to port
it as a new `L96Weak4DVar`-style class rather than patch `Weak4DVar` in place.
**Verification:** `pytest tests/test_baselines_l96weak4dvar.py -q` (12
passed, `fdv` env); `pytest tests/test_baselines_enkf.py
tests/test_baselines_strong4dvar.py tests/test_baselines_weak4dvar.py
tests/test_baselines_l96weak4dvar.py tests/test_energy_score.py -q -m "not
slow"` (30 passed, 7 pre-existing unrelated failures -- all
`AttributeError: 'NoneType' object has no attribute 'step'` from
`Weak4DVar()`/`Strong4DVar()`/`EnKF()` constructed with no `dynamics` arg,
a stale Lorenz-63-default assumption broken elsewhere too, e.g.
`tests/test_numerical_equivalence.py`'s collection error; confirmed
pre-existing via `git diff --stat evaluation/baselines.py` showing a pure
197-line addition, zero deletions/modifications to existing code); `ruff
check evaluation/baselines.py tests/test_baselines_l96weak4dvar.py` clean.
**Not yet done:** an actual S0/S1 benchmark run with this class (only unit
tests so far, per "start small" -- the next step is a small-window sanity
run analogous to the earlier `Weak4DVar`/`Strong4DVar` experiments, then
scaling up if it reaches DA-baseline-comparable RMSE).
