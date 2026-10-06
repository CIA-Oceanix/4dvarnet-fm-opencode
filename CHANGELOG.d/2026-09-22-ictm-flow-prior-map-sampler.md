## 2026-09-22: ICTM (Flow Priors) MAP sampler, standalone from SDA

**Summary:** Implemented Iterative Corrupted Trajectory Matching (ICTM) --
Zhang et al., "Flow Priors for Linear Inverse Problems via Iterative
Corrupted Trajectory Matching" (NeurIPS 2024, `reports/FlowPrior.pdf`) --
as its own module, `evaluation/ictm_sampler.py::ictm_map_sample`, applied to
this repo's DA observation setup. This is a deterministic MAP point
estimator (Algorithm 1: K inner Adam gradient steps per Euler step on a
local-MAP objective), distinct in kind from `evaluation/sda_sampler.py`'s
SDA ensemble sampler (a single DPS-style normalized-gradient nudge per
step) -- `sda_sampler.py` is untouched by this change.

**Files modified/added:**
- `models/interpolant.py` -- `LinearInterpolant.score(x_tau, v, tau, sigma)`,
  the closed-form Tweedie/Proposition-1 prior score from the paper, needed
  by ICTM's local-prior gradient term (Eq. 11). Derived from scratch for
  this repo's tau=0-noise/tau=1-data convention; algebraically reduces to
  the paper's separate t=0 closed-form case, so Algorithm 1's branching
  collapses into one formula here.
- `evaluation/ictm_sampler.py` (new) -- `hutchinson_trace` (differentiable
  Skilling-Hutchinson trace estimator for the log-density Riemannian-sum
  term, standard in continuous normalizing flows / FFJORD) and
  `ictm_map_sample(model, batch, N_outer, K, lam, step_size, n_members,
  obs_indices, obs_channel_mask, trace_samples)`, matching
  `sda_guided_sample`'s `(x_1, n_forward)` return convention. The "corrupted
  trajectory" auxiliary path (`y_tau`) reuses `LinearInterpolant.mix`
  (fed `(x0, y)` instead of `(x0, x1)`); the observation cost reuses
  `evaluation/sda_sampler.py::guided_obs_cost` unchanged (`R_var=1.0`, `lam`
  alone weights it, matching the paper's own choice to replace the exact
  noise-variance coefficient with a tuned hyperparameter).
- `tests/test_interpolant.py` -- `test_score_matches_closed_form_gaussian_mixture`
  (validates the score formula against an exactly-solvable Gaussian-mixture
  case, no trained network involved) and `test_score_at_tau0_ignores_v`
  (confirms the branch-collapse claim above).
- `tests/test_ictm_sampler.py` (new) -- `hutchinson_trace` validated two
  ways (exact zero gradient for a linear `v`, since `dv/dx` is then
  x-independent; Monte Carlo convergence to the exact trace for a
  nonlinear `v`, via `torch.autograd.functional.jacobian`), plus
  `ictm_map_sample` shape/finiteness, `n_members` stacking, higher `lam`
  reducing observation cost, `K>1` changing the trajectory and NFE count,
  running under a caller's `torch.no_grad()`, and seed-determinism. All run
  against the real `UnconditionalPriorCFM`/`UNet1D` stack (not a toy linear
  model), which also serves as the regression check that this architecture
  supports the double-backward the trace term's gradient needs.

**Rationale:** Requested as a from-scratch implementation of the actual
ICTM algorithm (Algorithm 1: local-MAP objective + K-step inner refinement
+ Hutchinson trace term), rather than folding a single piece of it
(the Prop. 1 score) into the existing SDA sampler as a guidance-gradient
add-on (that earlier approach was implemented and then reverted at the
user's request in favor of this standalone version). Kept fully separate
from `sda_sampler.py` so the two algorithms -- SDA's ensemble guidance vs.
ICTM's deterministic MAP refinement -- stay independently reasoned about
and neither can accidentally destabilize the other's tests/benchmarks.

**Verification:** `pytest tests/test_ictm_sampler.py tests/test_interpolant.py
tests/test_sda_sampler.py tests/test_eval_sda_l96.py -q` -- 43 passed.
`git diff --stat evaluation/sda_sampler.py` is empty (confirmed untouched).
`ruff` was not available in this local environment (no `ruff` module/binary
on `PATH` or in the `intern` conda env) -- not run; ruff is CI-informational
(`continue-on-error: true`) per project policy, so this doesn't block. No
eval script (`eval_sda_l63.py`/a new `eval_ictm_l63.py`) was wired up yet --
`ictm_map_sample` is a MAP point estimator (no ensemble spread), so it
doesn't fit `eval_sda_l63.py`'s CRPS/ensemble-spread harness as-is; left as
a follow-up pending the caller's choice on `lam`/`K`/`step_size` (none
swept yet, unlike SDA's `guidance_weight=20`) and metric set (RMSE/R2 only,
most likely).
