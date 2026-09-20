## 2026-09-16: fix true_dynamics coupling_exponent -- true ODE prior used the wrong forcing coupling

**Summary:** `FourDVarNetSolver`'s `self.true_dynamics = Lorenz96Dynamics(...)` (built for the `_FULL_STATE_UPDATE_INPUTS` modes: `subgrad+state+trueprior`/`subgrad+trueprior`) never passed `coupling_exponent`, silently defaulting to `Lorenz96Dynamics`'s own class default of `1.0`. Every training/test window's actual ground-truth trajectory, however, is generated with `coupling_exponent=1.6` (`data/lorenz96.py`'s `Lorenz96Config.coupling_exponent_truth`, used throughout window generation). So the "true" ODE prior in every trueprior run to date (jobs 53564, 53595, 53681) was subtly wrong -- not literally the dynamics that generated the data.
**Files modified:**
- `models/fourdvarnet.py` -- `FourDVarNetSolver.__init__` gains
  `true_dynamics_coupling_exponent=1.6` (defaults to the TRUE value, not
  `Lorenz96Dynamics`'s own generic `1.0` -- matches the same "default
  already equals the project's one canonical L96 config" convention as
  `true_dynamics_NO=8`/`J=4`/`h=1.0`), threaded into the
  `Lorenz96Dynamics(...)` construction.
- `tests/test_fourdvarnet.py` -- `test_true_dynamics_coupling_exponent_defaults_to_truth_value`,
  `test_true_dynamics_coupling_exponent_is_configurable`.
**Rationale:** Discovered while sanity-checking whether a classical
(non-learned) Weak4DVar-style direct optimizer could reach reasonable S0
performance using the same var_cost weighting as the failed neural
`subgrad+trueprior`+var_cost run (job 53681) -- confirming Weak4DVar's own
S0 dynamics convention uses `coupling_exponent=1.6`, which the FDV
trueprior code never matched. User asked to fix it once identified.
**Impact on existing results:** jobs 53564/53595/53681 all used the
slightly-wrong (`coupling_exponent=1.0`) true ODE -- their reported
RMSE/EV numbers stand as valid measurements of what was actually run, but
should not be read as measurements of the LITERALLY-true-ODE-prior
scheme. Re-running with this fix is a natural follow-up, not yet done.
**Verification:** `pytest tests/test_fourdvarnet.py -q` (140 passed,
`fdv-monai-proto` env); `ruff check models/fourdvarnet.py
tests/test_fourdvarnet.py` clean.
