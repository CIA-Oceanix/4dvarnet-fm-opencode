## 2026-09-15: CFM velocity affine decomposition — derivation + diagnostic on trained checkpoints

**Summary:** Added a research note deriving the exact decomposition of the CFM
velocity into an amortized-mean term, a linear term, and a nonlinear remainder,
plus the diagnostic probe that measures the remainder on trained L96
checkpoints. Under the linear interpolant the velocity reduces exactly to
`v = (mu - x_tau)/(1-tau)`; projecting `mu` onto `span{m(y), x_tau}` gives
`G(tau) = tau*Pbar*[(1-tau)^2*sigma0^2*I + tau^2*Pbar]^-1` with
`Pbar = E_y[Cov(x1|y)]`, and the residual is exactly the non-Gaussianity of the
posterior. Measured on V3 (`PredictStateCFM`) and FDV1CFM
(`FourDVarNetPredictStateCFM`): the closed-form gain predicts the least-squares
fitted coefficient to ~1.5%/~4% from a single parameter, and the predicted peak
location `tau* = sqrt(s/(s+p))` matches the observed argmax on both.

**Files modified:**
- `docs/cfm_affine_velocity_decomposition.md` — new research note (derivation, measurement protocol, results, conclusions)
- `reports/l96/probe_cfm_affine_decomposition.py` — new diagnostic; fits scalar/per-channel (A,B) for `mu ~ A*m + B*x_tau` over a tau grid and reports residual ratios

**Rationale:** Gates a proposed unrolled-solver-as-mean-estimator CFM
parameterization before implementing it. The affine schedules turn out to be
free (predicted in closed form from the mean estimator's own MSE), but 19-31%
of the *velocity* amplitude is non-affine and roughly flat over
`tau in [0.1,0.8]`, so the residual network cannot be a thin correction head —
which revises the expected inference saving from ~5x down to ~3x. Analysis
only: no model, training or evaluation code is touched.

**Verification:** `ruff check` clean on both new files; probe run end-to-end on
the cached 200-window S0 and S1 test sets for both checkpoints (S1 reproduces
S0 to 3 decimals). Full CI gate test list: **454 passed, 4 skipped** (12m39s).

Note: a broader `pytest tests/ -m "not slow"` additionally errors in
`tests/test_equiv_report.py` at `evaluation/baselines.py:299`
(`_ESAccumulator.step` calls `.detach()` on a `numpy.ndarray`). This is
**pre-existing and unrelated** to this change — which adds three new files and
touches no library code — and that test file is not in the CI gate list. Worth
a separate fix; the ES accumulator path is reachable from
`Strong4DVar.assimilate`.
