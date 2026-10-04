## 2026-10-04: Strong and weak 4D-Var for the QG gyrostat DA benchmark (spectral climatological B)

**Summary:** This adds the variational baselines to the QG gyrostat DA
benchmark (forced and coupled; S0 and realistic S1).
- **New background covariance:** a spectral climatological B
  (`spectral_b_sqrt`). It replaces the diagonal grid-space B, which
  collapses on q₂ (val score 0.215).
- **Selected on val:** 10-day sub-windows, L-BFGS with 60 iterations, B × 1.
  The weak constraint uses model-error scale 0.1 at R × 1.
- **Test, forced / coupled:**
  - S0 strong: 0.840 / 0.847, best of all methods (+0.028 over the EnKS).
  - S1 strong: 0.516 / COUPLED_STRONG.
  - S1 weak: 0.612 / 0.619. That is +0.096 over strong, but −0.021 against
    the S1-tuned ETKF and −0.053 against the S1-tuned EnKS. The gap is in
    the unobserved lower-layer PV.
- **Safeguard:** a sub-window that is non-finite or raises the cost falls
  back to the background forecast. Fallbacks are counted per window.

**Files modified:**
- `evaluation/run_qg_baselines.py`:
  - `spectral_b_sqrt`;
  - `QG4DVar` gains `b_sqrt` (whitened control; the weak constraint uses
    Q = scale × B), a cost-increase guard, a non-finite fallback and
    `n_fallback`;
  - `run(..., fourdvar_b=)`.
- `evaluation/run_qg_specwind_da.py`: the `strong4dvar` / `weak4dvar`
  methods, the 4D-Var CLI options, and `n_fallback` per window.
- `batch/run_qg_specwind_da.sbatch`: the `VARB`, `BSCALE`, `QSCALE`, `WIN`
  and `ITER` env vars and their output-dir tags.
- `tests/test_qg_specwind_da.py`:
  - end-to-end 4D-Var tests (diagonal and spectral B);
  - a covariance test for `spectral_b_sqrt`;
  - a guard-rejection test.
- `reports/qg/generate_qg_specwind_da_report.py`:
  - a benchmark overview table;
  - 4D-Var rows in the S0, S1 and S1-tuned tables (its MAE shown, not
    ranked, in the CRPS column);
  - updated caveats.
  The report is regenerated.
- `docs/results/qg_specwind_4dvar.md` (new);
  `docs/results/qg_specwind_da_s0_s1_test.md` §3e; `reports/README.md`.

**Rationale:** The user asked for strong and weak 4D-Var developments
following the L96 weak-constraint 4D-Var. They complete the S0/S1 baseline
set before learned methods are compared.

**Verification:**
- SLURM 57998–58310 (val) and 58317–58349 (test): all completed.
- `pytest -m "not slow"` on the specwind DA, QG baselines and docs/report
  tests.
- `ruff check` on the touched files.
