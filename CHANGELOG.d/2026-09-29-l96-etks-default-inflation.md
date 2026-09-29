## 2026-09-29: ETKS row at the benchmark inflation (ETKF 1.15 / 2.5)

**Summary:** The ETKS on the 200 P1 test windows at the #295 benchmark ETKF inflation, paired with an ETKF rerun that reproduces #298's rows to within 0.3%. Result: regular S0 0.497 (−18.7%), S1 1.338 (−5.1%); random S0 0.572 (−15.6%), S1 1.348 (−4.2%).
**Files modified:**
- `reports/l96/generate_l96_benchmark_extended_report.py` — an "ETKS" row directly under the benchmark ETKF. The ETKS file pattern now takes the EnKF default in its name.
- `reports/l96/outputs/l96_benchmark_extended.md` — regenerated.
- `docs/results/l96_etks_default_inflation.md` (new); `docs/README.md`, `reports/README.md`, `docs/plans/tech/l96_etks_smoother.md` — index and status.
**Rationale:** The PR 2 rows (#297) were at λ 1.5 / 2.0 and 1.1 / 2.5, and neither is the benchmark default set in #295.
**Verification:** Runs from `batch/run_l96_etks_test.sbatch` tasks 2–3 (SLURM 56736); 4 input files added to the `benchmark_extended` bundle; report regenerated with its consistency checks passing; see the PR body for tests.
