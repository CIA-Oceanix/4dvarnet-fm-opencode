## 2026-11-09: L96 S1-full DA rows and the value of the conditioning information

**Summary:** Two new P1 result sets on the regular test set.
- **S1-full rows:** the S1 DA model keeps all 40 variables, with the biased parameters and corrupted forcing but no truncation. ETKF, EnKF, ETKS, Strong-4DVar and Weak-4DVar on the 200 test windows.
- **Value of conditioning:** EV of the DA schemes and of the 2D SDA2 / SDA3-fix priors (seed 1) when the parameters and/or forcing are withheld. The 1D SDA rows from the earlier run are included as well.

**Files modified:**
- `batch/run_l96_s1full_test.sbatch`: DA task plus 20 Weak-4DVar chunks.
- `batch/run_l96_conditioning_value_2d.sbatch`: runs the `feature/l96-2d-sda23` code at a pinned commit, with one output dir per prior and variant.
- `scripts/assemble_weak4dvar_bench.py`: `--cases` / `--name` options; defaults unchanged.
- `reports/l96/p1_s1full_conditioning.py` and its output `reports/l96/outputs/p1_s1full_conditioning.json`: scores as `p1_skill.py`.
- Earlier commits on this branch: the `s1_j` option in the sweeps, the S1-full validation sweep and the conditioning test sets.

**Rationale:**
- S1 combines a parametric error with a structural one, the DA model being truncated to the observed variables. S1-full separates the two.
- The conditioning sets measure how much each scheme uses theta and the forcing.

**Hyper-parameters:** RMSE-optimal on the 50 regular validation windows, the benchmark rule.
- ETKF lambda 1.3 (val RMSE 0.753); ETKS takes the ETKF value.
- EnKF 1.5 (0.773).
- Weak-4DVar q 0.03 (0.843 on 8 windows, against 0.759 for Strong-4DVar on the same windows).

**Verification:**
- The conditioning caches have the same truth and observations as the P1 S0 test windows, on all 200 windows.
- The Weak-4DVar chunks cover windows 0-199 once, with no resets. The rescored RMSE matches the chunks.
- `pytest tests/test_report_inputs.py` and `ruff check` pass on the touched files.
