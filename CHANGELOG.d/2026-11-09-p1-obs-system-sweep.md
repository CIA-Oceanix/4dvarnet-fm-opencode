## 2026-11-09: P1 observing-system figure on the benchmark schemes, in EV

**Summary:** The 16 cells of `fig_obs_system` are re-run with the benchmark schemes: ETKF, EnKF, ETKS and Strong-4DVar, plus the 2D learned schemes at 1200 epochs (seed 1): D-Unet, CFM, SDA2 and the D-Unet -> SDA3-fix hybrid. The figure now plots EV instead of RMSE.
**Files modified:**
- `batch/run_l96_obs_system_da.sbatch`: the DA run. CPU, all four methods, trajectories kept.
- `batch/run_l96_obs_system_learned.sbatch`: the learned run.
  - Evaluation only, on the cells' test sets, which hold the same obs as the DA cells.
  - SDA guidance uses the cell's true R.
  - The hybrid uses guidance weight 1.
  - The SDA2 / SDA3 tasks ran from `feature/l96-2d-sda23`, before #384 merged those model types.
- `reports/p1/p1_figures.py`: new `obs` step, writing `reports/p1/outputs/l96_obs_system.json`. EV is computed per channel and averaged over the 24 observed channels, as in the paper tables. `obs_system_figure` now draws all eight schemes from that JSON.
- `reports/p1/outputs/l96_obs_system.json`: the per-cell EVs.
**Rationale:** The paper figure was a preliminary version: 400-epoch learned models, filters only, RMSE.
**Verification:**
- The ETKF/EnKF re-runs use the same windows and layouts as the earlier cells. Their RMSE is within 0.02, the difference coming from the ensemble RNG on CPU vs GPU.
- The `obs` step asserts that the learned estimates' truth equals the DA truth in every cell.
- `pytest tests/test_docs_layout.py tests/test_reports_index.py tests/test_report_inputs.py` passes, as does `ruff check`.
- Outputs are bundled under `benchmark_extended/here/{obs_system_da,eval_obs_system}`.
