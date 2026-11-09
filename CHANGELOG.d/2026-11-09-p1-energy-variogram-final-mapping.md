## 2026-11-09: P1 energy/variogram scores on the final learned rows

**Summary:** The energy and variogram table now uses the final P1 learned mapping from #384: SDA = SDA2-2D-M seed 1 (gw 25), and D-Unet+SDA = DirectUNet-2D-M seed 1 -> SDA3-fix-2D-M seed 1 (tau0 0.1, gw 1). The SDA1-based rows of #386 are kept in the JSON as `sensitivity_sda1`. Learned FMSS_0.75 is now the seed-1 value (`per_seed`), matching the seed-1 scores.
**Files modified:**
- `batch/run_l96_p1_energy_variogram.sbatch`: tasks 4-5 for the two new rows.
- `reports/l96/p1_energy_variogram.py`: report step with the row-to-file mapping, the sensitivity entry and seed-1 FMSS.
- `reports/l96/outputs/fm_score/p1_energy_variogram.json`: regenerated.
**Rationale:** The table must use the same learned schemes as the P1 tables.
**Verification:** Job 71081. The ensemble-mean RMSE of the new rows is identical, to 4 decimals, to the stored evals. Same climatology sample and geometries as #386. `ruff check` passes.
