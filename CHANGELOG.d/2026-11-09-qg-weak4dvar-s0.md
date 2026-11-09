## 2026-11-09: QG Weak-4DVar at S0

**Summary:** Weak-constraint 4D-Var was tuned and run on the QG gyrostat benchmark with a perfect model (S0). The new row is in the QG P1 table.
**Files modified:**
- `reports/qg/generate_qg_p1_tables.py`: new S0 `Weak-4DVar` row.
- `reports/qg/outputs/qg_p1/qg_p1_rows.json`, `qg_p1_series.json`: regenerated; only the new row is added.
- `docs/results/qg_specwind_4dvar.md`: S0 weak-constraint section.
**Rationale:** The P1 table and budget figure had no QG Weak-4DVar at S0.
- **Validation:** 10 windows, model-error scale ∈ {0.005, 0.01, 0.02, 0.05, 0.1} (SLURM 70666–70668, 70724–70725). 0.01 gives the best mean field EV (0.827, vs 0.824 for the strong constraint).
- **Test:** 100 windows (SLURM 70744).
**Verification:**
- Test: EV 0.9906, MAE skill 0.882, against 0.9911 / 0.885 for Strong-4DVar.
- `generate_qg_p1_tables.py` reproduces every existing row unchanged.
- `pytest tests/test_docs_layout.py tests/test_reports_index.py` passes; `ruff check` is clean on the generator.
