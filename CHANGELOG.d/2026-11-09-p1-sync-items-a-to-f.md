## 2026-11-09: P1 paper synced with items a-f; slow/fast EV in the pooled skill report

**Summary:** Synced the P1 paper from Overleaf with the results of #384-#389:
- Tables 1, 2 and 5 now use the 2D SDA2/SDA3 rows (3 seeds).
- QG Weak-4DVar at S0 is added.
- New: the S1-full paragraph, the value-of-conditioning results, the observing-system figure in EV, and the energy/variogram comparison.
- The budget figures are regenerated.

`reports/l96/p1_skill.py` adds the slow/fast EV split quoted in Sect. 5.1, and `reports/l96/outputs/p1_l96_benchmark.md` gains that table.
**Files modified:**
- `docs/papers/p1_structural_hypotheses/`: `sections_v4/05_results.tex`, `sections_v4/06_discussion.tex`, `fig_budget_s0.pdf`, `fig_budget_s1.pdf`, `fig_obs_system.pdf`.
- `reports/l96/p1_skill.py`: slow (8) / fast (16) EV per row.
- `reports/l96/outputs/p1_l96_benchmark.md`: the slow/fast table, spliced from `p1_skill.section()`, identical to a full regeneration.
**Rationale:** Every number in the paper must trace to committed outputs. The S1-full and conditioning values come from `reports/l96/outputs/p1_s1full_conditioning.json` (PR #389).
**Verification:**
- `pytest tests/test_docs_layout.py tests/test_reports_index.py tests/test_report_inputs.py` passes; `ruff check` passes on `p1_skill.py`.
- The paper compiles.
