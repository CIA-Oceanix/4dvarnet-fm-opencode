## 2026-09-18: L96 — headline Summary table; freeze the pre-refactor report

**Summary:** the consolidated benchmark had six 6-column group-breakdown tables
and no single view of a scheme's standing. Adds a `## Summary` table — one row
per scheme, per-window mean ± std, all three metrics side by side, ranked by S0
RMSE — and freezes the pre-refactor report as `*.old.md`.

**Files modified:**
- `reports/l96/generate_l96_consolidated_report.py` — `fmt_summary_table`; header note rewritten for the three-state ES/CRPS convention
- `reports/l96/outputs/l96_consolidated_benchmark.md` — regenerated
- `reports/l96/outputs/l96_consolidated_benchmark.old.md` — **new**, frozen

**Rationale:** the group breakdowns answer "where does this scheme lose skill",
not "how do these schemes compare". The Summary answers the second, uses the
standardized per-window mean ± std for every cell, and is generated rather than
assembled by hand.

Stale prose also fixed in the same pass: the header note still claimed
*"ES for deterministic methods is the N=1 per-dim MAE proxy"* and still
referenced `L3 (ens30×10)` — both untrue after the 2026-09-18 standardization
and the pre-monai retirement. `DirectUNet(aug)+SDA3`'s description likewise
still described its ES/CRPS as proxy values when those cells now read
`pending`. The code had been changed without the narrative, so a reader would
have trusted the prose over the cells.

**On the frozen copy:** `*.old.md` keeps the 16 retired pre-monai rows, whose
numbers exist nowhere else, plus a header listing what changed. No `.old.py`
counterpart: git already holds the previous generator exactly, that version
cannot run (it builds `experiments/<run>/` paths directly — the defect that made
the report unrunnable from master), and a second generator carrying
`N1_ES_METHODS` would trip the "proxy must not return" assertion in
`tests/test_l96_report_consistency.py`.

**Verification:** 32 tests pass; report regenerates clean; Summary carries 28
scheme rows (27 scoring + the FDV2 unavailable row).
