## 2026-09-17: FM-sampler report computes its Strong-4DVar margin instead of hardcoding it

**Summary:** The report's conclusions claimed the cold sampler "beats
Strong-4DVar by 31%". That figure was hardcoded prose in the generator and
reconciled with no pairing in the report's own tables. It is now computed from
the data, and the correct value is **35%**.

**Files modified:**
`reports/l96/generate_l96_fm_sampler_report.py` — new `cold_vs_baseline()` and
`build_conclusions()` replace the static `CONCLUSIONS` constant, following the
`compare_blend_to_warm()` idiom already in the file ("Deltas computed from the
data, never written by hand"); degrades to an explicit "margin not computed"
when no full-test-set run is supplied, rather than emitting a stale number.
`reports/l96/outputs/l96_fm_sampler_benchmark.md` — regenerated bullet (the
source JSONs are gitignored SLURM outputs and no longer on disk, so the bullet
was written to match the generator's output exactly; a test enforces that).
`tests/test_l96_fm_sampler_report.py` — new, 5 tests.
`.github/workflows/ci.yml` — new test file added to the gate allowlist.

**Rationale:** The claim was unverifiable against its own report: the true
margins are 34.8% on `rmse_repo` (0.8116 vs 0.5295, 200 windows) and 29.9% on
`rmse_pooled`, neither of which is 31%. The bullet now also states its operands
and convention so the number can be checked without rerunning anything. Caught
while absorbing these results into the ML-venue scoping doc (#212), which
deliberately used the computed 35% rather than propagating the 31%.

**Verification:** `pytest tests/test_l96_fm_sampler_report.py` — 5 passed,
including a drift test asserting the committed report still matches what the
generator renders, and one asserting the margin moves when the input RMSE moves.
`ruff check` on the two touched Python files.
