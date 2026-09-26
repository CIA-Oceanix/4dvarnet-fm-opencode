## 2026-09-26: Re-land the coupled test-split record and the S0 DA design (lost from #282)

**Summary:** #282 was squash-merged while GitHub still showed a stale head
(`332241f`), so its last commit (`6e8532d`) never reached master. That
commit recorded the coupled test split in
`docs/results/qg_coupled_dataset.md` and added
`docs/plans/analysis/qg_specwind_da_s0.md` with its index row. It is
cherry-picked here onto current master. The design doc also records the
settled decisions (v2): staging to `/Odyssey` for a SLURM array, 3 columns
per day headlining, and 100 test windows per dataset (500 for D2/D3).

**Files modified:** `docs/plans/analysis/qg_specwind_da_s0.md` (new, v2);
`docs/results/qg_coupled_dataset.md`; `docs/README.md`;
`CHANGELOG.d/2026-09-26-qg-coupled-dataset.md` (the additions paragraph from
`6e8532d`).
**Rationale:** Restores content that the stale-head merge dropped. The
merged tree of #282 was checked to equal `332241f` exactly.
**Verification:** `pytest tests/test_docs_layout.py
tests/test_reports_index.py`: 65 passed.
