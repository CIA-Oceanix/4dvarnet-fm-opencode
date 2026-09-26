## 2026-09-26: Cross-links between docs/results and reports

**Summary:** Every note in `docs/results/` now has a `**Numbers:**` line right
after its `Status:`. It names the report, output folder or probe script its
numbers come from, or says `inline` or `none`. `reports/README.md` gains a
"Written findings that cite these outputs" table linking each output back to the
notes that cite it. Both READMEs open with the split: `reports/` holds generated
numbers, `docs/results/` holds the written findings that interpret them.

**Files modified:**
- all 15 `docs/results/*.md` — the `**Numbers:**` line;
- `reports/README.md` — the split statement and the back-link table;
- `docs/README.md` — the split statement and the `Numbers:` convention;
- `tests/test_docs_layout.py` — each note must have a `Numbers:` line whose cited
  paths exist, and every output it cites must be listed against it in
  `reports/README.md`;
- `tests/test_reports_index.py` — the index row regex now matches bare report
  filenames only, so the new table's full-path rows are not taken for index rows.

**Rationale:** "results" and "reports" read as the same thing, and nothing linked
a finding to the numbers it interprets. The split itself is kept on purpose,
because generated outputs and hand-written findings have different lifecycles
and different guards.

**Verification:**
- `pytest tests/test_docs_layout.py tests/test_reports_index.py tests/test_report_inputs.py`:
  100 passed.
- Negative checks both fail as expected: deleting a back-link row, and pointing a
  `Numbers:` line at a missing folder.
