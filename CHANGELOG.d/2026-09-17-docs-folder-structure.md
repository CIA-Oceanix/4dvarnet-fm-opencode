## 2026-09-17: Split `docs/` by lifecycle into `scoping/`, `results/`, `papers/`

**Summary:** Reorganized the flat 21-file `docs/` directory into three folders
split by lifecycle, and added an index that records which scoping doc is
current. Pure renames plus reference fixes — no doc content was rewritten.

**Files modified:** 20 files moved via `git mv` (all detected as renames, history
preserved) — `docs/scoping/` (9 forward-looking plans, `scoping_` prefix dropped
as redundant), `docs/results/` (6 measurement/investigation records),
`docs/papers/` (4 PDFs + `case_studies.tex`); `docs/worktrees.md` stays at root.
New `docs/README.md` (layout + citation conventions) and
`docs/scoping/README.md` (supersession index). 24 path references rewritten
across `PLAN.md`, 9 docs, and 5 Python files; 2 cross-directory bare references
fixed in `research_notes_cfm_da_originality_and_benchmarking.md`; the
already-broken sibling-relative link to `phase_B_l96_cfm_variants.md` fixed in
both `generate_l96_tweediecfm_report.py` and its checked-in output.

**Rationale:** The flat directory mixed three lifecycles with opposite
semantics — forward-looking plans that get *superseded in chains*, append-only
measurement records, and external artifacts. The "is the true prior the best
prior?" work is currently a three-link supersession chain and nothing in the
filenames said which link was live. The directory was already rotting:
`models/sda.py:24` cites `docs/phase_D_l96_sda.md`, a file that has never
existed on any branch. `docs/scoping/README.md` also records the Q2/Q3 question
numbering collision between the two scoping docs. `CHANGELOG.md` and pending
`CHANGELOG.d/` fragments were deliberately left pointing at the old paths: they
are historical records of what was true when written.

**Verification:** `pytest tests/ -m "not slow"` and `ruff check` on all five
touched Python files (all pass); `python -m py_compile` on the same. A scripted
checker confirms 0 remaining cross-directory broken references among all bare
`.md` mentions inside `docs/scoping/` and `docs/results/`.

**Known, not fixed here:** `models/sda.py:24` still cites the nonexistent
`docs/phase_D_l96_sda.md`; the correct target is not guessable, so it is left
for whoever knows what it meant rather than pointed somewhere plausible.
