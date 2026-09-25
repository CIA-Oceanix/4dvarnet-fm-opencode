## 2026-09-25: docs/ reorganized by purpose — plans/{paper,analysis,case_study,tech}, results/

**Summary:** `docs/scoping/` is split by what a doc is for:
- `plans/paper/`, with `superseded/`;
- `plans/analysis/`;
- `plans/case_study/`;
- `plans/tech/`.

`archive.md`, `worktrees.md` and the FDV torch.compile notes move to
`plans/tech/`. Results notes get a case-study prefix (`<case>_<topic>.md`).
`docs/scoping/README.md` is folded into `docs/README.md`, which is now the one
index of live docs. The pure "moved to" stub `results/fm_prior_sampler_results.md`
is deleted. A new `tests/test_docs_layout.py` enforces the layout.

**Files modified:**
- 29 docs moved (`git mv`, contents unchanged except the `Status:` lines
  below). The full old → new table is in the PR description.
- Every citation of a moved path is rewritten across `docs/`, `.py`, `.tex`,
  `config/`, `reports/` (generators **and** checked-in outputs), `AGENTS.md`,
  `PLAN.md`, `README.md`, `.github/workflows/ci.yml` and `.gitignore`.
  `CHANGELOG*` history is left as written.
- `Status:` lines were added to the four docs that lacked one, and the first word
  of eight others was normalized to their folder's vocabulary.
- 17 config comments that cited docs never committed to master are repointed
  to `reports/l96/outputs/p1_l96_benchmark.md`:
  - `docs/scoping/wave2_benchmark_plan.md`,
  - `docs/scoping/monai_default_config_benchmark_plan.md`,
  - `docs/results/l96_default_config_benchmark_synthesis.md`.
  These are comments only.
- `tests/test_docs_layout.py` (new) checks three things:
  - each plan or results doc has a `**Status:**` whose first word is in its
    folder's vocabulary;
  - nothing sits outside the layout;
  - every `docs/...` path cited in the tracked tree exists.
- `AGENTS.md` — a one-line pointer to the layout.
- Folded in #261, which merged while this PR was open:
  - `docs/results/cfm_tau_consistency_synthesis.md` →
    `docs/results/l96_cfm_tau_consistency_synthesis.md`;
  - the τ-consistency plan's CLOSED status moves into the `docs/README.md` index,
    which gains a "cite the synthesis" table;
  - `CLOSED` is added to the `plans/analysis/` and `plans/case_study/`
    vocabularies, for a plan that a synthesis in `results/` answers.

**Rationale:** Paper scoping, analysis plans, case-study proposals and tech notes
were mixed in one folder, with an index that had to explain what each file
was. The folder now says what a doc is for, and the prefix says which case
study it belongs to. Paper drafts stay in `docs/papers/`, so the P1 Overleaf
sync paths are unchanged. The citation check also surfaced 17 existing
citations of docs that never reached master.

**Verification:**
- `pytest tests/test_docs_layout.py tests/test_l96_report_consistency.py`: 47 passed.
- Every test file referencing `docs/` (`-m "not slow"`): 96 passed.
- `ruff check tests/test_docs_layout.py`: clean.
