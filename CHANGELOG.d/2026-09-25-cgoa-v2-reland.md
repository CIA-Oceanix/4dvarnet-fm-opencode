## 2026-09-25: Re-land CGOA scoping v2 (lost after PR #254 merged)

**Summary:** `docs/plans/case_study/cgoa_coupled_gyrostats.md` now carries v2
of the CGOA scoping note. The v2 commit (`ee803b9`) was pushed to
`feature/cgoa-scoping` after PR #254 had already merged, so master kept v1
while `docs/README.md` already described the doc as v2.

What v2 changes:
- **§5 no longer mirrors the L96 stack file for file.** CGOA adds only
  case-specific files: a generic quadratic-tensor engine, a gyrostat builder
  that exposes `component_slices`, and a `data/cgoa.py` that honours the
  existing window-dict contract.
- **Shared pieces are extracted first, in one no-op PR (new WP3):**
  - `L96Weak4DVar` → `WhitenedWeak4DVar`, with a per-component whitening
    scale;
  - grouped `evaluate_estimates`;
  - SCDA/WCDA via block localization on the existing `ETKF`/`EnKF`
    `loc_*_t` hook;
  - a generic DA loop;
  - a `cgoa` archive entry.
- **Correction to v1:** v1 said the plain `Weak4DVar` applies unchanged. On
  two-scale dynamics it diverges, as it did on L96.
- **New acceptance criterion:** no `eval_*_cgoa.py`, no per-run sbatch, no
  copied driver functions. Also adds decision 5, on where the extraction
  lands.

**Files modified:** `docs/plans/case_study/cgoa_coupled_gyrostats.md` — the
v2 diff applied at the file's post-#262 location. Doc paths were updated to
the new layout (`docs/plans/tech/multi_refactor_plan.md`,
`docs/plans/tech/archive.md`), and code line citations refreshed against
current master (`evaluation/baselines.py:775`, line 915; `train.py` lines 667
and 784).
**Rationale:** Master's doc and its index row disagreed on the version, and
the v2 implementation plan is the one to build on.
**Verification:** Docs only. `pytest tests/test_docs_layout.py` passes (35
tests).
