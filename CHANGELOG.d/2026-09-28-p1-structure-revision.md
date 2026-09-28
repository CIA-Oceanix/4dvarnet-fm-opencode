## 2026-09-28: P1 revised paper structure (scoping)

**Summary:** Scoping doc for a revised P1 structure: design matrix (F1 prior–likelihood split, F2 dynamics structure, F3 sequential inference, U-state, U-model) replaces the conditional-flow partition as the paper's instrument; restriction-vs-misspecification thesis; two-pass results (S0 then S1).
**Files modified:** `docs/plans/paper/l96_p1_structure_revision.md` — new SCOPING doc
**Rationale:** C4 and C5 of the current draft are not supported; the learned rows' S1 robustness is confounded by training on the true system.
**Verification:** `pytest tests/test_docs_layout.py -q`
