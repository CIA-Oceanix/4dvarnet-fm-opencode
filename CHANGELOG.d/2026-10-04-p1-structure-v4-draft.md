## 2026-10-04: P1 paper structure v4 (scoping doc) and a new LaTeX draft

**Summary:** Adds the revised P1 structure documents and a new `main.tex` that follows the latest of them (v4). The previous draft is kept as `main.old.tex`.
- `l96_p1_structure_revision.md` (v1) organises the paper by a design matrix.
- `_v2.md` reads the 2026-10-02 presentation: an error chain and three questions.
- `_v3.md` extends the error chain along τ with the flow-matching score FMS_τ.
- `_v4.md` is the detailed structure: v2's narrative with v3's metric, flow matching kept bounded.
- The new `main.tex` follows v4: error chain (irreducible / restriction / misspecification / approximation), the chain along τ with FMS_τ, Q1 perfect model / Q2 model error / Q3 form of the prior, current numbers, `\needsrun` markers, figure placeholders.

**Files modified:**
- `docs/plans/paper/l96_p1_structure_revision.md`, `_v2.md`, `_v3.md`, `_v4.md` — scoping.
- `docs/papers/p1_structural_hypotheses/main.tex` — new draft.
- `main.old.tex` — the previous `main.tex`, renamed and unchanged; it still builds with `sections/*.tex`.
- `README.md` — source of truth, the old draft, known gaps.

**Rationale:** The paper is re-scoped around the presentation's error-budget framing. The design matrix becomes labels (appendix). FMS_τ extends the budget from the mean to the reported uncertainty, with an exact split at every τ. The general "every DA scheme is a flow" programme goes to the ML paper.

**Verification:** `latexmk -pdf main.tex` (10 pages, no undefined references, no box overflowing by more than 20 pt) and `latexmk -pdf main.old.tex` both build; `pytest tests/test_docs_layout.py` passes.
