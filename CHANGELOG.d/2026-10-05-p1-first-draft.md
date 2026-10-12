## 2026-10-05: First prose draft of P1 on structure v4

**Summary:** `docs/papers/p1_structural_hypotheses/main.tex` now inputs one file per section from `sections_v4/`. The sections are an abstract, §1–§9 and appendix stubs, about 7k words. They are written from `docs/plans/paper/l96_p1_structure_v4.md`:
- the error chain (irreducible / restriction / misspecification / approximation), with exact splits;
- its extension along τ with FMS_τ, and the operator Ψ of each kind of scheme;
- the design: matched information, tuning parity, the estimator of each term;
- results Q1 (perfect model), Q2 (model error) and Q3 (form of the prior), with the current numbers and `\needsrun` markers;
- the discussion and conclusion.

**Files modified:** `docs/papers/p1_structural_hypotheses/main.tex` — preamble + `\input`s; `sections_v4/00_abstract.tex` … `09_appendix.tex` — new; `README.md` — layout and known gaps. `main.old.tex` and `sections/` are unchanged.

**Rationale:** The user asked for a first draft, on structure v4, to upload to Overleaf.

**Verification:** `latexmk -pdf main.tex` builds 17 pages with no undefined references and no box overflowing by more than 20 pt. `latexmk -pdf main.old.tex` still builds. `pytest tests/test_docs_layout.py` passes.
