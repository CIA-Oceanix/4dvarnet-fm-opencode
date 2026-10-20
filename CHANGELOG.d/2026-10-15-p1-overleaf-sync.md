## 2026-10-15: P1 paper draft synced from Overleaf

**Summary:** `docs/papers/p1_structural_hypotheses/` now mirrors the Overleaf project as of 2026-10-15 (revised sections, results restructured around normalised skill scores and error budgets, new figures, case-study appendix, Zotero bibliography replacing `refs.bib`).
**Files modified:** `docs/papers/p1_structural_hypotheses/` — `main.tex`, `sections_v4/*.tex`, `figures/*.pdf` (new), `zotero.bib` (new), `refs.bib` (removed); `.gitignore` — keep the figure PDFs while still ignoring the compiled `main.pdf`.
**Rationale:** The repository copy had not been synced since #326 (2026-10-06); the paper is edited on Overleaf.
**Verification:** `sync-overleaf.sh pull`; `sync-overleaf.sh status` reports no drift; `pytest tests/test_docs_layout.py`.
