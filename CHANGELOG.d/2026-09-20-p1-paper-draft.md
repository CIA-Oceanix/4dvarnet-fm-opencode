## 2026-09-20: First LaTeX draft of the P1 (DA-venue) paper

**Summary:** Adds `docs/papers/p1_structural_hypotheses/` — a complete, compiling
first draft of the DA diagnosis paper targeting JAMES, rendered from
`docs/scoping/da_paper_structural_hypotheses.md` (SCOPING v2). 11 pages,
abstract through conclusion, with every evidence table carrying its report
provenance in the caption.

**Files added:** `main.tex` (article class, portable to `agujournal2019` by one
line); `sections/01_introduction` .. `07_conclusion`; `refs.bib`; `README.md`
(build + Overleaf import instructions); `.gitignore` for LaTeX artefacts.

**Rationale:** Requested first version of an Overleaf draft. Written in-repo
rather than in Overleaf directly, since there is no Overleaf access from this
environment; the README documents the three import routes (git sync, zip upload,
copy-paste).

**The scoping doc remains the source of truth.** The draft renders its argument
and does not supersede it; numbers must be changed in both places, and the
standing caveat about the consolidated benchmark being regenerable applies here
too.

**Unresolved state is marked in the rendered PDF, deliberately.** Three macros —
`\todo`, `\needsrun`, `\caveat` — print in red/brown so that no claim resting on
an unrun experiment can be circulated unnoticed. The load-bearing one is
**C1 is blocked on D0**: weak-constraint 4D-Var is the classical method
*allowed* to know about model error, so until it is benchmarked C1 is argued
against strong-constraint and filtering methods only. `L96Weak4DVar` landed in
#229 but is wired to no driver and still defaults to `optimizer="adam"` (RMSE
~24 vs L-BFGS ~0.98).

**`refs.bib` is entirely unverified** — written from memory, every entry tagged
`UNVERIFIED`, and no `\cite` commands are used yet. It must be checked entry by
entry before submission.

**Verification:** `pdflatex -interaction=nonstopmode -halt-on-error main.tex`
succeeds; three passes give 11 pages with **zero** undefined references or
citations. Build artefacts are gitignored and none is committed.
