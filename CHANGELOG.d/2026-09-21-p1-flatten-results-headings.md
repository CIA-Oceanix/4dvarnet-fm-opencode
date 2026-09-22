## 2026-09-21: Flatten the P1 results to a single subsection level

**Summary:** The three `\subsubsection` headings introduced when the results were
reordered are promoted to `\subsection`, so §5 is nine peers rather than a
two-level tree.

**Files modified:** `docs/papers/p1_structural_hypotheses/sections/05_results.tex`
-- "Conditioning on model information is nearly inert", "ODE sensitivity is not
posterior sensitivity" and "Weak-constraint 4D-Var bounds the claim" are now
subsections. All `\label`s are unchanged, so every cross-reference from §1, §2,
§3 and §6 still resolves.

**Rationale:** Requested. The nesting also implied a subordination that does not
hold -- the two sensitivities and the weak-4D-Var bound are load-bearing results
in their own right, not elaborations of the sections they sat under.

**Verification:** `pdflatex` x3 -- 14 pages, 0 undefined references or citations;
every `\ref` to a `sec:`/`tab:`/`eq:` label checked against a matching `\label`,
none dangling. No `\subsubsection` remains anywhere in the document.
