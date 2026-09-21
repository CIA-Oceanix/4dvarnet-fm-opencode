## 2026-09-21: Fold weak-4D-Var into the marginal-value section; fix a wrong coverage claim

**Two changes, one of them a factual correction.**

**1. `sections/04_evaluation.tex` stated the wrong classical coverage for QG.**
The scheme-coverage table said QG carried `EnKF/ETKF`. It carries **all four**
classical schemes -- EnKF, ETKF, Strong-4DVar and Weak-4DVar -- at an identical
config to its neural rows (`reports/qg/outputs/qg_neural_report.md`, rows 9-12
and 26-29). The error understated classical coverage precisely where it matters
most, since weak-constraint 4D-Var is the paper's defence against "you compared
against a strawman". Corrected, and the table now also marks **Lorenz-96 as the
system with *no* weak-4D-Var**, which is the real gap.

**2. The dedicated weak-4D-Var results subsection is demoted to a paragraph**
inside "Model error destroys the marginal value of observations". §5 goes from
seven subsections to six.

Same test as the two preceding cleanups: the subsection owned no measurement.
Every number it quoted -- L63 `0.642/1.633`, QG `0.9660/0.9468` and
`-0.034/-0.501` -- is in the overview's robustness table. What it carried was a
*caveat on the collapse result*, and that is exactly where it now sits: directly
under the table it qualifies.

**The visibility argument, and why the coverage fix answers it better.** The one
real reason to keep a dedicated heading was that a DA referee looks specifically
for whether weak-constraint was considered, and a section title answers that at a
glance. But the right place for that signal is the evaluation framework, where
the schemes are enumerated -- and that table was *wrong*, which is a far worse
problem than a missing heading. With §4 corrected, weak-4D-Var is visible in the
scheme list, in the coverage table, and as a named paragraph under the claim it
bounds.

**Files modified:** `sections/04_evaluation.tex` (coverage table),
`sections/05_results.tex` (subsection to paragraph; `\label{sec:weak4dvar}`
retained so the §6 reference still resolves).

**Verification:** `pdflatex` x3 -- 14 pages, 0 undefined references or citations;
every `\ref` checked against a matching `\label`, none dangling.
