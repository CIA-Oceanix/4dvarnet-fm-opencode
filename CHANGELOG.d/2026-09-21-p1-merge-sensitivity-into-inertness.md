## 2026-09-21: Merge the ODE-vs-posterior sensitivity argument into the inertness section

**Summary:** "ODE sensitivity is not posterior sensitivity" is demoted from a
results subsection to a `\paragraph` inside "Conditioning on model information is
nearly inert". §5 goes from eight subsections to seven.

**Why.** The section stated exactly three numbers -- `<=1.5%`, `~0.2%` and
`1.68-1.80x` -- and **owned none of them**: the first two are measured by the
conditioning ladder immediately above it, the third comes from the overview's
robustness table. It was an interpretation of other sections' measurements
presented as if it were a result, and it had to cross-reference the section
directly above it for its own evidence.

**But unlike the §5.2 removal, the claim is kept in full.** C6 -- that assessing
parameter uncertainty by propagating it through the model overstates its effect
on the posterior whenever the observations are informative -- is a genuine
inference and the scoping doc's sharpest methodological point. What changed is
that it now lands immediately after the measurements it rests on, instead of in a
section that has to reach backwards for them. The text also now says plainly that
it is *an inference from two measurements rather than a separate experiment*, and
names the experiment that would measure it directly.

**Files modified:** `sections/05_results.tex`. `\label{sec:sensitivity}` is
retained on the paragraph, so the forward references from §1 and §2 still
resolve -- they now point at the merged section, which is where the argument
lives.

**Verification:** `pdflatex` x3 -- 14 pages, 0 undefined references or citations;
every `\ref` checked against a matching `\label`, none dangling.
