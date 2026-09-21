## 2026-09-21: Reorder the P1 results section to lead with a cross-system overview

**Summary:** The results section was ordered by hypothesis, so the headline was
never stated as one and two of the three case studies appeared late, each
supporting a single point. It now opens with a cross-system overview and then
proceeds by mechanism. 14 pages, compiles clean, zero undefined references.

**New structure.** §5.1 Overview -> §5.2 the model is the prior (with
conditioning inertness and the two sensitivities as subsections) -> §5.3
marginal-value collapse (with the weak-4D-Var bound) -> §5.4 representation ->
§5.5 the non-Gaussian deficit and blending -> §5.6 the confounded ordering.

**The overview is deliberately two tables, in this order.**
1. **Robustness**, `S0 -> S1` for the classical schemes on all three systems.
   This is *intra-classical*: each scheme is its own control, so the standard
   "your baselines were under-tuned" objection does not touch it.
2. **The gap**, best classical vs best learned. This is cross-class and carries
   the method-class caveat explicitly.

Leading with (1) matters. Opening on "learned beats classical" would read as a
method comparison and invite exactly the critique the diagnosis framing exists to
avoid; opening on the robustness failure is unobjectionable, and (2) then lands
with its caveat already attached.

**Three things this surfaces that the previous ordering buried:**
- **QG's variational potential-vorticity scores are negative** -- Strong-4DVar
  `-0.126` at S0 falling to `-0.857` at S1, Weak-4DVar `-0.034 -> -0.501`, while
  the ensemble filters hold near `0.4` and the neural rows reach `0.67`. Worse
  than climatology, and previously not shown at all. §5.4 ties it to the same
  `q ~ Laplacian(psi)` amplification as the psi-state result: a scheme that fits
  psi well can be arbitrarily bad in q.
- **Weak-constraint 4D-Var exists on QG as well as L63** (at an identical config
  to the QG neural rows), so it covers two of three systems and only the L96 run
  is missing. §5.3.1 now states this as the bound on C1 rather than a pending
  risk.
- **Q1 vs Q4 makes the method-class caveat concrete.** Q1 is obs-only so its S1
  numbers are bit-identical to S0 -- definitional, no information. Q4 is
  conditioned and genuinely holds. Adjacent, they demonstrate the caveat instead
  of asserting it.

**Stated gap:** L63 carries no learned schemes, so the robustness table spans
three systems while the gap table spans two. This is in the caption, not left for
a referee to notice. Metrics also differ across systems (RMSE for L63/L96, EV for
QG); they are kept separate and the *ratio* carries the comparison rather than a
forced common normalisation.

**Also:** §6's open-experiments list is marked provisional -- it was drawn up
against the old outline and should be re-derived from the consolidated structure
before any run is scheduled.

**Verification:** `pdflatex` x3 -- 14 pages, 0 undefined references or citations;
every `\ref` to a `sec:`/`tab:`/`eq:` label checked against a matching `\label`,
none dangling.
