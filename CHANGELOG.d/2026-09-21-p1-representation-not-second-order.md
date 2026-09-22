## 2026-09-21: Rewrite the P1 representation section -- it sits on the main axis, not beside it

**Summary:** The representation subsection was written as a QG curiosity about
state variables. Pulling the underlying report showed it is measured at **both
\Sz{} and \So{}**, which puts it on the paper's model-error axis and makes it
first-order rather than second.

**The finding the previous version missed.** It quoted only aggregate \Sz{}
numbers. The full table is a *trade between fields that model error widens by an
order of magnitude*:

| scenario | state variable | psi EV | q EV |
|---|---|---|---|
| S0 | q-state | +0.968 | +0.752 |
| S0 | psi-state | +0.977 | +0.583 |
| S1 | q-state | +0.328 | +0.428 |
| S1 | psi-state | +0.606 | **-2.925** |

At \Sz{} the swap buys `+0.009` on psi and costs `-0.169` on q. At \So{} it buys
`+0.278` and costs `-3.353`. **The representation choice is nearly free when the
model is correct and catastrophic when it is not** -- so H3a and H2 are not
independent axes: the error the representation amplifies via `q ~ Laplacian(psi)`
is exactly the error model discrepancy creates.

**Cross-class corroboration, now stated.** The same effect appears uncontrolled
across the overview table: the variational schemes optimise a psi-observation
cost and carry negative q scores (`-0.126`/`-0.034` at S0, `-0.857`/`-0.501` at
S1), the ensemble filters update in q and hold near `+0.4`, and the learned
schemes -- which reconstruct the target field directly rather than inheriting it
through an inversion -- are best on q (`0.60-0.67`). The observations are
identical across every row, so what varies is the representation, not the
information.

**The honest gap, in answer to "does this need the NN side too?".** Yes, and it
does not exist. All controlled evidence is on the DA side, where the state
variable is an explicit modelling choice. A `\needsrun` now asks for a matched
swap on the learned side (same architecture, data and budget, output space
varied), which would establish whether the effect is a property of DA or of
inverse problems generally.

**One result deliberately not used.** The existing L96 per-channel normalisation
ablation reports that retraining in a normalised space degrades skill badly
(EV `0.857 -> 0.027`). It is cited only as a `\caveat` and explicitly not relied
on: it was measured on a different backbone from this paper's schemes, whose best
configurations *do* normalise, so the two cannot be reconciled without a matched
rerun. Folding it in as NN-side representation evidence would have contradicted
the paper's own configurations.

**Verification:** `pdflatex` x3 -- 15 pages, 0 undefined references or citations;
every `\ref` checked against a matching `\label`, none dangling.
