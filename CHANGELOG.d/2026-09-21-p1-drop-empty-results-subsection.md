## 2026-09-21: Drop the P1 results subsection that carried no result

**Summary:** §5.2 ("The model is the prior: H2's unbuffered cost") is removed.
Its table moves into the overview; its one genuinely new observation is now
stated explicitly. §5 goes from nine subsections to eight.

**Why.** The section had no result of its own. Its table's classical rows
(0.8116/1.4617, 0.8883/1.4998, 0.9131/1.5381) were already in the overview's
robustness table, and its best-classical/best-learned pair (0.8116/1.4617 vs
0.4204/0.4183) was already in the overview's gap table. What remained was four
sentences restating H2's *definition* from §2 -- "the model is the prior, so a
parameter bias is a corrupted prior" -- which is an assertion, not a measurement.
The mechanism for H2 is actually carried by the two sections that follow
(conditioning inertness, and model-vs-posterior sensitivity), both of which
report measurements.

**What was kept, and one finding that was buried in it.** The table itself is
worth keeping -- it is the only per-scheme view of L96, and it carries the
operator-slot mapping -- so it moves into the overview as the detailed view. Its
three intermediate learned rows support a point nothing in the paper was making:
**occupying more operator slots does not by itself buy accuracy.** SDA3 fills all
three and is the weakest learned scheme (0.5365); DirectUNet fills `Psi_mean`
alone and beats it (0.5064); CFM fills all three and beats both (0.4810). The
slots determine what a scheme *can represent*, not how well it estimates -- which
is precisely why §5.7 finds accuracy and calibration traded against each other
rather than improving together. That sentence is now in the text, with the
forward reference.

**Files modified:** `sections/05_results.tex` (section removed, table and finding
folded into §5.1); `sections/02_background.tex` (the forward reference
`Sections~\ref{sec:price}--\ref{sec:marginal}` retargeted to `\ref{sec:inert}`,
since `sec:price` no longer exists).

**Verification:** `pdflatex` x3 -- 14 pages, 0 undefined references or citations;
every `\ref` to a `sec:`/`tab:`/`eq:` label checked against a matching `\label`,
none dangling.
