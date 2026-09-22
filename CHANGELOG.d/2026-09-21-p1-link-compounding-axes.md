## 2026-09-21: Make the compounding link between the two model-error results explicit

**Summary:** §5.3 (marginal-value collapse) now points forward to §5.4
(representation), naming the pattern they share: **model error is the multiplier
on the other structural choices, not one deficit among four.**

**Context -- a merge considered and declined.** The question was whether §5.4
should be folded into §5.3, as three earlier subsections were. It should not, and
the reason is the same test that justified those merges: *does the section own a
measurement?* §5.2, the sensitivity argument and the weak-4D-Var subsection all
failed it -- every number they quoted lived in the overview table. §5.4 passes:
it owns a controlled experiment (bit-identical physics, one variable changed,
free forecasts agreeing to `1e-6`) and its own table, on a different system from
§5.3's.

There is also a structural cost. The paper's organising principle (C5) is that
each deficit attaches to a **named hypothesis**: §5.3 is H2 x S-a, §5.4 is
H3a x H2. Merging puts two hypotheses under one heading, and since the
subsubsection level was deliberately removed, nothing would remain to mark the
distinction -- one section, two tables, two claims.

**What was right about the question.** The two results do share a spine, and that
was visible only once the reader was already inside §5.4. It is now stated where
they first meet it: observation sparsity and model error compound (§5.3), and so
do representation choice and model error (§5.4, the swap cost growing by an order
of magnitude between \Sz{} and \So{}).

**Files modified:** `sections/05_results.tex` -- one paragraph added.

**Verification:** `pdflatex` x3 -- 15 pages, 0 undefined references or citations.
