## 2026-11-06: P1 paper — L96 budget numbers on the 2D rows

**Summary:** Sections 5.1-5.2 now quote the L96 budget (floor 0.019, unattributed 0.062 for the 100-member ETKS, 4D-Var excess 0.22-0.24, learned excesses 0.009 / 0.004 / 0.058) and the L96 learned EV/CRPSS from the 2D-backbone rows, consistent with the tables and budget figures (synced from Overleaf).
**Files modified:** `docs/papers/p1_structural_hypotheses/sections_v4/05_results.tex`.
**Rationale:** The text still used the 1D numbers after the tables (#357) and the budget figures (#364) moved to 2D. Values from the pooled skill table of `reports/l96/outputs/p1_l96_benchmark.md`. Still 1D, under the section 5 `\todo`: L96 spread-skill ratios, the slow/fast split, the prior-structure prose of 5.4, the FMSS and observing-system figures.
**Verification:** `sync-overleaf.sh status` (no drift); `pytest tests/test_docs_layout.py`.
