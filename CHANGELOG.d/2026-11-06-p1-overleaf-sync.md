## 2026-11-06: P1 paper draft synced from Overleaf (QG on the 3D networks)

**Summary:** `docs/papers/p1_structural_hypotheses/` mirrors the Overleaf project as of 2026-11-06: QG table cells on the 3D networks with the climatological EV, QG CFM/SDA/SS cells, QG panels of the budget and section 5.3 figures, QG text of sections 5.1-5.4, design note on the QG EV, QG training details in the appendix, discussion.
**Files modified:** `docs/papers/p1_structural_hypotheses/sections_v4/{04_design,05_results,06_discussion,07_appendix}.tex`, `figures/fig_{budget_s0,budget_s1,fmss,ss_tau,g_tau}.pdf`.
**Rationale:** Keep the repository copy of the paper in step with Overleaf; the QG numbers come from `reports/qg/outputs/qg_p1/` (#362).
**Verification:** `sync-overleaf.sh pull`, then `sync-overleaf.sh status` (no drift); `pytest tests/test_docs_layout.py`.
