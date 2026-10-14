## 2026-10-14: L96 FDV2 x FM posterior — results note, plan closed after Phases 0-1

**Summary:** Adds `docs/results/l96_fdv2_fm_posterior.md`, which covers the validation sweeps and the held-out test of the training-free FM-posterior candidates against the FDV2->SDA hybrid and B5. Closes `docs/plans/analysis/l96_fdv2_fm_posterior.md` after Phases 0-1; Phase 2 was not run.
**Files modified:**
- `docs/results/l96_fdv2_fm_posterior.md`: new.
- `docs/plans/analysis/l96_fdv2_fm_posterior.md`: Status set to CLOSED.
- `docs/README.md`: the plan row and a new findings row.
**Rationale:** M1bI is perturbed-obs FDV2 starts at 0.75x the obs noise, a DPS stochastic FM sampler from tau0 = 0.7, and a val-tuned inflation. On 200 test windows per cell it has CRPS -4.2..-7.3% vs the hybrid's CRPS-best setting and -3.8..-6.9% vs B5, and it is calibrated in all 4 cells. RMSE ties the hybrid's best-RMSE setting on the random layout but is +0.6/+0.8% on the regular layout, so S-acc misses there.
**Verification:** `pytest tests/test_docs_layout.py`.
