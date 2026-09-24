## 2026-09-24: τ-consistency probes on the benchmark-default (L96B) flows, 400 vs 1200 epochs

**Summary:** Runs every τ-consistency probe (the ODE path, E1/B2/B4/A1/C1, truth-marginal NS1a–c, and the sampler grid) on the L96B PredictStateCFM-M and VanillaCFM-M checkpoints: 400 epochs (3 seeds) and 1200 epochs (seed 1), on the regular-grid and random-layout test sets.
**Files modified:**
- `docs/results/cfm_tau_consistency_l96b.md`: new results note.
- `reports/l96/outputs/cfm_tau_consistency/l96b/*`: 48 probe JSONs plus the array script.
- `docs/scoping/README.md`: pointer.

**Rationale:** Another session found that the random observing system plus longer training improves RMSE/CRPS. This checks whether that changes the τ-consistency diagnosis:
- The τ=0 bias persists (E1 ≈ 1) and grows relative to the endpoint with longer training (+14% → +18% PSC, +17% → +21% Van).
- First-moment consistency now holds from τ = 0.1.
- The variance collapse is mostly sampler-side: the ODE-limit spread/RMSE is 0.93–1.07 vs 0.64–0.81 for P1. Longer training takes a little dispersion back.
- PSC's late-τ Jacobian defect is structural.
- Early-fine N = 20 beats uniform N = 80 on CRPS for PSC, at a quarter of the cost.

**Verification:** Docs and outputs only; no code changed. SLURM array 55309, 12 tasks, all completed.
