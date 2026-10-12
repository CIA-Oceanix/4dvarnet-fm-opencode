## 2026-10-12: Plan — FDV2 with a score-based (denoising-score-matching) prior

**Summary:** Adds an implementation plan for training FDV2's prior Φ as a noise-conditional denoiser, so that one model provides the estimate, posterior samples and an error indicator. The plan covers:
- the denoiser parameterisation (EDM);
- the Tweedie-formula score;
- x − Φ(x, σ₀) as a scaled 4D-Var prior gradient;
- guided annealed sampling (DPS or ΠGDM) warm-started from the solver estimate.

It also records the 2026-10-12 uncertainty probes: Langevin on the current reconstruction prior and likelihood-weighted perturbed-observation ensembles were both negative.
**Files modified:** `docs/plans/analysis/l96_fdv2_score_prior.md` — the plan; `docs/README.md` — index row.
**Rationale:** The current Φ is a reconstruction prior with no density, which is why the uncertainty probes failed. A denoising-score-matching Φ makes the prior a proper score while keeping FDV2's solver input interpretable as a variational gradient.
**Verification:** `pytest tests/test_docs_layout.py`.
