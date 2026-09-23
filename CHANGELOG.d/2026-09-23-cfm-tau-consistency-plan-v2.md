## 2026-09-23: τ-consistency next-steps plan v2 (tower-property derivation)

**Summary:** Rewrites the draft plan in `docs/scoping/cfm_tau_consistency_next_steps.md` from the tower property. A truth-marginal consistency test (NS1) now comes before any training, the bootstrapped-target arm is split into a τ=0-only T2a and a general T2b, and a structural mean-slot fallback (T4) is added.
**Files modified:** `docs/scoping/cfm_tau_consistency_next_steps.md` — v2; `docs/scoping/README.md` — index row.
**Rationale:** The v1 probes evaluated B1/B4 on the ODE's own states, so a violation could not be attributed to the operator versus the sampler's distribution. NS1 tests per-τ orthogonality and the pairwise martingale on `x_tau` built from the test-set truth, which removes that ambiguity. With one `x1` per window, only the pairwise (martingale) form of the tower property can be trained on. That rules out a per-`y` tower penalty and motivates T2a: a lower-variance τ=0 target (≈ 3.5× at τ' = 0.2), unbiased when the teacher is exact.
**Verification:** Docs only; no code changed.
