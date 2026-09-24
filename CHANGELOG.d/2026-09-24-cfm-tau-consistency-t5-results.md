## 2026-09-24: T5 (second-order variance consistency loss) results: negative, cause identified

**Summary:** Records T5 (λ ∈ {10, 100} × seeds {1, 2}) and the T0b reseed, evaluated under both the uniform and the early-fine sampler, plus truth-marginal NS1 probes.
- λ = 10 is inside the baseline noise band with no calibration gain.
- λ = 100 diverged at switch-on, so its rows are pre-switch checkpoints.
- The early-fine sampler improves ES by 3.8–6.4% on all seven checkpoints.

**Files modified:**
- `docs/results/cfm_tau_consistency_t5.md`: new results note.
- `reports/l96/outputs/cfm_tau_consistency/t5/*`: eval JSONs, NS1 JSONs, training metrics.
- `docs/scoping/cfm_tau_consistency_next_steps.md`: T5 outcome note.
- `docs/scoping/README.md`: index row.

**Rationale:** The single-Rademacher-probe squared loss has expectation `(a − t)² + Var_u(â)`. The probe variance grows with the off-diagonal Jacobian, so the loss rewards shrinking the whole Jacobian: NS1c fell ≈ 10–15% at every τ, and the late-τ collapse was untouched. Specifies a corrected T5 (two-probe unbiased product, λ ramp, optional autograd JVP); nothing is launched.
**Verification:** Docs and outputs only; no code changed.
