## 2026-09-25: τ-consistency synthesis note; plan closed

**Summary:** Adds `docs/results/cfm_tau_consistency_synthesis.md`, the single citable reference that closes the τ-consistency line. It supersedes, for citation, the seven results notes (#245, #250, #251, #252, #255, #256, #260), and marks `docs/scoping/cfm_tau_consistency_next_steps.md` as CLOSED.
**Files modified:**
- `docs/results/cfm_tau_consistency_synthesis.md`: new.
- `docs/scoping/cfm_tau_consistency_next_steps.md`: status line.
- `docs/scoping/README.md`: index row.

**Rationale:** The findings were spread over seven notes whose conclusions evolved. The papers need one stable source covering:
- the self-refinement mechanism (P2);
- sampler-dominated under-dispersion, fixed by the early-fine default (P1);
- the one identified model failure, PredictStateCFM's late-τ Jacobian, with its proposed test;
- the negatives;
- the C4 result.

**Also:** `tests/test_t5_variance_and_sampler.py::test_variance_terms_match_an_autograd_jvp` now runs in float64. The float32 finite-difference check was hardware-dependent (1.4% on the CI runner vs 7e-5 locally; the CI CPU runs float32 convolutions at reduced precision) and blocked this docs-only PR.
**Verification:** Docs, plus that one test (12 passed locally). Every number was cross-checked against the source notes' tables and JSON outputs.
