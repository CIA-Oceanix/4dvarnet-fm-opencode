## 2026-09-23: τ-consistency next-steps plan v3 (controls, shared pair construction, execution plan)

**Summary:** v3 of `docs/scoping/cfm_tau_consistency_next_steps.md`. It adds the T1′ control for T2a, unifies T2a, T2b and NS1b in one x-space pair construction, replaces the mean-slot fallback with input zeroing (T4a), specifies the code changes, and adds an execution plan with parallel tracks.
**Files modified:** `docs/scoping/cfm_tau_consistency_next_steps.md` — v3; `docs/scoping/README.md` — index row.
**Rationale:**
- T2a adds training mass at τ = 0 and also changes the target. T1′ (the same mass, ordinary target) separates the two effects.
- Checking the code turned up two gaps that v2 had assumed away: `LitModel` has no EMA, and `training.seed` defaults to `None`, so the P1 baseline was unseeded.
- Costs are grounded in the measured 36-min A100 run (job 54694).
- Batch 1 does not depend on NS1, so diagnostics, training code and the T0 reseed can start together. NS1 gates only Batch 2.
**Verification:** Docs only; no code changed.
