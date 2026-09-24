## 2026-09-24: τ-consistency on true marginals (NS1) + sampler cost grid (NS0)

**Summary:** Track A of the τ-consistency plan (v3). A new probe tests the CFM operator on truth-built `x_tau`, i.e. its own training distribution: NS1a per-τ MMSE orthogonality, NS1b the pairwise martingale, and NS1c (added) second-order Tweedie/MMSE consistency. Opt-in NS0 flags on the path probe give an x0=0 path and a steps × members cost grid. Results are for both P1 M-tier flows.
**Files modified:**
- `reports/l96/probe_tau_consistency_truth_marginal.py` — new (NS1a/b/c; `pair_from_tau`).
- `reports/l96/probe_ode_mean_along_path.py` — opt-in `--x0-zero`, `--steps-grid`, `--members-grid`, `--skip-main`; the default output is unchanged.
- `tests/test_tau_pair_construction.py` — checks the law of the pair construction.
- `reports/l96/outputs/cfm_tau_consistency/{ns1_truth_marginal,ns0_cost_grid}_{psc_m,vanilla_m}_s0.json`.
- `docs/results/cfm_tau_consistency_ns1.md`.
**Rationale:** Separates operator inconsistency from ODE-marginal mismatch.
- The first-moment defect is confined to τ ≲ 0.2, and the martingale holds for s ≥ 0.2. So Batch 1 (T2a's teacher range τ' ∈ [0.2, 0.5]) is validated, and T2b is a no-go.
- NS1c shows second-order inconsistency on true marginals, with a model-dependent sign. A per-sample second-order objective is proposed as the Batch-2 candidate instead.
- NS0: one 2-step member beats the 30-draw τ=0 mean, and 3 steps × 8 members is within 2.6–4.7% of the 300-call ensemble mean.
**Verification:** `pytest tests/test_tau_pair_construction.py` — 5 passed; `ruff check` is clean on touched files; all probes ran end to end on both checkpoints (200 windows).
