## 2026-09-24: τ-consistency Batch 1 results (T0, T1′, T2a): negative

**Summary:** Records the Batch-1 runs of `docs/scoping/cfm_tau_consistency_next_steps.md` v3:
- T0: reseeded PredictStateCFM-M.
- T1′: 25% of each batch at τ=0.
- T2a: the same 25%, with EMA-teacher targets.

Both arms are worse than their paired T0 (ensemble-mean RMSE +19% / +10%, spread/RMSE 0.26 / 0.25 vs 0.38). Their τ=0 predictions get worse, not better.
**Files modified:**
- `docs/results/cfm_tau_consistency_batch1.md`: new results note.
- `reports/l96/outputs/cfm_tau_consistency/batch1/*.json`: the `ens30_no10` evaluations and both probes per run.

**Rationale:**
- Adding training mass at τ=0 makes the network memorise each training window's `x1` (T1′'s training loss falls well below the posterior variance while its validation loss stalls).
- The teacher target helps directionally (T2a beats T1′ by 7.5% RMSE) but cannot compensate.
- P1 vs T0 (identical configs) differ by 10.6%, so seed noise dominates single-run comparisons.
- Proposes a "replace, don't add" T2a′, retiring `tau0_frac`, and ≥ 2 seeds per arm. Nothing is launched.

**Verification:** Docs and JSON outputs only; no code changed.
