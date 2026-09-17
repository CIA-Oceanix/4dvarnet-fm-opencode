## 2026-09-17: Scoping doc absorbs the FM-prior sampler results (#211)

**Summary:** Revised the ML-venue scoping doc to absorb PR #211: closed §10
decision 4 (velocity fields for attribution, posterior scores as outcome —
not either/or), narrowed and re-pointed risk R6, added the orthogonality
condition §2.2's additivity actually rests on, and recorded the measured
calibration results in §7.

**Files modified:** `docs/scoping_ml_paper_exploiting_prior_knowledge.md` —
§2.2 additivity condition (forces L², and `Ψ_G` redefined as the L² projection
rather than the closed-form gain); §5 E1 now required to report two axes; §6 R6
rewritten; §7 four new measured bullets; §8 P0 gains the `Γ`×`N_outer` sweep;
§10 decision 4 decided.

**Rationale:** R6's "calibration half not yet readable" blocker is cleared by
#211's uniform N=30 spread/RMSE, and the decoupled-control negative result
supplies the evidence that settles decision 4. Three corrections surfaced while
checking the numbers: R6's "54% / 8.5%" were two differently-normalized relative
reductions, not shares of one gap (correctly ~93% / ~7%); the affine-decomposition
doc is on `master` since #211, not on a feature branch as cited; and the source
report's "beats Strong-4DVar by 31%" is hardcoded prose reconciling with no
pairing in its own tables (34.8% `rmse_repo` / 29.9% `rmse_pooled`).

**Verification:** Docs-only; no code touched. Every edit applied through a
checked replacement script that asserts each anchor matches exactly once
(7 + 4 anchors, all matched). Arithmetic recomputed from
`reports/l96/outputs/l96_fm_sampler_benchmark.md` and
`reports/l96/outputs/l96_consolidated_benchmark.md`.
