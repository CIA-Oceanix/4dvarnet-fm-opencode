# τ-consistency of the benchmark-default flows (random observing system, 400 vs 1200 epochs)

**Status:** RESULTS (2026-09-24). This re-runs every τ-consistency probe of `docs/results/cfm_tau_consistency.md`, `cfm_tau_consistency_ns1.md` and `cfm_sampler_schedule.md` on the flows trained under the benchmark-default preset (#242, #247). Those flows report better RMSE/CRPS after longer training.

**Runs probed** (PredictStateCFM-M = PSC, VanillaCFM-M = Van; monai backbone, 1000 training windows):

| tag | trained in | recipe | epochs | seeds |
|---|---|---|---|---|
| `psc400` | `4dvarnet-fm-sda-nanfix/experiments/L96B_predictstatecfm_monaiM_seed{1,2,3}` | L96B: random observing system (`obs_random_layout`, `obs_n_range` [10, 300], `obs_fast_range` [4, 16]), fixed random validation | 400 | 1, 2, 3 |
| `psc1200` | `…/L96B_predictstatecfm_monaiM_ep1200_seed1` | same, with `train_cache` | 1200 | 1 |
| `van400` | `4dvarnet-fm-bench-default/experiments/L96B_vanillacfm_monaiM_seed{1,2,3}` | same | 400 | 1, 2, 3 |
| `van1200` | `4dvarnet-fm-sda-nanfix/…/L96B_vanillacfm_monaiM_ep1200_seed1` | same | 1200 | 1 |
| P1 (reference) | the P1 M-tier checkpoints of the earlier notes | regular observation grid only | 400 | unseeded |

- **Test sets:**
  - `grid`: the regular-grid cache `l96_datasets_obsj2_int100_nwin200.pt`, the set used by every earlier τ-consistency note. All 8 checkpoints.
  - `rlay`: the canonical random-layout test set `l96_testset_rlayout_n10-100_k4-16_w200_d1.pt`. The seed-1 checkpoints only.
- **Protocol:** S0, 30 members, SLURM array 55309 (`batch/`-style script copied to `reports/l96/outputs/cfm_tau_consistency/l96b/probe_array.sbatch`).
- **Probes:** the four `reports/l96/probe_*` scripts on master. Every probe integrates its own **uniform** Euler grid, except where the sampler grid says otherwise, so the numbers are directly comparable with the earlier notes.
- The normalization stats are the same file for every run (md5 checked). Outputs are in `reports/l96/outputs/cfm_tau_consistency/l96b/`.

## 1. Summary table

Regular grid (`grid`); pooled physical RMSE; E1, B4, A1 and NS1 in normalized space. "Limit" is spread/RMSE with 80 uniform Euler steps, i.e. the ODE's own dispersion, which bounds what sampling alone can reach.

| run | `m_hat` RMSE τ=0 / end | τ=0 gap | E1 | B4(0.1)/B4(0.9) | A1 | NS1a τ=0.1 | NS1c τ=0.1 / 0.9 | spread/RMSE uniform N=10 → early-fine N=10 → limit |
|---|---|---|---|---|---|---|---|---|
| P1 PSC | 0.444 / 0.400 | +11% | 0.94 | 2.58 | 0.12 | 0.29 | 0.90 / 0.08 | 0.53 → 0.61 → 0.64 |
| psc400 (3 seeds) | 0.513–0.527 / 0.449–0.462 | +14% | 0.89–0.93 | 1.47–1.49 | 0.20–0.21 | 0.08–0.09 | 0.55–0.59 / 0.07 | 0.70 → 0.84 → 0.94 |
| psc1200 | 0.466 / 0.394 | **+18%** | 0.99 | 1.71 | 0.14 | 0.10 | 0.57 / 0.09 | 0.61 → 0.77 → 0.88 |
| P1 Van | 0.446 / 0.389 | +15% | 1.12 | 3.22 | 0.21 | 0.32 | 1.90 / 1.26 | 0.66 → 0.73 → 0.81 |
| van400 (3 seeds) | 0.558–0.565 / 0.475–0.481 | +17–18% | 1.06–1.14 | 1.91–2.10 | 0.30–0.31 | 0.08–0.13 | 1.45–1.66 / 1.32–1.43 | 0.77 → 0.92 → 1.04–1.07 |
| van1200 | 0.489 / 0.405 | **+21%** | 1.08 | 1.42 | 0.20 | 0.11 | 0.82 / 1.29 | 0.69 → 0.86 → 1.01 |

The random-layout test set (`rlay`, seed 1) reproduces every pattern below.

- τ=0 gap: +15% (psc400) → +18% (psc1200); +16% (van400) → +18% (van1200).
- NS1a at τ = 0.1: 0.06–0.08.
- Limit spread/RMSE: 0.92–0.97.

## 2. Findings

1. **The biased τ=0 prediction (defect 1) does not shrink with longer training; it grows relative to the endpoint.**
   - The E1 slope stays ≈ 1 (0.89–1.14) in every run.
   - The τ=0 gap rises from +14% (psc400) to +18% (psc1200), and from +17% (van400) to +21% (van1200).
   - Longer training improves the ODE endpoint more than the τ=0 prediction. The one-to-two-step self-refinement of `cfm_tau_consistency.md` §2 is a stable property of these flows, and it gets **more** valuable as the model improves.
   - The τ=0 prediction should not be used as a point estimator for any of these models.

2. **The randomized observing system confines the first-moment defect to τ = 0 exactly.**
   - NS1a at τ = 0.1 falls from 0.29–0.32 (P1) to 0.06–0.13 (L96B). From τ ≥ 0.2 it is ≈ 0 everywhere, as before.
   - The operator is first-moment consistent on its own training distribution everywhere except the τ=0 point itself. Longer training does not change this.

3. **With these models, the variance collapse (defect 2) is mostly a sampler effect, not an operator one.**
   - The ODE-limit dispersion (80 uniform steps) rises from 0.64 / 0.81 (P1 PSC / Van) to **0.93–0.95 / 1.04–1.07** (L96B, 400 epochs).
   - The operator's share of the under-dispersion, which dominated P1 (`cfm_sampler_schedule.md` §3), is almost gone after training on the random observing system.
   - What remains at the standard N = 10 is Euler discretisation: spread/RMSE 0.70 (PSC) / 0.77 (Van).
   - B4 decay drops accordingly, from 2.6–3.2× to 1.4–2.1×.

4. **Longer training takes some dispersion back.**
   - psc1200's ODE limit is 0.88 (vs 0.93–0.95 at 400 epochs), and its uniform-N=10 spread/RMSE is 0.61 (vs 0.69–0.70). Van shows the same direction, smaller (limit 1.01 vs 1.04–1.07).
   - This matches the benchmark evaluation (spread/RMSE 0.50 → 0.43 for PSC). The accuracy gain of 1200 epochs comes with a sharper, slightly over-confident ensemble.
   - B4 moves the same way for PSC (1.47 → 1.71). For Van it improves (≈ 2.0 → 1.42), together with Van's early-τ Jacobian over-statement (NS1c at τ = 0.1: 1.45–1.66 → 0.82).

5. **The late-τ Jacobian defect of PSC is structural.**
   - NS1c at τ = 0.9 is 0.07–0.09 in every PSC run: P1, 400 and 1200 epochs, both test sets. Its Jacobian is ≈ 12× too small late in the path regardless of data or training length.
   - It does not prevent near-calibration at the ODE limit, so it is not the main cause of under-dispersion for these models.
   - Van instead over-states the Jacobian late (≈ 1.3–1.4) and ends slightly over-dispersed at the limit (1.01–1.07).

6. **The early-fine sampler helps even more on these models.**

   | run | CRPS uniform N=10 | early-fine N=10 | early-fine N=20 | uniform N=80 |
   |---|---|---|---|---|
   | psc400 (3 seeds) | 0.1830–0.1899 | −4.0% to −4.7% | −4.5% to −5.2% | −0.9% to +0.2% |
   | psc1200 | 0.1548 | −5.4% | **−5.9%** | −1.0% |
   | van400 (3 seeds) | 0.1920–0.1995 | −1.5% to −3.0% | −1.5% to −3.9% | −1.6% to −4.2% |
   | van1200 | 0.1528 | −3.1% | −4.3% | −4.4% |

   - RMSE also improves slightly in every early-fine row.
   - **Early-fine N = 20 beats uniform N = 80 on CRPS for PSC, at a quarter of the cost, and ties it for Van.** Its spread/RMSE is 0.86–0.92 (PSC) and 0.95–1.02 (Van).
   - The same holds on `rlay`.

7. **A1 (the τ=0 prediction's dependence on x0) is larger under L96B** (0.20–0.31 vs 0.12–0.21 for P1) and shrinks with longer training (0.14 / 0.20). C1 asymmetry is unchanged (0.37–0.42 at τ = 0.9).

## 3. What this changes

- **For the benchmark:** the flows' calibration gap at N = 10 is now mostly a sampler budget question.
  - The early-fine default (#253) closes most of it.
  - **N = 20 with p = 0.5** would bring PSC to 0.86–0.92 and Van to 0.95–1.02 spread/RMSE, with the best CRPS in every row, at twice today's cost.
  - Worth proposing as the benchmark sampler (a decision; not changed here).
- **For the τ-consistency plan (v4):**
  - The operator share of defect 2 that T5 targeted is small for the L96B models, so a corrected T5 has even less to gain.
  - The remaining operator-side items are PSC's late-τ Jacobian (structural, but not limiting) and the slight dispersion loss from longer training.
- **For the papers:** the self-refinement mechanism (defect 1) is robust across recipe and training length, which supports making it a P2 claim. The P1 calibration story becomes: under-dispersion was mostly a sampler artefact once training covers the observing-system variability.

## Caveats

- **Seeds:** one seed at 1200 epochs per model. The 400-epoch band is 3 seeds and is narrow for every probe metric (see the ranges above). The 1200-vs-400 differences in the gap, A1, NS1c at τ=0.1 and the limit spread lie outside that band; the B4 difference for PSC (1.71 vs 1.47–1.49) is also outside it.
- **Test sets:** these models were trained on the random observing system. The regular-grid set is inside their training range (it contains the full 16-channel, 30-time layout), but it is not their benchmark test set. The `rlay` rows are the in-distribution check, and they agree.
- **Units:** the sampler grid's spread/RMSE and CRPS are pooled physical; the benchmark report's are per-window. Compare within this note.
