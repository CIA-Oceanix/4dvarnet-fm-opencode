# S1 — sampler-side test of the variance collapse (no retraining)

**Status:** RESULTS (2026-09-24). Step S1 of `docs/plans/analysis/l96_cfm_tau_consistency_next_steps.md` v4.

**Numbers:** `reports/l96/probe_stochastic_sampler.py` → `reports/l96/outputs/cfm_tau_consistency/sampler/`.

- Checkpoints: the P1 M-tier monai flows, PredictStateCFM-M (PSC) and VanillaCFM-M (Vanilla), archived under `4dvarnet-fm-fdv-tau-aware/experiments/`.
- Data: L96 S0, 200 cached test windows, 30 members.
- Script: `reports/l96/probe_stochastic_sampler.py`. Outputs: `reports/l96/outputs/cfm_tau_consistency/sampler/`.
- Metrics are pooled over windows × time × channels, in physical units. The CRPS is the fair ensemble CRPS per coordinate. These numbers are **not** the P1 table's per-window convention; compare rows within this doc only.

**Question.** v4 §2 splits the collapse (B4 decays ≈ 2.7×; spread/RMSE ≈ 0.4–0.55) into an operator part and a sampler part. How much of it can sampling alone fix?

**Answer.**
- **Stochasticity makes it worse.**
- **Step count helps, and saturates.**
- **Putting the fine steps early is a free win at the same cost:** CRPS −4.4% (PSC) and −2.0% (Vanilla), RMSE better, spread/RMSE +15% / +11%.
- What remains after saturation (spread/RMSE ≈ 0.65 PSC, ≈ 0.81 Vanilla) is the operator's share, so T5 stays needed, especially for PSC.

## The samplers

The DDIM-η family on the linear path. One step s → τ:

  `x_tau = tau D + sqrt(b_tau² − sigma²)·eps_hat + sigma·xi`, with `eps_hat = (x_s − s D)/b_s` and `sigma = eta·sqrt(v)`.

- η = 0 is exactly the current Euler step; η = 1 is ancestral. Every η preserves the path marginal for an exact `D` (`tests/test_stochastic_sampler.py`, 17 tests).
- Step grid: `tau_k = 1 − (1 − k/N)^p`.
  - p = 1 is uniform (the current sampler at N = 10).
  - p > 1 concentrates steps near τ = 1.
  - p < 1 concentrates steps near τ = 0.

## 1. Stochasticity (η), uniform steps

| N | η | PSC RMSE | PSC CRPS | PSC spread/RMSE | Vanilla RMSE | Vanilla CRPS | Vanilla spread/RMSE |
|---|---|---|---|---|---|---|---|
| 10 | 0 (current) | 0.3997 | 0.1704 | 0.530 | 0.3886 | 0.1531 | 0.660 |
| 10 | 0.5 | 0.3993 | 0.1713 | 0.516 | 0.3870 | 0.1528 | 0.639 |
| 10 | 1 | 0.4006 | 0.1763 | 0.450 | 0.3845 | 0.1545 | 0.543 |
| 20 | 0 | 0.4001 | 0.1691 | 0.593 | 0.3862 | 0.1496 | 0.743 |
| 20 | 1 | 0.4053 | 0.1773 | 0.509 | 0.3828 | 0.1504 | 0.629 |

**Noise re-injection lowers the spread** on both models. An ancestral step discards the carried noise and re-draws it, so the endpoint depends more on the late-τ predictions. Those under-state the posterior variance: NS1c (`cfm_tau_consistency_ns1.md`) found PSC's late Jacobian 12× too small. The deterministic ODE instead carries the initial noise through `eps_hat`. **Rejected** as a fix.

## 2. Step count and where the steps go (η = 0)

| schedule | N | PSC RMSE | PSC CRPS | PSC spread/RMSE | Vanilla RMSE | Vanilla CRPS | Vanilla spread/RMSE |
|---|---|---|---|---|---|---|---|
| uniform (p = 1) | 10 | 0.3997 | 0.1704 | 0.530 | 0.3886 | 0.1531 | 0.660 |
| uniform | 20 | 0.4001 | 0.1691 | 0.593 | 0.3862 | 0.1496 | 0.743 |
| uniform | 40 | 0.4017 | 0.1697 | 0.625 | 0.3853 | 0.1482 | 0.789 |
| uniform | 80 | 0.4043 | 0.1714 | 0.637 | 0.3851 | 0.1475 | 0.814 |
| late (p = 2) | 10 | 0.4098 | 0.1854 | 0.397 | 0.3944 | 0.1603 | 0.542 |
| late (p = 2) | 40 | 0.4143 | 0.1817 | 0.559 | 0.3863 | 0.1495 | 0.748 |
| late (p = 3) | 10 | 0.4178 | 0.1970 | 0.321 | 0.3997 | 0.1668 | 0.475 |
| **early (p = 0.5)** | **10** | **0.3951** | **0.1629** | **0.612** | **0.3863** | **0.1501** | **0.732** |
| **early (p = 0.5)** | **20** | **0.3949** | **0.1622** | **0.650** | 0.3850 | 0.1484 | 0.781 |
| early (p = 0.7) | 10 | 0.3969 | 0.1659 | 0.579 | 0.3872 | 0.1512 | 0.704 |
| early (p = 0.7) | 40 | 0.3979 | 0.1656 | 0.649 | 0.3850 | 0.1478 | 0.802 |

**Readings:**
1. **Euler discretisation is part of the collapse.** With uniform steps, spread/RMSE rises with N and **saturates**: PSC ≈ 0.64, Vanilla ≈ 0.81. The ensemble-mean RMSE barely moves (±1%). The step-limit value is the ODE's own spread; the gap from it to 1 is the **operator's share**.
2. **The variance is lost early in the path.** Coarse early steps (p > 1) are clearly worse, and fine early steps (p < 1) clearly better. This matches the B4 decay over τ ∈ [0.1, 0.4] and the NS1a/NS1c defects at low τ. It is the opposite of the late-concentrated schedule suggested in `cfm_affine_velocity_decomposition.md`, which was motivated by the first-moment non-affine residual, not by variance.
3. **Early-fine is a free win.** At the current cost (N = 10), p = 0.5 improves every metric on both models:
   - PSC: RMSE −1.2%, CRPS −4.4%, spread/RMSE 0.53 → 0.61.
   - Vanilla: RMSE −0.6%, CRPS −2.0%, spread/RMSE 0.66 → 0.73.
   - For PSC it beats uniform steps at every N up to 80. For Vanilla, N = 10 at p = 0.5 matches uniform N = 20 at half the cost.
4. **PSC has the larger operator share** (a limit of ≈ 0.65 vs ≈ 0.81), consistent with NS1c's under-stated late Jacobian.

## 3. Decision (v4 §3 S1 rule)

This is the **"mixed" outcome**: spread/RMSE rises but stays well below 1, and RMSE is not hurt.

- **Adopt early-fine steps (p = 0.5) as the candidate evaluation sampler.** Use it for T5 (next to the P1 Euler protocol, for comparability) and propose it for the P1 benchmark (v4 §6, decision 1).
- **T5 (second-order loss) stays the main lever** for the remaining operator share, especially for PSC.
- A smaller p (≤ 0.5) and N = 5 are cheap follow-ups that have not been tried.

## Caveats

- One checkpoint per model, S0 only, one member seed.
- Seed noise between identical training configs is ≈ 10% (`cfm_tau_consistency_batch1.md`). The sampler comparison is **within** one checkpoint, with the same members' starting noise across schedules at a given N. That removes training-seed noise from these differences, but a schedule's effect could still differ on other checkpoints.
- Spread/RMSE uses the pooled RMSE of the ensemble mean. It is a calibration indicator, not a proper score; the CRPS column is the proper score.
