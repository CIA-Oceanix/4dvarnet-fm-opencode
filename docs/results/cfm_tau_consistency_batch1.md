# τ-consistency Batch 1 (T0, T1′, T2a) — results

**Status:** RESULTS (2026-09-24). **Negative** for both training arms as designed.

- Plan: `docs/scoping/cfm_tau_consistency_next_steps.md` v3. Training code: #249. Diagnostics: `docs/results/cfm_tau_consistency.md` and `docs/results/cfm_tau_consistency_ns1.md` (#250).
- All numbers are L96 S0, 200 cached test windows, PredictStateCFM-M, P1 recipe.
- One run per arm, all with `training.seed: 1`, so each arm is paired with the T0 reseed on initialisation and batch order.

## Runs

| tag | config | job | train time | best val epoch |
|---|---|---|---|---|
| P1 | `A2_predictstatecfm_monaiM_l96` (unseeded, the P1 benchmark run) | — | — | — |
| T0 | same, `++training.seed=1` | 55119 | 39.5 min | 290 |
| T1′ | `A2_predictstatecfm_monaiM_tau0frac_l96` (25% of each batch at τ=0, target `x1`) | 55126 | 38.9 min | 213 |
| T2a | `A2_predictstatecfm_monaiM_boot0_l96` (the same 25%, target `sg D_EMA(x_tau', tau')`, τ' ~ U[0.2, 0.5], teacher from epoch 50) | 55127 | 43.7 min | 258 |

- T2a's logged `teacher_active` is 0 for epochs 0–49 and 1 from epoch 50, as designed.
- Checkpoint = `stage1_best.ckpt` (best `val_loss`). `val_loss` is the plain uniform-τ loss for every arm (#249).
- Checkpoints are **not archived**. They live in the `4dvarnet-fm-tau-t0` / `4dvarnet-fm-tau-runs` worktrees under `experiments/A2_predictstatecfm_monaiM_{l96,tau0frac_l96,boot0_l96}_seed1/`.

## 1. Training curves: the τ=0 mass overfits

| run | best val loss | final val loss | final train loss |
|---|---|---|---|
| T0 | 0.0082 | 0.0087 | 0.0098 |
| T1′ | 0.0131 | 0.0220 | 0.0114 |
| T2a | 0.0128 | 0.0222 | 0.0094 |

- **T1′ memorises at τ=0.** Its validation loss stops improving at about epoch 200 while its training loss keeps falling. A quarter of T1′'s training rows sit at τ=0, where the irreducible loss is about the posterior variance (P ≈ 0.05–0.09, normalised). A training loss of 0.011 therefore means the network fits each training window's own `x1` at τ=0 instead of `E[x1|y]`. With 1000 windows and one `x1` each, τ=0 is a plain regression that can memorise.
- This is also the most likely reason the τ=0-only control (#239, 0.489) is as poor as DirectUNet.
- **T2a's teacher targets reduce the memorisation but do not remove it.**

## 2. Ensemble evaluation (`ens30_no10`, P1 protocol)

`eval_neural_l96.py` metrics, all_obs group (in `batch1/ens30_*.json`):

| run | ensemble-mean RMSE | Energy Score | spread/RMSE |
|---|---|---|---|
| P1 | 0.377 | 0.173 | 0.37 |
| T0 | 0.417 | 0.199 | 0.38 |
| T1′ | 0.496 | 0.258 | 0.26 |
| T2a | 0.459 | 0.234 | 0.25 |

- **Seed noise is large.** P1 and T0 have identical configs and differ by **10.6%**. That is consistent with the 15–21% checkpoint-selection noise P1 measured on this family. Single-run differences below that level are not interpretable.
- **Both arms are worse than their paired T0**: T1′ by 19%, T2a by 10%. Both are more under-dispersed. This violates the guard-rail in plan §4.
- **At equal τ=0 mass, T2a beats T1′** (RMSE −7.5%, ES −9%). The teacher target is directionally better than `x1`, as the target-variance argument predicts. But the paired difference is one run, below the seed-noise level.

## 3. τ-consistency probes

These use the probe conventions of `docs/results/cfm_tau_consistency.md`: pooled physical RMSE for the per-step predictions, and normalised space for E1/B4/A1. The P1 row is that doc's number.

| run | RMSE of `m_hat` at τ = 0 / 0.1 / 0.3 / endpoint | τ=0 gap vs endpoint | E1 slope | B4(0.1)/B4(0.9) | A1 |
|---|---|---|---|---|---|
| P1 | 0.444 / 0.409 / 0.396 / 0.400 | +11.1% | 0.94 | 2.58 | 0.121 |
| T0 | 0.476 / 0.443 / 0.433 / 0.444 | +7.3% | 0.79 | 2.68 | 0.139 |
| T1′ | 0.508 / 0.497 / 0.488 / 0.513 | −0.9% | 0.47 | 3.73 | 0.075 |
| T2a | 0.493 / 0.485 / 0.470 / 0.481 | +2.4% | 0.63 | 3.36 | 0.061 |

**Readings:**
1. **The consistency metrics "improve" for the wrong reason.** In both arms the E1 slope, the τ=0 gap and A1 fall. But the τ=0 prediction gets **worse** in absolute terms (0.476 → 0.508 / 0.493), and so does every later τ. The gap closes because the whole operator degrades toward the τ=0 prediction's level, not because the τ=0 prediction improves. **Success criterion 1 (τ=0 RMSE closes ≥ 50% of the gap) fails.**
2. **Defect 2 gets worse.** B4 decays 3.4–3.7× instead of 2.7×, matching the lower spread/RMSE in §2.
3. **The endpoint can be worse than mid-path.** In every seed-1 run the member-averaged prediction is best at τ ≈ 0.3–0.4 and degrades toward the endpoint: +2.5% for T0, +5% for T1′. The P1 run shows it only marginally. Using `m_hat` at mid-τ as the point estimator is a free, if small, gain to note.
4. **T2a vs T1′** favours the teacher target on every probe metric. This is consistent with §2 and has the same single-run caveat.

## 4. Conclusion and what it changes

- **Adding training mass at τ=0 is harmful at this data size (1000 windows).** It turns the τ=0 prediction into a memorising regressor. The uniform-τ network's τ=0 bias measured earlier is partly a *regularised* compromise, not only a defect.
- **The bootstrap target is directionally useful** (T2a > T1′), but it cannot pay for the extra τ=0 mass.
- Implications for the plan (to be decided; nothing launched):
  - **T2a′ ("replace, don't add").** Put teacher targets only on the rows the uniform draw already places at τ ∈ [0, 0.1], with no extra τ=0 mass. That low-τ range is also where NS1 locates the residual first-moment defect. It needs a small code change: a mode where `boot_*` relabels natural-τ rows instead of adding rows.
  - **T1′ and `tau0_frac`: retire.** T4a (`tau0_zero_input`) is built on T1′'s mass, so it would need re-basing on uniform τ before it is worth running.
  - **Seeds: at least 2 per arm from now on**, given the 10.6% seed gap.
  - **Batch 2: unchanged no-go for T2b** (NS1). Defect 2 needs the second-order objective proposed in `cfm_tau_consistency_ns1.md`.

## Caveats

- One run per arm; S0 only.
- `val_loss`-based checkpoint selection may interact with the overfitting: the arms' best epochs (213, 258) are earlier than T0's (290).
- `P_from_truth` rises in the arms (0.066 → 0.088 / 0.078) simply because their ensemble mean is worse. It is an upper bound on the posterior variance, as before.
