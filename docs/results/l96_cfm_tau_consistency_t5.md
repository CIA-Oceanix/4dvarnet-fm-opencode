# T5 — second-order (variance) consistency loss: results

**Status:** RESULTS (2026-09-24). **Negative as implemented; the cause is identified.** Step T5 of `docs/plans/analysis/l96_cfm_tau_consistency_next_steps.md` v4.

- Code and configs: #253 (`var_*` options on `PredictStateCFM`, `var_start_epoch`, `val_var_ratio`).
- Runs: SLURM array 55263, launched from a frozen worktree at `d78bd60`. Follow-ups: 55280, 55288.
- Protocol: L96 S0, 200 cached test windows, PredictStateCFM-M on the P1 recipe, `ens30_no10` evaluation under both the uniform Euler grid (p = 1, P1-comparable) and the early-fine default (p = 0.5).
- Outputs: `reports/l96/outputs/cfm_tau_consistency/t5/`. Checkpoints are not archived; they are in the `4dvarnet-fm-tau-t5-runs` worktree under `experiments/`.

## 1. Results

Ensemble-mean RMSE, Energy Score and spread/RMSE (all_obs, `eval_neural_l96.py` conventions):

| run | seed | p = 1: RMSE / ES / sp/RMSE | p = 0.5: RMSE / ES / sp/RMSE |
|---|---|---|---|
| P1 baseline | unseeded | 0.3774 / 0.1733 / 0.37 | 0.3727 / 0.1661 / 0.44 |
| T0 baseline | 1 | 0.4173 / 0.1990 / 0.38 | 0.4093 / 0.1863 / 0.46 |
| T0b baseline | 2 | 0.3840 / 0.1774 / 0.38 | 0.3802 / 0.1707 / 0.45 |
| T5 λ = 10 | 1 | 0.4313 / 0.2066 / 0.37 | 0.4228 / 0.1953 / 0.43 |
| T5 λ = 10 | 2 | 0.3886 / 0.1793 / 0.37 | 0.3838 / 0.1723 / 0.43 |
| T5 λ = 100 | 1 | 0.6094 / 0.2991 / 0.44 | 0.5953 / 0.2833 / 0.51 |
| T5 λ = 100 | 2 | 0.5975 / 0.2909 / 0.45 | 0.5814 / 0.2736 / 0.52 |

**Baseline noise band (3 runs):** RMSE 0.377–0.417, ES 0.173–0.199.

**Paired by seed:**
- λ = 10 is +3.4% / +3.8% (RMSE / ES) vs T0 at seed 1, and +1.2% / +1.1% vs T0b at seed 2.
- Its spread/RMSE is unchanged (0.37 vs 0.38).
- λ = 10 is therefore **inside the noise band on accuracy, slightly worse in both pairs, and does nothing for calibration**, which was its purpose.

## 2. λ = 100 diverged at switch-on

| epoch | 49 | 50 | 51 | 55 | 100 | 399 |
|---|---|---|---|---|---|---|
| train loss | 0.027 | 1.9e6 | 1.4e8 | 4.5e10 | 1.1e14 | 2.6e17 |
| val loss | 0.031 | 115 | 594 | 7.0e3 | 3.1e5 | — |

- The term switches on at epoch 50 (`var_start_epoch`) and the run diverges immediately, despite gradient clipping at 10.
- **Checkpoint selection by best `val_loss` kept an epoch-≤49 model.** So the λ = 100 rows measure an under-trained network that never used the loss, not the loss itself.
- **Likely causes:**
  - an abrupt switch-on at a large weight;
  - a finite-difference objective, whose `J u` the optimiser can exploit at the ε scale;
  - the objective bias in §3, which rewards moving the Jacobian freely.

## 3. Why λ = 10 moves the Jacobian the wrong way

`val_var_ratio` (pooled Jacobian term over residual, target 1) was 1.13 at epoch 49. After switch-on it drops to ≈ 0.81–0.89 and stays there. The truth-marginal NS1c ratio (the #250 probe, per τ) shows the same shift on both seeds:

| run | τ = 0.1 | 0.3 | 0.5 | 0.7 | 0.9 |
|---|---|---|---|---|---|
| T0b (baseline) | 0.91 | 1.24 | 1.14 | 0.57 | 0.07 |
| T5 λ = 10, seed 1 | 0.80 | 1.06 | 0.98 | 0.49 | 0.06 |
| T5 λ = 10, seed 2 | 0.75 | 1.05 | 1.04 | 0.56 | 0.07 |

- λ = 10 **scaled the Jacobian term down by ≈ 10–15% at every τ**.
- It **left the real defect untouched**: the late-τ collapse (≈ 0.07 at τ = 0.9, 12× too small) is identical.

**Cause: the objective is biased.**
- T5 used one Rademacher probe `u` per sample: `L_var = (â − t)²`, with `â` the Hutchinson estimate of the diagonal Jacobian term.
- Its expectation over `u` is `(a − t)² + Var_u(â)`.
- The probe variance `Var_u(â)` grows with the **off-diagonal** Jacobian entries, i.e. the couplings across time steps and channels, which are large for a trajectory-level UNet.
- So the loss rewards **shrinking the whole Jacobian**, which lowers `a` as well, and for PredictStateCFM that is exactly the wrong direction.
- The bias also explains why a larger λ is unstable: the loss is minimised partly by driving the Jacobian structure around, not only by matching `a` to `t`.

The first-moment consistency (NS1a) is essentially unchanged: the τ=0 slope is 0.76 / 0.95 vs 0.99 for T0b, and ≈ 0 from τ = 0.2.

## 4. The early-fine sampler replicates

All seven evaluated checkpoints improve under p = 0.5 at the same N = 10:
- **ES −3.8% to −6.4%**;
- **RMSE −1.0% to −2.3%**;
- **spread/RMSE +0.06 to +0.08.**

This confirms `docs/results/l96_cfm_sampler_schedule.md`, which rested on one checkpoint per model, and supports the default adopted in #253.

## 5. What a corrected T5 would need (not launched)

1. **An unbiased objective.** Use two independent probes, `(â₁ − t)(â₂ − t)`, whose expectation is `(a − t)²` with no probe-variance term. Or use the exact diagonal, which is expensive at T × D = 72k.
2. **A ramp instead of a switch.** Take λ linearly from 0 to its target over about 50 epochs.
3. **Optionally an autograd JVP** (double backward) instead of finite differences, so the optimiser cannot exploit the ε-scale structure.
4. **The same protocol:** 2 values of λ (e.g. 10, 30) × 2 seeds against the T0/T0b pair, judged on spread/RMSE and ES first.

**Honest prior.**
- The per-channel identity is only a *diagonal, time-averaged* constraint. The measured defect sits in the late-τ structure of a low-rank, time-correlated posterior covariance, which the time-averaged diagonal barely sees (NS1c caveat in #250).
- Even an unbiased T5 may move NS1c toward 1 without moving the ensemble spread.
- The early-fine sampler already recovers part of the gap for free.

## Caveats

- Two seeds per arm, S0 only. The baseline band is three runs.
- The λ = 100 rows are pre-switch checkpoints (§2) and say nothing about the loss at that weight.
- `val_var_ratio` is pooled over τ ∈ [0.05, 0.95] and all channels. NS1c (§3) is the per-τ view.
