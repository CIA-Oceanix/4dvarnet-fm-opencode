# τ-consistency of the CFM operator — next steps

**Status:** DRAFT for review, v1 2026-09-23. Nothing implemented. Follows
`docs/results/cfm_tau_consistency.md`; section references (§2, §3) and property
labels (A1, B4, D1, E1…) are that doc's.

## 1. What we are trying to fix, and why it matters

Two defects of the learned operator `D(x, tau, y) ≈ E[x1 | x_tau, y]`, on both P1
M-tier flows:

1. **Biased τ=0 prediction** (E1 slope ≈ 1, A1 ≠ 0).
   - `D(·, 0, y)` is 11–15% worse (pooled RMSE) than the ODE ensemble mean, which corrects it within 1–2 Euler steps.
   - Consequence: the flow's advertised mean component is not the flow's best mean estimate. The "tau=0 mean" rows of the P1 table understate the flow family by this amount.
2. **Variance destroyed along τ ∈ [0.1, 0.4]** (B4 decays 2.6–3.2×).
   - This is the origin of the ensemble under-dispersion (spread/RMSE ≈ 0.4–0.55), the main weakness P1 reports for the flows.
   - The transport is self-consistent (B2), so the fix has to act on `D` at low/mid τ, not on the sampler.

The two defects live in the same τ range, so one intervention may address both.
That is the working hypothesis to test.

## 2. Proposed work, in order

All training arms use **PredictStateCFM-M** with the exact P1 recipe:
- monai backbone, `data.normalize: true`, cosine LR, 400 epochs, batch 16, lr 1e-3, clip 10.
- Config: `config/experiment/A2_predictstatecfm_monaiM_l96.yaml` plus one change per arm.

PredictStateCFM outputs `D` directly, which makes T2's target construction the simplest.

### NS0 — no retraining (≈ 1 GPU-hour; can start immediately)

- **NS0a. Deterministic path** (`x0 = 0`, one member).
  - If it reaches the ensemble-mean RMSE (~0.40 pooled), the ODE gain is refinement, not ensembling.
  - Adds one flag to `probe_ode_mean_along_path.py`.
- **NS0b. Martingale check (D1).**
  - From members at `tau_s ∈ {0, 0.1, 0.2}`, run K=8 *stochastic* continuations (reverse SDE with the forward-noising kernel in `z = x/tau` coordinates) to τ=1, and average them.
  - The average must equal `D(x_s, tau_s)`. This localises where the martingale breaks, without truth.
- **NS0c. Reseed baseline.** Re-probe a second PredictStateCFM-M seed, if one is trained for T0 below. This separates checkpoint noise (15–21% on this family, per P1) from arm effects.

### T0 — baseline reseed (1 run)

This is the P1 PredictStateCFM-M config with a new `training.seed` (the seed field
exists since #242). Without it, no single-run arm difference below ~5% is interpretable.

### T1 — low-τ sampling density (1 run; smallest code change)

- Replace `tau ~ U[0,1]` in `PredictStateCFM.compute_loss` with a mixture: `(1 - f)·U[0,1] + f·U[0, tau_max]`, with `f = 0.5`, `tau_max = 0.4`.
- New fields on the model schema (`conf/schema.py`), default `uniform`, so every existing config is untouched: `tau_sampling: uniform | low_mix`, `tau_low_frac`, `tau_low_max`.
- **Hypothesis:** more low-τ capacity, a less biased τ=0 prediction, and possibly less contraction in [0.1, 0.4].
- **Risk:** the known see-saw (fitting one τ range hurts another, Zhang et al. 2025) could shift error to high τ. The probes will show it.

### T2 — bootstrapped low-τ target (1–2 runs; the principled fix)

This is Daras-style consistency, used as a variance-reduced target.

- For a fraction of each batch with `tau < tau_b = 0.4`, build a pair from the same `x1`:
  - draw `tau' = tau + Δ`, with Δ ~ U[0.1, 0.3];
  - form `x_tau' = tau' x1 + (1 - tau') x0'`;
  - obtain `x_tau` by forward-noising in `z = x/tau` coordinates: `z_tau = z_tau' + sqrt(λ_tau² - λ_tau'²) ξ`, with `λ = (1 - tau) s0 / tau`. This gives `x_tau` exactly the right conditional law given `x1`.
- Loss for those samples: `‖D(x_tau, tau) − [(1 − α) x1 + α · sg(D_teacher(x_tau', tau'))]‖²`, with `α ∈ {0.5, 1.0}`.
  - Teacher: an EMA of the online weights if `LitModel` has one; otherwise stop-gradient of the online network (check before implementing).
- **Why it should help:** at the exact optimum, `E[D(x_tau', tau') | x_tau] = D(x_tau, tau)`, so the target is unchanged in expectation but has lower variance than `x1`.
  - The τ=0 prediction currently regresses onto the noisiest possible target. This is the same argument as Stable Target Field (Xu et al., ICLR 2023).
  - Bootstrapping from τ' ≈ 0.2–0.5, where §2 shows the network is already good, transfers that accuracy down to τ=0.
- **Risk:** teacher bias is self-reinforcing. Hence `α < 1`, an EMA teacher, and an early check of E1 on a 100-epoch checkpoint.

### T3 — variance-conservation (B4) penalty (deferred)

- A truth-free penalty on `(V(tau) − V(tau'))²`, where `V = Var_{x0}(D) + (b²/tau)·tr J / n`.
- Needs K ≥ 4 draws of `x0` per window in-batch plus a finite-difference JVP: ≈ 3–5× the training cost of T1, and a noisy estimator.
- Only worth it if T1/T2 fix the bias (defect 1) but not the variance decay (defect 2).

## 3. Evaluation (identical for every run)

1. The standard P1 `ens30_no10` evaluation: RMSE, CRPS, spread/RMSE, on S0 only.
2. `reports/l96/probe_ode_mean_along_path.py` and `reports/l96/probe_tau_consistency.py`, unchanged, on S0.

**Success criteria vs the T0 reseed pair:**

| metric | target |
|---|---|
| τ=0 prediction RMSE | closes ≥ 50% of the gap to the ODE ensemble mean |
| E1 slope | drops from ≈ 1 toward 0 |
| ODE ensemble-mean RMSE / CRPS | no regression beyond the T0 reseed spread |
| B4 flatness, `B4(0.1) / B4(0.9)` | falls from 2.6 toward 1 |
| spread/RMSE (P1 convention) | rises from 0.40 |

**Reading the outcomes:**
- **Bias fixed, ODE unchanged:** the flow's gain was compensation, and the τ=0 prediction becomes a one-call point estimator as good as the flow's ensemble mean. That is a large cost win: 1 call vs 300.
- **Bias fixed and ODE better:** the refinement stacks on a better start.
- **B4 flatter and spread up:** a calibration result, which is directly P1-relevant.

## 4. Cost

- T0 + T1 + T2 (single α) = **3 P1-recipe training runs**, plus about 1 GPU-hour of probes per run (NS0 included).
- The second T2 value of α adds one run.

## 5. Decisions needed

1. **Approve NS0 + T0 + T1 + T2(α = 0.5)?** Alternatively, T1 alone first, as the cheapest signal.
2. **Seeds:** one per arm plus the T0 pair (proposed), or two per arm (≈ 2× cost, but interpretable at the 15–21% noise level).
3. **T3:** defer (proposed), or run in parallel.
4. **Paper home.** The self-conditioning reading ("the flow is an unrolled refinement of its own τ=0 estimate") is structurally P2's thesis (`docs/scoping/p2_unrolled_solvers_and_flows.md`: an unrolled solver is the `Psi_mean`-only member of the flow family). The calibration defect is P1's. Proposed: the mechanism goes to P2, and the B4 variance-decay diagnosis goes to P1's calibration section.
