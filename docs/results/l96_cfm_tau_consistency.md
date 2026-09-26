# Why the CFM ODE beats its own τ=0 mean, and which τ-consistency properties the learned operator breaks

**Status:** RESULTS (2026-09-23). Two probes on the P1 M-tier monai flows, L96 S0,
200 cached test windows. Analysis only — no model, training or evaluation code is
changed. Next steps: `docs/plans/analysis/l96_cfm_tau_consistency_next_steps.md`.

**Numbers:** `reports/l96/probe_tau_consistency.py`, `reports/l96/probe_ode_mean_along_path.py` → `reports/l96/outputs/cfm_tau_consistency/` (`tau_consistency_*.json`, `ode_mean_path_*.json`).

**Question.** In the P1 benchmark (`reports/l96/outputs/p1_l96_benchmark.md`) the
ODE ensemble mean of every flow beats that same flow's τ=0 prediction
`mu(x0, 0, y)`, averaged over 30 draws. For PredictStateCFM-M it is 0.358 vs 0.408,
and for VanillaCFM-M 0.345 vs 0.410. At τ=0 the exact CFM target *is* `E[x1|y]`, the
MSE-optimal estimator, so any gain must come from the learned τ=0 prediction's
approximation error. Why does the flow do better, and is it "averaging a noisy
estimate of `E[x1|y]` over τ"?

**Short answer.** It is not averaging over τ. The τ=0 prediction is a *biased*
(not noisy) estimate of `E[x1|y]`, and the flow corrects it within the first one or
two Euler steps by feeding the network its own previous estimate through `x_tau`.
This is implicit self-conditioning, i.e. an unrolled refinement. Separately, the
learned operator violates variance conservation along τ, and that is where the
known under-dispersion (spread/RMSE ≈ 0.4–0.55) is generated.

---

## 1. Setup and notation

Matching `models/vanilla_cfm.py` and `docs/results/l96_cfm_affine_velocity_decomposition.md`:

- `x_tau = tau x1 + (1 - tau) x0`, with `x0 ~ N(0, s0² I)`, `s0 = sigma_prior = 0.5`, in the normalized space. Write `b_tau = (1 - tau) s0`.
- `D(x, tau, y) = E[x1 | x_tau = x, y]`. PredictStateCFM outputs `D` directly; VanillaCFM outputs `v`, and `D = x + (1 - tau) v`.
- `m(y) = E[x1|y]` and `P(y) = Cov(x1|y)`.
- Euler sampler with uniform steps (N=10, `tau_k = k/N`).

**The endpoint is not an average of the per-step predictions.** With uniform
Euler steps, the last step multiplies `(D - x)/(1 - tau)` by exactly `1/N`, so
`x_N = D(x_{N-1}, tau_{N-1}, y)`: the output is the *last* prediction.

Unrolling gives `x_k = (1 - tau_k) x0 + tau_k · sum_{j<k} w_jk D_j`, with weights
`w_jk ∝ 1/((N-j)(N-j-1))` that grow toward the most recent steps. Earlier
predictions reach the output only through the state fed back to the network.

**Checkpoints.** Both come from the P1 benchmark, archived under
`4dvarnet-fm-fdv-tau-aware/experiments/`:
- `A2_predictstatecfm_monaiM_l96/checkpoints/stage1_best.ckpt`
- `A1_vanillacfm_monaiM_l96/checkpoints/stage1_best.ckpt`

**Control.** The τ=0-only PredictStateCFM-M, `A2_predictstatecfm_monaiM_tau0_l96`
(#239), has the same recipe with `train_tau_0_only: true`.

**Scripts.**
- `reports/l96/probe_ode_mean_along_path.py` (§2): pooled all_obs RMSE, **physical units**.
- `reports/l96/probe_tau_consistency.py` (§3): **normalized space**, where `s0` is defined.

Both probes use 30 members, seed 0. Raw outputs are in `reports/l96/outputs/cfm_tau_consistency/`.

> **Convention note.** The pooled RMSE here (`sqrt` of the mean over windows × time
> × channels) is larger than P1's per-window, per-channel convention: the τ=0 mean
> is 0.444 here vs 0.408 in P1. The ordering and relative gaps are the same.

---

## 2. Where along the path the gain happens

`m_hat_tau` is the member-averaged prediction `D(x_tau, tau, y)` at step `tau`.
Under the tower property it estimates `E[x1|y]` for every τ.

| RMSE (S0, physical, pooled) | PredictStateCFM-M | VanillaCFM-M |
|---|---|---|
| τ=0-only PredictStateCFM-M (control, #239) | 0.489 | – |
| `m_hat` at τ = 0 | 0.444 | 0.446 |
| `m_hat` at τ = 0.1 / 0.2 | 0.409 / 0.398 | 0.406 / 0.394 |
| `m_hat` at τ = 0.5 | 0.396 | 0.389 |
| ODE ensemble mean (= `m_hat` at τ = 0.9) | 0.400 | 0.389 |
| endpoint for N = 1 / 2 / 5 / 10 / 20 | .444 / .416 / .403 / .400 / .400 | .446 / .413 / .395 / .389 / .387 |
| uniform average of `m_hat` over τ / cross-validated optimal weights | 0.398 / 0.395 | 0.392 / 0.388 |
| error correlation vs τ=0 / among τ ≥ 0.2 | 0.90–0.96 / ≥ 0.99 | same |
| single member (any τ ≥ 0.2) | ≈ 0.45 | ≈ 0.46 |
| member spread of the prediction at τ = 0 → 0.9 | 0.095 → 0.21 | 0.16 → 0.26 |

**Findings:**
1. **About 80% of the gain is reached by τ ≈ 0.2.** N=2 already recovers two-thirds of it. After τ ≈ 0.3, `m_hat_tau` is flat.
2. **Averaging over τ does not help.** Per-step errors are highly correlated, and an explicit average (even with cross-validated optimal weights) gains about 0.3% over the best single step.
3. **Monte-Carlo noise at τ=0 is not the cause.** Averaging 30 draws with a 0.095 scatter leaves about 0.017, negligible against a 0.045 gap. The τ=0 prediction is *biased*.
4. **Training on all τ already improves the τ=0 prediction itself** (0.489 → 0.444 vs the τ=0-only control), before any integration.
5. **A single member ties the τ=0 prediction.** Its RMSE ≈ `sqrt(ensmean² + spread²)`. The member average over x0 at fixed τ is a genuine variance reduction, but it only recovers what the sampling spread added.

**Mechanism.** At τ=0.1 the network sees `x_0.1 ≈ 0.9 x0 + 0.1 m_hat_0`, i.e. its own
previous estimate plus noise, and corrects it. This is the self-conditioning
reading of the flow: an unrolled refinement of the τ=0 estimate. It is the same
structural idea as a 4DVarNet unrolled solver.

---

## 3. τ-consistency properties of the exact operator, measured

The exact minimiser of the CFM loss satisfies every identity below. The loss
enforces each τ *separately*, so a finite network need not. The groups are labelled
as in the scoping discussion:
- **A**: boundary / pointwise
- **B**: moment identities along the path (tower property)
- **C**: differential (second-order Tweedie)
- **E**: MMSE orthogonality

The remaining groups are not measured here:
- **C2**: the Fokker–Planck / Burgers PDE
- **D1**: the martingale property under the stochastic sampler
- **E2**: conditional vs unconditional Bayes consistency

The identities, exact for this path:

- **E1.** `E[(x1 - D(·,0,y)) g(y)] = 0` for every `g`. Take `g = ODE mean - D0`.
- **B2.** `E[x_tau|y] = tau m`, and `Cov(x_tau|y) = b_tau² I + tau² P`.
- **B4.** `P = Cov_{x_tau}(D) + E_{x_tau}[(b_tau²/tau) ∇_x D]` at every τ (law of total variance plus second-order Tweedie, `Cov(x1|x_tau,y) = (b_tau²/tau) ∇_x D`).
- **A1.** `∇_x D(·, 0, y) = 0`.
- **A3.** `∇_x D = tau P / s0² + O(tau²)`, i.e. the Jacobian term tends to `P` as τ → 0⁺.
- **C1.** `∇_x D` is symmetric (the score is a gradient; hence the exact velocity is a gradient field) and PSD.

**Estimators:**
- Jacobian traces: Hutchinson Rademacher probes, one per member per step, with `J u` by central finite differences.
  - Finite differences with `eps = 1e-2` match exact autograd to < 0.1% at τ ∈ {0, 0.5, 0.9}.
- `J^T u`: autograd, 5 members.
- `P` from truth: `MSE(ensemble mean) / (1 + 1/M)`. This is an **upper bound** on the posterior variance, since it includes any bias of the ensemble mean, so ratios against it are conservative.
- E1 uses split halves: the residual uses the D0 of half A; the correction uses the endpoint and D0 of half B, so Monte-Carlo noise does not bias the numerator.
- 95% CIs come from a window bootstrap (200 resamples).

| property (target) | PredictStateCFM-M | VanillaCFM-M |
|---|---|---|
| RMSE τ=0 prediction → ODE ensemble mean (normalized) | 0.262 → 0.236 | 0.263 → 0.231 |
| **E1** slope of the τ=0 residual on the ODE correction (0) | **0.94** [0.92, 0.96]; noise-corrected 1.03 | **1.12** [1.10, 1.14]; noise-corrected 1.34 |
| **E1** correlation | 0.43 | 0.48 |
| **B2** squared bias of the state mean vs τ·m, relative to (τ·m)² (0) | ≤ 1.0%, → 0 by τ ≥ 0.6 | ≤ 0.7%, → 0 |
| **B2** state variance vs `b² + τ²·P_self` (equal) | within 2–8% | within 2–5% |
| **B2** state variance vs `b² + τ²·P_truth` (equal) | 2.8× too small at τ = 0.9 | 2.0× too small |
| **B4** total variance at τ = 0.1 / 0.3 / 0.6 / 0.9 (constant ≈ P_truth) | 0.041 / 0.026 / 0.019 / 0.016 (P_truth 0.054) | 0.075 / 0.031 / 0.025 / 0.023 (P_truth 0.052) |
| **A3** Jacobian term at τ=0.1 ÷ `P_self` (1) | 1.8 | 2.4 |
| **A3** same ÷ `P_truth` (1): all / slow / fast | 0.53 / 1.03 / 0.50 | 1.08 / 4.8 / 0.90 |
| **A1** Jacobian Frobenius RMS at τ=0 (0) | 0.12 | 0.21 |
| **C1** asymmetry `‖(J−Jᵀ)u‖/‖Ju‖`, τ = 0 → 0.9 (0; √2 = unstructured) | 1.25 → 0.35 | 1.18 → 0.40 |

**Readings:**

1. **The τ=0 prediction is the inconsistent object (E1, A1).**
   - For a minimum-MSE τ=0 prediction the E1 slope would be 0; it is about 1 for both models.
   - So the τ=0 residual lies along the direction the ODE moves, and going the full distance is close to optimal. The explained energy (R² ≈ 0.18–0.23) matches the RMSE gain.
   - The τ=0 prediction also responds to its noise input: A1 is 0.12–0.21, comparable to the Jacobian size at other τ.
2. **The transport is self-consistent (B2).**
   - The ensemble state mean is linear in τ, and its variance follows `b² + τ²·P_self` within a few percent.
   - The ODE faithfully carries whatever variance the predictions supply, so integration error is not the source of under-dispersion.
3. **Variance is destroyed along the path, mostly over τ ∈ [0.1, 0.4] (B4).**
   - The total-variance identity drops 2.6× (PredictStateCFM) and 3.2× (VanillaCFM), where it should be flat.
   - The Jacobian term shrinks faster than the member spread of the prediction grows, i.e. the network over-contracts toward the mean in that range.
   - This is where the under-dispersion comes from. It is a quantitative version of the "high-noise misfit leads to collapse" mechanism (Zhang et al. 2025, arXiv 2508.16154).
   - VanillaCFM overshoots at τ=0.1 (0.075 > 0.052), so the error is not one-signed.
4. **The small-τ Jacobian carries more variance than the sampler delivers (A3).**
   - It implies 1.8–2.4× the ensemble's own variance.
   - Against the truth-based bound it is right in some groups (PredictStateCFM slow 1.03, VanillaCFM fast 0.90) and not uniformly.
   - The information exists early and is lost later, so an inflation step after sampling would treat the symptom.
5. **The Jacobian is far from symmetric at every τ (C1).**
   - The learned field is not the gradient of a potential, as the exact one is.
   - This does not affect B4 or A3, which use only the trace, and the trace is blind to the antisymmetric part.
   - The small late-τ trace (diag J ≈ 0.05–0.08) is **not** a violation: white-noise probes mostly hit directions where the posterior covariance, low-rank and smooth in time, is tiny.

---

## 4. Relation to the literature

These references were checked on 2026-09-23, except where marked *(recall)*.

- **Exact estimators can't beat MMSE.** An exact posterior sample has MSE = 2·MMSE, and the mean of N samples (1 + 1/N)·MMSE (Blau & Michaeli 2018 *(recall)*). A gain over the τ=0 prediction is therefore a statement about approximation error only.
- **Multi-step refinement vs regression.**
  - Deblurring via stochastic refinement (Whang et al., CVPR 2022): averaging samples slightly beats an MSE regressor.
  - GenCast (Price et al. 2024): the ensemble mean beats deterministic models at longer lead times.
  - FlowDAS (Chen et al. 2025): stepwise stochastic-interpolant updates beat one-shot generation in DA.
  - Counter-evidence: InDI (Delbracio & Milanfar, TMLR 2023) finds one step gives the best PSNR.
- **The high-noise end is fit worst.**
  - Zhang et al. 2025 (collapse errors from the deterministic sampler).
  - Esser et al. 2024 (SD3, logit-normal τ sampling) and Lee et al., NeurIPS 2024 (U-shaped τ sampling).
  - Liu et al. 2022: one-step rectified flow gives the conditional mean.
- **τ-consistency as a training signal.**
  - Daras et al., NeurIPS 2023 (the posterior-mean prediction is a martingale; D1).
  - Lai et al., ICML 2023 (FP-Diffusion; C2).
  - Lai et al. 2023: equivalence of consistency-type objectives *(recall)*.
  - Xu et al., ICLR 2023: Stable Target Field, a reduced-variance denoising target *(recall)*.
- **The endpoint as weighted per-step errors.** Xu, Liu & Kong 2025 (arXiv 2510.21792) is an error-*accumulation* model. No paper found claims cancellation across steps, consistent with §2.

---

## 5. Caveats

- One checkpoint per model and S0 only. P1 measured checkpoint-selection noise of 15–21% on this family, so absolute numbers can move under a reseed. The *pattern* (E1 ≈ 1, B4 decaying ~3×, B2 self-consistent) is identical on two independently trained parameterisations.
- `P_truth` is an upper bound (includes the bias²). A3 and B4 ratios against it are conservative.
- The E1 split-half slope is attenuated by Monte-Carlo noise in the denominator. The noise-corrected value relies on the member-variance estimates.
- Hutchinson traces use 30 probes per point, pooled over 200 windows × 3000 steps. The asymmetry uses 5 members.
- The probes use the P1 checkpoints as archived. Regenerating needs those `experiments/` directories (not in git) plus `experiments/l96_datasets_obsj2_int100_nwin200.pt`.

## Regenerate

```bash
PY=/Odyssey/private/rfablet/miniforge3/envs/fdv-monai-proto/bin/python
CK=<experiments dir>/A2_predictstatecfm_monaiM_l96/checkpoints/stage1_best.ckpt
$PY reports/l96/probe_ode_mean_along_path.py --checkpoint $CK --batch-size 50 \
    --out reports/l96/outputs/cfm_tau_consistency/ode_mean_path_psc_m_s0.json
$PY reports/l96/probe_tau_consistency.py --checkpoint $CK \
    --out reports/l96/outputs/cfm_tau_consistency/tau_consistency_psc_m_s0.json
```

On one RTX 8000, each run takes ~20–30 min per model. Run the two models one after
the other: the Jacobian-transpose pass OOMs a 48 GB card when two jobs share it.
