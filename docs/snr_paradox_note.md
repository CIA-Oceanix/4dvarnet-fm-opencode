# SNR Paradox Metrics for Ensemble DA Schemes
## Research note — 4DVarNet-FM-opencode / CIA-OceaniX

---

### 1. Context and motivation

The **forecast ensemble paradox** (Eade et al. 2011, 2014) refers to the empirical finding that ensemble prediction systems correlate better with reality than with their own members: the ensemble mean tracks the observed truth more faithfully than individual members track the ensemble mean. The implication is that the predictable signal is systematically underestimated by the model ensemble — the signal-to-noise ratio (SNR) assigned by the model is too low.

In the context of **data assimilation and reanalysis**, the same structural problem appears:
- Classical DA schemes (EnKF, 4DVar) assume specific error covariance structures.
- If the model's prior over-estimates noise (inflated Q, wrong forcing, sampling error), the posterior ensemble spread is excessive relative to the actual predictable content.
- Gaussian schemes are structurally blind to the non-Gaussian component of the posterior (Ψ_NG ≡ 0 by construction), which can carry predictable signal in non-linearly structured form.

Within the **flow-matching (FM) framework**, this maps cleanly onto the three-way decomposition of the conditional flow-matching (CFM) operator:

```
Ψ(X_τ, Z, τ) = Ψ_mean(Z) + Ψ_G(X_τ, Z, τ) + Ψ_NG(X_τ, Z, τ)
```

The SNR paradox manifests as:
- **Wrong Ψ_G**: the Gaussian gain K_τ is computed from a miscalibrated covariance Σ_{1|Z}, under-weighting the signal.
- **Missing Ψ_NG**: non-Gaussian predictable structure (regimes, multimodality, physical boundaries) is structurally excluded from Gaussian methods.

The information-theoretic consequence: the Cramér-Rao Lower Bound Tr(I⁻¹(τ)) — representing the true irreducible error — is smaller than what the model ensemble thinks it is. The ensemble is not saturating the CRB; it is calibrated to a wrong, looser one.

---

### 2. Classical and ensemble DA schemes

Before turning to FM-based schemes, it is worth characterising the classical baselines in Ψ-decomposition terms, and spelling out how to extend deterministic 4DVar into an ensemble scheme — which is necessary to apply the full metric suite (M1–M5).

#### 2.1 EnKF and variants (LETKF, iEnKS)

All ensemble Kalman methods produce N posterior members via a stochastic or deterministic update of the form:

```
x^(i)_a = x^(i)_f  +  K (y^(i) − H x^(i)_f)
```

where K = B_ens H^T (H B_ens H^T + R)^{-1} is the ensemble Kalman gain, and y^(i) = y + ε_i^o (for stochastic filters like EnKF) or y^(i) = y (for deterministic square-root filters like LETKF).

In Ψ-decomposition terms:
- **Ψ_mean** = x̄_a (ensemble mean after update)
- **Ψ_G** = K (x_τ − β_τ x̄_f), the Gaussian gain applied to the ensemble anomaly
- **Ψ_NG** = 0 exactly — the update is linear in x, so no non-Gaussian structure is generated

**On ETKF/LETKF vs EnKF for a focused SNR paradox study**: both are Gaussian with Ψ_NG = 0 and the same structural limitation. Their differences (deterministic vs stochastic ensemble update, localisation, inflation implementation) affect the *accuracy* of the Gaussian covariance estimate but not the conceptual story. For the SNR paradox, the relevant dimension is Gaussian vs non-Gaussian, not which variant of the ensemble Kalman update is used. **A focused study can use EnKF as the representative Gaussian ensemble filter.** LETKF is already reported in the preprint tables and can be shown alongside EnKF without a separate dedicated analysis — it provides a useful sanity check that the EnKF results are not an artefact of perturbed-observation sampling noise, but it does not change the conclusions about Ψ_NG.

One practical difference worth noting: stochastic EnKF introduces additional sampling noise via the perturbed observations, which slightly inflates the ensemble spread beyond the true posterior. This makes EnKF's SSR slightly worse than LETKF's at the same N — a technical detail, but relevant when reporting M1 and M3 carefully.

#### 2.2 Deterministic 4DVar

The current repo implementation minimises the variational cost J_θ(x, y) = J_obs + J_reg + J_bg to obtain a single MAP estimate x̂. In Ψ-decomposition terms:
- **Ψ_mean** = x̂ (the MAP, the only output)
- **Ψ_G** = implicit in the curvature of J_θ at x̂, but never materialised
- **Ψ_NG** = 0 by construction (the minimisation is deterministic)

4DVar as currently implemented cannot contribute to the ensemble metric suite (M1, M2, M3, M4 require N ≥ 2 members). The two extensions below fix this.

#### 2.3 Ensemble 4DVar — EDA/RML branch

The most direct extension: run N independent 4DVar instances with perturbed inputs.

For member i = 1,…,N, define:
```
J^(i)(x) = ||y^(i) − H(x)||²_R  +  ||x − x_b^(i)||²_B
x^(i)_a  = argmin J^(i)(x)
```
where y^(i) = y + ε_i^o, ε_i^o ~ N(0, R) and x_b^(i) = x_b + ε_i^b, ε_i^b ~ N(0, B).

Under linear-Gaussian dynamics this samples *exactly* from the posterior (Randomised Maximum Likelihood / Ensemble of Data Assimilations result). Under nonlinear dynamics the approximation error is O(cubic) in the nonlinearity — well-characterised and typically small for moderate nonlinearity.

In Ψ-decomposition terms:
- **Ψ_mean** = x̄_a (mean of the N MAP estimates)
- **Ψ_G** = ensemble anomaly covariance from the N solutions, weighted by K_τ
- **Ψ_NG** ≈ small but non-zero under nonlinear dynamics (unlike EnKF, the nonlinear minimisation can produce weakly non-Gaussian ensemble shapes when the cost landscape is non-quadratic)

This is the cleanest extension of 4DVar to ensemble mode for the SNR paradox analysis: minimal new code (a loop and input perturbations around the existing minimiser), directly comparable to EnKF, and produces a genuine posterior ensemble for all metrics M1–M5.

#### 2.4 Ensemble 4DVar — Laplace sampling branch

Run 4DVar *once* to get x̂, then approximate the posterior as Gaussian centred at the MAP:

```
p(x|y) ≈ N(x̂, P_a)     with P_a = [J_θ''(x̂)]^{-1}
x^(i) ~ N(x̂, P_a)
```

In Ψ-decomposition terms:
- **Ψ_mean** = x̂ (exact MAP)
- **Ψ_G** = K_τ (x_τ − β_τ x̂) with K_τ derived from P_a (the cleanest possible Gaussian update)
- **Ψ_NG** = 0 exactly by construction

This makes Laplace sampling the ideal **null hypothesis** for the SNR paradox and for M5: it is the theoretically optimal Gaussian ensemble centred on the 4DVar solution. Any FM scheme that beats it in total D = D_mean + D_G + D_NG is accessing genuinely non-Gaussian information. The Laplace ensemble defines the Cramér-Rao floor for Gaussian methods.

**Practical Hessian approximation** (in order of implementation effort):

1. **L-BFGS curvature reuse** — if 4DVar uses L-BFGS, the inverse Hessian approximation H̃^{-1} is available at convergence from the last m correction vectors. Near-zero additional cost; gives a low-rank P_a in the subspace explored during minimisation.

2. **Randomised Hessian-vector products** — sample r random directions v_k ~ N(0,I), compute J_θ''(x̂) v_k via two adjoint passes or forward-mode AD, build a rank-r eigendecomposition of P_a. Cost: 2r adjoint evaluations. For L96 (D=40), r=20 suffices to capture the leading posterior structure.

3. **EnKF/LETKF covariance surrogate** — use the ensemble covariance Σ_EnKF ≈ P_a. Cheapest to implement but conflates the two methods; useful as a consistency check only.

#### 2.5 Scheme hierarchy and the Ψ decomposition

The four classical/ensemble schemes form a clean hierarchy of increasing posterior fidelity:

```
EnKF  →  4DVar EDA/RML  →  4DVar Laplace  →  CMP + SDA
  |              |                 |                |
Ψ_NG=0       Ψ_NG≈small       Ψ_NG=0 (exact)   Ψ_NG explicit
  |              |                 |                |
Gaussian     Gaussian+ε       optimal Gaussian   full posterior
```

Each step adds one capability:
- EnKF → 4DVar EDA: same Gaussian class but nonlinear minimisation allows weak Ψ_NG
- 4DVar EDA → Laplace: better-calibrated P_a (MAP-centred, curvature-informed), Ψ_NG forced to zero
- Laplace → CMP+SDA: adds explicit Ψ_NG estimation via SDA

This hierarchy is itself the central organising claim of the SNR paradox analysis: the gap between Laplace (optimal Gaussian) and CMP+SDA (full posterior) quantifies how much non-Gaussian signal is accessible in the L96 system but invisible to all Gaussian methods.

---

### 3. FM scheme taxonomy

The repo implements three generative FM-based DA schemes. Their relationship to the Ψ decomposition is what determines how each scheme handles SNR.

#### 3.1 Vanilla CFM (conditional flow matching)

Standard conditional flow matching (Lipman et al. 2023): a neural operator Ψ_θ(x_τ, y, τ) is trained end-to-end to approximate E[x₁|x_τ, y] via MSE regression over the ODE trajectory. The parameterisation is purely state-based — it receives (x_τ, y) with no explicit energy gradient, and does not exploit the variational prior or state-space structure.

**Operator decomposition**: all three components (Ψ_mean, Ψ_G, Ψ_NG) are implicitly fused in the single learned operator. There is no architectural separation; the decomposition must be extracted post-hoc via projection (see Annex A5). Crucially, Ψ_NG is not driven to zero by the architecture, so the scheme can in principle represent non-Gaussian posterior structure — but with no explicit inductive bias toward it.

**SNR behaviour**: because the Gaussian gain is not explicitly parameterised, the scheme's response to miscalibrated priors depends entirely on what it saw at training time. If trained under a specific noise level, it may not generalise to inflated or wrong priors.

#### 3.2 Score-based DA (SDA)

Following Rozet & Louppe (2023): a prior score model s_θ(x_τ, τ) ≈ ∇_{x_τ} log p(x_τ) is trained offline on unconditional samples. At inference time, the posterior score is assembled as:

```
∇_{x_τ} log p(x_τ|y) = s_θ(x_τ, τ)  +  ∇_{x_τ} log p(y|x_τ)
```

For Gaussian observations y = H x₁ + ε, ε ~ N(0, R), the likelihood score can be computed analytically (or approximated via Tweedie). Sampling proceeds via a reverse SDE/ODE driven by this composite score.

**Operator decomposition**: the prior score encodes both Ψ_G and Ψ_NG; the likelihood score contributes primarily to Ψ_G (Gaussian linear update from y). This decoupling is the key structural advantage: the prior model and the observation operator are independent, so the observation configuration can change without retraining.

**SNR behaviour**: the prior score carries the model's implicit SNR estimate. If the prior is wrong (wrong forcing F, inflated climatological variance), the prior score is wrong, and the likelihood update must compensate — possibly partially. Unlike EnKF, SDA can represent non-Gaussian posterior modes via the prior score's non-linear part.

#### 3.3 Hybrid CMP + SDA (ConditionalMeanPrediction + Score-based DA)

A two-stage scheme:
1. **CMP stage**: a conditional mean predictor estimates μ̂_{1|y} = E[x₁|y] deterministically (e.g., a 4DVarNet-style solver or the Gaussian-equivalent operator Ψ_G). This gives the Ψ_mean + Ψ_G part of the full operator.
2. **SDA stage**: the SDA sampler is then initialised near μ̂_{1|y} and used to sample the posterior residual, effectively estimating Ψ_NG.

**Operator decomposition**: this is the most explicit realisation of the Tweedie-based decomposition. Ψ_mean is directly available (output of stage 1); Ψ_G is the Gaussian correction applied in stage 1; Ψ_NG is the SDA residual added in stage 2.

**SNR behaviour**: because the mean is explicitly separated, SNR miscalibration in the prior affects only the SDA stage's residual sampling, not the mean estimate. This makes CMP+SDA the most robust of the three to wrong prior SNR — and also the most interpretable for diagnosing *where* the information deficit lies.

#### 3.4 Summary: structural properties across all schemes

| Scheme | Ψ_mean | Ψ_G | Ψ_NG | N members | Prior reconfigurable? |
|--------|--------|-----|------|-----------|----------------------|
| EnKF / LETKF | ensemble mean | Kalman gain K | 0 (exact) | N ≥ 2 | Yes (inflation) |
| 4DVar (deterministic) | MAP x̂ | implicit in B | 0 (exact) | 1 only | Yes (B matrix) |
| 4DVar EDA/RML | mean of MAP ensemble | from member covariance | ≈ 0 (nonlinear ε) | N ≥ 2 | Yes (perturb B, R) |
| 4DVar Laplace | MAP x̂ | from P_a = J_θ''⁻¹ | 0 (exact) | N ≥ 2 | Yes (via B) |
| Vanilla CFM | fused in Ψ_θ | fused | fused | N ≥ 2 | No (retrain) |
| SDA | prior + likelihood | likelihood score | prior score | N ≥ 2 | Partial (y only) |
| CMP + SDA | explicit (stage 1) | explicit (stage 1) | explicit (stage 2) | N ≥ 2 | Yes (either stage) |

---

### 4. SNR paradox metric suite (scheme-agnostic)

These metrics apply to any scheme producing N posterior members {x^(i)}_{i=1..N} and a reference truth x*.
Let x̄ = (1/N) Σ_i x^(i). All metrics are aggregated over a validation period or space-time domain.
Detailed numerical recipes for each metric are in the Annex.

---

#### M1 — Spread-Skill Ratio (SSR)

```
SSR = σ_ens / RMSE_mean
```

- SSR = 1: perfectly calibrated
- SSR > 1: over-dispersive — model underestimates SNR → **paradox regime**
- SSR < 1: overconfident

Applicable to all schemes. For 4DVar (single member), SSR is undefined; report RMSE only.

---

#### M2 — Predictability Ratio (PR)

```
PR = ρ(x̄, x*)  /  ρ_self
```

where ρ_self = (1/N) Σ_i ρ(x̄, x^(i)) is the mean correlation of the ensemble mean with its own members.

- PR ≈ 1: no paradox
- PR > 1: **SNR paradox** — the mean predicts truth better than it predicts members
- PR < 1: overconfident

Direct analog of the Eade et al. diagnostic, adapted to reanalysis/reconstruction. In forecast mode, evaluate PR as a function of lead time: a scheme with better Ψ_NG should sustain PR > 1 at longer leads.

---

#### M3 — CRPS decomposition

```
CRPS = REL − RES + UNC
```

- **REL**: reliability (miscalibration). High REL → spread-skill mismatch.
- **RES**: resolution (sharpness). Low RES → uninformative ensemble.
- **UNC**: intrinsic difficulty; scheme-independent.

SNR paradox signature: **REL increases faster than RES decreases** as the SNR is degraded (inflation sweep Axis A). Numerical recipe: Annex A3.

---

#### M4 — Ensemble SNR index

```
SNR_ens = Var(x̄) / ( (1/N) Σ_i Var(x^(i) − x̄) )
SNR_obs = Var(x̄) / Var(x̄ − x*)
Π = SNR_obs / SNR_ens
```

- Π > 1: **paradox** — observations reveal more signal than the ensemble acknowledges
- Π ≈ 1: consistent
- Π < 1: overconfident

Π is a single scalar per scheme × scenario, easy to track across the experimental sweep. Numerical recipe: Annex A4.

---

#### M5 — Distribution score decomposition (FM-based)

Available when at least one of the three FM schemes is used as a reference or comparator:

```
S^{G,NG}_{Z,p}(X₁, X̃₁) = D_mean + D_G + D_NG
```

where:
- **D_mean** = S_{Z,p}(Ψ_mean, Ψ̃_mean): error in the deterministic background
- **D_G** = S_{Z,p}(Ψ_G, Ψ̃_G): Gaussian miscalibration — captures wrong covariance / SNR
- **D_NG** = S_{Z,p}(Ψ_NG, Ψ̃_NG): non-Gaussian information deficit — exactly 0 for EnKF/4DVar by construction

**Scheme-specific predictions for the SNR paradox scenario:**

| Scheme | D_mean under inflation | D_G under inflation | D_NG under inflation |
|--------|----------------------|---------------------|----------------------|
| EnKF / 4DVar | stable | ↑ monotone | 0 (fixed) |
| Vanilla CFM | stable | ↑ (fused estimate) | small or unstable |
| SDA | stable (likelihood anchors mean) | modest ↑ | ↑ (prior score wrong) |
| CMP + SDA | stable (stage 1 robust) | modest ↑ | ↑ (stage 2 residual grows) |

The critical test: does any FM scheme maintain lower total D than the optimal Gaussian baseline, specifically via non-zero D_NG recovery? If yes, it is accessing non-Gaussian predictable signal that Gaussian methods cannot see. Numerical recipe: Annex A5.

---

### 5. Experimental design: SNR paradox scenarios on L96

**System**: Lorenz-96, F=8 (true), 40 dimensions, standard DAPPER configuration (20/40 observed components, σ_obs = 1.0).

Three SNR degradation axes, applied to classical baselines; FM schemes evaluated under the same ground truth and observation draws.

#### Axis A — Covariance inflation sweep

Vary EnKF multiplicative inflation α ∈ {1.00, 1.02, 1.05, 1.10, 1.20, 1.50}.
For FM schemes: vary the prior noise level σ₀ in the state-space prior (eq. 5 of the JAMES preprint) to create an equivalent over-dispersive prior. This controls the SNR the FM scheme "believes" during sampling.

Expected signature: SSR and Π increase with α; PR > 1 emerges; REL grows while RES stays roughly stable.

Key comparison: Vanilla CFM vs SDA vs CMP+SDA — which one's D_G grows least under inflation? CMP+SDA is hypothesised to be most robust because its mean estimate (stage 1) is decoupled from the inflation.

#### Axis B — Wrong model forcing

Run ensemble with F_model ∈ {6, 7, 8} against F_true = 8.

F_model < F_true → model dynamics less chaotic → ensemble underestimates Lyapunov growth → effective noise Q is wrong. Mimics NWP models that underestimate predictability of slow modes.

For FM schemes trained on F_true=8 trajectories: evaluate against assimilation cycles using a model with F_model=6 or 7. This tests whether the learned prior (SDA's score model, or CMP's conditional mean) degrades gracefully when the assimilation model is wrong but the generative prior is correct.

Expected separation: SDA (prior decoupled from assimilation model) should degrade more gracefully than Vanilla CFM (prior fused with model assumptions at training time).

#### Axis C — Observation density sweep

Vary p ∈ {0.10, 0.20, 0.50, 1.00} (fraction of observed L96 components), σ_obs = 1.0 fixed.

Low p → prior dominates → miscalibrated prior SNR matters more → paradox amplified.
This maps the regime transition from observation-dominated (high p) to model-dominated (low p) posterior.

For CMP+SDA: at low p, the SDA stage (Ψ_NG) must carry more weight because the Gaussian observation update is weaker. Track D_NG as a function of p — it should grow as p decreases, confirming that non-Gaussian structure matters most in the data-sparse regime.

---

### 6. Reanalysis → forecast extension

The repo currently focuses on reconstruction (smoothing within a window). The SNR paradox is classically defined for forecasts, but the connection is:

1. **Within-window (reconstruction)**: compute M1–M4 over the assimilation window using the posterior ensemble. Check PR > 1 as a baseline.

2. **Out-of-window (forecast)**: propagate the posterior ensemble forward with the L96 model for lead times t ∈ {1, 2, 4, 8, 16} Lyapunov units. Track RMSE, CRPS, SSR, PR as functions of lead time.

3. **Degradation rate as scheme diagnostic**: the lead time at which PR returns to ≈ 1 (paradox extinguished by chaotic growth) measures how much predictable information was encoded in the posterior. A scheme with lower D_NG at the end of the assimilation window should sustain higher PR at short forecast leads. CMP+SDA is the hypothesis to test here: better Ψ_NG → better PR persistence.

---

### 7. Summary: recommended metric table per experiment

**By metric** (what each metric can see):

| Metric | Symbol | Paradox signature | Gaussian schemes (N≥2) | FM schemes |
|--------|--------|-------------------|------------------------|------------|
| Spread-skill ratio | SSR | > 1 | ✓ (EnKF, 4DVar EDA, Laplace) | ✓ |
| Predictability ratio | PR | > 1 | ✓ (N≥2) | ✓ |
| CRPS reliability | REL | ↑ faster than RES ↓ | ✓ | ✓ |
| Ensemble SNR index | Π | > 1 | ✓ (N≥2) | ✓ |
| Gaussian score divergence | D_G | ↑ monotone with inflation | approx. via Σ̂_ens | ✓ (native) |
| Non-Gaussian score divergence | D_NG | > 0 for FM, = 0 for Gaussian | 0 by construction | ✓ (native) |

**By scheme** (what each scheme contributes):

| Scheme | M1 SSR | M2 PR | M3 CRPS | M4 Π | M5 D_G | M5 D_NG |
|--------|--------|-------|---------|------|--------|---------|
| EnKF / LETKF | ✓ | ✓ | ✓ | ✓ | approx | 0 |
| 4DVar (deterministic) | — | — | — | — | — | — |
| 4DVar EDA/RML | ✓ | ✓ | ✓ | ✓ | approx | ≈ 0 |
| 4DVar Laplace | ✓ | ✓ | ✓ | ✓ | native (P_a) | 0 |
| Vanilla CFM | ✓ | ✓ | ✓ | ✓ | native | native |
| SDA | ✓ | ✓ | ✓ | ✓ | native | native |
| CMP + SDA | ✓ | ✓ | ✓ | ✓ | native | native |

4DVar (deterministic) produces a single MAP estimate and cannot contribute to any ensemble metric; it serves as a performance floor for M3 CRPS (single-member limit) only.

---

### 8. Open questions / threads

- **Spectral SNR**: MSE as a function of wavenumber (eq. 6 of the distribution score note) provides a frequency-resolved SNR. Is the paradox scale-selective in L96? Expected to be stronger at large scales (high spectral power S(ω)) where the predictable signal concentrates.

- **R_score as an online SNR probe**: the Tweedie residual R_score in SDA and CMP+SDA is non-zero precisely where the Gaussian update is insufficient. Can its magnitude serve as a real-time indicator of non-Gaussian predictable content remaining in the posterior? This would link D_NG directly to PR at forecast leads.

- **Training loss implications**: if the true process has higher SNR than the model prior, should the D_G term be explicitly targeted in the training loss (rather than MSE only)? The CMP stage in CMP+SDA is the natural place to enforce SNR calibration.

- **SDA prior vs assimilation model mismatch** (Axis B): when F_model ≠ F_true, SDA's decoupled prior (trained on F_true data) may actually outperform EnKF (calibrated to F_model). This would be a striking result: the generative prior compensates for assimilation model error.

---

---

## Annex: Numerical computation of metrics M1–M5

Throughout: N = ensemble size, T = number of verification time steps, D = state dimension (D=40 for L96). Ensemble members are x^(1),...,x^(N) ∈ ℝ^D; truth is x* ∈ ℝ^D; ensemble mean is x̄ = (1/N) Σ_i x^(i).

---

### A1 — Spread-Skill Ratio (M1)

**Per time step t:**
```
σ²_ens(t)  = (1/N) Σ_i ||x^(i)_t − x̄_t||²   / D     (per-dimension average)
err²(t)    = ||x̄_t − x*_t||²                  / D
```

**Aggregated SSR:**
```
SSR = sqrt( (1/T) Σ_t σ²_ens(t) )  /  sqrt( (1/T) Σ_t err²(t) )
    = σ̄_ens / RMSE_mean
```

**Alternative (per-dimension):** Compute SSR_d = σ̄_{ens,d} / RMSE_d for each dimension d, then report mean and std across dimensions. This reveals whether the SNR paradox is spatially uniform or concentrated in specific modes.

**Python:**
```python
# members: (T, N, D); truth: (T, D)
spread = members.std(axis=1).mean()          # scalar, averaged over T and D
rmse   = ((members.mean(axis=1) - truth)**2).mean()**0.5
SSR    = spread / rmse
```

---

### A2 — Predictability Ratio (M2)

Correlations are computed along the time axis (T time steps), separately for each dimension d, then averaged.

**Define for each dimension d:**
```
ρ_truth_d = corr_T( x̄[:,d], x*[:,d] )
ρ_self_d  = (1/N) Σ_i corr_T( x̄[:,d], x^(i)[:,d] )
PR_d      = ρ_truth_d / ρ_self_d
```

**Report:** mean PR = (1/D) Σ_d PR_d, and the distribution over d.

**Note on small ensembles**: ρ_self is biased upward when N is small because x̄ is computed from the same members it is correlated with. Apply a leave-one-out correction:

```
ρ_self_corrected_d = (1/N) Σ_i corr_T( x̄_{-i}[:,d], x^(i)[:,d] )
```

where x̄_{-i} = (1/(N−1)) Σ_{j≠i} x^(j) is the jackknife mean excluding member i.

**Python:**
```python
import numpy as np

def predictability_ratio(members, truth):
    # members: (T, N, D); truth: (T, D)
    T, N, D = members.shape
    mean = members.mean(axis=1)                      # (T, D)

    def corr(a, b):                                  # corr along T axis
        a = a - a.mean(0); b = b - b.mean(0)
        return (a * b).mean(0) / (a.std(0) * b.std(0) + 1e-12)  # (D,)

    rho_truth = corr(mean, truth)                    # (D,)

    # leave-one-out self-correlation
    rho_self = np.zeros(D)
    for i in range(N):
        mean_loo = (mean * N - members[:, i, :]) / (N - 1)
        rho_self += corr(mean_loo, members[:, i, :])
    rho_self /= N

    PR = rho_truth / (rho_self + 1e-12)             # (D,)
    return PR.mean(), PR                             # scalar + per-dimension
```

---

### A3 — CRPS decomposition (M3) — Hersbach (2000)

For scalar observations: apply per-dimension and average.

**Step 1 — CRPS for one ensemble, one time step (scalar):**
```
CRPS(F, x*) = (1/N) Σ_i |x^(i) − x*|  −  1/(2N²) Σ_{i,j} |x^(i) − x^(j)|
```

**Step 2 — Hersbach decomposition** (aggregated over T time steps, per scalar dimension):

Sort members at each t: x_{(1),t} ≤ x_{(2),t} ≤ … ≤ x_{(N),t}.
Define N+1 bins (k=0,…,N) with nominal CDF value p_k = k/N.

For each bin k and time t, compute the fraction of the bin lying to the left (α) and right (β) of the observation x*_t:

```
k = 0   (bin: (−∞, x_{(1)}]):
    α_{0,t} = 0
    β_{0,t} = max(x_{(1),t} − x*_t, 0)

k = 1,...,N−1   (bin: [x_{(k)}, x_{(k+1)}]):
    α_{k,t} = max(min(x*_t, x_{(k+1),t}) − x_{(k),t}, 0)
    β_{k,t} = max(x_{(k+1),t} − max(x*_t, x_{(k),t}), 0)

k = N   (bin: [x_{(N)}, +∞)):
    α_{N,t} = max(x*_t − x_{(N),t}, 0)
    β_{N,t} = 0
```

Verify the reconstruction: CRPS_t = Σ_k [ α_{k,t} p_k² + β_{k,t} (1−p_k)² ].

Average over T:
```
ā_k = (1/T) Σ_t α_{k,t}
b̄_k = (1/T) Σ_t β_{k,t}
ō_k = (1/T) Σ_t 1[ x*_t ≤ x_{(k),t} ]     (empirical hit rate at quantile p_k)
```

Then:
```
REL = Σ_{k=0}^{N} (ā_k + b̄_k) (ō_k − p_k)²
RES = Σ_{k=0}^{N} (ā_k + b̄_k) (ō_k − ō)²
UNC = (1/(2T²)) Σ_{t,s} |x*_t − x*_s|        (CRPS of empirical climatology)
```

where ō = (1/(N+1)) Σ_k ō_k (marginal observation frequency).

Identity check: CRPS_mean ≈ REL − RES + UNC (should hold to numerical precision).

**SNR paradox signature**: plot REL and RES separately as functions of inflation α (Axis A). REL ↑ monotonically; RES stays roughly flat. Their ratio REL/RES is a direct scalar measure of the paradox intensity.

**Python (via properscoring):**
```python
import properscoring as ps
import numpy as np

def crps_decomposition(members, truth):
    # members: (T, N); truth: (T,)
    crps = ps.crps_ensemble(truth, members)           # (T,) CRPS per step
    # Hersbach decomposition (manual or via threshold_brier_score)
    # quickest route: use xskillscore for the full decomposition
    import xskillscore as xs
    import xarray as xr
    obs_da  = xr.DataArray(truth,   dims=['time'])
    fct_da  = xr.DataArray(members, dims=['time', 'member'])
    result  = xs.crps_ensemble(obs_da, fct_da, member_dim='member', decomposition=True)
    return result  # contains crps, reliability, resolution, uncertainty
```

Or implement the Hersbach loop explicitly for full control (recommended when you want per-bin diagnostics).

---

### A4 — Ensemble SNR index (M4)

All variances are computed along the time axis T, then averaged over the D dimensions.

```
# signal variance = variance of the ensemble mean over time
signal_var = Var_T( x̄ )                            # (D,) then average → scalar

# intra-ensemble noise variance = average member spread over time
noise_ens  = (1/N) Σ_i Var_T( x^(i) − x̄ )        # (D,) then average

# observed noise variance = mean squared error of ensemble mean
noise_obs  = Var_T( x̄ − x* )                       # equivalently MSE corrected for bias
```

Then:
```
SNR_ens = signal_var / noise_ens
SNR_obs = signal_var / noise_obs
Π       = SNR_obs / SNR_ens
```

**Practical note**: Var_T(x̄) and Var_T(x̄ − x*) should be computed with the same denominator (T or T−1 consistently). For comparison across schemes with different N, report Π as the primary scalar.

**Python:**
```python
def snr_paradox_index(members, truth):
    # members: (T, N, D); truth: (T, D)
    mean    = members.mean(axis=1)                   # (T, D)
    sig_var = mean.var(axis=0).mean()                # scalar
    
    noise_ens = ((members - mean[:, None, :]) ** 2).mean(axis=(0, 1)).mean()  # scalar
    noise_obs = ((mean - truth) ** 2).mean()                                  # scalar
    
    SNR_ens = sig_var / (noise_ens + 1e-12)
    SNR_obs = sig_var / (noise_obs + 1e-12)
    Pi      = SNR_obs / SNR_ens
    return Pi, SNR_ens, SNR_obs
```

**Spectral variant**: replace Var_T with spectral power at wavenumber ω to get a frequency-resolved Π(ω). For L96, compute the DFT of x̄_t and x*_t over the D=40 spatial dimensions at each t, then aggregate power spectra over T. This identifies which scales drive the paradox.

---

### A5 — Distribution score decomposition (M5)

This metric requires access to the flow-matching operator Ψ_θ(x_τ, y, τ) or the score function s_θ(x_τ, y, τ), which is available for all three FM schemes in the repo. It is not directly available for classical schemes (EnKF, 4DVar); for those, Ψ_G and Ψ_NG must be estimated (see below).

#### A5.1 — Extracting Ψ_mean, Ψ_G, Ψ_NG from each scheme

**Vanilla CFM:**
The trained operator Ψ_θ approximates E[x₁|x_τ, y] directly.

1. Ψ_mean(y): estimate by averaging Ψ_θ(x_τ, y, τ=0) over many draws x_τ ~ N(0,I), or equivalently draw many posterior samples and compute their mean: μ̂_{1|y} = (1/M) Σ_m x̂^(m)₁.
2. Σ̂_{1|y}: empirical covariance of the M posterior samples around μ̂_{1|y}.
3. Ψ_G(x_τ, y, τ) = K_τ (x_τ − β_τ Ψ_mean(y)), where K_τ = β_τ Σ̂_{1|y} (β²_τ Σ̂_{1|y} + α²_τ I)^{-1}.
4. Ψ_NG(x_τ, y, τ) = Ψ_θ(x_τ, y, τ) − Ψ_mean(y) − Ψ_G(x_τ, y, τ).

Requires M >> N samples (M ≈ 100–500 for stable Σ̂ in D=40).

**SDA:**
The composite score s(x_τ, y, τ) = s_prior_θ(x_τ, τ) + s_likelihood(x_τ, y, τ) is directly accessible.

Via Tweedie's formula (eq. 13 of the distribution score note):
```
Ψ(x_τ, y, τ) = x_τ / β_τ  +  (α²_τ / β_τ) s(x_τ, y, τ)
```

Decompose the score into Gaussian and non-Gaussian parts:
1. Gaussian score: s_G(x_τ, y, τ) = −Σ_τ^{−1} (x_τ − β_τ μ_{1|y}), where Σ_τ = β²_τ Σ̂_{1|y} + α²_τ I.
2. Non-Gaussian score residual: R_score(x_τ, y, τ) = s(x_τ, y, τ) − s_G(x_τ, y, τ).
3. Then: Ψ_G(x_τ, y, τ) = (α²_τ / β_τ) s_G, and Ψ_NG = (α²_τ / β_τ) R_score.

**CMP + SDA:**
Components are directly available from the two-stage architecture:
- Ψ_mean = μ̂_{1|y} (output of stage 1 / CMP)
- Ψ_G = K̃_τ (x_τ − β_τ μ̂_{1|y}) with K̃_τ = β_τ (β²_τ + α²_τ ν²)^{-1} from the spherical covariance prior
- Ψ_NG = (α²_τ / β_τ) R_score_θ(x_τ, y, τ) from the SDA stage (eq. 14 of the preprint)

**Classical schemes (EnKF, 4DVar) — approximate:**
1. Ψ_mean ≈ x̄ (ensemble mean or 4DVar solution).
2. Ψ_G: construct from the ensemble covariance Σ̂_ens and the gain K_τ = β_τ Σ̂_ens (β²_τ Σ̂_ens + α²_τ I)^{-1}.
3. Ψ_NG = 0 exactly (by construction for Gaussian methods).

#### A5.2 — Computing the similarity measure S_{Z,p}

The similarity measure between two posteriors with operators Ψ and Ψ̃ is (eq. 9 of the distribution score note, p=2):

```
S*_{Z,2}(X₁, X̃₁) = ∫₀¹ E[ ||Ψ(x_τ, y, τ) − Ψ̃(x_τ, y, τ)||² ] dτ
```

where the expectation is over (x_τ, y) drawn from the true joint distribution.

**Numerical approximation:**
1. Draw test conditions (x*_t, y_t) from the validation set, t=1,...,T_val.
2. For each t and a grid of τ values {τ₁,...,τ_K} (e.g. K=10, uniform on [0,1]):
   a. Sample x_τ = β(τ) x*_t + α(τ) ε, ε ~ N(0,I).
   b. Evaluate Ψ(x_τ, y_t, τ) and Ψ̃(x_τ, y_t, τ).
   c. Compute squared difference d²(t,k) = ||Ψ − Ψ̃||² / D.
3. Approximate the integral:
```
S* ≈ (1/T_val) Σ_t  (1/K) Σ_k  d²(t,k)
```
4. Symmetric measure: S = S*(X₁, X̃₁) + S*(X̃₁, X₁).

**Decomposed measure:**
Replace Ψ, Ψ̃ by their components (Ψ_mean, Ψ_G, Ψ_NG from A5.1) to get D_mean, D_G, D_NG separately.

**Practical note on K**: the integral is typically stable with K=5–10 quadrature points because the integrand varies smoothly in τ under OT scheduling (α=1−τ, β=τ). The SNR paradox diagnostic is most sensitive at intermediate τ (τ ≈ 0.3–0.7) where the Gaussian gain K_τ is largest.

**Python (schematic):**
```python
import torch

def compute_distribution_score(psi_fn, psi_tilde_fn, x_true, y_obs,
                                alpha_fn, beta_fn, tau_grid):
    # psi_fn, psi_tilde_fn: callable (x_tau, y, tau) -> estimate of E[x1|x_tau, y]
    # x_true: (T, D); y_obs: (T, D_obs); tau_grid: (K,)
    T, D = x_true.shape
    K = len(tau_grid)
    total = 0.0
    for tau in tau_grid:
        a, b = alpha_fn(tau), beta_fn(tau)
        eps   = torch.randn_like(x_true)
        x_tau = b * x_true + a * eps
        psi   = psi_fn(x_tau, y_obs, tau)
        psi_t = psi_tilde_fn(x_tau, y_obs, tau)
        total += ((psi - psi_t) ** 2).mean()   # average over T and D
    return total / K
```

---

*Note generated from discussion session, 2026-09-11 (CIA-OceaniX / 4DVarNet-FM-opencode context).*
