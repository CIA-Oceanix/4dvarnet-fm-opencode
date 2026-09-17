# Conditional sampling from an FM prior on L96: the blended-flow family

**Status:** measurements, 2026-09-17. Companion to
`docs/cfm_affine_velocity_decomposition.md` (the operator decomposition) and
`docs/psi_decomposition_results.md` (its measurement). This note covers the
*sampler*: what it takes to get a calibrated conditional ensemble out of a
flow-matching prior, one positive result and six diagnosed negatives.

All numbers here were produced by the scripts named in §6 against checkpoints in
`experiments/l96/`. No number is quoted from any draft or preprint.

---

## 1. Setup and conventions

**Prior.** `SDA3_monai_cond_noisy_l96_norm` — an observation-conditional CFM trained
under noisy parameters/forcings — unless stated otherwise. `SDA1_monai_prior_l96_norm`
(unconditional) is shown once for contrast. Linear interpolant
`x_tau = (1-tau) x_0 + tau x_1`, `x_0 ~ N(0, sigma_0^2 I)`, `sigma_0 = 0.5`.

**Guidance.** SDA-style variance-weighted likelihood guidance (Rozet & Louppe):
at each step the Tweedie estimate `x_hat_1 = x_tau + (1-tau) v` is scored against
the observations with weights

    w(tau) = 1 / (2 (R + Gamma alpha_tau^2 / beta_tau^2)),   Gamma = 1e-2,

and the state is nudged by `dt alpha_tau^2 / (beta_tau (1-tau))` times the gradient.
`N_outer = 50` Euler steps.

**Mean estimator.** `L1b_monai_unet_s0s1_norm_obsdensity` (the observation-density
augmented DirectUNet-M) in the blend experiments; the un-augmented
`L1b_monai_unet_s0s1_norm` in the earlier negatives. Their full-test-set RMSEs are
0.4715 and 0.5014 respectively (consolidated-benchmark rows: 0.4727 / 0.5011).

**Metrics.** `rmse_repo` is the repo's `evaluate_estimates` convention — the mean
over state dimensions of the per-dimension RMSE — and is the number comparable to
the consolidated benchmark. `rmse_pooled` is the pooled RMSE over all entries; the
two differ by ~8% on L96, so they are reported side by side. `spread` is
`sqrt(mean over entries of the ensemble variance)` (biased, `correction=0`).

**Reliability target.** For a perfectly reliable `N`-member ensemble with biased
variance, `spread/RMSE = sqrt((N-1)/(N+1))` — 0.845 at `N = 6`, 0.905 at `N = 10`.
A ratio far below target is over-confident; far above, over-dispersive.

**Caveat on window counts.** Runs marked *(4 win)* or *(10 win)* use only the first
4 or 10 test windows and run systematically ~10% pessimistic against the full
200-window value. Only §4's table is benchmark-comparable.

---

## 2. A deterministic estimator is a flow

### 2.1 The constant operator

Write the FM operator as `Psi(x_tau, y) = E[x_1 | x_tau, y]`; the Tweedie identity
gives the velocity `v = (Psi - x_tau)/(1-tau)`. Now take *any* deterministic
estimator `m_hat(y)` and read it as an operator that ignores its state argument:

    Psi_det(x_tau, y) = m_hat(y)    =>    v_det = (m_hat - x_tau) / (1 - tau).

Integrating `v_det` from `x_0` gives, exactly,

    x_tau = (1 - tau) x_0 + tau m_hat ,

which is precisely the SDEdit / warm-start state. **The warm start is not a separate
trick — it is the trajectory of the degenerate flow whose operator is constant.**
A deterministic estimator and a generative prior therefore live in the same space
and can be mixed at the level of velocities.

### 2.2 The family

    v_lam(x_tau, tau) = (1 - lam(tau)) v_FM(x_tau, tau) + lam(tau) v_det(x_tau, tau),
    lam(tau) = lam_0 (1 - tau)^p .

* `lam_0 = 0` is the cold (pure FM) sampler.
* `lam(tau) = 1{tau < tau_0}` — the discontinuous member — is exactly the warm start:
  below `tau_0` the flow is purely deterministic, so it *reaches* `x_{tau_0} =
  (1-tau_0) x_0 + tau_0 m_hat` and integrates the FM flow from there.
* **`p >= 1` is required.** `v_det` carries a `1/(1-tau)` factor, so the blended
  velocity contains `lam(tau)/(1-tau) = lam_0 (1-tau)^{p-1}`, which is bounded as
  `tau -> 1` only for `p >= 1`. `p = 0` (constant `lam`) diverges. This is not a
  numerical nuisance: it is the statement that a constant operator cannot be blended
  in with fixed weight all the way to `tau = 1`, where the FM operator must take over.

### 2.3 Why the weight should decay: the estimator-combination reading

At the warm-start time the state is `x_{tau_0} = (1-tau_0)x_0 + tau_0 m_hat`, and the
FM operator's Gaussian part maps it through the gain `K_{tau_0}`. The effective weight
the final estimate places on `m_hat` is therefore `w = tau_0 K_{tau_0}`, with

    K_tau = tau P / ((1-tau)^2 sigma_0^2 + tau^2 P)

(the isotropic closed form validated in `docs/psi_decomposition_results.md`). `w` is
the classical two-estimator combination weight: the deterministic estimate is trusted
in proportion to how much prior variance `P` remains unexplained at that time. Since
`K_tau` rises and the prior's own information grows with `tau`, the *continuous*
family should put its mass at small `tau` — which is what `(1-tau)^p` does, smoothly,
instead of the hard cut.

### 2.4 Relation to the CFM literature

Time-varying mixing of two velocity fields is standard practice under a different
name. Classifier-free guidance is `v = (1+w) v_cond - w v_uncond`, i.e. an affine
combination of two flows with a weight that implementations routinely schedule in
`tau`. What is *not* standard here is that one of the two flows is a **pre-trained
deterministic estimator re-read as a degenerate flow**, rather than a second
generative model. That reading is what makes the warm start a special case rather
than an unrelated heuristic, and it is the part worth writing up.

---

## 3. Sweep over (lam_0, p)

SDA3 prior + obsdensity mean, `N = 6`, 10 windows, `N_outer = 50`.
Reliability target 0.845. No post-hoc recentering anywhere.

| config | rmse_repo | rmse_pooled | spread | ratio |
|---|---|---|---|---|
| mean-only | 0.4839 | 0.5084 | 0 | — |
| cold (`lam_0 = 0`) | 0.5951 | 0.6365 | 0.6452 | **1.014** |
| warm (`tau_0 = 0.3`) | 0.4048 | 0.4376 | 0.1124 | 0.257 |
| `lam_0 = 0.5, p = 1` | 0.4022 | 0.4336 | 0.1498 | 0.346 |
| **`lam_0 = 0.5, p = 2`** | **0.3898** | **0.4223** | 0.2224 | **0.527** |
| `lam_0 = 0.9, p = 1` | 0.4300 | 0.4597 | 0.0631 | 0.137 |
| `lam_0 = 0.9, p = 2` | 0.4043 | 0.4360 | 0.1163 | 0.267 |

Reading:

* The **cold sampler is the only well-calibrated one** (1.014 against a target of
  0.845) and is the *worst* on RMSE. This is the honest baseline: an FM prior with
  variance-weighted guidance samples a genuinely dispersed posterior, it just does
  not concentrate as well as a discriminatively-trained mean.
* The warm start buys 0.19 RMSE but destroys the spread (0.257).
* `lam_0 = 0.5, p = 2` **dominates the warm start on both axes simultaneously** —
  lower RMSE *and* twice the spread. That is the one Pareto improvement in this
  entire body of work, and it is what the continuous family was predicted to buy.
* Larger `lam_0` and smaller `p` both push toward the warm start's behaviour, as
  the family predicts (`p = 1` decays slowest at small `tau`, so it commits harder).

None of these is calibrated. The blend trades dispersion for accuracy; only the cold
sampler is reliable. This is stated plainly rather than papered over: **we do not
have a sampler that is simultaneously benchmark-competitive on RMSE and calibrated.**

---

## 4. Full test set

### 4.1 Reference rows (from `reports/l96/outputs/l96_consolidated_benchmark.md`)

S0 RMSE, repo `evaluate_estimates` convention, 200 windows:

| scheme | S0 RMSE |
|---|---|
| Strong-4DVar | 0.8116 |
| ETKF / EnKF | 0.8883 / 0.9131 |
| SDA3(monai), guided from pure noise | 0.5365 |
| DirectUNet-M(monai,cos) | 0.5064 |
| DirectUNet-M(monai,cos,obsdensity) | 0.4727 |
| DirectUNet+SDA3 (non-augmented mean) | 0.4204 |
| **DirectUNet(aug)+SDA3** | **0.3889** |
| subgrad+state-Stier+SDA3(monai) — table best | 0.3410 |

**`DirectUNet(aug)+SDA3` is the direct comparator**: same obsdensity-augmented
DirectUNet-M mean, same SDA3 prior, same SDEdit warm start at `tau_0 = 0.3`. It
differs from the runs below in three respects that must be stated before any
comparison is drawn:

| | benchmark row | this note |
|---|---|---|
| guidance | repo DPS, normalized gradient, `guidance_weight = 2.0` | SDA variance-weighted, `Gamma = 1e-2` |
| Euler steps | `N_outer = 10` | `N_outer = 50` |
| members | 30 | 10 |

So the rows below are internally apples-to-apples (identical settings across
configurations) and only *indicatively* comparable to the benchmark row. The
`warm(tau_0 = 0.3)` row is the bridge: it is the same *scheme* as
`DirectUNet(aug)+SDA3` under this note's guidance and integration settings, so the
gap between it and 0.3889 measures the effect of those settings, and the gap between
it and the blend measures the effect of the blend.

### 4.2 Measured

Full 200-window test set, `N = 10` members, `N_outer = 50`, SDA3 prior +
obsdensity-augmented DirectUNet-M mean. Reliability target 0.905.

| config | rmse_repo | rmse_pooled | spread | ratio |
|---|---|---|---|---|
| mean-only | 0.4715 | 0.4934 | 0 | — |
| cold (`lam_0 = 0`) | 0.5583 | 0.5979 | 0.6599 | **1.104** |
| warm (`tau_0 = 0.3`) | 0.3990 | 0.4280 | 0.1168 | 0.273 |
| `lam_0 = 0.5, p = 1` | 0.3944 | 0.4224 | 0.1540 | 0.365 |
| **`lam_0 = 0.5, p = 2`** | **0.3813** | **0.4101** | 0.2285 | **0.557** |

**Harness validation.** `mean-only` scores 0.4715 against the benchmark's
`DirectUNet-M(monai,cos,obsdensity)` row of 0.4727 — a 0.25% difference on the same
checkpoint and the same 200 windows. The evaluation path in this note is therefore
measuring the same quantity as the consolidated benchmark, which is what makes the
rest of the table comparable at all.

**The internal result, which is the clean one.** Against its own warm-start baseline
at identical settings, the blend improves **both** axes at once:

| | RMSE | spread | ratio |
|---|---|---|---|
| warm (`tau_0 = 0.3`) | 0.3990 | 0.1168 | 0.273 |
| blend (`lam_0 = 0.5, p = 2`) | 0.3813 (**-4.4%**) | 0.2285 (**+96%**) | 0.557 |

4.4% lower RMSE with twice the dispersion. This is the one Pareto improvement in
this body of work, it survives the move from 10 windows to the full test set, and it
is a single scalar schedule with no retraining.

**Against the benchmark, stated carefully.** The closest row,
`DirectUNet(aug)+SDA3 = 0.3889`, has the same three ingredients as the blend. The
blend's 0.3813 is 2.0% better — but the settings differ (guidance rule, `N_outer`,
member count), so this is *not* a clean claim. The `warm` row is what makes the
decomposition legible: at 0.3990 it is the **same scheme** as the 0.3889 benchmark
row, run under this note's settings, and it is 2.6% *worse*. So this note's
SDA-variance-weighted guidance at `N_outer = 50` is mildly worse than the repo's
normalized-gradient DPS at `guidance_weight = 2.0, N_outer = 10` for a warm start,
and the blend's contribution (+4.4%) is what carries it past the benchmark row. The
honest summary is that **the blend is worth ~4% on RMSE and ~2x on spread relative to
the warm start it replaces**, and that this happens to land slightly ahead of the
benchmark row rather than the blend being a 2% improvement over it.

**Calibration — correcting an earlier reading.** The cold sampler's ratio is 1.104
against a 0.905 target, i.e. **~22% over-dispersive**, not calibrated. The same holds
at `N = 6` (1.014 against 0.845, ~20% over). Earlier in this investigation the cold
configuration was described as "calibrated"; it is better stated as *mildly
over-dispersive, and by far the closest to reliable of anything measured here*. Every
other configuration is severely over-confident (0.27-0.56 against 0.905).

---

## 5. Negative results, with diagnoses

Each of these was implemented, run, and rejected. They are recorded because the
diagnoses are the substantive content — several are consequences of the operator
decomposition rather than implementation faults.

### 5.1 Variance-corrected warm start — negligible

The warm-start state `(1-tau_0)x_0 + tau_0 m_hat` has second moment
`(1-tau_0)^2 sigma_0^2` from the noise but omits `tau_0^2 P_hat` from the estimator's
own error. Adding it back changed nothing (0.4744 -> 0.4767 rmse_repo, 4 win) because
the missing term is small: `tau_0^2 P_hat = 0.011` against
`(1-tau_0)^2 sigma_0^2 = 0.123`, i.e. ~8%.

### 5.2 Operator mean-shift — the algebra is output-only

The preprint's Eq. (11) suggests replacing the operator's mean, `Psi* = Psi(x_tau) +
(1 - tau K) Delta_m`. Applied directly this was catastrophic (1.043 / 1.124 rmse_repo,
4 win). The diagnosis is **not** out-of-distribution network queries, which was my
first (wrong) guess. It is that the update is *output-only* and is valid only if `Psi`
is affine. The correct mean-shifted operator shifts the input too:

    Psi*(x_tau) = Delta_m + Psi(x_tau - beta_tau Delta_m)
    =>  v*(x_tau) = v(x_tau - beta_tau Delta_m) + Delta_m ,

which, integrated, reduces **exactly** to shifting the whole trajectory — i.e. to
post-hoc addition of `Delta_m`. There is nothing to gain from doing it inside the ODE.

### 5.3 Post-hoc recentering and a diagonal covariance update

Recentering the ensemble on `m_hat` gives the mean-only RMSE by construction (0.5547,
4 win) while keeping the cold sampler's spread — ratio 1.145, over-dispersive, since
the sampler's own spread reflects *its* error, not the better estimator's. The
diagonal covariance update of Eq. (11) applied on top brings the ratio to 0.801
(spread 0.6743 -> 0.4721). This *works* and is the cheapest route to a calibrated
ensemble around a good mean, but it is a two-stage construction, not a sampler: the
RMSE is entirely `m_hat`'s.

### 5.4 Low-rank / iterative covariance — rank deficiency

Replacing the diagonal target with `Sigma_hat = (1-lam)D + lam A A^T` (ensemble
anomalies rescaled so `diag(Sigma_hat) = p_hat`, applied by Woodbury) *inflated*
rather than calibrated: ratio 0.854 -> 1.964 at `lam = 0.1`, 1.190 -> 2.187 at
`lam = 0.5`, and iterating did not recover. With `N = 6` members in a 72000-dimensional
state the sample covariance is hopeless; the ensemble contributes sampling noise, not
correlation structure. Localisation would be the standard remedy and was not pursued.

### 5.5 Desroziers innovation calibration — the estimator fits the observations

Truth-free calibration via `E[d_oa . d_ab] = H A H^T` drove the spread monotonically
down (1.145 -> 0.660 over 3 iterations) past the target rather than onto it. Cause:
the diagnostic assumes the analysis error is independent of the observation noise,
but `m_hat` is *trained to fit the observations*, so its departures understate its
error — measured `dep_fit = 0.147` against a known `R = 0.163`. The diagnostic is
being fed a quantity it is not valid for.

Two implementation faults found and fixed along the way, both worth remembering:
computing the update from the fitted departure `||y - H x_bar_a||^2` instead of the
Desroziers product, and computing the diagnostics on the *recentered* ensemble, which
erases the very dependence being iterated on.

### 5.6 Observation cross-validation — brittle under temporal holdout

Holding out 25% of observed timesteps restores the independence §5.5 lacks, at the
cost of feeding `m_hat` a thinned observation set it was not trained for. The
resulting error estimate is a gross over-estimate (~15x), so the calibrated ensemble
is wildly over-dispersive: ratio 1.145 -> 1.692 (un-augmented mean), 1.212 -> 2.040
(obsdensity-augmented mean — *worse*, since that model exploits density even more).
The direction of the bias is the safe one, but the magnitude makes it useless.

### 5.7 Decoupled mean / gain / non-Gaussian control — `Psi_NG` is not invariant

The `lam`-blend moves three things at once:

    Psi_lam = [(1-lam) mu_p + lam m_hat] + (1-lam) K_p (x - beta mu_p) + (1-lam) Psi_NG

so blending toward a better mean also shrinks the gain *and* the non-Gaussian term —
which is where the dispersion goes. Giving each component its own knob (`a` for the
mean, `K~` for the gain, `c` for the NG amplitude) should therefore recover the RMSE
gain without the spread loss. It does not:

| config (10 win, N = 6) | rmse_repo | spread | ratio |
|---|---|---|---|
| blend `lam_0 = .5, p = 2` | **0.3898** | 0.2224 | 0.527 |
| decoupled `a_0 = .5` | 0.6098 | 0.1917 | 0.312 |
| decoupled `a_0 = .9` | 0.4635 | 0.1501 | 0.311 |
| blend + `K_hat` | 0.7517 | 0.1311 | 0.174 |

Worse on **both** axes. The cause is the caveat flagged when the unified form was
derived and then underweighted: `Psi_NG` is *defined relative to the pair
`(mu_p, K_p)`* — it is the residual of the network's operator after subtracting that
specific affine part. Re-using the network's `Psi_NG` underneath a different gain
`K~` is inconsistent, and the inconsistency is large enough to dominate. Any real
decoupling would have to retrain `Psi_NG` against the new affine part.

The regression rows (`cold` 0.5951/1.014, `blend` 0.3898/0.527) reproduce the
`lam`-sweep to the digit, which confirms the general velocity implementation is
correct and the result is a genuine negative rather than a bug.

---

## 6. Reproduction

Environment: **`fdv-monai-proto`** (torch 2.8.0, monai 1.6.0) — required for every
monai checkpoint. The `fdv` environment (torch 2.4.1) has no monai and fails with a
late `ModuleNotFoundError` inside `train.py::model_factory`. Scripts live outside the
package tree, so `PYTHONPATH` must point at the repo root.

```bash
export PATH="/Odyssey/private/rfablet/miniforge3/envs/fdv-monai-proto/bin:$PATH"
export PYTHONPATH=$PWD

# positive result: the blended-flow family
python reports/l96/run_blended_flow_sampler.py \
  --prior-ckpt experiments/l96/SDA3_monai_cond_noisy_l96_norm/checkpoints/stage1_best.ckpt --cond \
  --mean-ckpt experiments/l96/L1b_monai_unet_s0s1_norm_obsdensity/checkpoints/stage1_best.ckpt \
  --windows 200 --chunk 10 --members 10 --lam0s 0.5 --ps 1 2 \
  --output lambda_full200.json

# negative result: decoupled component control
python reports/l96/probe_sampler_covariance_control.py \
  --prior-ckpt experiments/l96/SDA3_monai_cond_noisy_l96_norm/checkpoints/stage1_best.ckpt --cond \
  --windows 10 --members 6 --output decoupled.json
```

Shared pieces are in `reports/l96/fm_sampler_common.py`. A batch wrapper is at
`batch/run_l96_blended_flow_sampler.sbatch` (`MEMBERS`/`WINDOWS`/`N_OUTER` overridable).

**Refactor regression.** `run_blended_flow_sampler.py`, which moves the scoring and
gain estimation into the shared module, reproduces the original scratchpad sweep
**bit-for-bit** on all seven rows of §3 (10 windows, `N = 6`): `cold` 0.5951/1.014,
`warm` 0.4048/0.257, `lam_0=0.5,p=2` 0.3898/0.527, and so on to four decimals. The
§3 and §4 tables are therefore the same code path.

**Lesson baked into that module.** `P_prior` — the variance the gain `K_p` is
evaluated at — must come from an **unguided** pass. Estimating it on a guided
ensemble was done three separate times in this work; the symptom is `P_prior = 0.153`
instead of `0.804`, which silently moves a calibration ratio from 0.854 to 1.190.
`estimate_prior_gain_variance` therefore inlines the unguided integration and exposes
**no** `guidance` flag, so there is no longer a flag to forget.

---

## 7. Where this leaves the sampler question

* An FM prior with variance-weighted guidance **is** a working conditional sampler:
  the cold configuration is the closest to reliable of anything measured here
  (ratio 1.104 vs. a 0.905 target — ~22% over-dispersive, not calibrated) and its
  RMSE of 0.5583 beats Strong-4DVar (0.8116) by 31% and both EnKF (0.9131) and ETKF
  (0.8883) by more. It does not beat the benchmark's own standalone `SDA3(monai)`
  row (0.5365), which uses the repo's DPS guidance at `N_outer = 10`.
* It does **not** reach the accuracy of a discriminatively-trained mean estimator,
  let alone the benchmark hybrid. Every scheme that closes that RMSE gap does so by
  importing the mean estimator, and pays for it in spread.
* The blended-flow family is the best trade found: a single scalar schedule that
  strictly dominates the warm start on both axes (-4.4% RMSE, +96% spread on the full
  test set), with a clean interpretation (a
  deterministic estimator is a degenerate flow; the warm start is its discontinuous
  member; the decay rate is the estimator-combination weight `tau_0 K_{tau_0}`).
* The unexplored direction that the negatives point to is **training** rather than
  post-hoc surgery: §5.7 shows the non-Gaussian term cannot be re-used under a
  changed affine part, which is exactly the argument for learning `Psi_NG` against a
  prescribed `(mu, K)` — the unrolled-mean architecture of
  `docs/cfm_affine_velocity_decomposition.md` §5.
