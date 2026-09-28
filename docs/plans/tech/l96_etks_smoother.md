# Plan: ensemble transform Kalman smoother (ETKS) on the L96 ETKF

**Status:** DESIGN — ETKS built on the existing unlocalised L96 ETKF, to measure the filter-vs-smoother (sequential inference, F3) restriction term for P1. PR 1 implemented (`ETKS` in `evaluation/baselines.py`, `tests/test_baselines_etks.py`); PR 2 (wiring, validation tuning, benchmark rows) not started. Prerequisite done: the ETKF square-root transform now keeps the unobserved anomaly directions (PR #291). Revised 2026-09-28: exact `correct` retrospective-inflation factor, its effective-lag consequence, and the tests that pin it. Revised again 2026-09-28 after the prototype (`docs/results/l96_etks_prototype.md`): `correct` confirmed as the default, `none` restricted to L ≤ 2, ensemble RTS dropped, PR 2 must score with the benchmark metric and tune λ first.

## Prototype outcome (2026-09-28)

A throwaway prototype on all 200 P1 test windows (`docs/results/l96_etks_prototype.md`,
regular grid, benchmark ETKF and inflation):

| | S0 | S1 |
|---|---|---|
| best ETKS vs ETKF, RMSE | −17.7% (`correct`, L = 2) | −6.1% (`correct`, L = 1) |
| same, CRPS | −14.5% | −2.3% |
| windows where the ETKS is better | 200/200 | 200/200 |

Decisions taken from it, applied below:

1. **`correct` is the default** and needs no lag tuning: it is flat from L = 2 to the full window.
2. **`none` only at L ≤ 2.** Beyond that it over-corrects (S1 CRPS +5.7% vs the filter at L = 2).
3. **`inflate` is not implemented.**
4. **The ensemble RTS (PR 1b) is dropped.** In S0 (D = 40 > N − 1) it diverges without heavy truncation of P_f⁺ and is still worse than `correct` after it. In S1 it only matches `correct`.
5. **The effective lag is 1–2 analyses, not the 3–6 cycles predicted below from λ^(−m).** It is set by the decay of the ensemble cross-covariances on L96, not by the inflation correction. Keep the λ^(−m) analysis as the upper bound it is.
6. **The S0 calibration identity fails at λ = 1.5** (variance drop 2.3–2.7× the MSE drop, filter spread/RMSE 1.11). λ must be tuned on validation before the "restriction reading" test means anything.
7. **Score with the benchmark metric in PR 2.** The prototype used the per-window RMS over variables, which runs 1–10% above the benchmark's mean per-variable RMSE.
8. **Pre-inflation state at the analysis time.** In `correct` mode the state at the analysis time t_k itself is the pre-inflation ensemble, and the first later analysis then has m = 1. The linear toy shows that the post-inflation ensemble is wrong there.

## Why

P1's analysis matrix separates three factorisations: prior–likelihood split (F1),
Markov/one-step prior (F2), and sequential inference (F3, the draft's H1). Every
learned scheme and Strong-4DVar (full 3000-step window by default) estimates the
window from *all* its observations; ETKF/EnKF only use past ones. So:

- the benchmark's DA-vs-learned gap at S0 (ETKF 0.687 vs ~0.34, regular grid)
  mixes the cost of filtering with everything else;
- the F3 restriction term, E||m_filter − m_smoother||², cannot be measured without
  a smoother (only bounded).

The ETKS isolates F3 at fixed model, ensemble, inflation and observations.

## Why it is cheap on the existing filter

`evaluation/run_l96.py` builds the L96 ETKF without `loc_radius`, so every
analysis goes through the ensemble-space branch of `ETKF.assimilate(_batch)`
(`evaluation/baselines.py`):

```python
ens_b = mu + w @ A + Tmat @ A          # then: mu + λ (ens_b − mu)
```

That is a linear map in ensemble space (members as rows). With C = 11ᵀ/N and
P = I − C:

    X^a_k = G_k X^f_k,        G_k = C + (1 w_kᵀ + T_k) P          (analysis)
    X^a,λ_k = M_λ G_k X^f_k,  M_λ = C + λ P                        (inflation)

T_k is symmetric and 1 is one of its eigenvectors (1ᵀ Y = 0), so
M_λ G_k = C + 1 w_kᵀ P + λ T_k P. Before PR #291, T_k was built from the thin
SVD alone and had rank od < N − 1 on L96 (24 obs, N = 30). Every G_k was
rank-deficient, so any product of them was too. The fixed transform has full
rank, which the smoother products and the RTS check (test 4) need.

With no state-space localisation, member-wise deterministic dynamics (the forcing
W is shared by all members) and `etkf_additive = 0`, the smoother is exact
bookkeeping: apply each later transform G_j retrospectively to the stored
ensemble at every earlier step. No new dynamics, no adjoint; extra cost
O(K·N³ + T·N²·D) (O(K·L·N³) for the `correct` mode, see Algorithm).

## Algorithm

1. **Forward**: unchanged ETKF. At each analysis time t_k, also record the
   *pre-inflation* ingredients (w_k, and T_k's eigenpairs (V_k, τ_k)), not the
   assembled map, because the `correct` mode rescales them per lag. Keep λ,
   and the post-analysis, post-inflation member trajectory X_s (what
   `store_ensemble` already produces, (B, N, T, D)).
2. **Backward**: for s in [t_k, t_{k+1}),

       X^smooth_s = Π_k X_s,   Π_k = G_{k+L}^{(L−1)} ⋯ G_{k+2}^{(1)} G_{k+1}^{(0)}

   (latest analysis leftmost, truncated at lag L and at K). The superscript m
   is the number of inflations applied between s and t_j, i.e. m = j − k − 1;
   see the next section for G^{(m)}. All steps of a segment share Π_k
   (member-wise forecasts between obs times).
   - `none` / `inflate`: the factor does not depend on m. At L = ∞, Π_k
     accumulates backwards in one pass (Π_{k−1} = Π_k G_k), O(K·N³). A finite
     L needs a sliding product, O(K·L·N³) in the simple form.
   - `correct`: the factor for analysis j differs per segment k, so each Π_k
     is its own product of ≤ L factors, O(K·L·N³). This is still cheap:
     K ≤ 300, N = 30, and the effective L is small (below).
3. **Outputs**: smoothed mean, variance, analysis-ensemble CRPS; members optional
   (via `members_store`, node-local `/tmp` by default).

The lag L is a continuous F3 knob: L = 0 is the filter, L = ∞ the fixed-interval
smoother over the window; RMSE(L) is the restriction curve.

## Main design risk: inflation in the backward pass

The ETKF inflates anomalies by λ after every analysis (per-case default: S0 1.5,
S1 2.0). For a later analysis t_j, the retrospective update at s uses the
cross-covariance between the members at s and the forecast anomalies at t_j.
Those anomalies were inflated at each of the m = j − k − 1 analyses in between,
while the ones at s were not. To first order, the ensemble cross-covariance
C_{s,j} = A_sᵀ Y_j / N1 is λ^m times what the dynamics alone would carry from s.
Whether that factor is wrong depends on what inflation stands for, which is why
the mode is a tuned choice rather than a derivation:

- **inflation ≈ additive model error Q, independent of x_s**: only the
  propagated part correlates with x_s, the true cross-covariance is C_{s,j}/λ^m,
  and the plain update over-corrects s → `correct`.
- **inflation ≈ a fix for sampling error / under-dispersion of a correct
  model**: the inflated ensemble is the better estimate of the joint
  distribution too, and the plain update is right → `none`. This is arguably
  the S0 reading (perfect model), while the Q reading fits S1.

### The `correct` factor, exactly

Divide the cross-covariance by c = λ^(−m), keep P_s and the innovation
covariance S_j = Y_jᵀY_j/N1 + R as they are. The Kalman update of s with gain
K_s = c C_{s,j} S_j⁻¹ is, in ensemble space:

- mean: x̄_s ← x̄_s + c · w_jᵀ A_s (the plain update's increment, times c);
- anomalies: P_s ← P_s − c² C_{s,j} S_j⁻¹ C_{s,j}ᵀ. The plain update reduces P_s by
  A_sᵀ(I − T_j²)A_s/N1 = C_{s,j} S_j⁻¹ C_{s,j}ᵀ, so the corrected transform is

      T_j^{(m)} = V_j diag( sqrt(1 − (1 − τ_{j,i}²) · c²) ) V_jᵀ,

  on T_j's eigenpairs (V_j, τ_j). It is symmetric, keeps 1 as an eigenvector,
  and is the identity at c = 0 and T_j at c = 1.

      G_j^{(m)} = C + (c · 1 w_jᵀ + T_j^{(m)}) P,        c = λ^(−m)

Both identities were checked numerically (random ensembles, od < N, λ = 2, m = 3:
the mean increment and the updated covariance match the exact Kalman update to
float64 tolerance). The inflation of the smoothed ensemble itself is left out on
purpose: X_s already carries the inflation applied at t_k.

### Consequence: at the default λ, `correct` is a short fixed-lag smoother

The mean increment from analysis j carries weight λ^(−m), and the covariance
reduction λ^(−2m):

| m (analyses between s and t_j) | 1 | 2 | 4 | 6 | 8 |
|---|---|---|---|---|---|
| λ = 1.5 (S0), mean weight | 0.67 | 0.44 | 0.20 | 0.088 | 0.039 |
| λ = 2.0 (S1), mean weight | 0.50 | 0.25 | 0.063 | 0.016 | 0.004 |

Beyond m ≈ 6 (S0) or m ≈ 3 (S1), later observations contribute under 10%. The
"full-interval" `correct` smoother is therefore effectively a fixed-lag
smoother with L ≈ 3–6 cycles, set by λ and not by the observations. On the
regular grid (100 steps between analyses) that is ~300–600 of 3000 steps. On
random obs times (10–300 per window) the effective lag shrinks in time as
density grows. Two consequences:

- in `correct` mode, the F3 terms and RMSE(L) curves measure
  "filter vs ~L_eff-lag smoother". The report must state this, and must not
  call it the fixed-interval restriction term;
- the λ sweep in PR 2 matters twice: a lower λ lengthens the effective lag as
  well as changing the filter. Report L_eff (the m at which λ^(−m) < 0.1) next
  to each row.

### Modes

| mode | backward factor for analysis j at s | expected behaviour |
|---|---|---|
| `none` | G_j (pre-inflation) | **measured:** as good as `correct` at L = 1, over-corrects from L = 2; allowed for L ≤ 2 only |
| `correct` (default) | G_j^{(m)} above | exact per factor under "inflation ≈ Q" (chained: 2–10% off the exact linear smoother in the toy); **measured:** best on S0 and S1, flat in L |
| `inflate` | M_λ G_j (inflation applied to lagged states) | not implemented (dropped 2026-09-28) |

λ is tuned on the validation windows (`scripts/make_l96_validation_sets.py`),
matching how the SDA guidance weight was tuned; L is fixed (full window for
`correct`, L = 1 for `none`) and checked, not tuned.

**Ensemble RTS smoother: dropped (2026-09-28).** The backward recursion
treats inflation as forecast error by construction (Raanes 2016). It was the
candidate exact reference under that reading, but on L96 it gains nothing over
`correct`. It also needs P_f⁺ in state space, which is rank-deficient in S0
(D = 40 > N − 1 = 29) and made it diverge (20-window RMSE 6.1 ± 21) unless the
SVD keeps only singular values above 3–10% of the largest
(`docs/results/l96_etks_prototype.md`, finding 4).

## PR 1 — code and tests

- `ETKS(ETKF)` in `evaluation/baselines.py`, options `lag: int | None = None`
  and `retro_inflation: str = "correct"` (`"correct"` or `"none"`; `"none"`
  with `lag > 2` raises, see the prototype outcome). Raises
  `NotImplementedError` on the non-linear paths: localisation,
  `etkf_additive > 0`, `loc_mode != "square_root"`.
- `correct` uses the pre-inflation ensemble at each analysis time (m counted
  from before its inflation), the post-inflation one between analyses.
- Minimal hook in `ETKF.assimilate` / `assimilate_batch`: when
  `self._record_transforms` is set, store the already-computed `w` and the
  eigenpairs of `Tmat` (from `_etkf_sqrt_transform`, PR #291) per analysis;
  `ETKS` assembles G_j or G_j^{(m)} from them. Filter arithmetic untouched; the
  golden-value test must pass bit-for-bit (it already moved once, in #291).
- The NaN-replacement branches break member correspondence: count them per
  window and flag/exclude affected windows rather than smoothing through them.
- Missing obs channels (random observing system, `seen` subsetting) need nothing
  special: G_k lives in ensemble space.
- Memory: (B, N, T, D) float32 ≈ 2.9 GB at B = 200; smooth per evaluation batch
  chunk, CPU side.
- `tests/test_baselines_etks.py`:
  1. ETKF outputs unchanged with the hook off/on (+ `tests/test_da_golden_l96.py`).
  2. L = 0 reproduces the filter exactly.
  3. At and after the last analysis time, smoother = filter.
  4. Linear-Gaussian check: small linear system, λ = 1, large N; smoothed mean
     and covariance match an exact RTS smoother within Monte-Carlo tolerance.
     Needs od < N somewhere in the setup, so it covers the #291 fix.
  5. `assimilate` and `assimilate_batch` agree.
  6. Random layout with NaN channels runs; smoother beats filter on a short L96
     window.
  7. `retro_inflation="correct"` with λ = 1 reduces to `none` (trivial but
     cheap).
  8. The `correct` factor itself: for random ensembles and m ≥ 1, the mean
     increment and updated covariance of G_j^{(m)} equal the exact Kalman
     update with cross-covariance scaled by λ^(−m) (float64 tolerance); c = 0
     gives the identity. This, not test 7, is what pins the λ > 1 behaviour.
  9. `correct` at the full window agrees with `correct` at L = 4 within a small
     tolerance on a short L96 window (the flat-in-L behaviour measured in the
     prototype).
  10. Regression against the prototype: on P1 test windows 0–19 (S0, seeded as
      in the prototype), the per-window-RMS RMSE is 0.791 for the filter and
      0.639 for `correct` L = 2, to a loose tolerance. Marked slow; it needs the
      P1 cache.

## PR 2 — wiring and runs

- `evaluation/run_l96.py`: "ETKS" in the method pool, cache and trajectory paths.
  `evaluate_all_l96.py`: `--etks-lag`, `--etks-retro-inflation`; per-case
  inflation defaults unchanged.
- Validation tuning (regular and random, S0 and S1): λ ∈ {1.1, 1.2, 1.3, 1.5, 2.0}
  for the filter and for ETKS `correct` (full window), selected separately on CRPS.
  The filter's λ is not necessarily the smoother's: S0 at λ = 1.5 is over-dispersed.
  Check `correct` L ∈ {1, 2, 4, full} and `none` L = 1 at the chosen λ.
- Score with the benchmark metric (mean over variables of per-variable RMSE,
  `evaluation/run_l96.py::evaluate_baseline`) so the rows sit next to the
  published ones; the prototype's per-window RMS is 1–10% higher.
- Benchmark rows on the shared test windows: regular/random × S0/S1 (RMSE, CRPS,
  spread/RMSE); the #243 obs-count factorial sweep; time-resolved RMSE(t) (the
  filter's sawtooth vs the smoother).
- ETKS row in `reports/l96/outputs/l96_benchmark_extended.md`; CHANGELOG.d
  fragment per PR.

## What it measures for P1

Restriction term of F3, per case and density, three ways:

| quantity | identity if both are Bayes-consistent |
|---|---|
| MSE_F − MSE_S | = E‖m_F − m_S‖² |
| spread²_F − spread²_S | = the same term |

- **S0**: agreement validates the restriction reading; the size says how much of
  DA's 2× deficit against the window-level learned schemes is filtering vs
  smoothing (a fairness fix for the DA baseline). **Prototype:** RMSE −17.7%,
  i.e. roughly 0.687 → ~0.57 against ~0.34 for the learned schemes, so a
  minority of the gap. The identity does *not* hold at λ = 1.5 (ratio
  2.3–2.7, over-dispersed filter); retest at the validation-tuned λ.
- **S1**: prediction — the identity breaks (misspecification is not a restriction);
  the smoother back-propagates biased cross-covariances and may gain less, or lose.
  **Prototype:** it gains less (RMSE −6.1%, CRPS −2.3%) and the ratio is
  0.7–0.8 with a strongly under-dispersed filter; consistent with the
  prediction, but not yet separable from the filter's miscalibration.
- Contrasts enabled: ETKS vs ETKF isolates F3; ETKS vs Weak-4DVar isolates
  ensemble vs variational U-state at fixed time factorisation; ETKS vs SDA is the
  cleanest mechanistic-vs-learned-prior contrast (both window-level, explicit R).

## Optional follow-ups

- EnKS from the EnKF: the unlocalised stochastic update is also linear in ensemble
  space (G_k includes the perturbed-obs innovations). Same machinery.
- Results note in `docs/results/` on the restriction-term consistency check at
  the tuned λ (the prototype note is `docs/results/l96_etks_prototype.md`).

## Cost and order

PR 0 (ETKF square-root transform, #291): merged. Prototype: done
(`docs/results/l96_etks_prototype.md`; ~2.5 min per 20 windows and case on one
RTX 8000, including all smoother variants). PR 1: ~1 day of code and tests, no
GPU. PR 2: validation sweep + benchmark rows, a few GPU-hours (ETKF-scale). Independent of the M-trained learned arms; runs in
parallel with them.
