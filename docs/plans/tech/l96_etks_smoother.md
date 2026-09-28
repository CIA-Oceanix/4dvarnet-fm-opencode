# Plan: ensemble transform Kalman smoother (ETKS) on the L96 ETKF

**Status:** DESIGN — ETKS built on the existing unlocalised L96 ETKF, to measure the filter-vs-smoother (sequential inference, F3) restriction term for P1. Not started.

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

That is a linear map in ensemble space (members as rows):

    X^a_k = G_k X^f_k,   G_k = 11ᵀ/N + (1wᵀ + T_k)(I − 11ᵀ/N)

With no state-space localisation, member-wise deterministic dynamics (the forcing
W is shared by all members) and `etkf_additive = 0`, the smoother is exact
bookkeeping: apply each later transform G_j retrospectively to the stored
ensemble at every earlier step. No new dynamics, no adjoint; extra cost
O(K·N³ + T·N²·D).

## Algorithm

1. **Forward**: unchanged ETKF. At each analysis time t_k, also record G_k
   (N×N per window and analysis time) and λ_k, and keep the post-analysis,
   post-inflation member trajectory X_s (what `store_ensemble` already
   produces, (B, N, T, D)).
2. **Backward**: for s in [t_k, t_{k+1}),

       X^smooth_s = Π_k X_s,   Π_k = G_{min(K,k+L)} ⋯ G_{k+1}

   accumulated backwards (Π_K = I, Π_{k−1} = Π_k G_k; truncated at lag L).
   All steps of a segment share Π_k (member-wise forecasts between obs times).
3. **Outputs**: smoothed mean, variance, analysis-ensemble CRPS; members optional
   (via `members_store`, node-local `/tmp` by default).

The lag L is a continuous F3 knob: L = 0 is the filter, L = ∞ the fixed-interval
smoother over the window; RMSE(L) is the restriction curve.

## Main design risk: inflation in the backward pass

The ETKF inflates anomalies by λ after every analysis (per-case default: S0 1.5,
S1 2.0). For a later analysis t_j, the retrospective update at s uses the
cross-covariance between the members at s and the forecast anomalies at t_j,
which were inflated at every intermediate analysis while those at s were not:
the cross-covariance is over-estimated by ~λ^(j−k−1). At λ = 2 over 29 cycles, a
naive fixed-interval ETKS will over-correct the early window.

One option, `retro_inflation`:

| mode | backward update at s | expected behaviour |
|---|---|---|
| `none` | plain G_j | over-corrects at large lag; needs small L |
| `correct` (default) | anomaly part of G_j divided by the accumulated λ^(j−k−1) (inflation treated as independent model error added after s) | consistent with "inflation ≈ Q"; allows full-interval smoothing |
| `inflate` | full map incl. inflation applied to lagged states | the usual naive EnKS; over-dispersed smoothed spread |

(L, mode), then λ, are tuned on the validation windows
(`scripts/make_l96_validation_sets.py`), matching how the SDA guidance weight was
tuned. Fallback if no variant is stable at full lag: an ensemble RTS smoother
(backward recursion on stored forecast/analysis ensembles at obs times, which
treats inflation as forecast error by construction; Raanes 2016). Build it only
if needed.

## PR 1 — code and tests

- `ETKS(ETKF)` in `evaluation/baselines.py`, options `lag: int | None` and
  `retro_inflation: str = "correct"`. Raises `NotImplementedError` on the
  non-linear paths: localisation, `etkf_additive > 0`, `loc_mode != "square_root"`.
- Minimal hook in `ETKF.assimilate` / `assimilate_batch`: when
  `self._record_transforms` is set, build G_k from the already-computed `w` and
  `Tmat` plus λ_k. Filter arithmetic untouched; the golden-value test must pass
  bit-for-bit.
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
  5. `assimilate` and `assimilate_batch` agree.
  6. Random layout with NaN channels runs; smoother beats filter on a short L96
     window.
  7. `retro_inflation="correct"` with λ = 1 reduces to `none`.

## PR 2 — wiring and runs

- `evaluation/run_l96.py`: "ETKS" in the method pool, cache and trajectory paths.
  `evaluate_all_l96.py`: `--etks-lag`, `--etks-retro-inflation`; per-case
  inflation defaults unchanged.
- Validation tuning (regular and random, S0 and S1): L ∈ {0, 1, 2, 4, 8, ∞} ×
  three modes at the filter's λ; then λ ∈ {1.2, 1.5, 2.0} at the best (L, mode).
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
  smoothing (a fairness fix for the DA baseline).
- **S1**: prediction — the identity breaks (misspecification is not a restriction);
  the smoother back-propagates biased cross-covariances and may gain less, or lose.
- Contrasts enabled: ETKS vs ETKF isolates F3; ETKS vs Weak-4DVar isolates
  ensemble vs variational U-state at fixed time factorisation; ETKS vs SDA is the
  cleanest mechanistic-vs-learned-prior contrast (both window-level, explicit R).

## Optional follow-ups

- EnKS from the EnKF: the unlocalised stochastic update is also linear in ensemble
  space (G_k includes the perturbed-obs innovations). Same machinery.
- Ensemble RTS smoother, only if the `retro_inflation` variants fail at full lag.
- Results note in `docs/results/` on the restriction-term consistency check.

## Cost and order

PR 1: ~1 day of code and tests, no GPU. PR 2: validation sweep + benchmark rows,
a few GPU-hours (ETKF-scale). Independent of the M-trained learned arms; runs in
parallel with them.
