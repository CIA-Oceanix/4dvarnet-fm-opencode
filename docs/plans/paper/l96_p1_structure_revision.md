# P1 — revised paper structure

**Status:** SCOPING — draft revision of the P1 paper structure (2026-09-28 positioning discussion). Supersedes the outline of `docs/papers/p1_structural_hypotheses/` where they conflict; the LaTeX has not been touched yet.

## 1. What changes, in one paragraph

The current draft is built on four hypotheses (H1–H3b), two symptoms (S-a
marginal value of observations, S-b impoverished posterior) and the conditional-
flow partition Ψ_mean + Ψ_G + Ψ_NG as the measuring instrument. Two of its
pillars fell: C4 (S-b) is not supported, and the partition attributes nothing
(DirectUNet ≈ CFM; the filters' "Ψ_NG ≡ 0" is not visible, and is not even true
of their analysis ensemble). The revision replaces the instrument by a **design
matrix**: every scheme is a point in (F1, F2, F3, U-state, U-model), and results
are one-factor contrasts across it. A two-component split Ψ_mean + Ψ_anom and an
error decomposition (irreducible / restriction / misspecification /
approximation) give the formal core. Model error stays the headline, but as a
stress test applied across the matrix.

## 2. Thesis and working title

> Every structural commitment of data assimilation is either an **information
> restriction**, which costs variance honestly and preserves calibration, or a
> **misspecification**, which costs bias silently and destroys it. Model error
> turns commitments that are harmless restrictions under a perfect model into
> misspecifications. Representing model uncertainty is the act of converting the
> second back into the first — and each representation succeeds only over a
> range of model errors we can measure.

Still a **diagnosis paper**: no new method; learned estimators are instruments
that relax one commitment at a time.

Title candidates:
- *What data assimilation's structural choices buy and cost: a controlled comparison with learned estimators under model error*
- *Restriction or misspecification? Dissecting classical and learned data assimilation on a two-scale Lorenz-96*

## 3. Formal core (new §2–§3)

**Commitments, stated as choices that apply to every scheme:**

| | question | options |
|---|---|---|
| **F1** prior–likelihood split | is a prior kept separate from p(y∣x) at inference? | factorised (H, R explicit) / amortised |
| **F2** dynamics structure | how is the prior over the window represented? | one-step transition (Φ, or learned local) / joint window prior / no separate prior; **source**: equations vs samples (from M or the true system) |
| **F3** sequential inference (= old H1) | is inference recursive in time? | sequential / whole window |
| **U-state** | how is state uncertainty represented and carried? | none / explicit (B, P, ensemble), carried through Φ / implicit (learned), window-level |
| **U-model** (= old H2 split) | how is modelling uncertainty represented? | none / in the state (Q, inflation) / in the parameters (θ spread, noisy-θ conditioning) / in the structure |

Dependence to state up front: F3 is exact only if F2 holds in x and obs errors
are white; model error (bias, OU-correlated forcing) breaks F2 in x, hence F3.

**Design matrix (the paper's Table 1):**

| scheme | F1 | F2 | F3 | U-state | U-model |
|---|---|---|---|---|---|
| ETKF / EnKF | yes | one-step Φ | yes | ensemble, through Φ | inflation |
| ETKS (new) | yes | one-step Φ | window | ensemble, through Φ | inflation |
| Strong-4DVar | yes | one-step Φ (hard) | cycled 500-step sub-windows (6 per window) | static B at each sub-window start, point output | none |
| Weak-4DVar | yes | one-step Φ + Q | window | B, Q, point output | Q |
| SDA1 / SDA2 / SDA3-fix | yes | joint window, learned | window | implicit + explicit R | train spread / θ-conditioned / noisy θ |
| DirectUNet | no | none | window | none | train spread |
| CFM (Vanilla, PredictState) | no | none | window | implicit | train spread |
| Hybrid DU → SDA3-fix | partly | joint window | window | implicit + explicit R | noisy θ |
| each learned row, trained on M (new) | as above | as above, source = M | window | as above | σ = 0 / σ = 20% |

**Ψ_mean + Ψ_anom.** Ψ_anom = E[x₁∣x_τ,C] − E[x₁∣C] has zero mean over x_τ,
equals a τ-dependent Kalman gain in the Gaussian case, and classifies U-state
form: 0 (point: 4D-Var, DirectUNet), ensemble-based (filters), learned
(CFM/SDA). "A deterministic estimator is a degenerate flow" and "blending
composes Ψ_mean from one scheme with Ψ_anom from another" survive from the
current §3; the three-term partition and the "structural ceiling" do not.

**Error decomposition and identity.** For the mean,
MSE = E tr Cov(x₁∣C) + restriction E‖m − m_S‖² + misspecification + approximation.
For the anomaly, the same four terms via the law of total covariance (second
moments) or any proper score (log score: conditional mutual information for the
restriction; CRPS: Cramér distance). Identity: a restriction raises MSE and
spread² by the same term; a misspecification raises MSE only. Diagnostic:
unexplained share 1 − (spread/RMSE)². Classification of every matrix cell as
restriction / misspecification / approximation:

Chain of estimators, all targeting the state over the 3000-step window given
C = (y over the window, per-window θ, forcing z) (θ_da and the corrupted forcing
at S1): m = E[x∣C] under the true system (**irreducible**) → m_S = E[x∣S] under
the true system, S a summary of C (**restriction**) → m̃ = E_p̃[x∣S], the
scheme's infinite-resource limit under its own model p̃ (**misspecification**)
→ m̂, what is computed (**approximation**).

| | EnKF (N = 30) | EnKS (ETKS, lag L) | Strong-4DVar (6 × 500 steps) | SDA1 / SDA2 / SDA3-fix |
|---|---|---|---|---|
| restriction | filtering: x_k from y up to k only | removed up to lag L; residual beyond (L_eff ≈ 6 / 3 cycles at S0 / S1 with λ-corrected inflation) | cycling: a sub-window ignores later sub-windows' obs; carried only by a static background | SDA1 ignores θ, z (marginalises the training spread); SDA2/3 condition on them |
| misspecification, S0 and S1 | Gaussian (linear) update; multiplicative inflation | + linear-Gaussian backward cross-covariances; an assumed reading of inflation | hard constraint (Q = 0); Gaussian static B; mode not mean; no posterior (anomaly ≡ 0) | guidance approximates ∇log p(y∣x_τ); gw = 25 tempers the likelihood (≈ shrunk R) |
| misspecification, added at S1 | biased Φ and forcing through every forecast; inflation cannot represent a parameter error | the same, propagated backwards | biased Φ as a hard constraint (floor ≈ 1.4) | SDA1: none (true-system training — the oracle confound); SDA2: clean-θ training given θ_da (bias leaks); SDA3-fix: noisy-θ training turns it into a restriction |
| approximation | 30 members for 40 dims, no localisation, perturbed-obs noise, NaN repairs | + transform products | non-convex L-BFGS over 500 chaotic steps | finite network/training (400 ep), 10-step sampler, 30 members |

EnKF vs EnKS differ only in restriction (the F3 identity test). Along SDA1 →
SDA2 → SDA3-fix, restriction and misspecification swap roles (the U-model
thesis inside one family). Already measured: θ-restriction at S0 ≈ 0 (SDA1
0.501 vs SDA2 0.500); dynamics misspecification S1 − S0 (ETKF +0.79,
Strong-4DVar +0.73, SDA1 0); conditioning misspecification (hybrid prior SDA2
0.333 vs SDA3-fix 0.307 at S1). Pending: cycling restriction of 4D-Var (dws 500
vs 3000), F3 (ETKS), likelihood tempering (gw sweep rescored for spread).

**Pooled calibration (measured 2026-09-28).** The identity is about pooled
second moments, E[var] against E[squared error of the ensemble mean] over
windows × time × channels; calibrated = sqrt(N/(N+1)) = 0.984 for N = 30.
The benchmark's per-window ratio (mean of time-averaged std over mean of
per-window RMSE) is **not** that quantity and must not be used for the identity.
Benchmark protocols (flows 30 × 20 early-fine; SDA gw 25, 10 steps; hybrid
τ₀ 0.1, gw 2); members re-sampled in memory
(`reports/l96/probe_pooled_calibration.py`, array
`batch/run_l96_pooled_calibration.sbatch`); DA from the stored per-time
ensemble variance. Unexplained share = 1 − pooled² · (N+1)/N. Learned rows:
mean over 3 seeds [range]; hybrids vary the DirectUNet seed with the SDA3-fix
seed-1 prior, as in the report. Per-window ratios (regular, seed 1) for
comparison: ETKF 0.98 / 0.37, EnKF 0.97 / 0.33, PSC 0.63, Van 0.72, SDA1 0.43,
SDA2 0.54 / 0.64, SDA3-fix 0.46, hybrid 0.53.

| scheme | layout | RMSE S0 / S1 | **pooled S0** | **pooled S1** | unexplained S0 / S1 |
|---|---|---|---|---|---|
| ETKF (pre-#291; rerun queued) | regular | 0.687 / 1.478 | **1.08** | **0.37** | −20% / 86% |
| EnKF (rerun queued) | regular | 0.711 / 1.514 | **1.13** | **0.33** | −32% / 89% |
| ETKF, EnKF | random | — | ⏳ | ⏳ | — |
| PredictStateCFM-M (1200 ep) | regular | 0.341 / 0.338 | **0.85** [0.82–0.88] | **0.86** [0.83–0.89] | 25% / 24% |
| | random | 0.444 / 0.446 | **0.86** [0.82–0.88] | **0.84** [0.80–0.87] | 24% / 28% |
| VanillaCFM-M (1200 ep) | regular | 0.351 / 0.349 | **0.93** [0.91–0.95] | **0.94** [0.91–0.96] | 10% / 9% |
| | random | 0.462 / 0.467 | **0.86** [0.84–0.87] | **0.85** [0.84–0.86] | 23% / 25% |
| SDA1-M | regular | 0.501 / 0.500 | **0.59** [0.59–0.59] | **0.59** [0.59–0.59] | 64% / 64% |
| | random | 0.613 / 0.623 | **0.55** [0.54–0.55] | **0.53** [0.53–0.54] | 69% / 71% |
| SDA2-M | regular | 0.500 / 0.509 | **0.76** [0.76–0.76] | **0.90** [0.90–0.91] | 40% / 15% |
| | random | 0.605 / 0.627 | **0.75** [0.74–0.75] | **0.86** [0.85–0.87] | 42% / 24% |
| SDA3-fix-M | regular | 0.501 / 0.501 | **0.64** [0.62–0.65] | **0.66** [0.65–0.68] | 58% / 55% |
| | random | 0.610 / 0.619 | **0.60** [0.60–0.60] | **0.62** [0.62–0.62] | 63% / 60% |
| Hybrid DU-M(1200) → SDA3-fix-M | regular | 0.312 / 0.307 | **0.63** [0.63–0.63] | **0.64** [0.64–0.65] | 60% / 57% |
| | random | 0.388 / 0.385 | **0.52** [0.52–0.52] | **0.52** [0.52–0.52] | 72% / 72% |

RMSE in the report's convention (mean over windows and channels of per-channel
RMSE); every learned row reproduces the report's 3-seed RMSE. Readings:
- **Filters**: over-dispersed at S0 once pooled (ETKF slow variables 1.40,
  fast 1.05: inflation 1.5 over-inflates the slow block), then the
  misspecification signature at S1 (86–89% unexplained). The draft's "filters
  are the best-calibrated schemes at S0" is a per-window artefact.
- **Amortised flows (F1 = no)**: close to calibrated (0.84–0.94) on both
  layouts and invariant to model error (approximation ≈ 10–28%). Previously
  reported as strongly under-dispersed — a per-window artefact. VanillaCFM
  loses some dispersion off the regular grid (0.93 → 0.86); PredictStateCFM
  does not.
- **Factorised learned (SDA, hybrid; F1 = yes)**: genuinely under-dispersed
  (55–72% unexplained), worse on the random layout, invariant to model error;
  consistent with guidance tempering the likelihood (gw > 1). The hybrid
  inherits it from its SDA sampler (0.52 on the random layout). Seed ranges are
  ≤ 0.03 everywhere, so none of this is seed noise.
- **SDA2**: spread *rises* at S1 (0.76 → 0.90 regular, 0.75 → 0.86 random) with
  RMSE up 2–4%: conditioned on the biased θ, the prior widens (larger F); better
  calibration by accident, not recognition of model error. SDA3-fix (noisy-θ
  training) does not move.
- To redo: ETKF/EnKF on master after #291 (both layouts; SLURM 56384, branch
  `feature/l96-da-post291`); the gw sweep rescored for pooled spread.

## 4. Section outline

Evidence status: ✅ exists (source report), 🟡 partial, ⏳ pending run.

**Abstract** — rewritten around the thesis; one number per claim.

**§1 Introduction**
- Commitments of DA as choices, not errors (keep the current opening).
- The question: which commitments are restrictions, which become misspecifications under model error, and what representing model uncertainty buys.
- "What this paper is, and is not" (keep). Scope: state target only; unrolled solvers out (P2).
- Contributions = K1–K8 below, compressed.
- Testbed: two-scale L96 (one system, all five columns; the price of matched information).

**§2 DA's commitments, stated as choices** — state-space model and context C
(keep current §2.1, eqs. dynamics/obs/ssm/context); the five choices; the
F3←F2 dependence; a buys/costs table (update of `tab:hypotheses`).

**§3 A common description of estimators** — design matrix; Ψ_mean + Ψ_anom
(half a page, interpolant/ODE equations kept only to define Ψ); error
decomposition, identity, cell classification.

**§4 Experimental design**
- 4.1 L96 system and observing systems: regular 30-obs grid, canonical random layout, #243 factorial (n_obs × k), out-of-range probes. ✅
- 4.2 **Matched-information rule**: every main arm gets the same y and the same model information M (model family, biased per-window θ_da, DA forcing); nothing sees true-system samples. **Reference bounds**, labelled as such: perfect model (S0), learned-on-true-system (today's rows). ⏳ for the M-trained arms
- 4.3 Model-error axis: parametric bias δ ∈ {0, 5, 10, 20, 40%} crossed with stated uncertainty σ ∈ {0, 20%}, plus one structural error (forcing coupling exponent / quartic coupling). ⏳ (δ = 10% ✅)
- 4.4 Schemes and tuning parity: every inference hyper-parameter tuned on the validation windows (ETKF/ETKS inflation and lag, Weak-4DVar Q, SDA guidance weight, hybrid τ₀); training recipe = benchmark default (1200 epochs, monai M, cosine). ✅ recipe; ⏳ SDA 1200-epoch priors
- 4.5 Metrics: RMSE, CRPS, spread/RMSE, unexplained share, marginal value dRMSE/d log n_obs, sensitivity dRMSE/dδ, restriction-term triplet (MSE gap, spread² gap, ‖m_F − m_S‖²).

**§5 Results A — with a perfect model: what each choice costs** (restriction terms)
- 5.1 Overview / regime map: all schemes, S0 and S1, vs density; DA best ≤ 10 obs at S0, learned from ~20; learned degrade out of range. ✅ `l96_benchmark_extended.md` §1, §3, §5
- 5.2 F3: ETKF vs ETKS(L) vs Strong/Weak-4DVar; RMSE(L) restriction curve; identity check; share of the DA–learned gap due to filtering. At the default inflation the inflation-corrected ETKS reaches back only ~6 / 3 cycles (S0 / S1), so it measures filter vs short-lag smoother; the full-window term needs the ensemble RTS smoother (triggered follow-up in the ETKS plan). ⏳ ETKS (plan `l96_etks_smoother.md` on branch `feature/l96-etks`); ✅ ETKF/Strong-4DVar
- 5.3 F2 and its source: mechanistic one-step prior (Weak-4DVar, ETKS) vs learned joint prior (SDA), both factorised, both window. 🟡 (SDA 0.501 vs Strong-4DVar 0.703; needs Weak-4DVar, ETKS, SDA 1200 ep). Optional SDA-local (learned one-step) isolates F2 itself.
- 5.4 F1: factorised (SDA) vs amortised (DirectUNet/CFM): in-distribution accuracy vs robustness to the observing system (×1.21–1.22 vs ×1.24–1.32, fixed-grid ×2.7–3.3) and out-of-range densities (SDA 0.35→0.32 vs DirectUNet 0.17→0.34 from 300 to 1000 obs). ✅ with SDA budget caveat ⏳
- 5.5 U-state at S0: pooled calibration (§3 table: filters over-dispersed 1.08–1.13, flows 0.86–0.95, SDA/hybrid 0.59–0.76); unexplained share; one paragraph on rank histograms (no class-specific shape deficit). ✅ `docs/results/l96_rank_histograms_c4.md`

**§6 Results B — under model error: which choices become misspecifications**
- 6.1 Marginal value of observations vs δ (headline): Strong-4DVar flat at the model-error floor (1.43→1.38 from 30 to 1000 obs; 7.0× collapse 15→30), filters partial (1.6–1.7×), Weak-4DVar, ETKS, M-trained learned. Main figure: panel per scheme, x = density, line per δ. 🟡
- 6.2 U-model: δ ≤ σ vs δ > σ vs structural. SDA2 (clean θ) leaks, SDA3-fix (noisy θ) flat (hybrid 0.310→0.333 vs 0.312→0.307) ✅; perturbed-parameter ETKF ⏳; M-trained σ = 0 / 20% ⏳; Weak-4DVar Q ⏳.
- 6.3 Calibration under model error: explicit-through-Φ breaks (ETKF pooled 1.08→0.37, unexplained −20%→86%; EnKF 1.13→0.33), implicit invariant (flows ≤ 25% unexplained; SDA/hybrid under-dispersed ~60% at both); SDA2's accidental S1 widening; guidance-weight/inflation symmetry (rescore the gw sweep for spread). ✅ / ⏳ rescoring
- 6.4 Value of true-system data: learned-on-truth minus learned-on-M (the "oracle gap"); what reanalysis-trained ML implicitly spends. Absorbs old C6 (model sensitivity ≠ posterior sensitivity) once de-confounded. ⏳
- 6.5 Complementarity: DirectUNet → SDA3-fix hybrid as the composition the matrix predicts (0.312/0.307 regular, beats ETKF in the sparsest bin). ✅ (rerun with M-trained prior ⏳)

**§7 Discussion**
- 7.1 Claims K1–K8, each tied to a matrix column and a restriction/misspecification reading.
- 7.2 What classical DA gets right: sparse regime, observing-system transfer (F1), calibration when its assumptions hold, no training set.
- 7.3 Limitations: idealised single system; circularity (now partly measured by §6.4; self-supervised `var_cost` adaptation left to future work or companion); tuning parity; no true posterior (differences measurable, absolute terms bounded by the best scheme); no cross-quoting between test sets / training protocols (keep).
- 7.4 Positioning (keep, verify citations): weak-constraint 4D-Var, model-error/inflation literature, unstable-subspace DA, SDA and flow-based DA, amortised inference. Gap statement rewritten: no existing controlled study of *how* model information and its uncertainty enter, at matched information, with the marginal value of observations as the dependent variable.

**§8 Conclusion** — rewritten around the thesis.

**Appendices**: A. flow representation details (interpolant, Gaussian-case Ψ_anom); B. tuning sweeps (inflation/lag, Q, gw, τ₀); C. rank histograms; D. optional L63 weak-4DVar; E. optional QG transfer table.

## 5. Claims

| claim | column | reading | status |
|---|---|---|---|
| **K1** Filtering is a restriction: smoothing recovers part of the DA deficit at S0, and the identity holds. | F3 | restriction | ⏳ ETKS |
| **K2** Keeping H, R explicit buys observing-system and out-of-range robustness; amortisation buys in-distribution accuracy. | F1 | amortised under shift = misspecification | ✅ (SDA budget ⏳) |
| **K3** A learned window prior beats the mechanistic one-step prior at a perfect model. | F2 / source | approximation vs restriction | 🟡 |
| **K4** Model error makes the hard constraint a misspecification: its marginal value of observations collapses; filters and soft constraints keep part of it. (headline) | U-model × F2 | misspecification | 🟡 (Strong ✅, Weak ⏳, sweep ⏳) |
| **K5** Representing model uncertainty converts misspecification into restriction, only while δ ≤ σ; state-space (Q, inflation) converts partially. | U-model | restriction ↔ misspecification | 🟡 (SDA2/3 ✅) |
| **K6** Explicit uncertainty carried by Φ is tunable to calibration when its assumptions hold (here over-inflated) and overconfident when they fail; implicit amortised uncertainty is near-calibrated and invariant; factorised learned samplers are under-dispersed (tempered likelihood) but invariant. | U-state × F1 | misspecification vs approximation | ✅ pooled, seed 1 (oracle caveat; ETKF post-#291 ⏳) |
| **K7** Oracle gap: the value of true-system data. | source | — | ⏳ |
| **K8** The best scheme composes an amortised Ψ_mean with a factorised, explicit-R sampler. | F1 × U-state | — | ✅ |

## 6. What leaves the paper

| item | goes to |
|---|---|
| C5 (partition as attribution), "structural ceiling" | dropped |
| C4 (Ψ_NG and dispersion) as a claim | §5.5 paragraph + appendix C |
| C2 / §5.4 (QG state variable, q ≈ ∇²ψ) | out of P1 (P2 or separate note) |
| QG overview rows, "QG separates the definitional case" | out, or appendix E |
| L63 (overview rows, weak-4DVar) | appendix D, only if needed after L96 Weak-4DVar |
| Old "open experiments" list | replaced by §7 below |
| Affine-velocity decomposition, τ = 0 mean probes | ML paper |

## 7. Relocation map (current LaTeX → new sections)

| current | new |
|---|---|
| abstract | rewritten; keep the "diagnosis, not a method" sentence |
| §1 opening, "What this paper is, and is not", Scope | §1 (kept) |
| §1 "Conditional flow matching as the instrument", Contributions, Testbeds | §1 rewritten |
| §2.1 state-space formulation, context | §2 (kept) |
| §2.2 H1–H3b, `tab:hypotheses`, "H2 is load-bearing", "H3b makes H2 bind" | §2 rewritten as F1–F3 / U-state / U-model |
| §3 interpolant, ODE | §3 / appendix A (shortened) |
| §3 partition, `tab:zeroing`, "structural ceiling" | replaced by design matrix + Ψ_mean + Ψ_anom |
| §3 "degenerate flow", "blending composes" | §3 (kept) |
| §4.1 three case studies | §4.1 L96 only |
| §4.2 schemes, §4.3 L96 protocol | §4.1, §4.4 (kept, extended) |
| §4.4 model-error axis, "caveat stated not cashed in" | §4.2–4.3 (matched-information rule) |
| §4.5 metrics | §4.5 (extended) |
| §5.1 overview | §5.1 (L96 part); L63/QG parts out/appendix |
| §5.2 conditioning inert/decisive | §6.2; "ODE sensitivity is not posterior sensitivity" → §6.4 |
| §5.3 marginal value, "Weak-4DVar bounds this claim" | §6.1 |
| §5.4 representation (QG) | out |
| §5.5 dispersion, "what this licenses", "blending" | §5.5, §6.3, §6.5 |
| §5.6 motivating ordering | §5.2 |
| §5.6 "the observing system: what H1 buys" | §5.4 (re-attributed to F1) |
| §6.1 claims C1–C6 | §7.1 K1–K8 |
| §6.2 limitations | §7.3 (circularity promoted into design; coverage paragraph dropped) |
| §6.3 positioning | §7.4 (gap rewritten) |
| §6.4 open experiments | §8 of this doc |
| §7 conclusion | §8 rewritten |

## 8. Runs, by the section they unblock

| run | sections | blocks submission |
|---|---|---|
| ETKS (plan on `feature/l96-etks`) | 5.2, 5.3, 6.1 | yes |
| Weak-4DVar on L96 (`evaluation/tune_l96_weak4dvar.py`, then benchmark) | 5.3, 6.1, 6.2 | yes |
| SDA priors at 1200 epochs | 5.3, 5.4 | yes |
| M-trained DirectUNet / PredictStateCFM / SDA, σ = 0 and 20% | 6.1, 6.2, 6.4, 6.5 | yes |
| δ sweep (≥ 3 levels) + one structural error, all schemes | 6.1, 6.2 | yes |
| Perturbed-parameter ETKF | 6.2 | desirable |
| Pooled spread/RMSE: learned rows done (3 seeds, both layouts); DA rerun on master after #291 running (both layouts) | 3, 5.5, 6.3 | DA rerun: yes |
| Strong-4DVar dws 3000 vs 500 (cycling restriction) | 3, 5.2 | desirable |
| Guidance-weight sweep rescored for spread | 6.3 | desirable (cheap) |
| SDA-local prior | 5.3 | optional |
| `var_cost` self-supervised adaptation | 7.3 | optional / companion |

Part A (§5) is writable now except §5.2; Part B (§6) is where the pending runs land.

## 9. Open decisions

| decision | default proposed here |
|---|---|
| Results layout | two passes, S0 (§5) then S1 (§6) |
| QG | out; appendix transfer table only if cheap |
| L63 | appendix only if L96 Weak-4DVar is inconclusive |
| Flow representation | short §3 + appendix A; full derivation to the ML paper |
| Adaptation arm (`var_cost`) | future work |
| Title | first candidate |

## 10. Risks

- **M-trained learned arms are robust at every δ** (the ±20% training spread covers a +10% bias): the structural-error arm then carries K5; build it in from the start.
- **ETKS unstable at full lag** under λ = 1.5–2.0 inflation: fixed lag or ensemble RTS fallback (see the ETKS plan).
- **"Your Q was under-tuned"** for Weak-4DVar: tune on validation windows with the same budget as the SDA guidance weight and report the sweep.
- **Single system**: stated as the price of matched information; QG transfer table is the mitigation if a reviewer insists.
