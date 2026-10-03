# P1 — paper structure v3: the error chain, read through flow matching

**Status:** SCOPING — third revision of the P1 structure (2026-10-03). It combines the **error decomposition** of the 2026-10-02 presentation (irreducible / restriction / misspecification / approximation; read in `l96_p1_structure_v2.md`) with **flow matching as a generic representation of any DA scheme**, including the definition of the FM metric FMS_τ (`docs/results/l96_p1_fm_score.md`). It supersedes the outlines of v1 (`l96_p1_structure_revision.md` §4) and v2 (§3); v2 §5 (slide numbers to update) still applies. The LaTeX has not been touched.

## 1. The idea in one paragraph

Every DA scheme, classical or learned, outputs (an approximation of) a conditional distribution `q(x | ·)`. On the flow-matching path `x_τ = τ x₁ + (1 − τ) x₀` (x₀ ~ N(0, I)), any such `q` defines an operator `Ψ_q(x_τ, τ) = E_q[x₁ | x_τ]`. At τ = 0 that operator is the scheme's mean. For τ > 0 it also carries the scheme's uncertainty: `Ψ_q = Ψ_mean + Ψ_anom`. A point estimate (4D-Var, DirectUNet) is a degenerate flow (`Ψ_anom ≡ 0`). An ensemble or Gaussian (EnKF, ETKF, ETKS) has a closed-form `Ψ_anom`. CFM and SDA learn theirs. Scoring the operator against the truth gives FMS_τ, which is the MSE at τ = 0 and a strictly proper distributional score for τ ∈ (0, 1). FMS_τ splits along the error chain **at every τ**. The restriction step is an exact orthogonal component, and so is the joint misspecification-plus-approximation term. The paper's exhibit is therefore not one budget table per question but a **budget curve over τ**: how each scheme's error splits between the four terms, for its mean (τ = 0) and for its uncertainty (τ > 0). This is how the paper answers both "do DA schemes inform the irreducible uncertainty?" and the subtitle "do they quantify their own uncertainty?".

## 2. Title and thesis

- **Title**: *Restriction or misspecification? A flow-matching decomposition of classical and neural data assimilation errors* (alt.: keep the presentation title, with the subtitle *do data assimilation schemes quantify their estimation uncertainty?*).
- **Thesis** (unchanged): a restriction costs variance honestly and preserves calibration; a misspecification costs bias silently and destroys it. Model error turns restrictions into misspecifications. FMS_τ makes the distinction measurable: a restriction raises the curve without changing its shape, while a misspecification bends it upward at large τ (calibration loss).
- Still a **diagnosis paper**: no new DA method. Flow matching is the common language and the metric, not a contender.

## 3. Formal core (new §2–§4 of the paper)

### 3.1 Target and error chain (τ = 0)

`p(x | C) ∝ p(y | x) p(x | θ, z)`, with `C = (y, θ, z)`. For a scheme that uses the information `S ⊆ C`, believes the model `p̃`, and computes `m̂`:

`m = E[x|C] → m_S = E[x|S] → m̃ = E_p̃[x|S] → m̂`

`MSE = E tr Cov(x|C) + E‖m − m_S‖² + E‖m_S − m̂‖²`. Both splits are exact: `m_S = E[m | S]` (tower property), and `m̂` depends on S and on the scheme's own randomness only. The last term is `misspecification + approximation + 2 × cross`. The cross term vanishes only under extra conditions, so the two are separated by controlled contrasts (§3.4), not by an identity.

### 3.2 Every DA scheme is a flow

| scheme | q | Ψ_mean | Ψ_anom |
|---|---|---|---|
| Strong/Weak-4DVar, DirectUNet | point mass at m̂ | m̂ | 0 (degenerate flow) |
| 4DVar + Laplace (optional) | N(m̂, H⁻¹) | m̂ | Gaussian closed form |
| ETKF / EnKF / ETKS | ensemble → Gaussian N(m, v) or empirical / KDE mixture | ensemble mean | Gaussian: `Ψ = m + snr·v/(1 + snr·v) · (x_τ/τ − m)`; mixture: posterior-weighted member average |
| VanillaCFM / PredictStateCFM | learned | `Ψ(·, τ = 0)` (the τ=0 head) | learned |
| SDA (score prior + guidance) | guided sampler | from its samples | from its samples (Tweedie: score ↔ Ψ) |
| Hybrid | DirectUNet mean + SDA sampler | DirectUNet | SDA3-fix (composition) |

Blending (Ψ_mean of one scheme, Ψ_anom of another) is the hybrid. It is defined in this language and nowhere else.

### 3.3 The FM metric FMS_τ and its decomposition

`FMS_τ(q) = E‖Ψ_q(x_τ, τ) − x₁*‖²`
- τ = 0: MSE of the mean.
- τ ∈ (0, 1): strictly proper.
- Gaussian q: minimised at `v = E e²` for every τ (spread/skill).
- τ → 1: universal floor plus the Hyvärinen score.
- Integrated over dsnr: KL(p‖q) up to a constant (Verdú 2010), i.e. the log score.
- A point estimate scores its MSE at every τ.

**Error chain along τ.** Let `Ψ_p` (information C) and `Ψ_{p_S}` (information S) be the true operators. If the scheme's output depends only on S and on randomness independent of the truth, then for every τ:

`FMS_τ(q) = mmse_p(τ) + R_τ(S) + D_τ(q ‖ p_S)`

- `mmse_p(τ) = E‖Ψ_p − x₁*‖²` — irreducible at τ
- `R_τ(S) = E‖Ψ_p − Ψ_{p_S}‖²` — restriction at τ
- `D_τ = E‖Ψ_q − Ψ_{p_S}‖²` — misspecification + approximation at τ, ≥ 0, and 0 iff q = p_S

Both splits are exact, by the same tower-property argument as at τ = 0, now conditioning on (x_τ, S). At τ = 0 this reduces to §3.1. Consequences used in the paper:
- Differences between schemes are exact operator-divergence differences. `mmse_p(τ)` is unknown but common to all schemes.
- **Generalised filter/smoother identity**: for exact Bayesian filter and smoother on the same ensemble, `FMS_τ(F) − FMS_τ(S) = E‖Ψ_F − Ψ_S‖²` at every τ. If it holds, the gap is restriction; if it fails, there is misspecification. The τ = 0 version is v1 §3 (holds approximately at S0 with 100 members, fails at S1).
- **Calibration loss** (the share of FMS_τ removed by the best scalar variance rescaling `c·v`, with optimum `c*(τ)`) reads the U-state commitment. Its growth from S0 to S1 is the misspecification signature.
- **Shape gain** (KDE vs Gaussian at the same variance) reads non-Gaussian structure.
- Integrated summaries: uniform dτ, and dsnr-weighted, which is truncated log score.

### 3.4 How each term is estimated (no true posterior)

| term | estimator | τ = 0 status | τ > 0 status |
|---|---|---|---|
| irreducible `mmse_p(τ)` | upper bound = best scheme at each τ (hybrid at τ ≤ 0.5, flows at τ = 0.75) | ✅ hybrid RMSE 0.293 regular S0 | ✅ PredictStateCFM-M FMS₀.₇₅ 0.0195 (z-units) |
| restriction, filter → smoother | generalised identity on one ensemble (ETKS with its own filter pass) | ✅ MSE gap 0.159 (N = 30), 0.162 (N = 100), regular S0 | ⏳ FMS of the captured filter pass |
| restriction, context ignored | SDA1 vs SDA2/SDA3-fix (θ); SDA ± forcing z; DirectUNet ± (θ, z) | 🟡 θ ✅; z, DirectUNet(θ, z) ⏳ | 🟡 SDA2 calibration loss 7% vs SDA1 18% |
| approximation: finite ensemble | N = 30 → 100 (re-selected inflation) | ✅ ETKS 0.497 → 0.393 (MSE −38%), S1 unchanged | ⏳ FMS of the 100-member rows |
| approximation: optimisation | 4D-Var LBFGS 10 → 40 iterations | ✅ Strong 0.721 → 0.545 (validation) | n/a (point estimate) |
| approximation: training | 400 → 1200 epochs; 1000 → 3000 windows | ✅ DirectUNet 0.380 → 0.340 → 0.334 | ✅ PredictStateCFM calibration loss 11% → 2% |
| misspecification: model error | S0 → S1 per scheme; δ sweep; U-model arms (Weak-4DVar Q, SDA3-fix, perturbed-parameter ETKF) | ✅ ETKS MSE ×7; 🟡 Weak-4DVar Q (val S1 1.473 → 1.099) | ✅ ETKS calibration loss 34% → 63% (τ = 0.75); ETKF 14% → 23%; learned invariant |
| misspecification: structural at S0 | residual of D_τ after the approximation contrasts; Gaussian vs mixture (shape) | 🟡 | ✅ DA ensembles short-tailed (+4–8% at S0, +17–30% at S1) |

## 4. Paper outline

Numbers: regular / random test set, benchmark framework (1200 epochs, 3 seeds), from `p1_l96_benchmark.md` and `docs/results/l96_p1_fm_score.md`.

**§1 Introduction**
- Motivation (presentation slides 3–4): GLORYS vs DUACS; a learned prior beating DA with the true ODE (JAMES 2020); neural-DA skill claims. Is it model error, the algorithm, or something else?
- Approach: every scheme as a flow; one score, FMS_τ, decomposed along the error chain. A model-based and a learned scheme can reach the same RMSE with different budgets, and different budgets at τ > 0.
- Questions Q1 (perfect model), Q2 (model error), Q3 (form of the prior).
- Contributions:
  - (i) the error chain along τ, with exact orthogonal restriction and divergence terms;
  - (ii) FMS_τ as the metric for any DA output;
  - (iii) the controlled L96 study.
- Scope: state target, no new method; unrolled solvers belong to P2.

**§2 DA as estimation of a conditional distribution** — state-space model, context C, posterior, the error chain at τ = 0 (§3.1), and where each DA commitment lands in the chain (presentation slide 6).

**§3 Every DA scheme is a flow** — the path, `Ψ_q`, `Ψ_mean + Ψ_anom`, the table of §3.2 (as Table 1, with the design-matrix columns F1–U-model as labels), blending.

**§4 The flow-matching score and its decomposition** — FMS_τ and its properties; the decomposition `mmse_p + R_τ + D_τ`; the generalised identity; calibration loss and c*(τ); shape gain; integrated summaries; the estimator table of §3.4.

**§5 Experimental design** — two-scale L96 S0/S1; regular and random observing systems; schemes; benchmark framework; matched information and the oracle confound; tuning parity; scoring (Gaussian and KDE forms, Monte Carlo over x₀, window bootstrap).

**§6 Q1 — with a perfect model, where does each scheme's error go?**
- 6.1 Budget at τ = 0:
  - **DA**: restriction (filter vs smoother, identity), then approximation (N = 30 → 100 removes 38% of the ETKS MSE).
  - **Learned**: approximation near saturation; restriction from ignoring (θ, z).
  - **Irreducible**: bounded by the hybrid.
  - Reading: the DA deficit at S0 is mostly honest (restriction + approximation).
- 6.2 Budget along τ: ETKS beats ETKF at τ ≤ 0.5 but loses at τ = 0.75. The smoother buys its mean with an over-confident ensemble (calibration loss 34% vs 14%). The flows lose 0–3%, and SDA 7–18%.
- 6.3 Observing system and density (crossover, out-of-range). F1 robustness reads as D_τ under observing-system shift.

**§7 Q2 — what model error does to the budget**
- 7.1 The S1 budget at τ = 0: misspecification dominates DA (ETKS MSE ×7; 100 members change nothing). Learned rows are flat, as an oracle bound.
- 7.2 Along τ, the misspecification signature: DA calibration loss grows (ETKS 34 → 63%, c* ≥ 8; ETKF 14 → 23%), and so does tail over-confidence (+17–30%). Learned rows are invariant, except SDA2's accidental widening.
- 7.3 Marginal value of observations (headline): Strong-4DVar collapses 7.0× and sits on a floor; filters keep 1.8–1.9×; Weak-4DVar ⏳.
- 7.4 Representing model uncertainty moves D_τ back toward restriction: Weak-4DVar Q (validation S1 −25%), SDA3-fix vs SDA2, δ ≤ σ vs δ > σ (δ sweep ⏳).
- 7.5 Value of true-system data (oracle gap) ⏳.

**§8 Q3 — does the form of the prior drive performance?**
- 8.1 Same likelihood, two priors. The learned joint prior (SDA 0.455) beats the true-ODE hard constraint (Strong-4DVar 0.703) and the 30-member smoother (0.497), but not the 100-member smoother (0.393). The ordering is decided by approximation, not prior fidelity. Along τ, the SDA operator is better calibrated than the smoother's.
- 8.2 Composition. The hybrid has the best mean (τ ≤ 0.5) but not the best distribution: the flows overtake it at τ = 0.75 (0.0195 vs 0.0200 regular, 0.025 vs 0.029 random). Ψ_mean from one scheme and Ψ_anom from another, separated by the score.
- 8.3 Conditioning the prior (θ, z) as restriction terms.

**§9 Discussion**
- Lessons (slide 15) restated as budget statements.
- What classical DA still gets right.
- Back to the motivation.
- Limitations:
  - `mmse_p(τ)` is unknown, so only differences and bounds are available;
  - DA scored in Gaussian form;
  - snapshot (24-channel) scoring degenerates at τ = 0.75;
  - one idealised system;
  - the oracle confound.
- Positioning: proper scores (CRPS, energy, log score), mismatched estimation (Verdú), flow/score-based DA, amortised inference.

**§10 Conclusion.**

**Appendices**:
- A. Ψ_q in closed form (Gaussian, mixture), the decomposition proof;
- B. design matrix (F1–U-model);
- C. tuning sweeps;
- D. ensemble-size, iteration and training-budget sensitivities;
- E. the P1-protocol tier study (`p1_l96_benchmark.md` annex A);
- F. FMS estimator details (Monte Carlo, KDE, snapshot ESS, bootstrap).

## 5. What changes relative to v2

| v2 | v3 |
|---|---|
| Flow matching in the outlook only | Flow matching is the representation (§3) and the metric (§4). Every result is read through it |
| One budget table per question (τ = 0) | Budget curves over τ; the τ = 0 table is one slice |
| Calibration as a separate outcome | Calibration is the large-τ part of the same score, with the misspecification signature built in |
| Four additive terms (presentation slides) | Two exact splits (irreducible / restriction / D_τ), with D_τ split into misspecification and approximation by contrasts — stated honestly |
| Hybrid = best scheme | Hybrid = best mean, not best distribution (score-separated composition) |

## 6. Runs and analyses, by section

| item | serves | status |
|---|---|---|
| FMS of ETKS with its own filter pass (generalised identity, S0/S1, N = 30 and 100) | §4, §6.1–6.2, §7.2 | ⏳ (re-use `scripts/etks_ensemble_check.py` + `evaluation/fm_score.py`) |
| FMS of the 100-member ETKS rows | §6.1–6.2 | ⏳ |
| Integrated summaries (uniform dτ, dsnr-weighted / log score) | §4, all results | ⏳ (analysis) |
| Budget curves: pooled MSE / FMS per term, per scheme, both test sets | §6–8 | ⏳ (analysis, no GPU) |
| Weak-4DVar benchmark rows (point estimate: FMS = MSE) | §7.3–7.4, §8.1 | running (SLURM 58092/58093) |
| 4D-Var Laplace (Hessian) flow | §3.2 optional | optional |
| SDA ± forcing z; DirectUNet ± (θ, z) | §6.1, §8.3 | ⏳ |
| M-trained learned arms, δ sweep, perturbed-parameter ETKF | §7.4–7.5 | ⏳ |
| Fix the "under-dispersed (0.63)" wording in `p1_l96_benchmark.md` (per-window ratio, not pooled) | report | ⏳ small PR |

## 7. Open decisions

| decision | default proposed here |
|---|---|
| Main metric | FMS_τ with τ ∈ {0, 0.5, 0.75} in the tables + the integrated score; RMSE = FMS₀ |
| DA operator | Gaussian form in the main text (the only one for EnKF / Strong-4DVar); KDE as the shape check |
| Units | physical units² with window-bootstrap 95% intervals (as in `fm_score_bootstrap.md`) |
| Where the proof goes | §4 statement + appendix A |
| Design matrix | Table 1 columns + appendix B |
| Title | first candidate of §2 |
