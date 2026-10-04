# P1 — paper structure v4 (detailed draft)

**Status:** SCOPING — detailed structure of the P1 paper (2026-10-04). It takes the **narrative of v2** (motivation from practice, three questions, error-budget tables; `l96_p1_structure_v2.md`) and the **metric of v3** (the error chain extended along τ with the flow-matching score FMS_τ; `l96_p1_structure_v3.md`). Flow matching is kept bounded: it is used as a scoring language and as the notation for the hybrid. The general "every DA scheme is a flow" programme goes to the ML paper. This doc supersedes the outlines of v1–v3. v1 §3 (formal core) and v2 §5 (slide numbers to update) still apply. The LaTeX (`docs/papers/p1_structural_hypotheses/`) has not been touched.

Evidence marks: ✅ exists (source named), 🟡 partial, ⏳ pending. Unless stated, numbers are per-window RMSE on the 24 observed channels, mean over 200 test windows, **regular / random** observing system, benchmark framework (1200 epochs, 3 seeds; DA at the #295 inflation, 30 members), from `reports/l96/outputs/p1_l96_benchmark.md`. FMS numbers are from `docs/results/l96_p1_fm_score.md`, in z-units unless stated.

---

## 0. Positioning in one page

**Problem.** In practice, model-based data assimilation often loses to estimators that use less physics:
- observation-only optimal interpolation beats an ocean reanalysis on sea-surface height (DUACS vs GLORYS);
- a DA scheme with a learned prior beats the same scheme run with the true ODE (Fablet et al., JAMES 2020);
- neural DA papers routinely report better skill than classical DA.

The usual explanations — model error, a poor algorithm, poor tuning — are rarely separated.

**Approach.** Treat every DA scheme, classical or learned, as an estimator of the conditional distribution `p(x | C)` of the state given the context `C = (y, θ, z)`. Decompose its error along a chain of idealisations:
- **irreducible**: what no scheme can beat;
- **restriction**: the scheme uses less information than C;
- **misspecification**: the scheme believes a wrong model;
- **approximation**: the scheme computes its own target only approximately.

Do it for the mean (MSE) and for the uncertainty, with a single proper score that reduces to the MSE at τ = 0.

**Thesis.**
- A **restriction** costs variance honestly and preserves calibration.
- A **misspecification** costs bias silently and destroys calibration.
- Model error turns harmless restrictions into misspecifications.
- Representing model uncertainty converts some of them back, over a range we can measure.

**What is new.**
1. The error chain applied to classical and learned DA under matched information, with each term estimated by a controlled contrast.
2. Its extension along τ with FMS_τ. The split into irreducible, restriction and the remainder is **exact at every τ**. A restriction raises the score curve; a misspecification bends it upward at large τ.
3. A controlled two-scale Lorenz-96 study: a perfect model (S0) and model error (S1), two observing systems, five DA schemes and five learned ones, tuning parity.

**What it is not.**
- No new DA method.
- The target is the state only: no parameter estimation.
- Unrolled / 4DVarNet solvers are out (P2).
- The general flow-matching representation of DA algorithms is out (ML paper).

**Target venue.** A DA / geoscience-methods journal (QJRMS, Tellus A, or JAMES). Length about 9–11k words, 8–9 figures, 5–6 tables.

**Title.** *Restriction or misspecification? Dissecting the errors of classical and neural data assimilation.*
Alternative subtitle: *… — do data assimilation schemes quantify their estimation uncertainty?*

**Positioning against the literature** (§9.5):
- **Model-error DA**: weak-constraint 4D-Var, inflation, parameter perturbation, unstable-subspace DA. This paper measures *where* model information enters and what representing its uncertainty buys, at matched information, with the marginal value of observations as the dependent variable.
- **Learned / score-based DA**: amortised networks; SDA (Rozet & Louppe 2023); flow-based DA. These schemes are placed in the same error budget as classical DA, rather than compared on RMSE alone.
- **Proper scores**: CRPS, energy score, log score; the calibration–resolution decomposition (Bröcker 2009); mismatched estimation (Verdú 2010). FMS_τ is chosen for three reasons, stated explicitly:
  1. it is computable for samples, Gaussians *and* point estimates, whereas the log score is undefined for ensembles and point masses;
  2. it is resolved over τ, separating mean from uncertainty;
  3. it equals the MSE at τ = 0, keeping continuity with DA practice.

  The chain decomposition itself is not unique to FMS: the expected log score splits the same way through KL. The paper says so.

---

## 1. Introduction (≈ 1.3k words, 1 figure)

**Purpose:** the practical puzzle, the decomposition idea, the three questions.

1.1 **The puzzle.**
- GLORYS vs DUACS on SSH. Fig. 1a: a Gulf Stream panel, or the effective-resolution numbers 239 km vs 151 km.
- JAMES 2020: a learned prior beats DA with the true ODE. Fig. 1b: reconstructions.
- Neural-DA skill claims.
- Three candidate explanations: model error, the algorithm (e.g. filtering, Gaussian updates, finite ensembles), or the information a scheme uses.

1.2 **The idea.**
- Every scheme estimates `E[x | C]` and, ideally, `p(x | C)`.
- Its error splits into irreducible, restriction, misspecification and approximation terms.
- Two schemes with equal RMSE can have different budgets, and only the budget says what to fix.
- The same chain extends to uncertainty with a single proper score.

1.3 **Questions.**
- **Q1** With a perfect model, how close does each scheme get to `E[x | C]`, and where does the rest go?
- **Q2** What does model error do to the budget, and to the uncertainty each scheme reports?
- **Q3** Does the form of the prior — derived from the model equations or learned from data — drive performance?

1.4 **Contributions** (three bullets, from §0) and **what the paper is not** (four bullets).

1.5 **Roadmap.**

Source for 1.1: presentation slides 3–4. ⏳ figure permissions / remake of Fig. 1.

---

## 2. DA as estimation of a conditional distribution (≈ 1.2k words, 1 table)

**Purpose:** set the target and the error chain at τ = 0. Every DA commitment lands on one link of the chain.

2.1 **State-space model and context.**
- Two-scale dynamics with per-window parameters θ and stochastic forcing z.
- Observations `y_k = H x_k + ε_k`, `ε ~ N(0, R)`.
- `p(x | C) ∝ p(y | x) p(x | θ, z)`.
- `E[x | C]` is the MSE-optimal estimator; `E tr Cov(x | C)` is the irreducible error.
- Kept from the current §2.1, with eqs. dynamics/obs/ssm/context.

2.2 **The error chain.** `m = E[x|C] → m_S = E[x|S] → m̃ = E_p̃[x|S] → m̂`

  `MSE(m̂) = E tr Cov(x|C) + E‖m − m_S‖² + E‖m_S − m̂‖²`
- Both splits are exact by the tower property, provided `m̂` depends only on S and on randomness independent of the truth.
- The last term is `misspecification + approximation + cross`. The paper says plainly that these two are separated by contrasts, not by an identity.

2.3 **Where DA's commitments land** (Table 1, from slides 6 and 8):

| commitment | link | example |
|---|---|---|
| filtering (`S = y_{t'≤t}`), cycling | restriction | EnKF/ETKF vs ETKS; 4D-Var sub-windows |
| ignoring θ, z | restriction | DirectUNet, CFM, SDA1 |
| model error in Φ, θ, z | misspecification | S1 for every model-based scheme |
| Gaussian update, mode instead of mean, hard constraint (Q = 0), tempered likelihood | misspecification (structural) | ETKF, Strong-4DVar, SDA guidance |
| finite ensemble, local optimisation, finite network and training data | approximation | N = 30; LBFGS; 1200 epochs |

The design-matrix factors (F1 prior–likelihood split, F2 dynamics source, F3 sequential, U-state, U-model) go in Table 2 (§5) as labels, with the definitions in appendix B. They are no longer the spine.

---

## 3. Scoring the uncertainty: the error chain along τ (≈ 1.5k words, 1 figure, 1 table)

**Purpose:** extend the chain to the reported uncertainty with one score. The flow-matching notation is kept to what the score and the hybrid need.

3.1 **The operator of a scheme.**
- Path `x_τ = τ x₁ + (1 − τ) x₀`, with `x₀ ~ N(0, I)` in z-scored units.
- For any output distribution q, `Ψ_q(x_τ, τ) = E_q[x₁ | x_τ] = Ψ_mean + Ψ_anom`.
- Three cases (Table 3):
  - **point estimate** (Strong/Weak-4DVar, DirectUNet): `Ψ = m̂`, `Ψ_anom ≡ 0`;
  - **Gaussian N(m, v)** (EnKF, ETKF, ETKS in Gaussian form): `Ψ = m + snr·v/(1 + snr·v)·(x_τ/τ − m)`, `snr = τ²/(1 − τ)²`;
  - **ensemble or learned sampler** (CFM, SDA, KDE of members): a posterior-weighted average over samples.
- One sentence notes that CFM learns Ψ directly. The general representation is pointed to the ML paper.

3.2 **The score.** `FMS_τ(q) = E‖Ψ_q(x_τ, τ) − x₁*‖²`. Properties (proofs in appendix A):
- (i) τ = 0: the MSE of the mean;
- (ii) strictly proper for τ ∈ (0, 1);
- (iii) a Gaussian is minimised at spread² = MSE for every τ;
- (iv) τ → 1: a universal floor plus the Hyvärinen score;
- (v) the dsnr-integral is the log score up to a constant;
- (vi) a point estimate scores its MSE at every τ.

3.3 **The chain at every τ.** `FMS_τ(q) = mmse_p(τ) + R_τ(S) + D_τ(q ‖ p_S)`
- `R_τ(S) = E‖Ψ_p − Ψ_{p_S}‖²` (restriction).
- `D_τ = E‖Ψ_q − Ψ_{p_S}‖² ≥ 0`, zero iff `q = p_S` (misspecification + approximation).
- Exact under the same condition as §2.2.
- Corollaries:
  - differences between schemes are exact operator divergences; `mmse_p(τ)` is unknown but common to all;
  - the **filter/smoother identity along τ**;
  - **reading the curve**: a restriction raises `FMS_τ`, a misspecification bends it at large τ.

3.4 **Derived diagnostics.**
- **Calibration loss** and `c*(τ)`: the share of `FMS_τ` removed by the best scalar variance rescaling.
- **Shape gain**: KDE vs Gaussian at the same variance.
- **Pooled spread/skill**, `sqrt(E v / E e²)`, and the unexplained share.
- The per-window spread/RMSE ratio is biased low and is **not** used.

3.5 **Relation to CRPS and the log score** (one paragraph; the positioning of §0).

Fig. 2: schematic of the chain and of `FMS_τ` curves for a restriction vs a misspecification (synthetic Gaussian example, appendix A).

Status:
- Theory: ✅ in `docs/results/l96_p1_fm_score.md`; the proof of the τ-chain still needs writing up (⏳ appendix A).
- Code: ✅ `evaluation/fm_score.py`.

---

## 4. Experimental design (≈ 1.3k words, 1 figure, 2 tables)

4.1 **System.**
- Two-scale L96: 8 slow + 32 fast variables, `dt = 0.001`, 3000-step windows.
- Per-window θ (±20%) and stochastic forcing z.
- 24 observed channels, R = 0.5.

4.2 **Cases.**
- **S0**: the assimilating model is the truth.
- **S1**: the model is told biased `θ_da` (+10%) and a corrupted forcing; the truth is unchanged.
- **δ sweep** (⏳): δ ∈ {0, 5, 10, 20, 40%} × stated σ ∈ {0, 20%}, plus one structural error.

4.3 **Observing systems.**
- The regular 30-obs grid.
- The canonical random layout: 10–100 obs at stratified times, 4–16 fast channels, the exact obs the DA schemes assimilate.
- The n_obs × k factorial and out-of-range probes, for §6.3.

4.4 **Schemes** (Table 2, with design-matrix labels):
- **DA**: ETKF, EnKF, ETKS (30 members; 100 as sensitivity), Strong-4DVar, Weak-4DVar.
- **Learned**: DirectUNet, VanillaCFM, PredictStateCFM, SDA1/SDA2/SDA3-fix, and the DirectUNet → SDA3-fix hybrid.
- **Training recipe** (benchmark framework): random-observing-system training with fresh noise, 1200 epochs, 3 seeds, monai M tier.

4.5 **Matched information and the oracle confound.**
- Every main arm sees the same y.
- Model-based arms get the S0/S1 model.
- Learned arms trained on the true system are a **reference bound** at S1. M-trained arms (σ = 0 / 20%) are the matched comparison (⏳).

4.6 **Tuning parity.** Every inference hyper-parameter is tuned on 50 validation windows:
- inflation per case;
- ETKS lag and retro-inflation;
- Weak-4DVar q and iterations;
- SDA guidance weight;
- hybrid τ₀ and guidance.

The sweeps go in appendix C.

4.7 **Scoring.**
- RMSE and pooled spread/skill.
- `FMS_τ` at τ ∈ {0, 0.25, 0.5, 0.75} plus the uniform-dτ integral: Gaussian form for every scheme, KDE for the sample-based ones; 4 x₀ draws with common random numbers.
- Physical units², 95% window-bootstrap intervals, paired tests.

Status:
- ✅ everything except the δ sweep, the M-trained arms, and Weak-4DVar on the random layout. The latter needs the NaN-mask fix in `L96Weak4DVar._obs_cost`.
- The Weak-4DVar regular-grid benchmark rows are computed (SLURM 58092/58093, 40/40 chunks) but not yet assembled.

---

## 5. Q1 — With a perfect model, where does each scheme's error go? (≈ 1.6k words, 2 figures, 1 table)

5.1 **Overview** (Table 4, both observing systems, S0):

| scheme | RMSE regular / random |
|---|---|
| ETKF | 0.610 / 0.679 |
| EnKF | 0.641 / 0.706 |
| ETKS | 0.497 / 0.572 |
| Strong-4DVar | 0.703 / 0.742 |
| Weak-4DVar | ⏳ |
| DirectUNet-M | 0.340 / 0.450 |
| PredictStateCFM-M | 0.341 / 0.443 |
| VanillaCFM-M | 0.351 / 0.462 |
| SDA1 / 2 / 3-fix | 0.464 / 0.453 / 0.455 (regular) |
| hybrid | 0.293 / 0.367 |

Learned schemes lead by 30–45% at S0 on this testbed. The section explains why. ✅

5.2 **The budget at τ = 0** (Table 5: rows = scheme classes, columns = the chain terms, cells = MSE estimates with the contrast named):
- **Irreducible ≤** hybrid MSE (RMSE 0.293). Upper bound only; no lower bound is available. ✅
- **Restriction (filtering).** The filter → smoother identity on one ensemble: MSE gap 0.159 (N = 30) and 0.162 (N = 100), regular S0. It is approached as the ensemble grows. Smoothing removes 16–19% of the RMSE. ✅
- **Approximation (finite ensemble).** N = 30 → 100 at re-selected inflation: ETKS 0.497 → 0.393 (−38% MSE). The 30-member DA deficit vs the best learned scheme falls from 1.46× to 1.16×. ✅
- **Approximation (optimisation).** 4D-Var LBFGS 10 → 40 iterations: 0.721 → 0.545 (validation). ✅
- **Approximation (training).** DirectUNet 400 → 1200 epochs: 0.380 → 0.340; with 3000 windows, 0.334. Near saturation. ✅
- **Restriction (context).** SDA1 vs SDA2/SDA3-fix (θ): 0.464 vs 0.453/0.455. ✅ SDA ± forcing z and DirectUNet ± (θ, z): ⏳.
- **Structural misspecification at S0.** The residual of `D_τ` after the approximation contrasts: Strong-4DVar (MAP, hard constraint) is the largest. 🟡

**Reading:** at S0 the DA deficit is mostly honest — restriction and approximation. It shrinks with smoothing, ensemble size and iterations.

5.3 **The budget along τ** (Fig. 3: `FMS_τ` curves per scheme, S0, both observing systems).

| scheme | FMS at τ = 0 / 0.5 / 0.75 (regular S0) | calibration loss at τ = 0.75 |
|---|---|---|
| ETKF | 0.187 / 0.125 / 0.050 | 14% |
| ETKS | 0.129 / 0.107 / 0.059 | 34% |
| PredictStateCFM-M | 0.054 / 0.040 / 0.0195 | 2% |
| hybrid | 0.039 / 0.034 / 0.0200 | 16% |

- **ETKS vs ETKF**: the smoother has the better mean and the worse distribution from τ = 0.75. It buys accuracy with an over-confident ensemble.
- **Flows**: near-calibrated (pooled spread/skill 0.85–0.93).
- **SDA**: calibration loss 7–18% (tempered likelihood).
- Generalised identity along τ: ⏳ (FMS of the ETKS's own filter pass).

5.4 **The observing system and density** (Fig. 4: RMSE vs number of obs and of fast channels).
- **Crossover**: DA is best at ≤ 10 obs per window; learned schemes win from ~20.
- **Observing-system transfer** (random / regular ratio): Strong-4DVar 1.06, ETKS 1.15, SDA 1.21–1.24, CFM 1.30, DirectUNet 1.32 — explicit H, R buys robustness.
- **Out of range**: from 300 to 1000 obs, DirectUNet 0.17 → 0.34 while SDA keeps improving, 0.30 → 0.25.

Read as `D_τ` under observing-system shift (an amortised map misspecified off-distribution). ✅ `l96_benchmark_extended.md` §3–5.

---

## 6. Q2 — What does model error do to the budget? (≈ 1.8k words, 2 figures, 1 table)

6.1 **The budget at S1, τ = 0** (Table 6).
- **DA**: ETKF 1.409, EnKF 1.416, ETKS 1.338, Strong-4DVar 1.436. The ETKS MSE is ×7 its S0 value.
- **100 members change nothing**: ETKS 1.338 → 1.338.
- **Smoothing gains only 5%**, and the filter/smoother identity **fails**: MSE gap 0.237 vs variance gap 0.534 and mean shift 0.913. Misspecification dominates.
- **Learned** (oracle bound): DirectUNet 0.339, PredictStateCFM 0.338, SDA3-fix 0.455, hybrid 0.289. All flat.

✅ M-trained arms ⏳.

6.2 **The misspecification signature along τ** (Fig. 5: `FMS_τ` S0 vs S1 per scheme).
- **DA calibration loss grows**: ETKS 34 → 63% (c* ≥ 8), ETKF 14 → 23% on the regular grid; 3 → 14% (ETKF) and 21 → 47% (ETKS) on the random layout.
- **Tails**: DA ensembles become short-tailed (KDE vs Gaussian +17–30% at τ = 0.75).
- **Learned rows are invariant**, except SDA2, whose loss falls (7 → 3%) from conditioning on the biased θ.
- **Aggregate calibration is not calibration**: on random S1 the ETKF's pooled spread/skill is 0.99, yet it loses 12% at τ = 0.75. A scalar inflation matches the total error but not where it occurs.

✅

6.3 **Marginal value of observations** (headline, Fig. 6: RMSE vs n_obs per scheme, S0/S1).
- **Strong-4DVar**: its gain from 15 → 30 obs collapses 7.0× under model error, then sits on a floor (1.43 → 1.38 from 30 to 1000 obs).
- **Filters** keep 1.8–1.9×.
- **Weak-4DVar, learned**: ⏳ / 🟡.

✅ Strong / filters; ⏳ Weak-4DVar row.

6.4 **Representing model uncertainty: from misspecification back to restriction** (Fig. 7: RMSE and calibration loss vs δ, per U-model arm).
- **Weak-4DVar Q** (validation; benchmark rows computed, to assemble):
  - S0: no gain (q 0.03: 0.553 vs Strong 0.545);
  - S1: −25% (q 0.3: 1.099 vs 1.473).
- **SDA2 (clean θ) leaks the bias; SDA3-fix (noisy θ) stays flat.** Hybrid with the 400-epoch priors: SDA2 0.310 → 0.333 vs SDA3-fix 0.312 → 0.307.
- **Perturbed-parameter ETKF**: ⏳.
- **δ ≤ σ vs δ > σ** (δ sweep): ⏳.

🟡

6.5 **Value of true-system data** (oracle gap): learned-on-truth minus learned-on-M. It measures what reanalysis-trained ML implicitly spends. ⏳

---

## 7. Q3 — Does the form of the prior drive performance? (≈ 1.2k words, 1 figure)

7.1 **Same likelihood, two priors** (Fig. 8, RMSE and `FMS_τ`).
- Model-derived prior: Strong-4DVar 0.703, ETKS 0.497 (30) / 0.393 (100), Weak-4DVar ⏳.
- Learned joint-window prior: SDA 0.453–0.464.
- The learned prior beats the true-ODE hard constraint and the 30-member smoother, **but not the 100-member smoother**. The ordering is set by the approximation term, not by the fidelity of the prior.
- Along τ, the SDA operator is better calibrated than the smoother's (7–18% vs 34% loss).

✅

7.2 **A better prior is not a better estimate.**
- SDA 400 → 1200 epochs gains 7–9%.
- The true ODE used as a hard prior (Strong-4DVar) is the worst DA row at S0.
- Back to the JAMES 2020 puzzle: a learned prior beats the true ODE through approximation and restriction terms, not through a better prior.

✅

7.3 **Conditioning the prior.** θ (SDA1 → SDA2/SDA3-fix) and forcing z (⏳) as restriction terms; SDA2 at S1 as the misspecification case. 🟡

7.4 **Composition** (one paragraph, flow notation).
- The hybrid composes DirectUNet's mean (Ψ_mean) with SDA3-fix's guided sampler (Ψ_anom).
- **Best mean everywhere**: 0.293 / 0.367, flat at S1.
- **Not the best distribution**: the flows overtake it at τ = 0.75 (0.0195 vs 0.0200 regular; 0.025 vs 0.029 random).
- The score separates the two components it composes.

✅

---

## 8. Discussion (≈ 1.5k words)

8.1 **Lessons as budget statements** (slide 15, made precise):
- **(a)** At a perfect model, the DA deficit is restriction + approximation, i.e. honest and reducible.
- **(b)** Under model error, misspecification dominates and shows in the large-τ score before it shows in the RMSE.
- **(c)** A better prior does not imply a better estimate; approximate priors can beat the true physics.
- **(d)** Model-based DA can misreport the information value of its conditioning variables.

8.2 **What classical DA still gets right.** The sparse-observation regime; observing-system transfer (explicit H, R); calibration when its assumptions hold; no training set.

8.3 **Back to the puzzle.** What the budgets suggest for GLORYS vs DUACS (misspecification under model error) and for learned-prior skill (approximation and restriction at S0). Stated as hypotheses for real systems.

8.4 **Limitations.**
- `mmse_p(τ)` is unknown: only differences and the upper bound are available.
- DA is scored in Gaussian form (KDE where members exist).
- One idealised system.
- The oracle confound (partly measured in §6.5).
- Tuning parity.
- The cross term between misspecification and approximation is only bounded by contrasts.

8.5 **Positioning** (§0, with verified citations).

8.6 **Outlook.**
- The flow representation of any DA scheme as a research programme (ML paper).
- Self-supervised adaptation of learned schemes to the observed system.
- Transfer to QG.

## 9. Conclusion (≈ 0.4k words)

The thesis and one number per question:
- **Q1**: smoothing −16–19% and 30 → 100 members −38% MSE at S0.
- **Q2**: ETKS calibration loss 34 → 63%; Strong-4DVar's marginal value of observations collapses 7×.
- **Q3**: SDA 0.455 vs ETKS 0.497 (30) / 0.393 (100).

## Appendices

- **A.** `Ψ_q` closed forms (point, Gaussian, mixture); proofs of the FMS properties and of the chain at every τ; synthetic restriction-vs-misspecification example (Fig. 2).
- **B.** The design matrix: F1, F2, F3, U-state, U-model, and every scheme's row.
- **C.** Tuning sweeps: ETKF/EnKF/ETKS inflation (30 and 100 members), ETKS lag, Weak-4DVar q × iterations, SDA guidance weight, hybrid τ₀ / gw.
- **D.** Sensitivities: ensemble size, LBFGS iterations, training budget and data, flow sampler steps.
- **E.** The P1-protocol tier study: S+ / M / L, VanillaCFM vs PredictStateCFM, P1 τ = 0 heads (`p1_l96_benchmark.md` annex A).
- **F.** FMS estimators: Monte Carlo over x₀, KDE bandwidth, snapshot (24-channel) blocks and their effective-sample-size degeneration at τ = 0.75, bootstrap.
- **G.** Rank histograms.

---

## 10. Exhibits

| # | exhibit | section | status |
|---|---|---|---|
| Fig. 1 | the puzzle: SSH products; JAMES 2020 reconstructions | §1 | ⏳ remake |
| Fig. 2 | chain schematic + synthetic FMS curves (restriction vs misspecification) | §3 | ⏳ |
| Fig. 3 | FMS_τ curves, S0, both observing systems | §5.3 | 🟡 data ✅, plot ⏳ |
| Fig. 4 | RMSE vs n_obs and fast channels, S0 | §5.4 | ✅ (`l96_benchmark_extended.md`) |
| Fig. 5 | FMS_τ S0 vs S1 per scheme | §6.2 | 🟡 data ✅, plot ⏳ |
| Fig. 6 | marginal value of observations, S0/S1 | §6.3 | 🟡 (Weak-4DVar ⏳) |
| Fig. 7 | RMSE and calibration loss vs δ per U-model arm | §6.4 | ⏳ |
| Fig. 8 | priors: model-derived vs learned, RMSE + FMS | §7.1 | 🟡 |
| Fig. 9 | reconstruction examples (best / median / worst windows) | §5.1 or App. | ✅ (`p1_l96_benchmark.md`) |
| Tab. 1 | where DA's commitments land | §2.3 | ✅ |
| Tab. 2 | schemes, with design-matrix labels and tuned hyper-parameters | §4.4 | ✅ |
| Tab. 3 | Ψ_mean / Ψ_anom per scheme class | §3.1 | ✅ |
| Tab. 4 | overview RMSE S0/S1, both observing systems | §5.1 | ✅ (Weak-4DVar ⏳) |
| Tab. 5 | budget at τ = 0, S0 | §5.2 | 🟡 needs pooled-MSE recomputation |
| Tab. 6 | budget at τ = 0 and τ = 0.75, S1 | §6.1–6.2 | 🟡 |

## 11. Claims (v1 K1–K8, re-homed)

| claim | home | status |
|---|---|---|
| K1 filtering is a restriction; the identity holds at S0 and fails at S1 | §5.2, §6.1 (+ along τ ⏳) | ✅ τ = 0 |
| K2 explicit H, R buys observing-system robustness; amortisation buys in-distribution accuracy | §5.4 | ✅ |
| K3 learned window prior vs mechanistic prior — **qualified**: it depends on approximation (N = 30 vs 100) | §7.1 | ✅ |
| K4 model error makes the hard constraint a misspecification (headline) | §6.3 | 🟡 Weak-4DVar row |
| K5 representing model uncertainty converts misspecification while δ ≤ σ | §6.4 | 🟡 |
| K6 explicit uncertainty is tunable in aggregate, not in structure; implicit is near-calibrated and invariant | §5.3, §6.2 (FMS) | ✅ |
| K7 oracle gap | §6.5 | ⏳ |
| K8 composition: best mean, not best distribution | §7.4 | ✅ (revised by FMS) |

## 12. What goes where

| item | P1 | elsewhere |
|---|---|---|
| FMS_τ definition, properties, chain along τ | §3 + App. A | — |
| Ψ closed forms for point / Gaussian / mixture | §3.1, App. A | — |
| "every DA scheme is a flow" as a programme; Laplace 4D-Var; score ↔ Ψ for SDA | outlook sentence | **ML paper** |
| Affine velocity decomposition; τ = 0 head probes (1200-epoch heads trail their samplers: 0.430 / 0.457 vs 0.341 / 0.351) | App. E mention | **ML paper** |
| P1-protocol tiers, CFM parameterization | App. E | — |
| QG | outlook | separate study |
| unrolled / 4DVarNet solvers | out | **P2** |

## 13. Work before submission, in order

| # | item | serves | cost |
|---|---|---|---|
| 1 | Assemble the Weak-4DVar benchmark rows (S0 q 0.03, S1 q 0.3; 40/40 chunks done) → extended + P1 reports | §5.1, §6.3–6.4, §7.1 | analysis + PR |
| 2 | Fix "under-dispersed (0.63)" in `p1_l96_benchmark.md`; state the spread/RMSE definition in the report columns | §3.4 | small PR |
| 3 | FMS of the ETKS with its own filter pass (N = 30 and 100, S0/S1) → identity along τ; FMS of the 100-member rows | §3.3, §5.2–5.3, §6.1 | ~4 GPU-h |
| 4 | Budget tables in pooled MSE (Tab. 5, 6); FMS figures (Fig. 3, 5); uniform-dτ integral | §5–6 | analysis |
| 5 | Weak-4DVar on the random layout (NaN-mask fix + test, then about 33 GPU-h) | §5.1, §5.4 | fix + runs |
| 6 | M-trained learned arms (σ = 0 / 20%) | §4.5, §6.1, §6.5 | training |
| 7 | δ sweep + one structural error, all schemes | §6.4, Fig. 7 | runs |
| 8 | SDA ± forcing z; DirectUNet ± (θ, z) | §5.2, §7.3 | training |
| 9 | Perturbed-parameter ETKF | §6.4 | runs |
| 10 | Appendix A proofs; Fig. 2 synthetic example | §3 | writing |

Items 1–4 make §2–§5 and §7 writable. Items 5–9 complete §6.

## 14. Open decisions

| decision | default |
|---|---|
| Venue | QJRMS or JAMES (DA audience); the ML-facing material stays in the ML paper |
| FMS columns in the main tables | τ ∈ {0, 0.5, 0.75} + the uniform-dτ integral |
| Units | physical units² with bootstrap intervals |
| Gaussian vs KDE for DA | Gaussian in the main text, KDE as a check |
| Section order | Q1, Q2, Q3 (Q3 last, so the hybrid and the priors close the results) |
| Title | the §0 candidate |
