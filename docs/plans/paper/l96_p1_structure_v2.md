# P1 — revised paper structure, v2 (from the 2026-10-02 presentation)

**Status:** SCOPING — second revision of the P1 structure, drawn from the presentation `docs/papers/Restriction or Misspecification DA under Model Error.pdf` (30 slides) and the current benchmark (`reports/l96/outputs/p1_l96_benchmark.md`, `l96_benchmark_extended.md`, after #311/#312). It supersedes the **outline** (§4) of `l96_p1_structure_revision.md`; that doc's formal core, claims K1–K8 and runs table stay the reference where this one does not restate them. The LaTeX has not been touched. **Its outline (§3) is superseded by `docs/plans/paper/l96_p1_structure_v3.md` (flow matching as the representation and the metric); §1 and §5 here still apply.**

## 1. What the presentation changes

The v1 structure is organised around a **design matrix** (F1, F2, F3, U-state, U-model): results are one-factor contrasts read across its columns. The presentation keeps the matrix but moves it to the supplementary slides, and builds the talk on something else:

1. **A motivation from practice, not from the commitments**: model-based reanalyses lose to observation-only products (GLORYS vs DUACS, SSH, Gulf Stream); learned priors beat DA run with the *true* ODE (Fablet et al., JAMES 2020); many neural studies report better skill than DA. *Why?* (slides 3–4)
2. **The error chain of the conditional expectation as the spine**: `m = E[x|C] → m_S = E[x|S] → m̃ = E_p̃[x|S] → m̂`, so `MSE = irreducible + restriction + misspecification + approximation`. Each scheme is read as an *allocation of an error budget* (slides 5–8): the central question becomes "do model-based and neural DA allocate the budget differently?".
3. **Three questions instead of eight claims** (slide 9):
   - **Q1** (S0) How do DA schemes inform the irreducible uncertainty when the dynamics are error-free?
   - **Q2** (S1) How does model misspecification change that?
   - **Q3** Does the form of the prior drive performance? (model-derived prior vs learned prior with the same likelihood: Strong/Weak-4DVar, ETKS vs SDA)
4. **One budget table per question** (slides 11, 12, 14): rows = scheme classes (filter, smoother, end-to-end neural, SDA without / with forcing), columns = the four terms, entries = order-of-magnitude shares.
5. **A synthesis that is a list of lessons, not claims** (slide 15): the decomposition goes beyond the irreducible term; a better prior does not necessarily give a better estimate; approximate priors can beat the true physics; model-based DA may misreport the information gain of its conditioning variables (y, θ, z).
6. **Flow matching moves to "What next?"** (slides 16–17): CFM as a generic representation of DA algorithms (`E[x1|x_τ,C] = Ψ_mean + Ψ_anom`, every scheme a flow) is the outlook, not the instrument of the paper.
7. **A sharper subtitle**: *Do DA algorithms quantify their estimation uncertainty?* — calibration becomes a first-class outcome next to RMSE.

**Proposed reading of this for the paper:** keep the thesis and the matched-information design of v1, but make the **error chain + three questions** the structure of the Results, the **budget tables** the main exhibits, and the design matrix a Methods table (Table 1) that labels every row. The paper stays a diagnosis paper.

## 2. Title, thesis, abstract spine

- **Title** (presentation): *Restriction or misspecification? Dissecting classical and neural data assimilation*. Alternative keeping the subtitle: *… — do data assimilation schemes quantify their estimation uncertainty?*
- **Thesis** (unchanged, slide 28): every structural commitment of DA is either an information **restriction** (costs variance honestly, preserves calibration) or a **misspecification** (costs bias silently, destroys calibration); model error turns harmless restrictions into misspecifications; representing model uncertainty converts some back, over a measurable range.
- **Abstract spine**: motivation (DA loses to learned / observation-only estimators in practice) → error chain → testbed with matched information → Q1, Q2, Q3 answers, one number each → lesson (the error budget, not the prior, decides; calibration tells restriction from misspecification).

## 3. Section outline

Evidence: ✅ exists (source), 🟡 partial, ⏳ pending run. Numbers are regular / random test set, S0, per-window RMSE, benchmark framework (1200 epochs, 3 seeds), unless stated.

**§1 Introduction**
- Motivation: GLORYS vs DUACS; learned prior beating the true ODE (JAMES 2020); neural-DA skill claims. Is it model error, the DA algorithm, or something else?
- The idea: read every scheme as an estimator of `E[x|C]` and decompose its MSE along the chain. A model-based and a learned scheme can reach the same MSE with different budgets, and only the budget says what to fix.
- Questions Q1–Q3; contributions; what the paper is not (no new method; state target only; unrolled solvers out → P2).

**§2 DA as estimation of the conditional expectation** (slides 2, 5–7)
- 2.1 State-space model, context `C = (y, θ, z)`, posterior `p(x|C) ∝ p(y|x) p(x|θ,z)`; `E[x|C]` as the MSE-optimal target; irreducible term `E tr Cov(x|C)`.
- 2.2 The error chain and its four terms; the orthogonality that makes the MSE additive along the chain (or the inequality form where it is not exact).
- 2.3 Where each DA commitment lands (slide 6): filter vs smoother → restriction (`S = y_{t'<t}`); MAP vs mean, Gaussian update, model error, hard constraint → misspecification; finite ensemble, local optimisation, finite training data and architecture → approximation.
- 2.4 **Table 1 — the error chain, scheme by scheme** (slide 8), with the design-matrix columns (F1 split, F2 dynamics source, F3 sequential, U-state, U-model) as extra columns: one table carries both the budget reading and the structural labels. The F1–U-model definitions go to a short paragraph + appendix (slides 29–30).

**§3 Measuring the chain** (new; v1 §3 formal core, reorganised by term)
The paper has no true posterior, so each term is measured by a controlled contrast:

| term | estimator | status |
|---|---|---|
| restriction (filter → smoother) | the identity `MSE_F − MSE_S = var_F − var_S = E‖m_F − m_S‖²` on one ensemble realisation (ETKS with its own filter pass) | ✅ regular S0: MSE gap 0.159 (N = 30), 0.162 (N = 100); holds approximately at S0 with 100 members, fails at S1 (`docs/plans/paper/l96_p1_structure_revision.md` §3) |
| restriction (context ignored) | conditioning contrasts: SDA1 vs SDA2/SDA3-fix (θ), SDA without / with forcing z (slide 14), DirectUNet with / without (θ, z) inputs | 🟡 θ ✅ (SDA2 0.453, SDA3-fix 0.455 vs SDA1 0.464); z ⏳; DirectUNet(θ, z) ⏳ |
| approximation (finite ensemble) | N = 30 vs 100 at re-selected inflation | ✅ ETKS 0.497 → 0.393 (MSE −38%), S1 unchanged (1.338 → 1.338) |
| approximation (optimisation) | Weak-/Strong-4DVar LBFGS 10 vs 40 iterations | ✅ val: Strong 0.721 → 0.545 at S0 |
| approximation (training) | 400 vs 1200 epochs, 1000 vs 3000 windows | ✅ DirectUNet 0.380 → 0.340 → 0.334: near saturation at 1200 epochs |
| misspecification (model error) | S0 → S1 at a fixed scheme; δ sweep; U-model arms (Weak-4DVar Q, SDA3-fix, perturbed-parameter ETKF) | ✅ S0→S1 (ETKS MSE ×7); 🟡 Weak-4DVar Q (val: S1 1.473 → 1.099 at q 0.3); ⏳ δ sweep |
| misspecification (structural, at S0) | MAP vs mean; tempered likelihood; Gaussian update — bounded by the gap to the best scheme | 🟡 |
| irreducible | **upper-bounded** by the best scheme: hybrid RMSE 0.293 regular S0 (MSE ≈ 0.086; recompute pooled) | ✅ bound; no lower bound (stated) |
| calibration signature | pooled spread/RMSE and the unexplained share `1 − pooled²(N+1)/N`: restriction keeps it, misspecification destroys it | ✅ (re-pool on the benchmark framework ⏳) |

The Ψ_mean + Ψ_anom flow representation is kept only as far as the hybrid needs it (half a paragraph); the rest goes to §8 Outlook / appendix.

**§4 Experimental setup** (slides 10, 18)
- Two-scale L96, S0 (perfect model) / S1 (biased θ_da +10%, corrupted forcing; same truth).
- Observing systems: regular 30-obs grid and the canonical random layout (10–100 obs, 4–16 fast channels) — both reported everywhere; factorial n_obs × k and out-of-range probes for the density analysis.
- Schemes: ETKF, EnKF, ETKS (30 members; 100 as sensitivity), Strong-4DVar, Weak-4DVar; DirectUNet, VanillaCFM, PredictStateCFM; SDA1 / SDA2 / SDA3-fix; DirectUNet → SDA3-fix hybrid.
- Benchmark framework: random-observing-system training with fresh noise, 1200 epochs, 3 seeds, M tier (`p1_l96_benchmark.md` §1–4).
- Matched-information rule and the **oracle confound**: learned rows trained on the true system are a reference bound at S1, not a matched arm (slide 10, footnote) — M-trained arms ⏳.
- Tuning parity: every inference hyper-parameter tuned on validation windows (inflation, lag, Q and iterations, guidance weight, τ₀).

**§5 Q1 — with a perfect model, how close does each scheme get to E[x|C]?** (slide 11)
- 5.1 Overview: DA 0.497 (ETKS) – 0.703 (Strong-4DVar); learned 0.340 (DirectUNet) ≈ 0.341 (PredictStateCFM); SDA 0.453–0.464; hybrid 0.293. ✅
- 5.2 **Budget table S0** (one row per class, both observing systems): filter → smoother restriction (identity, ✅); smoother's remaining excess over the irreducible bound split into finite-ensemble approximation (N = 30 → 100, ✅) and structural misspecification (the residual, 🟡); neural: approximation near saturation (✅), restriction from ignoring (θ, z) (⏳), structural misspecification 0 in distribution. *Reading:* at S0 the DA deficit is mostly approximation + restriction, i.e. honest: it shrinks with members and with smoothing.
- 5.3 Calibration at S0: filters/ETKS vs flows vs SDA (pooled spread/RMSE and unexplained share; on the benchmark framework the per-window spread/RMSE is 0.63–0.71 for the flows, 0.41–0.55 for SDA, 0.41–0.76 for the DA ensembles). ✅ / re-pool ⏳
- 5.4 The observing system and density (crossover: DA best ≤ 10 obs; learned from ~20; out-of-range behaviour; explicit H, R robustness). ✅ `l96_benchmark_extended.md` §3–5

**§6 Q2 — under model error, what does misspecification do?** (slide 12)
- 6.1 **Budget table S1**: DA misspecification dominates (ETKS MSE ×7 vs S0; the filter–smoother gain shrinks to 5%; 100 members change nothing at S1: 1.338 → 1.338) ✅; learned rows flat (0.339 / 0.338) but as an oracle bound ✅ / M-trained ⏳.
- 6.2 Marginal value of observations (headline figure): Strong-4DVar 7.0× collapse and flat floor; filters keep 1.8–1.9×; Weak-4DVar ⏳ (benchmark run in progress); learned. ✅ / 🟡
- 6.3 Representing model uncertainty converts misspecification into restriction: Weak-4DVar Q (val: −25% at S1, no gain at S0 — `q` 0.03 / 0.3 per case) 🟡; SDA2 (clean θ) leaks vs SDA3-fix (noisy θ) flat ✅; perturbed-parameter ETKF ⏳; δ sweep and structural error ⏳.
- 6.4 Calibration under model error: explicit-through-Φ uncertainty breaks (ETKS pooled 0.48 → 0.34 regular), implicit stays invariant. ✅
- 6.5 Value of true-system data (oracle gap): learned-on-truth vs learned-on-M ⏳.

**§7 Q3 — does the form of the prior drive the performance?** (slides 13–14)
- 7.1 Same likelihood, two priors: model-derived (Strong/Weak-4DVar, ETKS) vs learned joint-window prior (SDA). At S0 the learned prior beats the hard-constraint mechanistic one (SDA 0.455 vs Strong-4DVar 0.703) and the 30-member smoother (0.497), **but not the 100-member smoother (0.393)**: the ordering is decided by the approximation term, not by the prior's fidelity. ✅
- 7.2 "Improving the prior does not necessarily improve the estimate": SDA 400 → 1200 epochs gains 7–9%; the true ODE as prior (Strong-4DVar) is the worst DA row at S0. ✅
- 7.3 Conditioning the prior: SDA without / with forcing z (slide 14 rows), θ conditioning (SDA2 vs SDA3-fix at S1). 🟡 (z ⏳)
- 7.4 Composition: the hybrid (amortised mean + explicit-R learned sampler) is best on every test set (0.293 / 0.367, flat at S1). ✅

**§8 Discussion**
- 8.1 Synthesis (slide 15), each lesson tied to a budget-table cell.
- 8.2 What classical DA still gets right: sparse regime, observing-system transfer (explicit H, R), calibration when its assumptions hold, no training set (slide 23).
- 8.3 Back to the motivation: what the budgets say about GLORYS vs DUACS and about learned-prior skill (misspecification under model error vs approximation at S0).
- 8.4 Limitations: one idealised system; no true posterior (irreducible only upper-bounded); circularity (partly measured by §6.5); tuning parity.
- 8.5 Positioning (v1 §7.4, verified citations).
- 8.6 **Outlook — flow matching as a generic representation of DA algorithms** (slides 16–17): every scheme is a flow `Ψ_mean + Ψ_anom`; blending composes them.

**§9 Conclusion** — around the thesis and the three answers.

**Appendices**: A. design matrix and the five commitments (slides 26–30); B. tuning sweeps (inflation, lag, Weak-4DVar q and iterations, gw, τ₀); C. ensemble-size and training-budget sensitivities; D. P1-protocol tier study (S+ / M / L, CFM parameterization, SDA conditioning — `p1_l96_benchmark.md` annex A); E. rank histograms; F. flow representation.

## 4. Claims K1–K8 re-mapped to the questions

| v1 claim | v2 home | change |
|---|---|---|
| K1 filtering is a restriction | Q1 §5.2 | now one cell of the S0 budget |
| K2 explicit H, R buys robustness; amortisation buys accuracy | Q1 §5.4 | unchanged |
| K3 learned window prior beats the mechanistic one-step prior | Q3 §7.1 | **qualified**: true vs Strong-4DVar and the 30-member ETKS, false vs the 100-member ETKS |
| K4 hard constraint becomes a misspecification (headline) | Q2 §6.2 | unchanged; Weak-4DVar row pending |
| K5 representing uncertainty converts, while δ ≤ σ | Q2 §6.3 | Weak-4DVar Q evidence added |
| K6 explicit vs implicit uncertainty | Q1 §5.3 + Q2 §6.4 | split across the two questions |
| K7 oracle gap | Q2 §6.5 | unchanged |
| K8 best scheme composes | Q3 §7.4 | unchanged |

## 5. Numbers on the slides to update before reuse

| slide | as shown | current (benchmark framework) |
|---|---|---|
| 11–12 | DA filter S0 0.7–0.8, S1 1.40–1.50 | ETKF / EnKF 0.610 / 0.641 regular, 0.679 / 0.706 random; S1 1.41–1.48 (pre-#295 values on the slide) |
| 11–12 | DA smoother S0 0.57–0.60 | ETKS 0.497 regular, 0.572 random (the slide quotes the random layout) |
| 11–12 | neural S0 0.44–0.46, S1 0.45–0.47 | DirectUNet / CFM 0.340–0.351 regular, 0.443–0.462 random; S1 0.338–0.349 / 0.444–0.468 — **the rows mix test sets; one table per test set** |
| 14 | SDA 0.61–0.62 (400-epoch prior, gw 20, random) | SDA1/2/3-fix (1200 ep, gw 25): 0.453–0.464 regular, 0.550–0.572 random; S1 0.455–0.465 / 0.573–0.585 |
| 11–14 | shares "~30%", "> ~200%", "< ~15%" | to be recomputed as pooled-MSE shares from the §3 estimators (e.g. N = 30 → 100 removes 38% of the ETKS MSE at S0; S1 multiplies it by ~7) |
| 19 | filters pooled 1.08–1.13 (S0) / 0.33–0.37 (S1) | at the #295 inflation, regular: ETKF filter 0.77 → 0.60, ETKS 0.48 → 0.34; random 0.89 → 0.99 (filter) — the slide's values predate the retune |
| 20 | SDA 0.501 vs Strong-4DVar 0.703 | SDA3-fix 0.455 (1200 ep); also cite ETKS 0.497 / ETKS-100 0.393 (K3 qualification) |
| 21 | hybrid 0.310 → 0.333 vs 0.312 → 0.307 | 400-epoch priors; 1200-epoch hybrid 0.293 / 0.289 |
| 24 | ETKS, SDA 1200, ETKF/EnKF rerun "block submission" | done (#297/#299, #307/#308, #295/#298); Weak-4DVar benchmark running; M-trained arms and δ sweep still open |

## 6. Runs, by the question they serve

| run | serves | status |
|---|---|---|
| Weak-4DVar benchmark rows (S0 q 0.03, S1 q 0.3, 40 iterations, 200 windows) | Q2 §6.2–6.3, Q3 §7.1 | running (SLURM 58092/58093) |
| Weak-4DVar on the random layout (needs the NaN-mask fix in `L96Weak4DVar._obs_cost`) | Q1/Q2 random columns | ⏳ |
| Budget tables: pooled MSE per term, per scheme, both test sets, S0/S1 | §5.2, §6.1 | ⏳ (analysis only, no GPU) |
| SDA with forcing z (slide 14) | Q3 §7.3 | ⏳ (config to define) |
| DirectUNet with (θ, z) inputs | Q1 restriction of neural schemes | ⏳ (optional) |
| M-trained learned arms, σ = 0 / 20% | Q2 §6.5, oracle confound | ⏳ |
| δ sweep + one structural error | Q2 §6.3 | ⏳ |
| Perturbed-parameter ETKF | Q2 §6.3 | desirable |
| Pooled calibration re-run on the benchmark framework | §5.3, §6.4 | ⏳ (cheap) |

## 7. Open decisions

| decision | default proposed here |
|---|---|
| Spine of the Results | the three questions Q1–Q3 (presentation); the design matrix as Table 1 columns + appendix A |
| Main exhibits | one budget table per question, both test sets, pooled-MSE shares with the estimator named per cell |
| Flow matching in the paper | outlook (§8.6) + appendix F; only the hybrid uses it in the main text |
| Motivation | GLORYS vs DUACS and JAMES 2020 in §1; return to them in §8.3 |
| Calibration | first-class outcome (subtitle), but as the restriction/misspecification signature, not a separate claim |
| P1-protocol tier study | appendix D (as in the P1 report) |
| Title | *Restriction or misspecification? Dissecting classical and neural data assimilation* |
