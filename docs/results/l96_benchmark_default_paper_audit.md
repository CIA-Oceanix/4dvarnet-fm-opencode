# P1 paper draft vs the benchmark-default results — claims audit (2026-09-24)

**Status:** NOTES (2026-09-24). Claims audit of the P1 draft against `reports/l96/outputs/l96_benchmark_default.md`.

**Numbers:** `reports/l96/outputs/l96_benchmark_default.md`, `reports/l96/outputs/l96_benchmark_extended.md`, `reports/l96/outputs/l96_da_obs_count_dafw.md`.

Audit of `docs/papers/p1_structural_hypotheses/` (`main.tex`, `sections/01`–`07`)
against `reports/l96/outputs/l96_benchmark_default.md`. **Nothing in the paper is
edited here**; this is the list of what the new evidence changes, for the author.
Line numbers are per file at master `3a9ea31`.

## What changed in the evidence

| tag | fact |
|---|---|
| NEW-A | Under the benchmark default (random observing system redrawn every batch, fresh noise) the learned ranking flips: DirectUNet-M 0.380 < PredictStateCFM-M 0.409 < VanillaCFM-M 0.434 (regular S0, 3 seeds, sd ≈ 0.005). P1's DirectUNet was overfitting frozen per-window obs noise (a noise-redraw control recovers ~60% of its gain). |
| NEW-B | Random-observing-system test set: P1 fixed-obs checkpoints ×2.8–3.3, benchmark-default models ×1.25–1.30, SDA ×1.2, DA ×1.06–1.19. DA is the *least* sensitive to the observing system. |
| NEW-C | Learned beat DA at S0 on both test sets; learned S1/S0 ≈ 1.00, DA S1/S0 = 2.04–2.16 (was 1.69–1.95). |
| NEW-D | P1's L96 DA S0 was **not perfect-model**: per-window `fast_weights` never reached the DA forward model (it summed the fast variables unweighted), and S0 used the S1-tuned inflation 2.0. Fixed: ETKF 0.866 → 0.685, EnKF 0.894 → 0.711, Strong-4DVar 0.738 → 0.703. |
| NEW-E | SDA needs no retraining for the random set; guidance weight 20 was tuned on SDA1-M on the regular grid only. |

## Update 2026-09-24 — results since this audit was written

See `reports/l96/outputs/l96_benchmark_extended.md` for all numbers.

- **Marginal-value headline — re-run done.** With `fast_weights` passed, Strong-4D-Var's
  collapse *survives and grows*: 20.6% vs 3.2% (6.4×) on the original data configuration at
  the original inflation, 23.3% vs 3.4% (7.0×) on the benchmark data. The filters' "1.9×"
  does **not** survive: 1.1× (ETKF 11.9% vs 11.0%, EnKF 10.6% vs 9.9%) with only the fix,
  1.6–1.7× on the benchmark data at the per-case inflation. The 4D-Var-vs-filters contrast
  (C1, `tab:marginal`, abstract, conclusion) holds and sharpens; its filter numbers change.
- **The learned ranking depends on the training budget.** At the benchmark default's 400
  epochs DirectUNet-M led (0.380 vs 0.409 / 0.434); at 1200 epochs all three families gain
  9–19% and the gaps largely close (DirectUNet ≈ PredictStateCFM < VanillaCFM). Neither the
  P1 ranking (flows first) nor the 400-epoch ranking (DirectUNet first) should be quoted
  without the budget; ~85% of a 3000-window gain is training length.
- **SDA3's "noisy DA bias" arm was never tested in P1.** Its config inherited
  `biased: false`, so training windows had DA params equal to the true ones and SDA3 trained
  bit-identical to SDA2 at the same seed; the P1 SDA2-vs-SDA3 gap (`tab:ladder`, +1.3% vs
  +3.3%) was initialisation noise. Retrained with the bias active (SDA3-fix, 3 seeds), it is
  still no better than SDA1/SDA2 — "conditioning is inert" (`05_results.tex` L111–146) now
  rests on a real SDA3 arm and is **strengthened**; the ladder's SDA3 row must be replaced.
- **SDA guidance weight**: 25 is validation-optimal for all SDA variants (20 was within ~1%).
- **Best scheme: the DirectUNet-M → SDA2-M hybrid** (validation-tuned tau0 0.1, gw 2): best
  RMSE and CRPS on both test sets. Relevant to the conclusion's "most accurate scheme" claim
  (`07_conclusion.tex` L31–33), which should name it.
- **Flow numbers re-scored with the #257 benchmark sampler** (30 members × 20 early-fine
  steps, `ens30_no20`; the extended report uses it throughout). Rankings and conclusions
  are unchanged; flow RMSE drops ~1% (regular S0, 400 ep: PredictStateCFM 0.409 → 0.403,
  VanillaCFM 0.434 → 0.430), but calibration improves markedly: spread/RMSE 0.53 → 0.70
  (PredictStateCFM) and 0.62 → 0.79 (VanillaCFM). The under-dispersion range quoted
  below (≈1.6–2.4×) becomes ≈1.3–1.4× under the benchmark sampler — cite the sampler
  with any calibration number (`tab:calibration`, C4).
- **Observing-system dependence**: DA wins at ≤ 10 obs per window (S0), learned schemes from
  ~20 obs on and at every density under S1; learned schemes win at every number of observed
  fast channels. New material for H1 (`02_background.tex` L191–194) and the marginal-value
  surface (`06_discussion.tex` L145–148).

## Update 2026-09-25 — S1 conditioning bug: "conditioning is inert" is reversed

Every L96 S1 evaluation of the params-conditioned SDA priors (SDA2, SDA3; P1 and the
benchmark reports alike) fed them the TRUE parameters: the evaluation collate read the
plain `F/c1/...` keys, which hold the truth in S1 windows, while the DA baselines and the
training collate read the biased `*_da` values. Fixed and re-run (S0 unchanged: bit-for-bit for
every run first made with the current code; the 2026-09-22 P1 regular-set runs differ only in
their random draws, S0 0.5571 vs 0.5568).

- With the biased DA params, SDA2-M degrades at S1 (0.500 -> 0.509 regular) while
  SDA3-fix-M, trained on noisy DA params, does not (0.501 -> 0.502). Alone the gap is within
  seed noise (SDA3-fix seeds 0.494-0.514).
- As the DirectUNet-hybrid prior it decides robustness: DU(1200)->SDA2-M 0.310 / 0.333
  (S0 / S1, regular), DU(1200)->SDA3-fix-M 0.312 / 0.307 (random set 0.382 / 0.408 vs
  0.388 / 0.385). The SDA3-fix hybrid is the best scheme at S1 on both test sets and the
  better prior on the validation windows too.
- Paper impact: `tab:ladder` SDA2/SDA3 S1 cells and the "conditioning is inert" reading
  (`05_results.tex` L111-146) must be redone; the "most accurate scheme" claim
  (`07_conclusion.tex` L31-33) should name the DU->SDA3-fix hybrid, which is also flat under
  model error. SDA S1/S0 ratios quoted anywhere (e.g. `05_results.tex` L179-181) change for
  SDA2/SDA3 only.

## High priority — the marginal-value headline must be re-run (done 2026-09-24, see above)

`main.tex` L78–81, `01_introduction.tex` L68–72, `05_results.tex` L197–225
(`tab:marginal`), `06_discussion.tex` L8–17 (C1), `07_conclusion.tex` L9–15:
"doubling observations buys Strong-4D-Var 3.2% at S0 vs 19.7% … 6.2× collapse …
1.9× for filters".

Source: `reports/l96/outputs/s0_s1_obs_density_da_baselines.md` (2026-08-19). That
run predates `--da-fast-weights` (#241, 2026-09-23): its truth uses
`fast_weights=[1,1,0.1,0.1]` while the DA model summed the fast variables unweighted,
and S0 ran at inflation 2.0. Its S0 is therefore mis-specified, and the S0 gain that
the 6.2× ratio divides by is not a perfect-model number. Related evidence with
`fast_weights` passed (different protocol: 10 windows, random times, inflation 2.0,
`reports/l96/outputs/l96_da_obs_count_dafw.md`) gives ETKF 15→30 obs S0 −10.2%, S1
−9.3% — a ratio near 1.1, not 1.9. **Re-run the obs-density DA baselines with
`--da-fast-weights` and the per-case inflation before keeping this headline.**

## Contradicted

- `05_results.tex` L90–93 — "the conditional flows … beat both [SDA1 and DirectUNet]"
  (0.3445–0.3584 vs 0.4699): false under the benchmark default (NEW-A). "SDA1 is the
  weakest learned scheme" still holds.
- `07_conclusion.tex` L29–31 — "the components they zero are the ones the measurements
  show to matter": DirectUNet zeroes Ψ_G and Ψ_NG and is the most accurate learned scheme.
- `07_conclusion.tex` L31–33 — "blending … is the most accurate scheme measured here"
  (0.4202): beaten by DirectUNet-M 0.380 (and by P1 VanillaCFM-M 0.3445 already).
- `04_evaluation.tex` L412–413 — "S0 (perfect model)": false for every P1 L96 DA
  number (NEW-D); true only for the new DA runs.
- `05_results.tex` L455–458 and L459–463 — "sequential filters are worse than
  Strong-4D-Var" / "EnKF … trails a Ψ_mean-only variational scheme": on the regular
  grid ETKF 0.685 < Strong-4DVar 0.703 < EnKF 0.711; still true on the random set
  (4DVar 0.742 < ETKF 0.796 < EnKF 0.850). Weakened/contradicted depending on test set.

## Needs a number update

- `05_results.tex` L21–22 (`tab:overview-robust`, L96 row): DA S0 → S1 and ratios
  become Strong-4DVar 0.703→1.436 (2.04×), EnKF 0.711→1.514 (2.13×), ETKF 0.685→1.479
  (2.16×); Strong-4DVar is now the *least* degraded, so the bold moves. Caption L34
  cites the consolidated benchmark but the numbers come from the P1 report.
- `05_results.tex` L49 (`tab:overview-gap`): best learned becomes DirectUNet-M 0.380 /
  0.379 (benchmark default); best classical is ETKF at S0 but Strong-4DVar at S1;
  state the test set.
- `05_results.tex` L71–78 (`tab:price`): every L96 cell (DA per NEW-D, learned per NEW-A;
  SDA unchanged); ETKF/EnKF spread/RMSE are from the old run (DA CRPS/spread with the
  analysis ensemble is being re-measured).
- `04_evaluation.tex` L434–435 (`tab:classes`) and `06_discussion.tex` L47–52 (C6):
  "1.68–1.80×" → 2.04–2.16× (already inconsistent with `tab:price`'s 1.69–1.95).
- `05_results.tex` L98–100: L96 lower bound of the degradation range → 2.04.
- `05_results.tex` L179–181: DA side → 2.04–2.16× (and "≤1.5%" already conflicts with
  SDA3's +3.3% in `tab:ladder`).
- `05_results.tex` L394–400 (`tab:calibration`) and L405–408: DA rows stale; flows under
  the benchmark default are VanillaCFM 0.434 / 0.617, PredictStateCFM 0.409 / 0.534
  (spread/RMSE), so the under-dispersion range becomes ≈1.6–2.4×; P1 flows reach
  0.25 on the random set (≈3.8×, stronger out of distribution).
- `05_results.tex` L419–421: "2.5× less accurate … degrades 1.7×" → ≈1.6× (benchmark
  default) and 2.16×; the cited table has no flow rows.
- `06_discussion.tex` L27–42 (C4): calibration numbers stale; the downgrade stands.
- `04_evaluation.tex` L448–449: says RMSE is pooled; both reports use the mean of
  per-window RMSE (pre-existing mismatch).

## Needs a protocol description or caveat

- `04_evaluation.tex` L355–361 and L390–402: no observing system, training-obs protocol
  (frozen vs redrawn noise), seeds or SDA guidance weight is described — the new
  results hinge on exactly these. Define "the P1 protocol" (`05_results.tex` L81).
- `02_background.tex` L191–194 (H1 costs, "cannot adapt to the observation
  configuration"): DA is the most robust to configuration change (NEW-B); add it to
  what DA buys.
- `06_discussion.tex` L22–26 (C3): the distribution-aware half is strengthened (NEW-B),
  but "dominates both" needs care — on the regular grid the benchmark-default flows
  are worse than the P1 fixed-grid ones (0.409/0.434 vs 0.358/0.344).
- `06_discussion.tex` L66–74 ("baselines under-tuned"): NEW-D is exactly this objection
  confirmed for P1's L96 DA; disclose the fix (a 21% S0 change).
- `05_results.tex` L242–244 ("not a strawman"): weakened until NEW-D is disclosed.
- `05_results.tex` L124 and L136–140: SDA ladder conclusions (conditioning inert) hold on
  both test sets, but the guidance weight was tuned on SDA1 only (being re-tuned on
  validation windows), and the seed sd (≈1–2%) is comparable to the 1.3–3.3% effects —
  check the "15–21% selection noise" argument.
- `06_discussion.tex` L82–85 (no cross-quoting): add regular vs random test set and P1
  vs benchmark-default training as axes.
- `06_discussion.tex` L145–148 (marginal-value surface): the factorial random-layout
  grid now varies temporal and channel density on one protocol (partly addressed).

## Strengthened

- DA not robust to model error / learned flat under S1 (`05_results.tex` L28–29,
  L101–102; `02_background.tex` L212–225): DA S1/S0 now 2.04–2.16×.
- Learned beat DA at S0 (`05_results.tex` L54), now on both test sets (margin ≈1.8× vs 2.1×).
- "Occupying more operator slots does not by itself buy accuracy" (`05_results.tex` L90):
  a Ψ_mean-only scheme now tops the learned ranking.
- Conditioning is inert (`tab:ladder`): same sign, smaller effect on the random set.
- ODE vs posterior sensitivity (`05_results.tex` L183–190).

## Unaffected and central

S0/S1 means different things for model-free learned schemes (`04_evaluation.tex`
L419–425, `06_discussion.tex` L58–64 — carry this caveat into any new text), the
operator-partition theory (`tab:zeroing`, `tab:classes` as definitions), QG
representation results (§5.4), the C4 downgrade, circularity (`06_discussion.tex`
L87–97), weak-4D-Var as the highest-value missing run.

## Material the draft does not use yet

- SDA robust to the observing system with no retraining (×1.2), beating every P1
  fixed-obs learned model on the random set (0.614 vs 0.998–1.547).
- Learned models are sensitive to the observing system, DA is not — a fair counterweight
  to H1.
