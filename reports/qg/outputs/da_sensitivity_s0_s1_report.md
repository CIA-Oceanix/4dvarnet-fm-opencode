# QG DA Baseline Hyperparameter Sensitivity (S0 / S1)

Consolidates three related sensitivity studies into one place: ETKF/EnKF sensitivity to their own tunable hyperparameters (inflation, additive inflation, ridge regularization); a 4DVar (Strong/Weak) covariance-scale sensitivity negative result; and an ETKF/EnKF observation-density/configuration sensitivity study that went through a real mid-study correction after user pushback (both of its original headline findings were retracted once `loc_radius` was properly tuned per density -- see that section's own "before correction"/"correction" subsections for the full trail rather than only the corrected conclusion, per this project's convention of keeping the narrative honest). All three studies share the same underlying motivation: PV on the unobserved lower layer (q layer2) collapses under S1's model error, and each asks whether a different lever (inflation/ridge, 4DVar covariance scale, or observation configuration/density) can fix or explain it.

ETKF and EnKF sensitivity to their own tunable hyperparameters (inflation, additive inflation, ridge regularization), on both the S0 reference case (lag=5.0d, noise_frac=0.05) and the revised S1 combined-error case (S0's initial-uncertainty setup + param bias 0.1 + corrupted wind + `da_nx=32` structural resolution mismatch). All sweeps use **N=10 test windows** (the first 10 of the production 100-window test cache) -- cheap enough to sweep densely, but absolute EV values differ somewhat from the N=100 headline numbers in `qg_da_report.md`/`PLAN.md`'s S1 table. This report is about the *sensitivity pattern* per hyperparameter, not a replacement headline comparison.

Motivation: EnKF consistently beats ETKF on S1 (N=100: ETKF q EV 0.307 vs EnKF 0.331) and the reason was an open question. The natural first hypothesis -- ETKF is under/over-inflated -- is what this report's inflation/additive/ridge sweeps test.

## ETKF: multiplicative inflation

| inflation | S0 q EV | S1 q EV |
|---|---|---|
| _free forecast_ | -0.043 | -0.402 |
| 1.00 | **0.402** | **0.253** |
| 1.01 | *0.402* | *0.220* |
| 1.02 | 0.296 | 0.112 |
| 1.03 | 0.147 | -0.029 |
| 1.04 | -40.437 | -0.559 |
| 1.05 | -123.452 | -39.536 |
| 1.10 | -- | -250.810 |
| 1.20 | -- | -345.702 |
| 1.30 | -- | -375.857 |

(PV-q full-field EV; best per column **bolded**, second-best *italicized*.)

S0 (no model error) tolerates inflation somewhat better than S1, but both decline steeply well before 1.05; S1's decline becomes catastrophic divergence (note the scale change) from 1.05 on. `inflation=1.0` (effectively off) is the safe choice on both scenarios at this N.

## ETKF: additive inflation

`etkf_additive`: Gaussian noise added directly to the ensemble each step (raw q-field units), expressed here as a fraction of the truth q field's own std (~2.06e-5 for both S0 and S1 -- same truth trajectories, only the DA model differs). Never exercised before this sensitivity study.

| additive (% of q std) | S0 q EV | S1 q EV |
|---|---|---|
| _free forecast_ | -0.043 | -0.402 |
| 0% | 0.402 | **0.253** |
| ~1% | **0.412** | *0.251* |
| ~5% | *0.405* | 0.224 |
| ~10% | 0.317 | 0.138 |
| ~24% | 0.068 | -0.067 |

(PV-q full-field EV; best per column **bolded**, second-best *italicized*.)

Also monotonically degrades EV on both scenarios, but **gracefully** -- no catastrophic divergence at any tested magnitude, unlike multiplicative inflation's blow-up past 1.04-1.05. Neither inflation flavor *helps*; both are pure downside once past zero.

## ETKF: ridge regularization

`etkf_ridge`: multiplier on the Kalman-gain transform-matrix regularization (targets the deterministic square-root transform's matrix inversion directly, unlike inflation which targets ensemble spread). `etkf_ridge<=0` (the default used everywhere else in this report) internally floors to an implicit 1e-4-equivalent, so the "default" row below reuses the inflation=1.0 baseline files. Never exercised before this study.

| ridge | S0 q EV | S1 q EV |
|---|---|---|
| _free forecast_ | -0.043 | -0.402 |
| default (~1e-4) | 0.402 | 0.253 |
| 1e-3 | 0.419 | 0.258 |
| 1e-2 | 0.446 | 0.275 |
| 1e-1 | **0.466** | 0.303 |
| 1.0 | *0.463* | *0.332* |
| 2.0 | -- | **0.335** |
| 5.0 | -- | 0.323 |

(PV-q full-field EV; best per column **bolded**, second-best *italicized*.)

Unlike either inflation flavor, ridge **monotonically helps** up to a peak (S1 peaks at ridge=2.0, S0 plateaus around 0.1-1.0), then mildly declines (S1 ridge=5.0 < ridge=2.0). `ridge=1.0` is picked as one value near-optimal on both scenarios rather than tuning per-scenario.

## EnKF: multiplicative inflation (for comparison against ETKF above)

Same grid as ETKF's inflation sweep, run on EnKF, to test whether EnKF is robust because of its stochastic perturbed-observations update (which has built-in randomization against ensemble collapse that ETKF's deterministic square-root transform lacks) or just because `inflation=1.0`/`loc_radius=6.0` happen to suit it too. Never exercised before this study -- all prior EnKF runs used the fixed `inflation=1.0` default.

| inflation | S0 q EV | S1 q EV |
|---|---|---|
| _free forecast_ | -0.043 | -0.402 |
| 1.00 | **0.463** | **0.279** |
| 1.01 | *0.435* | *0.229* |
| 1.02 | 0.313 | 0.115 |
| 1.03 | 0.156 | -0.025 |
| 1.04 | -42.980 | -0.668 |
| 1.05 | -124.300 | -39.537 |
| 1.10 | -287.549 | -251.631 |

(PV-q full-field EV; best per column **bolded**, second-best *italicized*.)


## N=100 confirmation

Before committing to the expensive N=100 run, a quick N=10 check of whether ridge keeps helping past 1.0 found a **peak, not unbounded improvement** (S1 q EV: 0.253 default -> 0.332 @ ridge=1.0 -> **0.335 @ ridge=2.0, best** -> 0.323 @ ridge=5.0), and S0 (untested before) shows the same qualitative pattern. `ridge=1.0` was picked as one value near-optimal on both scenarios. Full N=100 ETKF run with `etkf_ridge=1.0`, against the canonical N=100 ETKF/EnKF baselines (note: the first interactive attempt at this silently OOM'd under this session's cgroup cap -- re-run via a real SLURM `sbatch` job, same fix as `run_qg_s1_full100.sbatch`'s prior note):

| method | S0 psi EV | S0 q EV | S1 psi EV | S1 q EV |
|---|---|---|---|---|
| ETKF (default) | **0.957** | *0.476* | **0.926** | **0.357** |
| EnKF | 0.947 | **0.481** | 0.896 | 0.331 |
| ETKF + ridge=1.0 | *0.957* | 0.476 | *0.926* | *0.357* |

(N=100, PV-q and psi full-field EV; best per column **bolded**, second-best *italicized*.)

**Confirmed at full scale**: ETKF+ridge=1.0 beats EnKF outright on **both** fields on S1, and beats it on psi (ties within noise on q, -0.005) on S0. No trade-off on the unobserved PV layer either (S0 q layer2 improved over plain ETKF's ~0.34 to 0.41, matching EnKF's ~0.40, not sacrificed). `etkf_ridge=1.0` is a strictly better ETKF config than the implicit default on this reference case -- **promoted to the actual default 2026-09-12** (was pending at first writing): `run_qg_baselines.py`'s/`sweep_qg_baselines.py`'s own default changed, and the canonical `etkf.json` reference numbers now hold these ridge=1.0 results (the old ~1e-4-floor numbers are archived as `etkf_ridge_default.json` in both directories, not deleted).

## 4DVar (Strong/Weak): covariance-scale sensitivity -- negative result

Same motivation as the ETKF-ridge work above: both 4DVar variants collapse on the unobserved PV q layer2 under S1 (N=100: Strong -0.857, Weak -0.501, vs ETKF/EnKF staying positive -- see `qg_da_report.md`'s S1 table) -- checked whether a covariance-weighting sweep could fix or explain it the way `etkf_ridge` did for ETKF. Swept Strong-4DVar's `b_var_scale` (background-covariance whitening scale) and Weak-4DVar's `q_var_scale` (per-step model-error weight) on S1, **N=5** (4DVar is ~15-20x more expensive per window than ETKF/EnKF -- a single strong4dvar window took 364s in a timing probe vs ETKF's ~20-47s; N=5 was a cost/noise tradeoff, one order of magnitude below the ETKF sweeps' N=10). Each window already gets its own `QG4DVar`/LBFGS instance inside `run()`'s loop.

| Strong-4DVar `b_var_scale` | S1 q EV | S1 psi EV |
|---|---|---|
| 0.3 | **-1.059** | **0.706** |
| 1.0 (default) | *-1.075* | *0.706* |
| 3.0 | -1.111 | 0.698 |

| Weak-4DVar `q_var_scale` | S1 q EV | S1 psi EV |
|---|---|---|
| 0.1 (default) | **-0.915** | **0.687** |
| 0.3 | *-0.986* | *0.606* |
| 1.0 | -1.090 | 0.545 |
| 3.0 | -1.113 | 0.544 |

`b_var_scale` is essentially flat across an order of magnitude (mild decline, not a peak like ridge showed). `q_var_scale` is monotonically *worse* the higher it's pushed above the existing default 0.1 -- giving the per-step model-error controls more freedom to correct actively **hurts** rather than helping, the opposite of the initial hypothesis (that S1's real model error would need *more* absorption capacity). Both methods' existing defaults were already at or near the best point in the explored direction; **no config change made** (unlike ETKF). Caveats: N=5 is noisy (absolute EVs differ from the N=100 canonical numbers, though the ranking direction is consistent), and only `q_var_scale` values *above* 0.1 were tested -- a small chance a lower value (0.01-0.03) does better still, not checked given the cost. Consistent interpretation: the 4DVar q-layer2 collapse looks like a **structural** limitation (a single optimized trajectory has no ensemble spread to exploit on the unobserved layer), not a fixable covariance-tuning gap the way ETKF's transform-regularization gap was.

Data: `reports/qg/outputs/qg_4dvar_sensitivity_sweep/*.json` (N=5, 7 files). Scratch driver (not committed): `qg_4dvar_sensitivity_sweep_scratch.py`.

## ETKF/EnKF obs-density sensitivity: psi2 observations, `cols_sampling="random"`, and a `loc_radius` correction

Motivated by the same q-layer2/unobserved-deep-layer weakness the ETKF-ridge and 4DVar sensitivity work above kept running into: does giving the DA method *any* direct information about the lower layer help, versus just adding more of the same (upper-layer-only) observation? Two new opt-in `QGConfig` capabilities were added to test this: `psi2_points_per_day` (independent lower-layer random-point observations, never directly observed before this work) and `cols_sampling="random"` (upper-layer column observations as random `(t, x)` draws across the whole day, removing the default `"sequential"` sampler's `steps_per_day` ceiling on `cols_per_day` -- which turned out to hang forever above that ceiling, a pre-existing bug now fixed with a clear `ValueError`). All sweeps below: N=10, ETKF default `etkf_ridge=1.0`, `N_ensemble=80`.

### Initial N=10 sweep (before correction)

| config (`loc_radius=6.0` throughout) | S0 q full | S0 q layer2 | S0 psi full | S1 q full | S1 q layer2 | S1 psi full |
|---|---|---|---|---|---|---|
| baseline (cols=4, sequential) | 0.467 | 0.399 | 0.907 | 0.328 | 0.231 | 0.829 |
| cols=8 (sequential) | **0.531** | *0.433* | *0.922* | 0.326 | 0.191 | 0.732 |
| cols=4, random (sanity check) | 0.465 | 0.397 | 0.908 | *0.338* | *0.242* | *0.835* |
| cols=16, random | *0.524* | 0.403 | 0.895 | 0.103 | -0.100 | -1.189 |
| psi1(4)+psi2(10) | 0.502 | **0.447** | **0.924** | **0.389** | **0.316** | **0.878** |

**Initially interpreted as two headline findings -- both later retracted, see the correction below**: (1) "more upper-layer density destabilizes S1" (cols=8 and especially cols=16-random degrade S1 psi, the latter catastrophically to -1.19); (2) "psi2 observations are uniquely valuable" (psi1+psi2 was the best or tied-best config on both scenarios). An EnKF cross-check at the same three configs found the identical graded destabilization pattern on S1, if anything more severe at the extreme (cols=16-random: EnKF psi -4.95 vs ETKF's -1.19) -- confirming the collapse (whatever its cause) is shared across both ensemble methods, not ETKF-specific.

### Correction, step 1: inflation ruled out

The user pushed back on finding (1) -- correctly -- and asked whether a too-large/too-weak inflation could explain it, and requested a `cols=64` run (expecting *more* data to help, not hurt). Sweeping inflation at `cols=16` (random, S1, `loc_radius=6.0` unchanged):

| inflation | S1 q full EV |
|---|---|
| 0.90 | -0.412 |
| 0.95 | *-0.355* |
| 1.00 (default) | **0.095** |
| 1.02 | -23.869 |
| 1.05 | -282.105 |

`inflation=1.0` was already the tested optimum -- ruling out inflation mismatch as the explanation.

### Correction, step 2: `cols=64` degrades S0 too

At `cols=64` (random, `loc_radius=6.0` still unchanged): S1 got worse still (q full -0.294, psi full -17.577), as expected if this were pure model-error reinforcement -- but critically, **S0 (no model error) also degraded** (psi full: 0.637, down from the baseline's ~0.907). A pure-model-error story predicts *no* S0 effect -- this was the tell that something else was going on.

### Correction, step 3: `loc_radius` sweep recovers and exceeds baseline

Localization exists specifically to counter ensemble sampling error given a *fixed, small* ensemble (`N_ensemble=80` throughout, deliberately not scaled up, matching the operational constraint the user named: ensemble size stays small in practice). Higher observation density means more *simultaneous* columns per assimilation step, so a `loc_radius` tuned for the sparse 4-8/day regime becomes under-resourced at higher density. Shrinking it (N_ensemble fixed at 80) fully recovers and then exceeds the original baseline, at both `cols=64` and `cols=16`:

| cols=64, loc_radius | S0 psi EV | S1 psi EV | S1 q full EV |
|---|---|---|---|
| 6.0 (unchanged) | 0.637 | -17.577 | -0.294 |
| 4.0 | 0.938 | 0.788 | 0.409 |
| 3.0 | 0.972 | 0.942 | 0.514 |
| 2.0 | **0.980** | *0.962* | *0.560* |
| 1.0 | *0.978* | **0.963** | **0.567** |

| cols=16, loc_radius | S0 psi EV | S1 psi EV | S1 q full EV |
|---|---|---|---|
| 6.0 (unchanged) | 0.895 | -1.189 | 0.103 |
| 4.0 | 0.958 | 0.927 | 0.473 |
| 3.0 | **0.962** | *0.939* | *0.498* |
| 2.0 | *0.962* | **0.943** | **0.506** |

At the best `loc_radius` (2.0 for cols=64, 3.0-4.0 for cols=16), both configs **beat** the original `cols=4`/`loc=6.0` baseline (S1 psi 0.829 baseline vs 0.96+ tuned). **Root cause**: classic ensemble sampling-error/rank-deficiency, not a physical/model-error effect -- `loc_radius` must be swept per obs-density config, not held at one project-wide default. **Finding (1) above is retracted**: properly localized, more upper-layer density is unambiguously better on both S0 and S1, as conventional DA wisdom predicts. (The other standard fix for the same problem, not tried here: serial/sequential per-observation processing instead of one large combined observation vector.)

### Correction, step 4: psi1-only vs. psi1+psi2 at a fair `loc_radius`

| psi1(4)+psi2(10), loc_radius | S1 q full EV | S1 psi EV |
|---|---|---|
| 6.0 (unchanged) | 0.389 | 0.878 |
| 5.0 | **0.395** | *0.881* |
| 4.0 | *0.394* | 0.879 |
| 3.0 | 0.392 | 0.880 |
| 2.0 | 0.384 | **0.884** |

The psi1+psi2 combined config was only ever tested at `loc_radius=6.0`; it turns out to already be near its own optimum there (its total density is modest). But once **psi1-only** configs are given their own fair, density-matched `loc_radius` (from step 3 above), they clearly outperform the psi1+psi2 mix at its much lower density:

| config (S1) | q full EV | psi EV |
|---|---|---|
| psi1(4)+psi2(10), loc=5 (its own optimum) | 0.395 | 0.881 |
| psi1-only, cols=16, loc=3 (tuned) | *0.498* | *0.939* |
| psi1-only, cols=64, loc=2 (tuned) | **0.560** | **0.962** |

**Finding (2) above is therefore also not supported**: once the comparison is fair, this only shows "more well-localized data beats less well-localized-but-mixed data," not "psi1 alone beats psi1+psi2 at equal density." **Genuinely open question, not yet answered**: does psi2 information add anything *at matched total density* against a properly-tuned pure-psi1 config (e.g. cols~54+psi2=10 vs cols=64, both individually `loc_radius`-tuned)? Not yet tested.

### N=100 confirmation (2026-09-14)

Both loc_radius-tuned high-density configs (cols=16 loc=2.0, cols=64 loc=1.0), for **both** ETKF and EnKF, run at full N=100 -- closing the "N=100 confirmation is the natural next step" caveat from the correction above. EnKF's own `loc_radius` optimum was checked separately at N=10 first (`qg_enkf_loc_check_scratch.py`) rather than assumed to match ETKF's, since the untuned (loc=6.0) EnKF cross-check collapsed *more* severely than ETKF at the same density -- it turned out to match ETKF's exactly at both densities.

| config (N=100) | S0 psi | S0 q | S0 q-layer2 | S1 psi | S1 q | S1 q-layer2 |
|---|---|---|---|---|---|---|
| ETKF baseline (cols=4, loc=6.0) | 0.957 | 0.476 | 0.410 | 0.926 | 0.357 | 0.266 |
| ETKF cols=16 (loc=2.0) | 0.982 | 0.644 | 0.555 | 0.974 | 0.526 | 0.428 |
| ETKF cols=64 (loc=1.0) | *0.990* | *0.738* | *0.630* | *0.983* | *0.585* | *0.480* |
| EnKF baseline (cols=4, loc=6.0) | 0.947 | 0.481 | 0.394 | 0.896 | 0.331 | 0.221 |
| EnKF cols=16 (loc=2.0) | 0.983 | 0.684 | 0.585 | 0.971 | 0.539 | 0.435 |
| EnKF cols=64 (loc=1.0) | **0.992** | **0.768** | **0.655** | **0.984** | **0.600** | **0.489** |

**Fully confirmed at N=100**: both tuned high-density configs decisively beat the canonical cols=4/loc=6.0 baseline, for both methods, on every field and both scenarios -- including the previously-collapsing unobserved deep layer (q layer2), which roughly **doubles** at cols=64 despite psi1 columns never directly observing it (S1: ETKF 0.266->0.480, EnKF 0.221->0.489). cols=64 beats cols=16 throughout, consistent with "more density, properly localized, is unambiguously better" holding at full scale too, not just N=10.

**One new wrinkle**: at these higher densities, **EnKF edges out ETKF+ridge=1.0** on both fields (cols=64 S1: EnKF q=0.600 vs ETKF q=0.585; cols=16 S1: EnKF q=0.539 vs ETKF q=0.526) -- a partial reversal of ETKF's advantage at the cols=4 baseline (where `etkf_ridge=1.0` was specifically tuned and promoted). The margin is modest (~0.01-0.02) so not necessarily decisive, but it means "ETKF+ridge=1.0 beats EnKF" is a cols=4-specific finding, not a universal one -- whether ETKF's own ridge/inflation should be re-tuned at higher density, rather than reusing the cols=4-tuned value, is untested.

**Not yet decided**: whether to promote one of these configs (most plausibly cols=64/loc=1.0) into the canonical S0/S1 benchmark, the way `etkf_ridge=1.0` was promoted after its own N=100 confirmation. Unlike that promotion, this one changes the *observation configuration* itself (16x the column density), not just a DA hyperparameter -- a benchmark-design decision, not a pure tuning one, so left open here rather than actioned.

Data: `reports/qg/outputs/qg_obs_density_sweep_n100/*.json` (N=100, 8 files: ETKF/EnKF x cols={16,64} x S0/S1). Scratch drivers (not committed): `qg_obs_density_n100_scratch.py`, `qg_enkf_loc_check_scratch.py` (the EnKF loc_radius check). Batch: `batch/run_qg_obs_density_n100.sbatch` (jobs 53537 [ETKF, 1h50m], 53538 [EnKF, 2h36m]).

Data: `reports/qg/outputs/qg_obs_density_sweep/*.json` (N=10, 57 files: the original ETKF/EnKF density sweep, the cols=4/random sanity check, the EnKF cross-check, the ETKF inflation and `loc_radius` sweeps at cols=16/64, the psi1+psi2 `loc_radius` sweep, and EnKF's own `loc_radius` check at cols=16/64 -- used to confirm EnKF's optimum before the N=100 run below rather than assume it matched ETKF's). Tests: `tests/test_qg_psi2_points.py` (15), `tests/test_qg_cols_sampling.py` (10) -- mechanical correctness only (shapes, determinism, no-hang, H-function/localization correctness), no scientific claim baked in. Scratch drivers (not committed): `qg_obs_density_sweep_scratch.py`, `qg_enkf_loc_check_scratch.py`.

## Synthesis

- **Inflation-driven divergence is shared, not ETKF-specific**: ETKF and EnKF collapse in near-lockstep under multiplicative inflation, on both S0 and S1 -- e.g. S0 q EV at inflation={1.00,...,1.05}: ETKF {0.402,0.402,0.296,0.147,-40.4,-123.5} vs EnKF {0.463,0.435,0.313,0.156,-43.0,-124.3}, virtually the same curve, same catastrophic threshold. This **rules out** "ETKF's deterministic transform makes it uniquely fragile to over-inflation" -- both ensemble methods share this fragility equally, so it cannot explain the EnKF>ETKF gap.
- **Additive inflation**: also monotonically neutral-to-harmful (never helpful) on ETKF, but fails gracefully with no catastrophic divergence at any tested magnitude, unlike multiplicative inflation. Neither inflation flavor *helps* --  both are pure downside once past zero, confirming "ETKF just needs the right inflation" is not the fix.
- **`etkf_ridge` is the real, actionable lever**: unlike inflation, increasing the Kalman-gain transform-matrix regularization **monotonically helps** -- S1 q EV rises from 0.253 (default, ~1e-4-equivalent) to **0.332 at ridge=1.0**, which *exceeds* both EnKF's own N=10 baseline here (0.279) and EnKF's N=100 headline number (0.331, from `qg_da_report.md`'s S1 table). This is a mechanism EnKF has no equivalent of (it has no deterministic transform-matrix inversion to regularize), so it's specific to ETKF's own weakness, not a general ensemble-DA fix.
- **Bottom line on the original open question -- CONFIRMED at N=100** (see the N=100 confirmation section above): the previously "unexplained" EnKF>ETKF gap on S1 was largely an artifact of running ETKF with an **under-regularized** transform-matrix inversion (the implicit ridge~1e-4 default), not a fundamental method limitation or an inflation-tuning problem. `etkf_ridge=1.0` at full N=100 beats EnKF outright on S1 (both fields) and ties/beats it on S0, with no trade-off on the unobserved PV layer. **Promoted to the default ETKF config** 2026-09-12 (the canonical `etkf.json` reference numbers in `qg_da_report.md`/PLAN.md's S0/S1 tables now hold these ridge=1.0 results; the old ~1e-4-floor numbers are archived, not deleted).
- **4DVar's covariance-scale sweep found no equivalent lever**: unlike ETKF's ridge, neither Strong-4DVar's `b_var_scale` nor Weak-4DVar's `q_var_scale` improves on the existing default in the tested range -- the q-layer2 collapse under S1 looks structural (no ensemble spread to exploit on the unobserved layer), not a covariance-tuning gap. No config change made.
- **Obs-density/configuration sensitivity: a `loc_radius`-tuning lesson, not a new physical effect**: an initial N=10 sweep looked like it found "more upper-layer density destabilizes S1" and "psi2 (lower-layer) observations are uniquely valuable" -- **both retracted** after the user's pushback led to an inflation check, a `cols=64` run, and a `loc_radius` sweep. The real story: `loc_radius=6.0` was tuned for the sparse 4-8/day regime and becomes an ensemble-conditioning bottleneck at higher density with a fixed small ensemble (N=80) -- shrinking it fully recovers and then exceeds the original baseline at both cols=16 and cols=64, on both S0 and S1. **Fully confirmed at N=100** for both ETKF and EnKF (EnKF's own `loc_radius` optimum checked separately, matched ETKF's exactly): both tuned high-density configs decisively beat the cols=4 baseline on every field, including the previously-collapsing q layer2 (roughly doubles at cols=64). One new wrinkle: EnKF edges out ETKF+ridge=1.0 at these higher densities (modest margin, ~0.01-0.02) -- "ETKF+ridge=1.0 beats EnKF" is a cols=4-specific finding, not universal. Whether psi2 information adds value *at matched total density* against a properly-tuned pure-psi1 config remains genuinely open; whether to promote a high-density config into the canonical benchmark is also not yet decided (unlike ridge, this changes the observation configuration itself, a benchmark-design call).
- **Still open**: `N_ensemble` has never been varied (hardcoded 80 everywhere) in any of the three studies above.

Data: `reports/qg/outputs/qg_repro_validation_s1/etkf_n10_*.json` (original inflation/additive S1 sweep) + `reports/qg/outputs/qg_da_sensitivity_sweep/*.json` (consolidated S0 + EnKF + ridge sweep, N=10) + `reports/qg/outputs/qg_repro_validation{,_s1}/etkf_ridge1.json` (N=100 confirmation) + `reports/qg/outputs/qg_4dvar_sensitivity_sweep/*.json` (4DVar covariance-scale sweep, N=5) + `reports/qg/outputs/qg_obs_density_sweep/*.json` (obs-density/configuration sweep with correction, N=10) + `reports/qg/outputs/qg_obs_density_sweep_n100/*.json` (N=100 confirmation). Scratch drivers (not committed): `qg_da_s1_scratch.py`, `qg_da_sensitivity_sweep_scratch.py`, `qg_n100_ridge_confirm_scratch.py`, `qg_4dvar_sensitivity_sweep_scratch.py`, `qg_obs_density_sweep_scratch.py`, `qg_obs_density_n100_scratch.py`, `qg_enkf_loc_check_scratch.py`.

