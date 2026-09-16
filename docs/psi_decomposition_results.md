# Psi_mean / Psi_G / Psi_NG measurements on L96 — results

**Status:** RESULTS (2026-09-16). Measurements only; no training was run. Companion
to `docs/cfm_affine_velocity_decomposition.md` (the derivation) and
`docs/scoping_ml_paper_exploiting_prior_knowledge.md` (which adopts this partition
as its organizing framework).

Decomposition, with `x_tau = (1-tau) x0 + tau x1`, `x0 ~ N(0, sigma0^2 I)`:

```
Psi(x_tau,y,tau) = E[x1|x_tau,y] = Psi_mean(y) + Psi_G(x_tau,y,tau) + Psi_NG(...)
  Psi_mean = m(y) = E[x1|y]
  Psi_G    = K_tau (x_tau - tau*m),   K_tau = tau*P[(1-tau)^2 sigma0^2 I + tau^2 P]^-1
  Psi_NG   = residual, orthogonal to span{m, x_tau}
```

`Psi_NG == 0` iff the posterior is Gaussian with y-independent covariance.

---

## 1. Psi_NG vs observation density (query-based)

`reports/l96/probe_psi_decomposition_vs_density.py`. S0, 200 cached test windows,
fast-Y channels thinned to `keep_k` of 16 (redrawn per observation time); values
averaged over `tau in [0.1,0.95]`.

| model | m RMSE @16 | NG/\|Psi\| (16 -> 0) | NG/\|v\| | NG/\|Psi_G\| |
|---|---|---|---|---|
| V2rerun (`TweedieCFM`, multi-tau) | 0.522 | 0.108 -> 0.189 | 0.237 -> 0.345 | 1.70 -> 0.62 |
| FDV1CFM (`FourDVarNetPredictStateCFM`) | 0.515 | 0.112 -> 0.216 | 0.268 -> 0.415 | 1.69 -> 0.62 |
| V3 (`PredictStateCFM`) | 0.652 | 0.137 -> 0.216 | 0.322 -> 0.438 | 1.41 -> 0.62 |

**Findings.**

1. **Thinning observations raises Psi_NG's share** (V2: +74% on average; at
   `tau=0.1`, 0.147 -> 0.381, a 2.6x rise). The canonical L96 observing network
   conditions the posterior strongly enough to suppress the non-Gaussian component.
2. **But Psi_NG's share *relative to* Psi_G falls** (1.70 -> 0.62). Sparser
   observations widen the posterior, which is primarily a *variance* effect --
   exactly what Psi_G encodes. At `tau=0.5`, Psi_G grows 2.8x while Psi_NG grows 1.2x.
   Consequence for method deficits: the penalty for being *deterministic*
   (missing Psi_G+Psi_NG) grows ~2.1x with sparsity, while the penalty for being
   *Gaussian* (missing Psi_NG only) grows ~1.2x.
3. **Three architectures converge to NG/G = 0.62 at `keep_k=0`** despite differing at
   full density -- consistent with the sparse-observation operator being dominated by
   problem structure rather than model choice.
4. **The late-tau blow-up is a model artifact.** V3/FDV1CFM show NG/\|v\| rising
   steeply past `tau=0.9`; V2rerun does not (0.204 at `tau=0.95`, *falling* to 0.174
   under sparsity). V2 and FDV1CFM have near-identical mean quality (0.522 vs 0.515),
   so this is a property of the **residual/velocity stage**, not the posterior.
   This retracts conclusion 5 of `cfm_affine_velocity_decomposition.md`.

**Caveat.** All three checkpoints were trained at full density, so at `keep_k<16`
they are out of distribution (V2 `m` RMSE degrades 0.522 -> 1.576, in line with the
non-augmented degradation in `l96_obs_density_augmented_training.md`). Part of the
measured Psi_NG at reduced density may be model confusion rather than genuine
posterior non-Gaussianity. The clean version needs a **density-augmented multi-tau**
model, which does not exist: the `*_obsdensity` checkpoints are DirectUNet and
tau=0 CFM, both `Psi_mean`-only by construction.

---

## 2. Method taxonomy from stored ensembles (sample-based)

`reports/l96/probe_psi_from_members.py`. For a method whose output ensemble
represents its implied posterior, the tau-marginal is an exact Gaussian mixture, so
Tweedie gives `Psi~(z) = sum_n w_n x1^(n)` with `w_n = softmax(-(z - tau x1^(n))^2 /
2 alpha^2)` -- a softmax-weighted member average, no KDE bandwidth, leave-one-out to
avoid self-conditioning. 60 windows, N=30, sigma0=0.5.

| method | RMSE | spread | spread/RMSE | \|Psi_G\|max | \|Psi_NG\|max |
|---|---|---|---|---|---|
| SDA3-monai (prior + guidance) | 0.573 | 0.423 | **0.738** | 0.390 | 0.244 |
| L3 (multi-tau CFM) | 0.594 | 0.358 | 0.603 | 0.331 | 0.171 |
| V2rerun (multi-tau CFM) | 0.504 | 0.227 | 0.450 | 0.209 | 0.099 |
| FDV1+SDA3-monai (hybrid) | **0.425** | 0.110 | **0.259** | 0.100 | 0.042 |
| L2b (tau=0 CFM) | 0.653 | 0.068 | 0.104 | 0.059 | 0.023 |

**Findings.**

1. **Every scheme is over-confident relative to its own error**, and the ordering is
   roughly *inverse* to accuracy. For a reliable N=30 ensemble the target ratio is
   `sqrt((N-1)/(N+1)) = 0.967` (using the biased variance estimator, as here).
2. **The warm-started hybrid has a quarter of SDA3's spread** (0.110 vs 0.423) and
   the best RMSE. The SDEdit warm start trades diversity for accuracy: starting at
   `mix(noise, m_hat, tau0=0.3) = 0.7*noise + 0.3*m_hat` scales injected noise to
   0.7*sigma0, anchors it with a mean of much larger magnitude, and runs 7 of 10
   steps.
3. Therefore **the hybrid is iterative refinement of a point estimate, not posterior
   sampling** -- its success belongs in the `Psi_mean` column. Corroborated by the
   consolidated benchmark: the hybrid's gain shrinks monotonically as the mean
   estimator improves (DirectUNet-M -17.0%, FDV1 -10.9%, FDV1-Stier -10.6%,
   subgrad+state-Stier -8.5%), which pure ensembling would not predict.

**Caveats.** (i) This is a **marginal** (per-coordinate) decomposition -- comparable
across the methods in this table, **not** against section 1's query-based numbers
(V2: 0.025 here vs 0.108 there). (ii) `spread/RMSE` conflates under-dispersion with
mean bias, since the reliability identity assumes the truth is exchangeable with the
members. It is a screening statistic, not a calibration verdict; rank histograms and
a binned spread-skill diagram (cheap from the same member files) would settle it.
The ordering argues against bias as the driver -- the *most accurate* mean has nearly
the *least* relative spread. (iii) sigma0=0.5 is an analysis choice applied uniformly.

---

## 3. Conditional samplers (investigation, unresolved)

`evaluation/sda_samplers_experimental.py` (**not production**),
`reports/l96/sweep_pigdm_guidance.py`. SDA1-monai, S0, 8 windows, 8 members.

The repo's sampler (`evaluation/sda_sampler.py`) nudges by the **normalized**
gradient, `-w*g/||g||`, so each step has fixed length `w` regardless of the
observation's informativeness. Two consequences: `R_var` is provably inert (it
cancels), and the step is not a likelihood-score step, so the ensemble has no strong
claim to be a *posterior* sample. Rozet & Louppe instead use a variance-weighted
term `~1/(sigma_y^2 + gamma r_tau^2)` whose magnitude decays with tau.

| config | RMSE | spread | sp/RMSE | mean step norm |
|---|---|---|---|---|
| unguided (w=0) | 1.746 | 0.555 | 0.318 | -- |
| repo baseline (w=40, tuned) | **0.569** | 0.404 | 0.709 | 40 |
| PiGDM-weighted, gamma=1.0 | 0.991 | 0.507 | 0.511 | 2.8 |
| PiGDM-weighted, gamma=0.03 | 1.620 | 1.374 | 0.848 | 16.8 |

**Findings.**

1. Variance-weighted guidance **works** (all gamma beat unguided; best 0.991, -43%)
   but remains well behind the tuned heuristic (0.569).
2. Two candidate explanations **eliminated**: the step clamp never bound (0.0% at
   every gamma), and per-channel `R_var/std_d^2` spans only 0.134-0.181 (+-10%), so
   the scalar approximation was immaterial.
3. The remaining suspect is the **tau-schedule**, not the scale. The principled
   coefficient `dt(1-tau)sigma0^2/tau` decays 81x from tau=0.1 to tau=0.9, so it
   front-loads guidance and applies almost none near tau=1; the baseline applies
   constant magnitude throughout, including the late steps that set the final state.
   That decay is correct for a well-specified prior -- so a constant-magnitude
   heuristic winning suggests the guidance is **compensating for prior
   misspecification** rather than conditioning. Open, testable: instrument the
   per-tau step profile and try a constant-magnitude PiGDM-direction step.
4. **Twisted SMC with a prior proposal is degenerate** at this dimension: resampling
   fired at every step, spread collapsed to 0.030, RMSE 1.72. With ~720 observed
   entries in a 72000-dim state, log-weight variance across 16 particles guarantees
   collapse. Asymptotic exactness is real but unreachable at feasible N without a
   **guided (twisted) proposal** with its density ratio, as in TDS. The
   prior-proposal simplification -- taken because it makes the incremental weight
   collapse to `g_new/g_old` -- is what kills it.

---

## 4. Reproduction

```bash
# section 1 (fdv env -- UNet1D-family checkpoints)
python reports/l96/probe_psi_decomposition_vs_density.py \
  --checkpoint experiments/V2_tweedie_cfm_l96_rerun/checkpoints/stage2_best.ckpt \
  --config config/experiment/V2_tweedie_cfm_l96_rerun.yaml \
  --keep-k 16 8 4 0 --batch-size 25 --output psi_density_v2_s0.json

# section 2 (either env; reads stored members only)
python reports/l96/probe_psi_from_members.py \
  --members experiments/V2_tweedie_cfm_l96_rerun/ens30_no10/members_s0.npz \
  --label V2rerun --max-windows 60 --output psi_mem_v2.json

# section 3 (fdv-monai-proto env -- monai-backbone checkpoints)
python reports/l96/sweep_pigdm_guidance.py --windows 8 --members 8 \
  --gammas 0.03 0.1 0.3 1.0 3.0 --output pigdm_sweep.json
```

**Environment note.** The repo has two incompatible environments split by backbone:
`fdv` (torch 2.4.1, no monai) runs the UNet1D-family checkpoints; `fdv-monai-proto`
(torch 2.8.0, monai 1.6.0) is required for every monai-backbone checkpoint. The
failure is a `ModuleNotFoundError` raised late, inside `train.py::model_factory`.

**Metric note.** `evaluation/estimate_metrics.py::evaluate_estimates` reports the
**mean over dimensions of the per-dimension RMSE**, not pooled RMSE. For
FDV1+SDA3-monai on S0 these are 0.3786 vs 0.4087 -- a 7.5% gap, since slow (0.162)
and fast (0.487) errors differ ~3x. Section 2 uses the repo convention; section 1's
ratios are internally consistent in either. `l96_consolidated_benchmark.md` describes
its convention as "pooled (`sqrt(mean sq err)` over all windows/timesteps)", which
does not match what the code computes and would not reproduce its own numbers.
