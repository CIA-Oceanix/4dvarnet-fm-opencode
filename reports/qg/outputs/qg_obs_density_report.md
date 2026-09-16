# QG Obs-Density Curve: ETKF/EnKF at cols/day = 1-64 (S0 / S1)

ETKF and EnKF performance across the full observation-density range this project has explored (1, 2, 4, 8, 16, 32, 64 upper-layer (psi1) columns/day), N=100, on both the S0 reference case and the revised S1 combined-error case. Every `loc_radius` value here was independently screened at N=10 and confirmed at N=100 for **both** methods separately (not assumed to transfer from one density or method to another) -- see PLAN.md's obs-density sensitivity sections and `da_sensitivity_s0_s1_report.md` for the full screening trail, including the initial (retracted) N=10 findings this superseded and the joint `loc_radius`/`etkf_ridge`/`inflation` re-tuning that followed.

**Config per row**: `N_ensemble=80`, `inflation=1.0` throughout (reconfirmed optimal at the new `loc_radius`, both methods); `etkf_ridge=0.1` for ETKF at every density (independently confirmed at both `loc_radius=2.0` and `loc_radius=1.0`, not re-screened separately at 8/16/32 -- the mechanism, regularizing conditioning that `loc_radius` already controls, doesn't depend on density beyond `loc_radius` itself). `cols_sampling="random"` for cols>12 (exceeds the sequential sampler's `steps_per_day` ceiling at the reference dt=7200s); `"sequential"` (default) otherwise.

## S0 (reference case, no model error)

| cols/day | loc_radius | ETKF psi | ETKF q | ETKF q-layer2 | EnKF psi | EnKF q | EnKF q-layer2 |
|---|---|---|---|---|---|---|---|
| 1 | 2.0 | 0.926 | 0.346 | 0.332 | 0.924 | 0.347 | 0.332 |
| 2 | 2.0 | 0.943 | 0.411 | 0.378 | 0.941 | 0.412 | 0.378 |
| 4 | 2.0 | 0.959 | 0.498 | 0.442 | 0.958 | 0.500 | 0.443 |
| 8 | 2.0 | 0.974 | 0.601 | 0.519 | 0.974 | 0.602 | 0.520 |
| 16 | 2.0 | 0.982 | 0.683 | 0.585 | 0.983 | 0.684 | 0.585 |
| 32 | 2.0 | *0.988* | *0.739* | *0.629* | *0.989* | *0.740* | *0.630* |
| 64 | 1.0 | **0.991** | **0.774** | **0.662** | **0.992** | **0.768** | **0.655** |

(PV-q full-field and layer2, psi full-field EV, N=100; best per column **bolded**, second-best *italicized* -- with a strictly monotonic curve this typically just marks cols=64/32, included for consistency with this project's table convention.)

## S1 (revised combined-error case)

| cols/day | loc_radius | ETKF psi | ETKF q | ETKF q-layer2 | EnKF psi | EnKF q | EnKF q-layer2 |
|---|---|---|---|---|---|---|---|
| 1 | 2.0 | 0.885 | 0.267 | 0.243 | 0.883 | 0.268 | 0.243 |
| 2 | 2.0 | 0.916 | 0.332 | 0.285 | 0.914 | 0.332 | 0.285 |
| 4 | 2.0 | 0.942 | 0.412 | 0.340 | 0.941 | 0.413 | 0.341 |
| 8 | 2.0 | 0.962 | 0.490 | 0.398 | 0.962 | 0.491 | 0.399 |
| 16 | 2.0 | 0.971 | 0.537 | 0.434 | 0.971 | 0.539 | 0.435 |
| 32 | 2.0 | *0.978* | *0.568* | *0.460* | *0.979* | *0.573* | *0.464* |
| 64 | 1.0 | **0.982** | **0.600** | **0.490** | **0.984** | **0.600** | **0.489** |

(PV-q full-field and layer2, psi full-field EV, N=100; best per column **bolded**, second-best *italicized* -- with a strictly monotonic curve this typically just marks cols=64/32, included for consistency with this project's table convention.)

## Synthesis

- **Clean, monotonic dose-response curve**: once `loc_radius` is tuned per density (rather than held at one project-wide default), more observation density is unambiguously better on every field, both scenarios, both methods -- no collapse, no non-monotonicity anywhere in the 1-64 range. This is the corrected picture after the 2026-09-14 retraction of the original (loc_radius=6.0, untuned) N=10 sweep, which had found the opposite (density *hurting* S1 above cols=8).
- **ETKF and EnKF are now nearly indistinguishable at every density** -- typically within 0.001-0.006 EV of each other on every field, at every density from 1 to 64 cols/day. The EnKF>ETKF gap that motivated the original 2026-09-11 ETKF-ridge sensitivity study, and the later ETKF>EnKF gap after `etkf_ridge=1.0`'s promotion, were both largely artifacts of running at an untuned `loc_radius=6.0` -- properly localized, the method choice barely matters at this reference case's scale.
- **The previously-collapsing unobserved deep layer (q-layer2) improves monotonically with density too**, despite psi1 columns never observing it directly -- S1 ETKF q-layer2 rises from 0.243 (cols=1) to 0.490 (cols=64), roughly doubling, purely through better-conditioned analysis updates propagating through the layer coupling.
- **Still open**: whether independent lower-layer (psi2) observations add value *at matched total density* against a properly-tuned pure-psi1 config -- not yet tested (see `da_sensitivity_s0_s1_report.md`). `N_ensemble` has never been varied (hardcoded 80 throughout this entire curve).

Data: `reports/qg/outputs/qg_obs_density_sweep_n100/*.json` (N=100, all 7 densities x 2 methods x 2 scenarios). Underlying N=10 screens: `reports/qg/outputs/qg_obs_density_sweep/*.json`. Scratch drivers (not committed): `qg_cols1_loc_scratch.py`, `qg_cols2_loc_scratch.py`/`qg_cols2_loc_lowgrid_scratch.py`, `qg_cols4_loc_scratch.py`, `qg_cols8_loc_scratch.py`, `qg_cols32_loc_scratch.py`, `qg_obs_density_n100_scratch.py` (original cols=16/64), `qg_ridge_at_loc2_scratch.py`/`qg_ridge_at_loc1_cols64_scratch.py`, `qg_inflation_at_loc2_scratch.py`, `qg_density_n100_scratch.py` (generic N=100 driver used for cols=8/32 and the ridge=0.1 re-runs at cols=1/2/16/64).

