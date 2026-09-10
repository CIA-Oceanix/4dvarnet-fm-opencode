# L96 Fast-Y Observation-Density Generalization

Inference-time-only generalization test (no retraining) for the 4
best-of-subcategory L96 monai-backbone schemes -- DirectUNet-L(cos),
CFM-M(flat), SDA3, DirectUNet+SDA3 -- under randomly reduced fast-Y
observation density. Of the 16 canonical fast-Y channels (2 per slow
node), only `keep_k` are kept, **redrawn independently at every
observation time** within each window; the 8 slow-X channels always
stay fully observed. `keep_k=16` is the full-density sanity check and
should reproduce the canonical `l96_consolidated_benchmark.md` numbers.

**Caveat:** DirectUNet/CFM consume `obs` only via
`torch.nan_to_num(obs, nan=0.0)`, with no separate mask channel --
trained only on whole-timestep NaN blocks, never partial-channel NaN
within an observed timestep. A dropped fast-Y channel is therefore
indistinguishable from a genuine near-zero observation for these two
schemes: this is genuinely out-of-distribution, not fixable without
retraining. SDA's guided-sampling cost, by contrast, never conditions
the network on raw obs -- the dropped channels are cleanly excluded
from the guidance cost term, with no zero-imputation ambiguity.

Each (method, case, keep_k) cell is `n_repeats=3` independent
seed reruns (fresh mask redraw, and fresh sampling noise for the
stochastic methods); values below are mean +/- std across those repeats.
Sampling settings: `{'n_outer': 10, 'n_members': 30, 'r_var': 0.5, 'sda_guidance_weight': 40.0, 'hybrid_guidance_weight': 2.0, 'hybrid_tau0': 0.3}`.

## RMSE / EV(all_obs) vs. fast-Y density

| keep_k | DirectUNet-L(monai,cos) (S0) | DirectUNet-L(monai,cos) (S1) | CFM-M(monai,flat) (S0) | CFM-M(monai,flat) (S1) |
|---|---|---|---|---|
| 16 | 0.473±0.000 / EV 0.914±0.000 | 0.476±0.000 / EV 0.913±0.000 | 0.462±0.000 / EV 0.917±0.000 | 0.462±0.000 / EV 0.917±0.000 |
| 8 | 0.704±0.003 / EV 0.789±0.002 | 0.704±0.005 / EV 0.787±0.003 | 0.690±0.001 / EV 0.799±0.001 | 0.690±0.003 / EV 0.798±0.002 |
| 4 | 0.898±0.002 / EV 0.644±0.002 | 0.905±0.004 / EV 0.635±0.003 | 0.899±0.005 / EV 0.644±0.004 | 0.904±0.004 / EV 0.637±0.003 |
| 0 | 1.081±0.000 / EV 0.472±0.000 | 1.085±0.000 / EV 0.465±0.000 | 1.108±0.000 / EV 0.450±0.000 | 1.112±0.000 / EV 0.443±0.000 |

## RMSE degradation vs. full density (keep_k=16)

| keep_k | DirectUNet-L(monai,cos) (S0) | DirectUNet-L(monai,cos) (S1) | CFM-M(monai,flat) (S0) | CFM-M(monai,flat) (S1) |
|---|---|---|---|---|
| 16 | 1.000x | 1.000x | 1.000x | 1.000x |
| 8 | 1.490x | 1.479x | 1.492x | 1.492x |
| 4 | 1.900x | 1.902x | 1.943x | 1.956x |
| 0 | 2.286x | 2.279x | 2.396x | 2.406x |
