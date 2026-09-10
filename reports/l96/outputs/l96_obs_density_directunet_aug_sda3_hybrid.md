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

| keep_k | DirectUNet+SDA3 (S0) | DirectUNet+SDA3 (S1) |
|---|---|---|
| 16 | 0.389±0.000 / EV 0.939±0.000 | 0.389±0.000 / EV 0.938±0.000 |
| 8 | 0.633±0.001 / EV 0.823±0.000 | 0.631±0.003 / EV 0.823±0.002 |
| 4 | 0.848±0.006 / EV 0.672±0.005 | 0.850±0.005 / EV 0.667±0.005 |
| 0 | 1.065±0.000 / EV 0.472±0.000 | 1.069±0.000 / EV 0.465±0.000 |

## RMSE degradation vs. full density (keep_k=16)

| keep_k | DirectUNet+SDA3 (S0) | DirectUNet+SDA3 (S1) |
|---|---|---|
| 16 | 1.000x | 1.000x |
| 8 | 1.627x | 1.621x |
| 4 | 2.180x | 2.185x |
| 0 | 2.739x | 2.747x |
