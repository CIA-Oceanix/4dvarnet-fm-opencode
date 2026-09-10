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

| keep_k | DirectUNet-L(monai,cos) (S0) | DirectUNet-L(monai,cos) (S1) | CFM-M(monai,flat) (S0) | CFM-M(monai,flat) (S1) | SDA3(monai) (S0) | SDA3(monai) (S1) | DirectUNet+SDA3 (S0) | DirectUNet+SDA3 (S1) |
|---|---|---|---|---|---|---|---|---|
| 16 | 0.486±0.000 / EV 0.907±0.000 | 0.487±0.000 / EV 0.906±0.000 | 0.481±0.000 / EV 0.910±0.000 | 0.478±0.000 / EV 0.910±0.000 | 0.537±0.000 / EV 0.887±0.000 | 0.536±0.000 / EV 0.887±0.000 | 0.420±0.000 / EV 0.927±0.000 | 0.418±0.000 / EV 0.927±0.000 |
| 8 | 0.906±0.002 / EV 0.666±0.002 | 0.905±0.006 / EV 0.663±0.004 | 0.922±0.003 / EV 0.662±0.003 | 0.922±0.002 / EV 0.658±0.001 | 0.781±0.002 / EV 0.742±0.001 | 0.782±0.002 / EV 0.738±0.001 | 0.800±0.003 / EV 0.717±0.002 | 0.802±0.001 / EV 0.712±0.001 |
| 4 | 1.104±0.003 / EV 0.516±0.003 | 1.107±0.001 / EV 0.508±0.001 | 1.131±0.002 / EV 0.506±0.002 | 1.131±0.003 / EV 0.501±0.002 | 0.921±0.003 / EV 0.629±0.003 | 0.922±0.003 / EV 0.625±0.002 | 0.976±0.003 / EV 0.575±0.003 | 0.979±0.003 / EV 0.569±0.003 |
| 0 | 1.292±0.000 / EV 0.370±0.000 | 1.291±0.000 / EV 0.363±0.000 | 1.321±0.000 / EV 0.354±0.000 | 1.319±0.000 / EV 0.349±0.000 | 1.082±0.001 / EV 0.479±0.001 | 1.081±0.001 / EV 0.476±0.001 | 1.138±0.000 / EV 0.423±0.000 | 1.138±0.000 / EV 0.420±0.000 |

## RMSE degradation vs. full density (keep_k=16)

| keep_k | DirectUNet-L(monai,cos) (S0) | DirectUNet-L(monai,cos) (S1) | CFM-M(monai,flat) (S0) | CFM-M(monai,flat) (S1) | SDA3(monai) (S0) | SDA3(monai) (S1) | DirectUNet+SDA3 (S0) | DirectUNet+SDA3 (S1) |
|---|---|---|---|---|---|---|---|---|
| 16 | 1.000x | 1.000x | 1.000x | 1.000x | 1.000x | 1.000x | 1.000x | 1.000x |
| 8 | 1.863x | 1.860x | 1.918x | 1.929x | 1.454x | 1.459x | 1.903x | 1.918x |
| 4 | 2.271x | 2.276x | 2.352x | 2.364x | 1.715x | 1.719x | 2.321x | 2.340x |
| 0 | 2.656x | 2.654x | 2.746x | 2.758x | 2.015x | 2.016x | 2.708x | 2.721x |
