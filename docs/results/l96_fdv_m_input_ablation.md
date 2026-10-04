# L96 FDV at the M tier: what the update network is fed (FDV1 vs FDV2 variants)

**Status:** NOTES (2026-10-03). Seed 1 only for every run; seed-2 runs of FDV1-M, `resid+state` and subgrad-M (aux 0) were submitted on 2026-10-03 and will be reported in a follow-up note. Part of Phase 1 of `docs/plans/analysis/l96_p2_fdv_benchmark.md`.

**Numbers:** inline. Per-run `neural_eval.json` / `estimates_*.npz` (`ens1_no10`, benchmark-default test sets) are in the run directories under `experiments/` of the worktrees that trained them (FDV1-M and subgrad-M aux 0: `4dvarnet-fm-p2-fdv`; the other three: `4dvarnet-fm-l96-etks`), not checked in.

## Question

At the S tier, FDV2 `subgrad+state` beats FDV1 (`obs+state`) by 9–10% when both use the aux prior loss (0.01). At the M tier, subgrad-M (aux 0) trailed FDV1-M by about 3%. Is the observation/prior information better used when it is fed as residuals and a trained prior, and if not, which ingredient costs accuracy?

## Variants (M tier, 1200 epochs, seed 1, otherwise identical)

The 127-key resolved configs differ only in the keys listed; the code differs only by these options (default off); evaluation settings are identical (`ens1_no10`, 200 windows per test set).

| run | update network input | prior Φ | aux prior loss |
|---|---|---|---|
| FDV1-M (`obs+state`) | x, y (missing = 0) | none | — |
| `resid+state` (new) | x, (x − y)·mask | none | — |
| subgrad-M, aux 0 | (y − x)·mask, x − Φ(x), x | untrained as a prior | 0 |
| subgrad aux 0.01 detached, prior input zeroed | (y − x)·mask, 0, x | trained, unused by the solver | 0.01, `aux_detach_x_final` |
| subgrad aux 0.01 detached | (y − x)·mask, x − Φ(x), x | trained | 0.01, `aux_detach_x_final` |

`resid+state` has the same parameters and, at the same seed, identical initial weights as FDV1-M (checked), so it isolates the observation representation.

## Results

RMSE (per window, 24 observed channels, mean over 200 windows) and paired change against FDV1-M with 95% interval; "w" = windows where the variant beats FDV1-M.

| run | regular S0 | regular S1 | random S0 | random S1 |
|---|---|---|---|---|
| **FDV1-M** | **0.245** | **0.240** | **0.291** | **0.291** |
| `resid+state` | 0.253 (+3.2% ± 1.1, w 68) | 0.248 (+3.7% ± 1.2) | 0.296 (+1.6% ± 1.7, w 78) | 0.294 (+1.0% ± 1.9) |
| subgrad-M, aux 0 | 0.252 (+3.0% ± 1.2, w 63) | 0.248 (+3.4% ± 1.2) | 0.299 (+2.8% ± 1.6, w 65) | 0.298 (+2.5% ± 1.7) |
| subgrad aux 0.01 det., prior input 0 | 0.259 (+5.6% ± 1.2, w 49) | 0.253 (+5.8% ± 1.3) | 0.304 (+4.4% ± 1.6, w 57) | 0.307 (+5.3% ± 2.0) |
| subgrad aux 0.01 det. | 0.263 (+7.3% ± 1.2, w 40) | 0.258 (+7.5% ± 1.4) | 0.308 (+5.9% ± 1.7, w 46) | 0.303 (+4.1% ± 2.1) |

## Findings

1. **No input variant beats FDV1 at M.** The best alternatives (`resid+state`, subgrad aux 0) are about 3% worse on the regular grid and 1–3% on the random layouts, where the intervals reach zero.
2. **The masked residual costs a little.** `resid+state` differs from FDV1-M only in feeding (x − y)·mask instead of y: +3% on the regular grid (significant), +1–2% on the random layouts (not). Its validation loss led FDV1-M early (−10% at epoch 100) and fell 3–6% behind after epoch ~800.
3. **An untrained prior channel adds nothing:** subgrad aux 0 ≈ `resid+state`. Without the aux loss Φ is not a prior (‖x − Φ(x)‖² ≈ ‖x‖², |Φ(x)| ≈ 0.2 on normalized states).
4. **A trained prior channel hurts at M.** With the aux loss on and detached, Φ explains about 45% of the state variance, and the run is 4–7% behind FDV1-M.
5. **Single-seed noise is about 2–3%.** The prior-input-zeroed run should behave like `resid+state` (Φ cannot affect its solver), yet it is 2.4 points worse on the regular grid. Its extra parameters change the solver's initial weights, and the aux term enters the loss that selects `stage1_best.ckpt`. Gaps of 3% or less need the seed-2 runs.

## The non-detached aux loss is unstable at M

subgrad-M with the aux prior loss as originally defined (0.01 × [‖x̂ − Φ(x̂)‖² + ‖x − Φ(x)‖²], gradient flowing into the solver through x̂) blew up at epoch ~320.
- Final-state MSE on test windows went from 0.049 at the best checkpoint (epoch 311) to 0.256 after the spike, while Φ's prior cost stayed at about 0.55. So the solver diverged, not Φ.
- This matches the 2026-09-12 FDV1-M ablation with the same loss, which plateaued permanently at val ≈ 0.38 (`CHANGELOG.d/2026-09-12-fdv1-xstier-auxpriorcost-ablation.md`).
- The likely mechanism is the term's pull on the solver toward Φ's fixed points. `aux_detach_x_final` stop-gradients x̂ in that term (Φ still trains on the solver's outputs): the detached run passed the same epochs without a spike.
- Note that with aux > 0 the aux term is also added to `val_loss` (about 0.01 here), so those runs' validation curves sit above the others at equal MSE.

## Checks

- Configs, code and evaluation settings: compared key by key across the five runs (above).
- Evaluation precision: FDV1-M and subgrad aux 0 were evaluated before #302, with `'medium'` float32 matmul precision. Re-evaluated at `'highest'`, both give identical RMSE to 4 decimals (per-window differences ≈ 2e-5).
- Interruptions: subgrad aux 0 resumed at epoch 937 (hung node) and subgrad aux 0.01 detached at epoch 806 (GPU fault). Resumes restore the model, optimizer and schedule, but not the data-order and dropout random streams.

## Code added

- `update_input: resid+state` (`models/fourdvarnet.py`).
- `zero_prior_input` (subgrad+state only).
- `aux_detach_x_final`.

All default off, with tests in `tests/test_fdv_resid_state.py` and `tests/test_fdv_zero_prior_input.py`. Configs: `L96B_fdvresid_monaiM`, `L96B_fdvsubgrad_monaiM_aux01` (the run that blew up), `…_aux01det`, `…_aux01det_noprior`. Script: `batch/run_l96b_fdv_subgrad_aux.sbatch` (`FDV_CFG`, `FDV_SEED`).
