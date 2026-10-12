# L96 FDV at the M tier: observation mask, frozen prior and unrolled-solver training

**Status:** RESULTS (2026-10-12). Two seeds for the main comparisons; single-seed ablations marked as such. Follows `docs/results/l96_fdv_m_input_ablation.md` (seed-1 input ablation, 2026-10-03). Part of Phase 1 of `docs/plans/analysis/l96_p2_fdv_benchmark.md`.

**Numbers:** inline. Run directories (`ens1_no10` estimates, benchmark-default test sets) are under `experiments/` of the worktree `4dvarnet-fm-l96-fdv-mask` (FDV1-M seed 2, `resid+state` and subgrad aux-0 runs: `4dvarnet-fm-l96-etks`; FDV1-M seed 1: main `experiments/l96/`, its regular-grid estimates regenerated with the benchmark command into the worktree's `experiments/reeval/`). Probe scripts and JSON outputs: the worktree's `experiments/diag_resid_collapse/` (not checked in).

## Question and answer

**Question.** FDV2 (`subgrad+state`: the update network sees the observation residual, the prior residual x − Φ(x) and x) trailed FDV1 (`obs+state`: x and y) by 3–5% at the M tier. How should FDV2 be parameterised to be at least as good as FDV1?

**Answer.**
1. **Give the update network the observation mask as an explicit input channel.** The masked residual (x − y)·m has no way to signal "observed and already fitted"; with the mask, the residual-input schemes match or beat FDV1.
2. **Train Φ as a prior, not as a solver feature:** keep Φ out of the solver's loss (`freeze_prior_in_solver`) and, for training stability, move it 10× slower than the solver (`prior_unet_lr_scale: 0.1`).
3. **Fine-tune the unrolled solver with deep supervision beyond its 10-iteration horizon** (loss on x_10 and x_20), with Φ coupled during the fine-tune. This also makes the solver stable and slightly refinable when iterated past 10 steps.

Over two seeds the resulting FDV2 is **2–4% better than FDV1-M** (pooled paired: −2.3 / −2.7 / −3.1 / −4.0% on regular S0 / S1 and random S0 / S1). The prior itself adds about 1–2.5%; FDV2 training from scratch is fragile (3 of 5 full-learning-rate runs diverged), which is the main practical cost.

## Setup shared by every run

M tier (MONAI `DiffusionModelUNet`, channels [64, 128, 256], 2 residual blocks per level, GroupNorm 32, dropout 0.1, self-attention at the 750-step bottleneck, 5.9 M parameters; Φ the same network without iteration conditioning). L96 benchmark default: random observing system in training, normalised states, 1200 epochs, Adam 1e-3 with cosine annealing, gradient clipping 10, batch 16. N_outer = 10 unrolled iterations, update x_{k+1} = clamp(x_k − U(u_k, τ_k)/10), τ_k = k/9 through a sinusoidal embedding. Evaluation `ens1_no10` on the 200-window regular and random test sets; RMSE = per-window mean of the per-channel RMSE over the 24 observed variables; paired changes are per-window with 95% intervals.

## 1. Second seed of the input ablation

| run | regular S0 | regular S1 | random S0 | random S1 |
|---|---|---|---|---|
| FDV1-M (`obs+state`), seed 1 / 2 | 0.245 / 0.241 | 0.240 / 0.235 | 0.291 / 0.286 | 0.291 / 0.283 |
| `resid+state`, seed 1 / 2 | 0.253 / 0.247 | 0.248 / 0.242 | 0.296 / 0.287 | 0.294 / 0.290 |
| subgrad-M aux 0, seed 1 / 2 | 0.252 / 0.258 | 0.248 / 0.253 | 0.299 / 0.301 | 0.298 / 0.300 |
| `resid+state` vs FDV1-M, 2 seeds pooled | +2.9% ± 0.9 | +3.3% ± 0.8 | +0.9% ± 0.9 | +1.8% ± 0.9 |
| subgrad aux 0 vs FDV1-M, 2 seeds pooled | +5.1% ± 1.0 | +5.4% ± 0.9 | +4.1% ± 1.0 | +4.4% ± 1.0 |

The ranking FDV1 < `resid+state` < subgrad holds on both seeds.

## 2. The masked residual needs the mask

- **Per-iteration probe (seed 1).** The residual at observed points does not collapse to zero: it plateaus at the observation-noise level (about 0.36 in normalised units, 70% of entries below one noise σ, as for pure noise). `resid+state` descends steadily from the first step; FDV1 barely moves for 3–4 iterations and corrects late.
- **`resid+mask+state`** (input [x, (x − y)·m, m]), seed 1: 0.2414 / 0.2352 / 0.2861 / 0.2851. Against `resid+state`: −4.4 / −5.3 / −3.2 / −3.1% (± 0.9–1.2). Against FDV1-M seed 1: −1.4 / −1.8 / −1.7 / −2.1% (± 1.1–1.3), within FDV1's seed spread. The slow-variable deficit of `resid+state` disappears (regular S0 slow 0.1366 vs FDV1 0.1363).
- **Test-time mask ablation** on that checkpoint (regular S0): true mask 0.2415; mask set to 0: 0.455; mask set to 1: 1.465; mask replaced by 1[|r| > ε]: 0.2422 (ε = 0.02), 0.2507 (0.1), 0.2758 (0.2), 0.3340 (0.36). The information was in r; what the network could not do was compute a sharp on/off indicator from |r|. The mask is used as a gate.
- **`residnorm+mask+state`** (residual rescaled by its per-sample RMS, plus a log-RMS channel) diverged at epoch 148 and was cancelled.

## 3. Update-head variants (all single seed, negative or neutral)

| head | run | regular S0 | random S0 | vs reference |
|---|---|---|---|---|
| gain: x ← x − m ⊙ σ(a) ⊙ (x − y) − c/N | `resid+mask+state` | 0.2580 | 0.3031 | +5.4% / +4.2% vs FDV1-M s1 |
| gated: x ← x − 2σ(a) ⊙ c / N | FDV1 (`obs+state`) | 0.2401 | 0.2898 | −1.9% / −0.4% vs FDV1-M s1; inside FDV1's seed spread (s2 0.2408 / 0.2861) |
| gated | FDV2 + mask, aux 0.01 det. | 0.2425 | 0.2875 | +0.4% / −0.8% vs the direct head |

The gain head writes raw observation noise into the state (learned gains: about 0.5 on fast variables mid-unroll, 0.02–0.15 on slow ones, decaying to 0.05 / 0.02 at the last iteration) and the free correction must undo it. Gating the free update is neutral. Structural update rules do not help; the mask helps as an input.

## 4. FDV2 with the mask: the prior and training stability

All `subgrad+mask+state` (input [(y − x)·m, x − Φ(x), x, m]), aux prior loss 0.01 × [‖sg(x̂) − Φ(sg(x̂))‖² + ‖x_true − Φ(x_true)‖²] unless stated.

| configuration | seed | regular S0 | random S0 | outcome |
|---|---|---|---|---|
| aux 0 (Φ not a prior) | 1 | 0.2566 | 0.2977 | mask does not help (+1.7% vs unmasked) |
| aux 0.01, Φ coupled to the solver loss | 1 | 0.2414 | 0.2898 | ties FDV1-M; −8.1% vs the same without mask |
| aux 0.01, Φ coupled | 2 | — | — | **diverged at epoch 101** |
| aux 0.01, Φ coupled with gradient scale β = 1e-3 | 2 | — | — | **diverged at epoch ~35** |
| aux 0.01, **Φ frozen in the solver** | 2 | 0.2337 | 0.2771 | best single run |
| aux 0.01, Φ frozen | 1 | — | — | **diverged at epoch ~148** |
| aux 0.01, Φ frozen, **Φ learning rate × 0.1** | 1 | 0.2450 | 0.2890 | stable; ties FDV1-M seed 1 |

- **Gradient balance on Φ** (coupled checkpoint, train mode): the solver's MSE pulls Φ 100–250× harder than the aux prior loss (700–1700× for the frozen-trained Φ, whose prior loss is near its optimum). A coupled Φ is therefore a task feature, not a prior: its truth reconstruction error is 0.0076 vs 0.0005 when frozen.
- **No FDV1-type run diverged** with the same optimiser (`obs+state`, `resid+state`, `resid+mask+state`, all heads, both seeds). The instability lives in FDV2's Φ pathway. Plausible mechanism: Φ keeps moving at the full learning rate through the aux loss, so the solver's x − Φ(x) input drifts. A 10× slower Φ removed the blow-up on the seed tried; a second slow-Φ seed would establish reliability.
- **J_Φ probe, correction of an earlier reading.** Scored with the sharp frozen Φ, every solver (FDV1 included) reaches the prior manifold after the first iteration (J_prior 0.009–0.018 at k = 1, 0.0004 at the end and at the truth); "projects first, then fits" is not specific to frozen-Φ FDV2.

## 5. Behaviour beyond 10 iterations and under other observing systems (pretrained models)

Regular S0, seed 1 unless stated; τ clamped at 1 and step kept at 1/10 for the extra iterations.

| condition | FDV1-M s1 | FDV1-M s2 | `resid+mask+state` | FDV2 + mask aux 0.01 | FDV2 frozen Φ (s2) |
|---|---|---|---|---|---|
| baseline | 0.245 | 0.240 | 0.242 | 0.242 | 0.233 |
| +10 iterations | 1.31 | 1.83 | 0.53 | 0.58 | 0.70 |
| +30 iterations | 3.24 | 3.09 | 1.07 | 1.19 | 1.21 |
| 2 of 16 fast channels | 0.75 | 0.74 | 0.74 | 0.74 | 0.72 |
| 1 in 4 observation times | 0.50 | 0.49 | 0.50 | 0.51 | 0.50 |
| noise variance × 4 | 0.54 | 0.53 | 0.42 | 0.42 | 0.41 |

No pretrained solver is a convergent iteration: all diverge past their trained horizon (inner cost J_obs and J_prior both rise past k = 10). Robustness to more iterations and to larger noise comes from the residual-plus-mask input, not from the prior.

## 6. Making the unroll iterable: burn-in vs deep supervision

Both as fine-tunes of `resid+mask+state` seed 1 (300 epochs, lr 1e-4, weights-only start via `training.stage1.init_from`), with `tau_schedule: clamp` (τ_k = min(k/9, 1), identical to training for k ≤ 9, step fixed at 1/10).

- **Burn-in** (with probability 0.5, M ~ U{1..30} iterations without gradient before the 10 trained ones). From scratch it never learned (validation loss 0.8–0.9 throughout, twice): an untrained solver's iterates are garbage. As a fine-tune: baseline +2.2 / +2.6 / +2.1 / +1.8% (± 0.4–0.5), +10 iterations 0.2469 (vs 0.5325), +30 iterations 0.2630 (vs 1.0652). Stable, but it learns to stop rather than refine. A warm-up curriculum (`burnin_start_epoch`, `burnin_ramp_epochs`) is implemented but was not run.
- **Deep supervision** (`train_n_blocks: 2`: 20 training iterations in two blocks of 10, state detached between blocks, MSE averaged over x_10 and x_20). Baseline +0.3 / +0.4 / +0.5 / +0.4% (± 0.3–0.4), +10 iterations 0.2413, +30 iterations 0.2571; best at 12–15 iterations (0.2390 regular S0).

Error and inner cost along the iterations (regular S0; J_obs = mean (x − y)²/R over observed entries, about 1.0 at the truth):

| k | 10 | 12 | 15 | 20 | 30 | 40 |
|---|---|---|---|---|---|---|
| pretrained: RMSE | 0.2415 | 0.272 | 0.367 | 0.533 | 0.825 | 1.065 |
| pretrained: J_obs | 0.873 | 0.901 | 1.019 | 1.308 | 2.027 | 2.788 |
| burn-in: RMSE | 0.2468 | 0.2431 | 0.2439 | 0.2469 | 0.2543 | 0.2630 |
| deep supervision: RMSE | 0.2423 | 0.2390 | 0.2390 | 0.2413 | 0.2484 | 0.2571 |
| deep supervision: J_obs | 0.879 | 0.863 | 0.853 | 0.848 | 0.849 | 0.856 |

After either fine-tune the iteration descends the implicit inner cost (J_obs decreases monotonically to about 0.85, J_prior stays at the truth level); its minimiser fits the observation noise somewhat (J_obs < 1), so accuracy peaks at 12–15 iterations. Observing-system robustness is unchanged (± 1%).

## 7. FDV2 with deep supervision, two seeds

Phase 1 frozen-Φ FDV2 (seed 2 full Φ learning rate; seed 1 Φ learning rate × 0.1), then the deep-supervision fine-tune of section 6 with Φ frozen or coupled during the fine-tune.

| model, mean of 2 seeds (N = 10) | regular S0 | regular S1 | random S0 | random S1 |
|---|---|---|---|---|
| FDV1-M | 0.2426 | 0.2376 | 0.2885 | 0.2868 |
| FDV2 + mask, frozen Φ | 0.2393 | 0.2336 | 0.2831 | 0.2796 |
| + deep-supervision fine-tune, Φ frozen | 0.2400 | 0.2346 | 0.2826 | 0.2787 |
| **+ deep-supervision fine-tune, Φ coupled** | **0.2371** | **0.2313** | **0.2795** | **0.2754** |

| pooled paired change (2 seeds) | regular S0 | regular S1 | random S0 | random S1 |
|---|---|---|---|---|
| FDV2 frozen Φ vs FDV1-M | −1.4% ± 0.8 | −1.7% ± 0.8 | −1.9% ± 1.0 | −2.5% ± 1.0 |
| fine-tune Φ coupled vs FDV1-M | −2.3% ± 0.8 | −2.7% ± 0.9 | −3.1% ± 0.9 | −4.0% ± 1.0 |
| fine-tune Φ coupled vs FDV2 frozen Φ | −0.9% ± 0.4 | −0.9% ± 0.3 | −1.2% ± 0.4 | −1.5% ± 0.4 |
| fine-tune Φ frozen vs FDV2 frozen Φ | +0.3% ± 0.3 | +0.5% ± 0.3 | −0.2% ± 0.3 | −0.3% ± 0.3 |

- Per seed, FDV2 frozen Φ vs FDV1-M: seed 1 ties (+0.2 / +0.2 / −0.7 / −1.2%), seed 2 −3.0 / −3.7 / −3.1 / −3.9%.
- Coupling Φ is unstable from scratch but stable as a 300-epoch fine-tune at lr 1e-4 on both seeds.
- Extra iterations after the fine-tune (coupled; regular S0): seed 1 0.2410 → 0.2369 at k = 15 → 0.2538 at k = 40; seed 2 0.2321 → 0.2287 → 0.2465; the pretrained models reach 1.05 / 1.21 at k = 40.
- On seed 1 the best FDV2 only ties `resid+mask+state` (0.2410 vs 0.2415 regular S0); the prior's advantage over the masked no-prior scheme is clear only on seed 2.

## Caveats

- The two FDV2 seeds differ in the Φ learning rate (the full-rate seed 1 diverged); "two seeds" means two stable runs of the scheme.
- Gains at 12–15 iterations come from probe averages, not paired evaluations.
- `val_loss` misranked models by a few percent several times (it includes the aux term for FDV2); every conclusion above uses test RMSE.

## Code added (all default off)

`models/fourdvarnet.py`: update inputs `resid+mask+state`, `residnorm+mask+state`, `subgrad+mask+state`; `update_head` `gain` / `gated` (with `gain_init`); `freeze_prior_in_solver`; `prior_solver_grad_scale`; `tau_schedule` `exp` / `clamp` (with `tau_scale`); `burnin_prob`, `burnin_max`, `burnin_start_epoch`, `burnin_ramp_epochs` (epoch set by `LitModel.on_train_epoch_start`); `train_n_blocks`. `training/resume.py::load_init_weights` and `training.stage1.init_from` (weights-only fine-tuning). `batch/run_l96b_fdv_subgrad_aux.sbatch` runs from the submitting worktree. Tests: `tests/test_fdv_resid_mask_state.py`, `test_fdv_gain_head.py`, `test_fdv_gated_head.py`, `test_fdv_freeze_prior_in_solver.py`, `test_fdv_burnin.py`. Configs: `config/experiment/L96B_fdvresidmask*`, `L96B_fdvsubgradmask*`, `L96B_fdv1gated_monaiM.yaml`.
