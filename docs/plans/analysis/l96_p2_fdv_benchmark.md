# Unrolled variational solvers (FDV) in the L96 benchmark — for P2

**Status:** PLAN v1 2026-09-27. Phase 0 (code + configs) in the PR that adds this doc; nothing trained yet. Phase 1 (1 seed per tier) trained by 2026-10-03: FDV1-M 0.245 / 0.291 (regular / random S0) is the best M-tier variant; M-tier input ablations in `docs/results/l96_fdv_m_input_ablation.md`; seed-2 runs pending.

Serves `docs/plans/paper/l96_p2_unrolled_solvers_and_flows.md` (P2). That doc puts the
unrolled solver in the `Ψ_mean`-only slot of the operator family (§2.2) and needs it for
M2 (the mean-slot comparison), M3 (the blend with `v_det` = FDV) and M4a (τ-aware FDV1CFM).
Today every FDV number it could cite comes from the SUPERSEDED consolidated report:
old protocol, no `data.normalize`, regular observing system, and (for FDV1CFM) the legacy
unet1d backbone.

## Decisions (2026-09-27)

1. **Benchmark-default recipe only** (`config/l96_benchmark_default.yaml`: random observing
   system redrawn every batch, normalize, cosine, 1200 epochs, batch 16, lr 1e-3, clip 10).
   No 400-epoch P1-recipe twins; the P1 table keeps its "unrolled schemes out of scope" note.
2. **FDV1 (`obs+state`) and FDV2 `subgrad+state` first.** `grad+state` (the real autograd
   4DVarNet cost gradient) and FDV1CFM (M4a) are deferred to a later phase.
3. **Tiers M and S, 3 seeds each** — 12 training runs.

## Scheme table

| config | update rule | tier | main-UNet params | aux prior loss |
|---|---|---|---|---|
| `L96B_fdv1_monaiM` | `concat(x, y)` | M `[64,128,256]`×2 | ≈ 5.89 M | 0 |
| `L96B_fdv1_monaiS` | `concat(x, y)` | S `[32,64,128]`×1 | ≈ 1.06 M | 0.01 |
| `L96B_fdvsubgrad_monaiM` | `concat(y−x masked, x−Φ(x), x)` | M | ≈ 5.89 M + prior Φ | 0 |
| `L96B_fdvsubgrad_monaiS` | same | S | ≈ 1.06 M + prior Φ | 0.01 |

Common: `N_outer = 10`, `init_state_var = 0.1`, monai backbone, `clip_range = 50`.

- **Aux prior loss is 0.01 at S, 0 at M.** At M it plateaued FDV1's val loss at about 0.38
  (0.20–0.29 without it); at S it did no harm and gave the old best rows
  (`CHANGELOG.d/2026-09-12-fdv1-xstier-auxpriorcost-ablation.md`). This is a recipe
  difference *between tiers*, so M vs S is not a pure capacity comparison. If subgrad-M
  underperforms, the first ablation is subgrad-M with aux 0.01 (1 seed).
- **Parameter reporting counts `prior_unet` too** for subgrad rows; the parity claim
  against DirectUNet-M is for the main UNet only and must say so.
- **Objective difference, stated not fixed:** DirectUNet trains MSE + 0.1 × temporal-gradient
  loss (`training.loss`); `FourDVarNetSolver.compute_loss` is plain MSE.
- **`R_var` is irrelevant for both chosen modes** (FDV1 has no cost; `subgrad+state`'s
  `g_obs = (y − x)·mask` carries no `R_var`), so the physical-vs-normalized `R_var` scale
  question only returns with `grad+state`.
- FDV1 feeds zero-filled obs to the UNet with no mask channel, exactly like DirectUNet and
  the CFMs — parity, not a bug.

## Phase 0 — code (this PR)

- **F0a, per-element obs mask.** Under the random observing system `obs_mask` is (B,T) and
  the dropped fast channels of an observed row are NaN. `FourDVarNetSolver` and
  `FourDVarNetPredictStateCFM` built a (B,T,1) mask and zero-filled obs, so `g_obs` and
  `_masked_obs_cost` treated every dropped channel as **observed at 0**. New
  `models/fourdvarnet.py::_observed_mask` ANDs the time mask with `isfinite(obs)`;
  identical on the regular grid. The full-state (`*trueprior`) path is unchanged.
- **F0c, `eval_neural_l96.py --n-outer`.** Default is now the solver's own `N_outer` for
  `FourDVarNetSolver` (was 1, which silently gave FDV1 RMSE 3.47 once).
- **Configs** above; smoke test = 2 epochs + a 25-window eval on the random test set per
  config: all four train and evaluate (eval picks `n_outer = 10` itself).

**Measured cost** (smoke test, RTX 8000, 63 steps/epoch). A100 column scaled by the
RTX 8000 / A100 ratio of 2.55 implied by the archived subgrad-S run (400 epochs in 5.2 h).

| config | s/epoch RTX 8000 | 1200 epochs, A100 (est.) |
|---|---|---|
| `L96B_fdv1_monaiM` | 148 | ≈ 19 h |
| `L96B_fdv1_monaiS` | 65 | ≈ 8.5 h |
| `L96B_fdvsubgrad_monaiM` | 305 | ≈ 40 h |
| `L96B_fdvsubgrad_monaiS` | 120 | ≈ 16 h |

≈ 84 A100-hours per seed, **≈ 250 A100-hours for Phase 1**; `batch/run_l96b_fdv_seeds.sbatch`
(array 0-11, 4 concurrent, 48 h limit) finishes in about 4-5 days wall-clock.

## Phase 1 — training and scoring

- 12 runs: `train.py --config-name experiment/<cfg> training.seed={1,2,3}` from the shared
  `l96b_trainval_n1000_v100.pt` cache, via `batch/run_l96b_fdv_seeds.sbatch` (modelled on `batch/run_l96b_ep1200_seeds.sbatch`).
- Score each on both benchmark test sets — regular (`l96_datasets_obsj2_int100_nwin200.pt`)
  and random layout (`l96_testset_rlayout_n10-100_k4-16_w200_d1.pt`) — deterministic,
  `--n-members 1`, `--members-file scores`, eval sub-dir `ens1_no10`.
- Comparison rows (already trained, same recipe, 3 seeds at 1200 epochs): DirectUNet-M,
  PredictStateCFM-M, VanillaCFM-M; DA rows from the random-layout DA report.
- Report as mean ± std across seeds and paired within-window tests against DirectUNet-M
  (checkpoint-selection noise on this family is 15–21%).
- Archive checkpoints + resolved configs to the main `experiments/l96/` and add the rows to
  `l96_benchmark_extended.md` (new FDV section) through its input bundle.

**Decision criteria.** FDV earns its place in P2 as the variational `Ψ_mean` if its best tier
matches or beats DirectUNet-M on either test set. If not, it is still the degenerate member
for M2/M3, but the paper must say the variational structure does not pay at this budget.

## Phase 2 — P2 experiments that use FDV (after Phase 1)

- **M3 with FDV as `v_det`:** hard warm start `τ₀ = 0.3` vs the continuous blend
  `(λ₀, p)` over PredictStateCFM-M, both axes (RMSE, spread/RMSE). Inference only.
- **Sparsity axis:** RMSE vs `n_obs` on the protocol of `l96_da_obs_count_dafw.md`, so DA and
  FDV curves share one axis (the shared obs-density protocol P2 §9 requires).
- **`N_outer` at inference:** 5 / 10 / 20 / 40 iterations on the trained solvers — the
  unrolled-iteration axis against flow NFE.
- **Deferred to Phase 3:** `grad+state` (needs the `R_var`-under-normalize check and, per the
  2026-09-07 notes, an H200), FDV1CFM ported to monai + normalize, and its τ-aware variant (M4a).

## Risks

- **Cost.** ≈ 250 A100-hours (table above); subgrad-M alone is ≈ 120 of them. If budget
  binds, run S tiers + FDV1-M first and subgrad-M last.
- **FDV2 M-tier history:** every earlier M-tier `subgrad+state` run collapsed on fast Y
  (variance ratio 0.5–0.6), though all predate the normalization and aux-loss fixes.
- **Random layout is new for FDV.** No FDV has trained on it; the mask bug above shows the
  path was untested.
