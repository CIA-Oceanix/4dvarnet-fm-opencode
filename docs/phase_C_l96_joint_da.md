# Phase C-adjacent — L96 Joint State-Parameter ETKF DA Baseline (design doc)

**Status:** designed for execution (branch `feature/l96-joint-da-benchmark`).
**System:** Lorenz-96 two-scale (NO=8, J=4), observed subspace 24D (`obs_j=2`, Obs30).
**Scope:** Redesign `JointETKFL96` from scratch and benchmark it against `ETKF` on the
cached S0/S1 test set (200 windows), producing a standalone DA report.

This is the L96 analogue of the L63 `docs/joint_estimation_progress.md`, scoped
specifically to the **ETKF** joint state-parameter problem (F, c1, hx, eps, w1..w4;
h fixed), evaluated apples-to-apples with the L7/L8/L9 joint neural models.

---

## Motivation

The state-only L1b/L2b neural models beat the best state-only DA baseline on S0/S1.
For the **joint** state-parameter problem, the L96 joint neural models (L7/L8/L9,
Phase C) are evaluated and benchmarked, but the **joint DA baselines have never been
run** (`experiments/l96_joint_comparison.json` does not exist; the neural report's DA
rows are `--`). This work closes that gap for the ETKF variant: determine whether a
joint ETKF (which has the forward model) recovers the state and the 8 parameters as
well as (or better than) the joint neural estimators.

## Parameters estimated (8, h fixed)

Following the DA baselines' convention (`_L96_JOINT_PARAM_DIM = 8`,
`evaluation/baselines.py:1893`): **F, c1, hx, eps + fast_weights**. The augmented
state layout is `[state(sd), F, c1, hx, eps, w1..wJ]`, where `J` = number of active
fast weights (4 for S0 full dynamics, 2 for S1 reduced dynamics). `h` is fixed at 1.0
(`_l96_h_fixed`), not estimated — matching the L96 joint DA convention.

### S1 param convention (w3/w4 default)

For S1, the DA forward model is the **reduced 24D dynamics (`J=2`)** — it only carries
2 fast weights. To keep the param-RMSE comparison apples-to-apples with the neural
models (which output all 8 params), the joint ETKF **reports all 8 param columns**:
`w1,w2` are estimated, and `w3,w4` default to the reference prior `[1.0, 0.1]`. This
explicitly disadvantages DA on `w3,w4` (params it cannot observe) and is documented in
the report with a `†` annotation.

## The 6 bugs in the existing `JointETKFL96` (redesigned out)

1. **`assimilate_batch` is inherited (not overridden)** — `JointETKFL96` defines only
   `assimilate`; it inherits vanilla `ETKF.assimilate_batch` which operates on the
   state block only (no augmented state, no param estimation). `evaluate_baseline`
   routes to `assimilate_batch` whenever `batch_size>1`, silently producing wrong
   joint results. → New override required.
2. **No Energy Score (ES)** — `BaselineResult.es` is `None` (no `_ESAccumulator`),
   unlike vanilla `ETKF.assimilate`.
3. **Ignores `etkf_ridge`** — `d = s2 + N1` (line 2129) drops the `+ etkf_ridge*s2.max()`
   term used by vanilla ETKF (line 833).
4. **No NaN-safety** post-analysis (vanilla has a `nan_mask` block, lines 784-790).
5. **Latent S1 shape bug** — `_init_ensemble` builds `fws` from `p["fast_weights"]`
   (length `da_J`, sliced to 2 on S1) but `param_dim=8` is hardcoded, so the augmented
   state is `[N, sd+6]` on S1 while `param_arr` is `[N, 8]` → ValueError.
6. **`h` handling unclear** — `_l96_h_fixed` always returns 1.0 and the `h` kwarg is
   accepted but ignored. Kept fixed (by design) but now documented.

## Design decisions

| Decision | Choice | Rationale |
|---|---|---|
| Scope | Full redesign (all 6 fixes) + ES + batch path | Correct, apples-to-apples first benchmark |
| S1 params | Report all 8; w3,w4 default to `[1.0, 0.1]` | Matches neural 8-param schema; documents DA's info limit |
| ES state space | State block (40D S0 / 24D S1); subsampled to 24D via `_subsample_es` | Matches the cached benchmark convention (neural ES is 24D) |
| Initial methods | Joint-ETKF + vanilla ETKF only | User scoped to ETKF initially; EnKF/Strong stay `--` |
| Output | Standalone DA report (`l96_joint_da_benchmark.md`), native JSON | Keeps neural report stable |
| Branch | `feature/l96-joint-da-benchmark` off `origin/master` | Independent of V2/V3 CFM work |

## Work packages

- **WP1 — Redesign `JointETKFL96`** (`evaluation/baselines.py:2036`): rewrite with the
  6 fixes + `assimilate_batch` override + `_ESAccumulator` on the state block.
- **WP2 — Fix comparator** (`eval_joint_comparison_l96.py`): default `--batch-size 1`;
  pad S1 param truth to 8 (`_true_param_vector`); ETKF-only scope this pass.
- **WP3 — Tests** (`tests/test_joint_estimation_l96.py`): ES finite, S1 shape, batch≡
  sequential, ridge applied, NaN-safety.
- **WP4 — sbatch** (`batch/run_l96_joint_comparison.sbatch`): 1 GPU task (rtx8000).
- **WP5 — Report** (`reports/l96/generate_l96_joint_da_report.py` +
  `reports/l96/outputs/l96_joint_da_benchmark.md`): state RMSE/EV/ES × group, per-param
  RMSE (+NRMSE), `†` marking for S1 w3/w4, consistency check, vs vanilla ETKF and
  (context) the L9 neural numbers.
- **WP6 — Verify + run**: `pytest tests/test_joint_estimation_l96.py
  tests/test_joint_estimation_l96_neural.py tests/test_energy_score.py -m "not slow"`,
  `ruff check`, then 200-window GPU run.
- **WP7 — Docs**: `PLAN.md` + `CHANGELOG.md`, PR → merge.

## Key config facts (verified from the cached dataset)

- `experiments/l96_datasets_obsj2_int100_nwin200.pt`: `true_state (3000,40)`,
  `obs (3000,24)`, `obs_mask (3000,)`, `forcing_true/corrupted (3000,)`, scalar
  `F/c1/h/hx/eps`+`true_*`, **list** `fast_weights` (len 4); S1 also `*_da` (biased +10%)
  + `param_bias=0.1`. No `w1..w4` scalar keys (pre-flattening cache).
- S0: `Lorenz96Dynamics(dt=0.001, coupling_exponent=1.6)` (40D) + `ObsOperator(40,
  obs_var_indices)` (rectangular 24D).
- S1: `Lorenz96Dynamics(dt=0.001, NO=8, J=2, ..., coupling_exponent=1.0)` (24D) +
  `ObsOperator(24, range(24))` (identity).
- DA params: S0 uses `true_*`; S1 uses `*_da`. `da_J = 4` (S0) / 2 (S1).

## Success criteria

- `JointETKFL96` passes the new unit tests (ES, S1 shape, batch≡sequential).
- Runs on all 200 S0/S1 windows without crashing/wrong-shape.
- Produces `experiments/l96_joint_comparison.json` + `l96_joint_da_benchmark.md`.
- State/param numbers reported; whether joint-ETKF beats the L9 neural baseline on
  state and on param recovery is the headline finding.

## Status

Design completed 2026-08-26; execution on `feature/l96-joint-da-benchmark`.
