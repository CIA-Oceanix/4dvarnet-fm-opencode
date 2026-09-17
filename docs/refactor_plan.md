# Contributor-facing refactor plan

Goal: make the repository navigable by a scientist who did not write it, without
a reorganization whose blast radius exceeds its benefit.

Status as of 2026-09-17: **Phase 0 in progress** on `feature/refactor-phase0-test-gate`,
forked from `origin/master` @ `fdf2c1c` (the squash-merge of PR #211).

## What this plan is not

A top-level `src/` reorganization, or a per-case-study package tree
(`l63/ l96/ qg/`). The case-study separation is already clean at module level:
`data/lorenz63.py`, `data/lorenz96.py`, `data/qg.py` and `models/*_dynamics.py`
are each single-system, and a keyword scan finds real cross-system mixing in only
four files (`train.py`, `conf/schema.py`, `data/dataloader.py`,
`evaluation/baselines.py`). Moving already-clean files would break 230 `batch/`
scripts, 105 config paths, 19 live worktrees and the `resolved_config.yaml`
paths recorded inside archived checkpoints, for cosmetic gain.

The problems worth fixing sit on a different axis.

## Findings that drive the plan

1. **Three model-construction paths, partially diverged.**
   - `train.py:103` `model_factory` — 281 lines, 18 model types (training side)
   - `evaluation/neural_inference.py:152` `load_checkpoint` — ~290 lines,
     reconstructs from the run's `resolved_config.yaml` (the primary eval path)
   - `evaluation/neural_inference.py:441` `resolve_model_class` + `create_model`
     — ~230 lines, 11 model types, used as the fallback branch at
     `neural_inference.py:703`

   `resolve_model_class` does not know `monai_*`, `param_head` or `tweedie`.
   **Measured 2026-09-17:** this does *not* break archived checkpoints —
   `load_model()` loads **17 of 17** `.ckpt` files under `experiments/l96/`
   (all `monai_*` variants) via the `load_checkpoint` path. So the divergence is
   real but currently latent, and `eval_monai_l96.py:28`
   `load_monai_direct_unet` is **historical**, not presently forced: it predates
   `load_checkpoint`'s monai support. An earlier draft of this plan asserted the
   reverse causal story; the measurement above supersedes it.

   Consequence for Phase 1: **lower risk than first scoped.** The shared path
   already covers every archived benchmark checkpoint, so the golden harness has
   a clean baseline and the work is consolidation (fold `model_factory` and the
   `resolve_model_class`/`create_model` fallback into one registry) rather than
   repair.

   Note the two checkpoint artifacts per run: `stage1_best.ckpt` (Lightning, has
   `state_dict`) and `stage1.pt` (a bare `OrderedDict`). `load_checkpoint`
   accepts only the former; `eval_monai_l96.py:42` accepts both. The registry
   should accept both.

2. **Entry-point sprawl.** 26 top-level scripts; the 8 `eval_*_l96.py` share
   78-128 identical lines pairwise out of 150-330-line files. `reports/l96/` holds
   13 further driver-ish scripts.

3. **Three parallel DA drivers** with the same function names
   (`_baseline_traj_path`, `fmt_rmse`, `evaluate_baseline`,
   `run_and_cache_baselines`): `evaluation/run.py` (L63), `evaluation/run_l96.py`
   (L96), `evaluation/run_qg_baselines.py` (QG, 1337 lines).

4. **`evaluation/baselines.py`: 3025 lines, 46 touches in 90 days** — biggest and
   hottest file, holding generic filters + joint-parameter subclasses + L96
   joint variants + QG localization helpers.

5. **The safety net is switched off.** The CI merge gate is a hand-maintained
   file list (`.github/workflows/ci.yml`) naming 24 of 53 test files. 29 never
   run, including `tests/test_fourdvarnet.py` (1184 lines, the main model
   family), `tests/test_lightning_module.py`, and all three
   `tests/test_baselines_*.py`.

6. **Two config systems.** Hydra for L63/L96 (`train.py`); argparse with 37 flags
   for QG (`train_qg_neural.py`, 796 lines). QG does not appear in
   `conf/schema.py` at all. `DataConfig` is 53 flat fields spanning all three
   systems.

## Phases

Phases 1 -> 2 -> 3 are strictly sequential: 2 needs the registry from 1, and 3
moves the files 2 rewrites. Only Phase 0's items are parallel-safe. Phase 3 also
needs a window in which no topic worktree is editing `evaluation/baselines.py`.

### Phase 0 — restore the safety net (in progress)

| id | task | state |
|---|---|---|
| 0a | Triage: run the full fast suite, list what actually fails among the 29 ungated files | running |
| 0b | Unbreak collection | **done** |
| 0c | Flip the CI gate to `pytest tests/ -m "not slow"` | blocked on 0a |
| 0d | README + `docs/CONTRIBUTING.md` with one worked path | todo |
| 0e | Root cleanup (untracked `slurm-*.out`, `*.log`, `checkpoint_stage1.pt`) | todo |

**0b outcome — the two "broken tests" were never tests.**
`tests/test_numerical_equivalence.py`, `tests/test_equiv_report.py` and
`tests/compare_rmse_slices.py` define **zero** test functions between them; all
their work runs at module import. Collection therefore executed each script and
crashed, which is why `pytest tests/ -m "not slow"` failed outright.
`check_equiv_report.py` additionally hardcodes `torch.device("cuda")` and runs a
full DA suite, so it could never have passed on a CPU CI runner.

The `TypeError: Lorenz63Dynamics.step() takes 3 positional arguments but 6 were
given` is **not a live regression**: `step(state, forcing, **kwargs)` takes its
parameters by keyword, the production path already does that, and only this dead
script still called it positionally.

Fix: moved to `scripts/verification/` with a README, and the stale positional
call repaired so the script runs again. Automated equivalence coverage already
lives in `tests/test_refactoring_equivalence.py` (4 test classes) — which is
itself one of the 29 ungated files, and gets picked up by 0c.

### Phase 1 — one model registry

`models/registry.py` exposing `build_model(cfg)` and `load_model(ckpt)`. Delete
`create_model` and `load_monai_direct_unet`. Highest-risk phase: 18 model types
with bespoke kwargs across the two hottest files, and it must stay bit-identical
on checkpoint load.

**Prerequisite — golden-output harness.** For all 13 archived checkpoints under
`experiments/l96/`, record `ckpt -> model -> forward-pass hash` under current
code. This is what makes every later phase safe to hand to a cheaper model.

### Phase 2 — collapse the eval entry points (revised after PR #211)

PR #211 already landed the pattern this phase needs: `evaluation/fm_sampler.py`
has one `sample()` integrator plus a `Scheme` hierarchy (`Cold`, `Warm`, `Blend`,
`Decoupled`), a frozen `SamplerConfig`, a `SchemeContext.require()` guard, and
228 lines of tests asserting an algebraic identity and seed reproducibility.

Two consequences:

- **Scope grows.** There are now *three* sampler modules, and the good one is not
  the one production uses:

  | module | style | consumed by |
  |---|---|---|
  | `evaluation/sda_sampler.py` (166 L) | functions | 5 of the `eval_sda_*_l96.py` |
  | `evaluation/sda_samplers_experimental.py` (174 L) | functions | report drivers |
  | `evaluation/fm_sampler.py` (375 L) | `Scheme` classes | tests + 1 report driver |

  Phase 2 must absorb the two function-style modules into the `Scheme` hierarchy
  *before* collapsing the CLIs, and must cover `reports/l96/` drivers too, or the
  sprawl simply relocates.

- **Difficulty drops.** The design question is settled; extending an established,
  tested hierarchy is pattern-following rather than design.

Missing piece: `Scheme.name` derives from the class name but there is no inverse.
A `SCHEMES: dict[str, type[Scheme]]` plus a `scheme_from_spec("blend:lam0=0.6,p=2.0")`
parser (~15 lines) is what a config-driven CLI needs.

### Phase 3 — split `evaluation/`

```
evaluation/da/      baselines (core) + joint/ + l96/ + qg localization + run drivers
evaluation/neural/  neural_inference, estimate_metrics, metrics, samplers
```

The two halves barely reference each other (8340 LOC total), so the split is
cheap and immediately tells a newcomer which half they are in. Split
`baselines.py` along the same seam.

### Phase 4 — optional, only if QG keeps growing

Unify QG onto Hydra; split `DataConfig`'s 53 flat fields into per-system
sub-configs.

## Delegation

| phase | agent / model | why |
|---|---|---|
| 0a | `Explore`, Sonnet | read-only fan-out, produces a list |
| 0b | general-purpose, Sonnet | contained, test-verifiable |
| 0c, 0e | Haiku | mechanical, gated, reversible |
| 0d | human or Opus | needs judgement about what the science is |
| pre-1 | general-purpose, Sonnet | highest-leverage delegation; unlocks the rest |
| 1 | **Opus** | 18 dispatch branches, two hot files, bit-identical load required |
| 2a | Sonnet (was Opus) | house pattern now exists — see PR #211 |
| 2b | general-purpose, Sonnet x2-3 in parallel | per-script ports, independently verifiable |
| 3a | Opus or `Plan` | getting the seam wrong means redoing the hottest file |
| 3b | Sonnet | moves + import rewiring under a strong test gate |
| 4 | Opus design, Sonnet migration | design-heavy then bulk-mechanical |

Delegate downward only behind a falsifiable gate, e.g. "byte-identical `.npz`
estimates on 200 windows", or "`git diff --stat` shows moves only".

## Branch hygiene

Fork each phase from `origin/master` **after** the previous phase squash-merges,
never from the previous branch tip — a squash-merge rewrites history, and
branching off the old tip produces a conflict that silently blocks CI with zero
runs and no error message. Diff the merge SHA against local HEAD before calling a
phase landed.
