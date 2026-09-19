# QG onto Hydra, without orphaning the checkpoints

**Status:** DRAFT, v1 2026-09-19. Design plan, nothing implemented.
**Depends on:** PR #226 (QG archive reproducibility), which supplies the
`resolved_config.yaml` this plan builds the migration's compatibility
contract on. If you are reading this before #226 lands, §0's file does not
exist yet.

L96 is Hydra-driven (`train.py`, `@hydra.main`, structured schemas in
`conf/schema.py`, composition through `config/lorenz96_default.yaml`).
QG is not: `train_qg_neural.py` is argparse plus a flat
`OmegaConf.load(config/experiment/<exp_id>.yaml)`, with 37 CLI flags that
override the YAML one field at a time. The two case studies share the Lightning
trainer factory (`training/pipeline.py:create_trainer`) and nothing above it.

This plans the move. The binding constraint is stated first because it decides
the shape of everything else.

## 0. The constraint: trained checkpoints must stay evaluable

Four archived runs (`Q1`-`Q4`, `experiments/qg/<experiment_id>/`) produce the
published benchmark table, plus three interrupted Q5 FDV1 runs. A migration
that makes those unloadable, or that silently changes what
`eval_qg_neural_s0_s1.py` builds, destroys the case study's results — and the
2026-09-19 archive work exists precisely so they can be re-derived.

**The compatibility surface is one file name and three section names.**
`evaluation/qg_runs.architecture()` reads `resolved_config.yaml`'s
`model.{model_type,param_dim,cond_extra_dim,ic_dim}`, and falls back to
`results.json` for runs archived before that file existed. So:

> A Hydra config whose `model`/`data`/`training` sections keep those key names
> is readable by the *current* evaluation code, and a run trained under Hydra is
> indistinguishable from one trained today as far as evaluation is concerned.

`write_resolved_config()` was deliberately given L96's own section layout
(`model`/`data`/`training`, the same three keys
`training/resume.py:config_fingerprint` hashes) for this reason. The migration
therefore **adopts the existing resolved-config schema as the Hydra schema**
rather than designing a new one. Nothing about the archived runs changes.

Two things must not drift and are worth stating as invariants:

- **I1.** `resolved_config.yaml` keeps its name, location (`<exp_dir>/`) and its
  `model`/`data`/`training` sections. It gains fields; it renames none.
- **I2.** `build_model()`'s mapping from those fields to a module stays
  byte-compatible — same `state_dict` keys for the same config. Any architecture
  refactor is a separate change from the config refactor, never the same PR.

## 1. What actually has to move

| piece | today | after |
|---|---|---|
| entry point | `argparse`, 37 flags | `@hydra.main(config_path="config", config_name="qg_default")` |
| config root | none (flat per-experiment file) | `config/qg_default.yaml` + groups `data/`, `model/`, `training/` |
| 13 experiment configs | flat YAML, read by `OmegaConf.load` | `# @package _global_` + `defaults: [/qg_default, _self_]`, as L96's already are |
| schema | none | `conf/qg_schema.py` dataclasses, registered in the `ConfigStore` |
| 67 sbatch scripts | `--nx 64 --epochs 200 ...` | `data.nx=64 training.epochs=200 ...` |
| `LitModel` | QG has its own `QGNeuralLightning` | unchanged (out of scope) |

The 67 sbatch scripts are the cost centre, and most of them are the reproducible
record of experiments already run. That rules out a hard cut.

## 2. Three options

**A — hard cut.** Hydra only; rewrite all 67 batch scripts and 13 configs in one
PR. Cleanest end state, largest blast radius, and it invalidates every command
line recorded in `PLAN.md` and in past changelog entries.

**B — bridge (recommended).** `@hydra.main` becomes the entry point; a
translation shim maps the legacy `--flag value` form onto Hydra overrides,
prints a one-line deprecation notice, and is deleted once the batch scripts are
converted. Old command lines keep working throughout; new work is Hydra-native
from day one. The shim is ~40 lines and one table (`--cols-per-day` ->
`data.cols_per_day`), which is exactly the table the flags already are.

**C — schemas only.** Keep argparse, adopt `conf/qg_schema.py` for validation
alone. Cheapest, and it does fix the `cond_mode: true` YAML-boolean trap
(`data/qg_neural.py` carries a bespoke error message for it today) — but no
composition, no group defaults, no `--multirun`, so the capacity sweep and the
obs-density sweeps stay hand-rolled. Not recommended except as a fallback if B
stalls.

## 3. Staging (B)

Four PRs, each independently revertible, each leaving the tree runnable.

**PR 1 — schema, no behaviour change.** Add `conf/qg_schema.py` mirroring the
sections `write_resolved_config()` already writes; register in the `ConfigStore`;
add a test that every checked-in `config/experiment/Q*.yaml` and every archived
`resolved_config.yaml` validates against it. This is where the dead fields get
settled (§4), and it produces the first real payoff on its own: the schema
rejects a typo'd key, which a flat `OmegaConf.load` silently ignores today.

**PR 2 — config tree.** `config/qg_default.yaml` plus groups; rewrite the 13
experiment configs as `defaults`-composing overrides. Assert per config that
composition reproduces the *current* effective values field by field — the
migration's real test, and a mechanical one.

**PR 3 — entry point.** `train_qg_neural.py` becomes `@hydra.main` with the
legacy-flag shim. `version_base="1.3"` (so `hydra.job.chdir` stays `False`, §4).
Wire in `training/resume.py` at the same time: `resolve_experiment_dir()` and
`ckpt_path=resume_ckpt_path(1)` need a `DictConfig` with `model`/`data`/
`training`, which is exactly what PR 1 defines. **This is the PR that pays for
the migration** — it closes the gap that cost the Q5 sweep three 24-hour jobs on
2026-09-16 (small/M/large reached epochs 359/253/257 of 400, `stage1_last.ckpt`
intact and unusable, because `trainer.fit()` is called with no `ckpt_path`).

**PR 4 — batch scripts.** Convert the 67 sbatch scripts, delete the shim. Can
be split by directory or left to attrition; nothing else depends on it.

## 4. Hazards

1. **Working directory.** Hydra <1.2 chdir'd into the run directory. Every QG
   relative path (`experiments/qg_psi_norm_stats.pt`, `reports/qg_cache`, the
   `experiments/<exp>` default) assumes the repo root. Pin `version_base="1.3"`,
   as `train.py` does, and assert `hydra.job.chdir=False` in a test rather than
   trusting the default.
2. **Dead YAML fields.** `model.hidden_channels` is present in every QG
   experiment config and read by nothing: `build_model()` hardcodes
   `DEFAULT_HIDDEN_CHANNELS`. A structured schema makes it *look* authoritative.
   Either wire it through or delete it from the configs — and if wiring it
   through, verify per config that its value equals the current default first
   (all 13 do today, so this is a no-op change in effect; the point is to
   confirm that rather than assume it). Same question for `model.param_dim`
   versus `data.cond_mode`, which are only consistent by convention.
3. **`cond_mode: true`.** In YAML that is the boolean `True`, not the string.
   The schema types it `str` and turns a live trap into a load-time error.
4. **The skip-check.** Both entry points return early when `results.json`
   exists, and both write `resolved_config.yaml` *before* that check, so
   re-running a finished experiment backfills the config instead of returning
   with its checkpoints undocumented. Preserve that ordering through the
   migration; it is why the archived L96 runs have configs at all.
5. **`experiment_id` provenance.** L96 recovers it from
   `HydraConfig.get().job.config_name` when invoked as `--config-name
   experiment/<id>`. QG takes it from `--exp-id`. The archive's directory naming
   depends on it (`experiments/qg/<experiment_id>/`), so the shim must map
   `--exp-id` onto the same mechanism, not onto a separate config key that
   drifts from it.

## 5. Acceptance: the migration is done when

- Every archived run still loads: for each of `Q1`-`Q4`,
  `qg_runs.architecture()` -> `build_model()` -> `load_state_dict(strict=True)`
  succeeds. (Already a check today; make it a gate.)
- A re-run of `batch/run_qg_neural_s0s1_eval.sbatch` reproduces every published
  EV to 4 decimals. Bit-identity is not available — the 2026-09-19 re-run
  measured a 1.7e-05 worst relative difference across 32 metrics from float32
  kernel selection alone, so 4 decimals *is* the tolerance.
- One QG scheme is retrained end-to-end under Hydra and its
  `resolved_config.yaml` validates against the same schema as the archived ones.
- `scripts/consolidate_qg_archive.py --check --portable` still exits 0.
- An interrupted run resumes: kill a training job, resubmit, confirm it restarts
  from `stage1_last.ckpt` rather than from epoch 0.

## 6. What this does not do

Unifying `QGNeuralLightning` with `training/lightning_module.LitModel`, or
folding `train_qg_neural.py` into `train.py` behind a `system:` switch, is a
larger change with its own risk to the L96 side, and the config migration does
not depend on it. `docs/scoping/refactor_plan.md` is the right home for that
question; this plan deliberately stops at the config layer.
