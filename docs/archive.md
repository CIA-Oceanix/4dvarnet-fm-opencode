# Archived run artifacts

**Status:** NOTES — describes the repo's artifact layout, not the science.

A "run" is one trained model plus everything needed to reproduce its row in a
report:

```
<run>/checkpoints/stage1.pt          bare OrderedDict (state dict only)
<run>/checkpoints/stage1_best.ckpt   Lightning checkpoint (has a "state_dict" key)
<run>/resolved_config.yaml           the config training actually used
<run>/estimates_s0.npz               stored state estimates + reference truth
<run>/estimates_s1.npz
<run>/neural_eval.json               metrics sidecar (optional)
<run>/manifest.json                  provenance of each artifact (optional)
```

Both checkpoint forms hold the same weights, but they are **not
interchangeable**: `evaluation.neural_inference.load_model` accepts only
`stage1_best.ckpt`, because `stage1.pt` has no `state_dict` key.

## Two layouts, one resolver

| layout | path | holds |
|---|---|---|
| canonical | `experiments/l96/<run>/` | checkpoints + configs, consolidated 2026-09-10 |
| legacy | `experiments/<run>/` | where runs were originally written |

Artifacts resolve **per file**, canonical first, because the 2026-09-10
consolidation deliberately moved checkpoints and configs but left the large
`estimates_*.npz` in whichever worktree trained the run. A single run's files
legitimately span both layouts.

`evaluation/archive.py` is the only place that knows this. Do not build an
artifact path anywhere else, and in particular **never derive one by walking a
fixed number of parents from another path**:

```python
# what eval_monai_l96.py used to do
Path(ckpt).resolve().parents[2] / "l96_norm_stats_obsj2.pt"
```

That was correct for `experiments/<run>/checkpoints/ckpt.pt` and silently wrong
for every archived checkpoint the moment `experiments/l96/` added a level — it
looked one directory too shallow, and `.resolve()` meant symlinking the old
paths could not rescue it. Use `archive.resolve_norm_stats(cfg, name=...)`,
which takes `cfg.data.norm_stats_path` first.

## Keeping it reproducible

```bash
# is every method the consolidated report names resolvable?
python scripts/consolidate_l96_archive.py --check

# link anything missing in from the sibling topic worktrees (hard links,
# so no extra disk); --manifest also records provenance per artifact
python scripts/consolidate_l96_archive.py --apply --manifest
```

`--check` exits non-zero when a method the report needs has no estimates in any
known location. Run it before regenerating the consolidated report; a topic
worktree that evaluates a new model writes its estimates locally, so drift
restarts every time and `--apply` is how you pull it back.

## Known issue: metrics sidecars can disagree with their own arrays

For at least `L1b_monai_unet_s0s1_norm`, re-scoring the archived
`estimates_*.npz` with current code does **not** reproduce the archived
`neural_eval.json` (`rmse.slow` 0.2720 vs 0.2670, ~1.9%). Ruled out: metric code
(unchanged — the diff since is purely additive), checkpoint choice (`stage1.pt`
and `stage1_best.ckpt` give identical output), and float32-vs-float64.

The checked-in `reports/l96/outputs/l96_consolidated_benchmark.md` follows the
sidecars, so regenerating it moves published numbers. **Cause not identified —
treat neither set as authoritative until it is.** `manifest.json` exists so that
a future re-score can at least be checked against what produced it.

## QG

Same resolver (`evaluation.archive.QG`), different conventions:

```
experiments/qg/<run>/stage1_best.pt              bare state dict, written when training finishes
experiments/qg/<run>/checkpoints/stage1_best.ckpt  Lightning's own (monitored val_loss)
experiments/qg/<run>/checkpoints/stage1_last.ckpt  ditto, last epoch
experiments/qg/<run>/resolved_config.yaml        the config training actually used
experiments/qg/<run>/config.yaml                 copy of the source experiment YAML
experiments/qg/<run>/results.json                in-training S0 metrics + effective config
experiments/qg/<run>/estimates_s0.npz
experiments/qg/qg_psi_norm_stats.pt              shared, and part of the archive
experiments/qg/qg_param_norm_stats.pt
experiments/qg/qg_forcing_norm_stats.pt
```

Run directories are named by `experiment_id` (`Q1_direct_unet_tchannels_s0`),
not by benchmark position. `Q1` has meant three different architectures across
the 2026-09 backbone migrations; a checkpoint directory named after the position
says nothing about which.

A run killed before it finished has only `checkpoints/*.ckpt` and no
`stage1_best.pt`. `evaluation.qg_runs.checkpoint()` takes either, so an
interrupted sweep stays evaluable instead of looking like a missing run.

**The normalization stats live in the archive**, not only in
`experiments/`. QG psi is z-scored globally before training, so a checkpoint
without its stats predicts in units nobody can invert: the Q1-Q4 checkpoints
were reachable from master for four days and unusable there, because
`experiments/qg_psi_norm_stats.pt` existed only in the training worktree.
`evaluation.qg_runs.norm_stats_paths(run)` resolves all three, config-first.

### An evaluation reads the architecture from the run

`evaluation.qg_runs.architecture(run)` returns `model_type`/`param_dim`/
`cond_extra_dim`/`ic_dim` from `resolved_config.yaml`, falling back to
`results.json` (runs archived before 2026-09-19) and then to
`config/experiment/<run>.yaml`. Do not re-declare those at a call site:
`eval_qg_neural_s0_s1.py` used to carry them as per-scheme literals, tied to the
checkpoint by nothing, and a config change would have loaded weights into a
differently-shaped model.

`cond_mode` is deliberately *not* part of that. It is an evaluation-time choice
(`"scenario"`), not a property of the checkpoint (trained `"true"`/`"noisy"`),
and taking the recorded value would quietly turn the S1 comparison into a
meaningless one.

### Keeping it reproducible

```bash
# can the published report be regenerated from HERE?
python scripts/consolidate_qg_archive.py --check

# ...and from anywhere else? (canonical archive only)
python scripts/consolidate_qg_archive.py --check --portable

# link anything missing into the archive (hard links, no extra disk)
python scripts/consolidate_qg_archive.py --apply --manifest
```

`--portable` is the one that matters and the one the old layout failed: an
artifact reachable only through the training worktree's own
`experiments/<run>` looks fine from there and does not exist anywhere else.

The canonical archive is one physical directory, in the master worktree. A topic
worktree reaches it with `ln -s <master>/experiments/qg experiments/qg`, and then
`--apply` run from either one hard-links into the same place. (`experiments/` is
gitignored, so this is local plumbing, not a committed path.)

### What this cost before it was fixed

`reports/qg/outputs/qg_neural_report.md`'s benchmark table lost all four neural
rows in #208 (2026-09-16): the report was regenerated in a worktree where
`qg_neural_s0_s1_cross_scenario/results_lag5_noise0.05_bias0.1.json` did not
exist, the generator returned `None` for the neural half, and the table was
published with the four DA rows alone. That JSON is now tracked, and
`tests/test_qg_report_consistency.py` fails if a published row and its input
disagree.
