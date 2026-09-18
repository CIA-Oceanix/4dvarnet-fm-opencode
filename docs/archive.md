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
