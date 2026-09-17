# Contributing

**Status:** NOTES — describes the repo, not the science (kept at `docs/` level alongside `docs/worktrees.md`).

A worked path through the repository, for a scientist who did not write it.
Everything below is a real command taken from `batch/` and the report scripts.

## 0. Environment

One conda env for everything:

```bash
export PATH="/Odyssey/private/rfablet/miniforge3/envs/fdv-monai-proto/bin:$PATH"
```

`torch >= 2.6` defaults to `weights_only=True`, which rejects this project's
pickled datasets and checkpoints. Project code passes `weights_only=False`
explicitly — keep doing so in new code.

## 1. The three case studies

| system | dynamics | data | DA baselines | neural entry point |
|---|---|---|---|---|
| Lorenz-63 | `models/lorenz63_dynamics.py` | `data/lorenz63.py` | `evaluation/run.py` | `train.py` (Hydra) |
| Lorenz-96 (two-scale) | `models/lorenz96_dynamics.py` | `data/lorenz96.py` | `evaluation/run_l96.py` | `train.py` (Hydra) |
| QG (two-layer) | `models/qg_dynamics.py` | `data/qg.py` | `evaluation/run_qg_baselines.py` | `train_qg_neural.py` (argparse) |

L96 is the main benchmark. Start there.

## 2. A full L96 loop

**Generate data and run the DA baselines.** This one script does data generation,
EnKF/ETKF/4DVar and an RMSE table:

```bash
python evaluate_all_l96.py --num-test-windows 20 --device cuda
```

**Train a neural scheme.** Configs live in `config/experiment/`; pick one and
override anything on the Hydra command line:

```bash
python train.py \
  --config-name experiment/FDV1_unrolled_monai_unet_l96 \
  hydra.run.dir=. hydra.output_subdir=null \
  ++data.num_train_windows=60 ++data.num_val_windows=10 ++data.num_test_windows=10 \
  training.stage1.epochs=1 \
  +fresh=true
```

Training writes a `resolved_config.yaml` next to the checkpoint. Evaluation
auto-prefers it, so you rarely pass `--config` by hand.

**Evaluate.** Inference and metrics are deliberately split: step 1 writes state
estimates to `.npz`, step 2 scores any `.npz` (neural or DA) identically via
`evaluation/estimate_metrics.py`. This is what makes the comparison
apples-to-apples.

```bash
python eval_neural_l96.py \
  --checkpoint experiments/<run>/stage1_best.pt \
  --num-windows 200 --cases s0,s1
```

**Render the benchmark table:**

```bash
python reports/l96/generate_l96_consolidated_report.py
```

## 3. Before you open a PR

```bash
pytest tests/ -m "not slow"     # the merge gate
ruff check <files you touched>  # informational in CI, but keep it clean
```

Add a `CHANGELOG.d/YYYY-MM-DD-<slug>.md` fragment — **never edit `CHANGELOG.md`
directly**, CI blocks that. One file per change means concurrent PRs never
conflict on it. Format is in `CHANGELOG.d/README.md`.

## 4. Things that will surprise you

- **Cosine LR scheduling is the deliberate default** (`use_cosine_scheduler: true`),
  not an accident. See the 2026-09-10 CHANGELOG entry before changing it.
- **`experiments/` is gitignored.** Checkpoints worth keeping get archived to
  `experiments/l96/` and referenced from a report.
- **Nested `4dvarnet-fm-*/` directories are git worktrees**, one per topic branch
  (`docs/worktrees.md`). They are not part of the source tree.
- **Not every test file runs in CI yet.** See `docs/scoping/refactor_plan.md` Phase 0.

## 5. Where the code is going

`docs/scoping/refactor_plan.md` has the contributor-facing refactor plan: what is being
consolidated, in what order, and why a per-case-study directory split is
explicitly *not* the plan.
