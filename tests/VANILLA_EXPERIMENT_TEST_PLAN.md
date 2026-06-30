# Vanilla Experiment Pre-Launch Test Plan

**Date:** 2026-06-30
**Scope:** E1-E3 (DirectUNet) and F1-F3 (VanillaCFM) sbatch launch readiness
**Related:** `batch/run_one_epoch_tests.sbatch`, `train.py`, `config/experiment/E*.yaml`, `config/experiment/F*.yaml`

---

## Test A: Config + Model Validation (CPU, ~30s)

Validates all 6 experiment configs load correctly and produce the right model type.

### Command
```bash
python -c "
import hydra, torch, sys
sys.path.insert(0, '.')
from train import model_factory
from models.direct_unet import DirectUNet
from models.vanilla_cfm import VanillaCFM

EXPS = [
    'E1_direct_unet_default', 'E2_direct_unet_small', 'E3_direct_unet_rand',
    'F1_vanilla_cfm_default', 'F2_vanilla_cfm_small', 'F3_vanilla_cfm_rand',
]
for exp in EXPS:
    with hydra.initialize(config_path='config'):
        cfg = hydra.compose('experiment/' + exp)
    model = model_factory(cfg, torch.device('cpu'))
    expected = DirectUNet if exp.startswith('E') else VanillaCFM
    assert isinstance(model, expected), f'{exp}: expected {expected.__name__}, got {type(model).__name__}'
    print(f'  OK  {exp}  ->  {type(model).__name__}')
"
```

### Pass Criteria
- All 6 configs load without Hydra errors
- `model_factory()` returns the correct model type
- No `KeyError` or `NameError` in schema resolution

### Failure Mode
- Schema mismatch (e.g., `DirectUNetConfig` renamed or removed)
- Config YAML references missing fields

---

## Test B: 1-Epoch Full-Data GPU Smoke (single sbatch, ~10 min)

Runs the largest DirectUNet config (E1) for 1 epoch with the **full 2000-window dataset** to verify GPU memory and dataset generation time.

### sbatch Script
```bash
#!/bin/bash
#SBATCH --account=odyssey
#SBATCH --partition=Odyssey_GPU
#SBATCH --gres=gpu:rtx8000:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --job-name=4dvarnet-preflight
#SBATCH --output=sbatch_logs/preflight_%j.out

set -euo pipefail
cd /Odyssey/private/rfablet/Python/4dvarnet-fm-opencode
export PATH="/Odyssey/private/rfablet/miniforge3/envs/fdv/bin:$PATH"
export PYTHONUNBUFFERED=1

echo "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv -i 0)"
python train.py --config-name experiment/E1_direct_unet_default \
    training.stage1.epochs=1 \
    hydra.run.dir=. \
    hydra.output_subdir=null
echo "Done"
```

### Pass Criteria
- Dataset generation completes within 10 min
- Peak GPU memory ≤ 22 GB (leaving headroom on 24 GB rtx8000)
- Training completes 1 epoch without OOM or NaN loss
- `results.json` and checkpoint written correctly

### If GPU Memory Fails
Reduce batch size: add `training.batch_size=16` to the command line and retest.

---

## Test C: Hydra CWD Compatibility (CPU, ~1 min)

Verifies that experiment configs work **without** `hydra.run.dir=.` and `hydra.output_subdir=null` overrides (i.e., using Hydra's default timestamped output directory). This is important if the sbatch script omits these for cleanliness.

### Command
```bash
# Run in a temp directory to isolate Hydra's output dir side effects
mkdir -p /tmp/hydra_test && cd /tmp/hydra_test
cp -r /Odyssey/private/rfablet/Python/4dvarnet-fm-opencode/train.py .
python train.py --config-name experiment/E2_direct_unet_small \
    training.stage1.epochs=1 \
    +data.num_train_windows=5 \
    +data.num_val_windows=2 \
    +data.num_test_windows=2
```

### Pass Criteria
- Hydra creates a timestamped output dir without error
- `experiments/E2_direct_unet_small/` is correctly populated
- No `FileNotFoundError` from `os.chdir()` or checkpoint saving

---

## Launch Sequence

```
Step 1: Run Test A (CPU, 30s)         -> if fail: fix schema/config
Step 2: Run Test B (GPU sbatch, 10m)  -> if fail: fix batch size / memory
Step 3: Run Test C (CPU, 1min)        -> if fail: add hydra overrides
Step 4: Launch full array job          -> sbatch batch/run_vanilla_experiments.sbatch
```

---

## Rollback

If the full array job fails:
- Check `sbatch_logs/vanilla_<array_job_id>_<task_id>.out` for each task
- The `train.py` skip-if-exists guard prevents wasted re-runs:
  ```python
  if os.path.exists(results_path): return
  ```
- Fix the issue, then `rm experiments/<exp_id>/results.json` for failed tasks and resubmit.
