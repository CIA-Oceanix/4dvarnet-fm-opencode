#!/bin/bash
#SBATCH --job-name=qg_q1_smoke
#SBATCH --partition=Odyssey_GPU
#SBATCH --gres=gpu:a40:1
#SBATCH --mem=96G
#SBATCH --time=00:30:00
#SBATCH --output=batch/logs/qg_q1_smoke_%j.out
#SBATCH --error=batch/logs/qg_q1_smoke_%j.err

set -e

# 2-epoch smoke test of the MONAI circular-2D-UNet + cosine-scheduler relaunch
# (2026-09-10), on the real production 1000/100/100 nx=64 cache (same windows
# so it hits the cache instead of re-paying the ~2-year-spinup rollout) --
# just to validate the full train_qg_neural.py wiring (YAML load, norm_stats
# load, dataset, Lightning fwd/bwd, eval) before committing to the full
# 200-epoch run. `direct_unet` now builds models.monai_unet_qg2d.
# MonaiDirectUNetQG (circular Conv2d over the doubly-periodic (ny,nx) grid),
# which requires the fdv-monai-proto env (see that module's docstring).
CACHE_DIR="/Odyssey/private/rfablet/Python/4dvarnet-fm-qg-100samples/reports/qg/outputs/qg_windows_1000_100_100/cache"
PY=/Odyssey/private/rfablet/miniforge3/envs/fdv-monai-proto/bin/python

$PY train_qg_neural.py \
    --model-type direct_unet \
    --exp-dir experiments/Q1_direct_unet_s0_smoke \
    --nx 64 --epochs 2 \
    --num-train 1000 --num-val 100 --num-test 100 \
    --batch-size 2 --cache-dir "$CACHE_DIR"
