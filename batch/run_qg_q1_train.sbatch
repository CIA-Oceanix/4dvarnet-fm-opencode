#!/bin/bash
#SBATCH --job-name=qg_q1_train
#SBATCH --partition=Odyssey_GPU
#SBATCH --gres=gpu:a40:1
#SBATCH --mem=96G
#SBATCH --time=24:00:00
#SBATCH --output=batch/logs/qg_q1_train_%j.out
#SBATCH --error=batch/logs/qg_q1_train_%j.err

set -e

# Q1: DirectUNet neural baseline on QG S0 (nx=64, 1000/100/100 train/val/test),
# reusing the pre-generated production truth cache from the sibling
# feature/qg-100sample-benchmark worktree (--train/--val/--test-seed default
# to that pipeline's SPLIT_SEED_BASE, and --num-windows default to its
# SPLIT_SIZE, so this hits the existing cache instead of re-paying the
# ~2-year-spinup rollout -- see train_qg_neural.py's module docstring).
CACHE_DIR="/Odyssey/private/rfablet/Python/4dvarnet-fm-qg-100samples/reports/qg/outputs/qg_windows_1000_100_100/cache"
PY=/Odyssey/private/rfablet/miniforge3/envs/fdv/bin/python

$PY train_qg_neural.py \
    --model-type direct_unet \
    --exp-dir experiments/Q1_direct_unet_s0 \
    --nx 64 --q-loss-weight 0.1 --epochs 200 \
    --num-train 1000 --num-val 100 --num-test 100 \
    --batch-size 2 --cache-dir "$CACHE_DIR"
