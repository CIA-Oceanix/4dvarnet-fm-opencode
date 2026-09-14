#!/bin/bash
#SBATCH --job-name=qg_q4_train
#SBATCH --partition=Odyssey_GPU
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=96G
#SBATCH --time=24:00:00
#SBATCH --output=batch/logs/qg_q4_train_%j.out
#SBATCH --error=batch/logs/qg_q4_train_%j.err

set -e

# Q4: DirectUNet on QG S0, conditioned on NOISY (per-draw resampled-severity
# corrupted) wind-forcing + params -- mirrors the L96 SDA3 CFM study's
# noisy-params conditioning (see PLAN.md's 2026-09-10 QG Q3/Q4 section and
# config/experiment/Q4_direct_unet_s1_noisy_cond.yaml). Same production
# cache / MONAI 2D circular arch / cosine-LR default as Q1
# (batch/run_qg_q1_train.sbatch) -- only the conditioning differs.
#
# --exp-id is required: Q1/Q3/Q4 all share model_type=direct_unet.
#
# experiments/qg_psi_norm_stats.pt AND experiments/qg_param_norm_stats.pt
# must exist first (batch/run_qg_param_stats.sbatch).
#
# --cpus-per-task=6 backs train_qg_neural.py's --num-workers default of 4
# (+ headroom for the main process/collate) -- num_workers=1 serialized
# each draw's CPU-bound forcing/param prep with GPU training, a measured
# ~2x epoch-time cost for Q4's noisy-forcing recomputation (caught by a
# training smoke test; also fixed at the source by caching the QGDynamics
# object instead of rebuilding it every draw, data/qg_neural.py's
# `_cached_qg_dynamics`).
CACHE_DIR="/Odyssey/private/rfablet/Python/4dvarnet-fm-qg-100samples/reports/qg/outputs/qg_windows_1000_100_100/cache"
PY=/Odyssey/private/rfablet/miniforge3/envs/fdv-monai-proto/bin/python

$PY train_qg_neural.py \
    --model-type direct_unet --exp-id Q4_direct_unet_s1_noisy_cond \
    --exp-dir experiments/Q4_direct_unet_s1_noisy_cond \
    --nx 64 --epochs 200 \
    --num-train 1000 --num-val 100 --num-test 100 \
    --batch-size 2 --cache-dir "$CACHE_DIR"
