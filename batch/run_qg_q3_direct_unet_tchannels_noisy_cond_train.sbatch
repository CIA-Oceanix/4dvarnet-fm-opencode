#!/bin/bash
#SBATCH --job-name=qg_q3_tchannels_noisy
#SBATCH --partition=Odyssey_GPU
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=96G
#SBATCH --time=24:00:00
#SBATCH --output=batch/logs/qg_q3_tchannels_noisy_%j.out
#SBATCH --error=batch/logs/qg_q3_tchannels_noisy_%j.err

set -e

# Q3 (T-channels DirectUNet, noisy-trained forcing+param conditioning) --
# see config/experiment/Q3_direct_unet_tchannels_s1_noisy_cond.yaml and
# PLAN.md's 2026-09-14 T-channels bench refresh section.
CACHE_DIR="/Odyssey/private/rfablet/Python/4dvarnet-fm-qg-100samples/reports/qg/outputs/qg_windows_1000_100_100/cache"
PY=/Odyssey/private/rfablet/miniforge3/envs/fdv-monai-proto/bin/python

$PY train_qg_neural.py \
    --model-type direct_unet_tchannels \
    --exp-id Q3_direct_unet_tchannels_s1_noisy_cond \
    --exp-dir experiments/Q3_direct_unet_tchannels_s1_noisy_cond \
    --nx 64 --epochs 200 \
    --num-train 1000 --num-val 100 --num-test 100 \
    --batch-size 2 --cache-dir "$CACHE_DIR"
