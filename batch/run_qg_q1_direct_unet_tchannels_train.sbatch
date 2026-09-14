#!/bin/bash
#SBATCH --job-name=qg_q1_tchannels
#SBATCH --partition=Odyssey_GPU
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=96G
#SBATCH --time=24:00:00
#SBATCH --output=batch/logs/qg_q1_tchannels_%j.out
#SBATCH --error=batch/logs/qg_q1_tchannels_%j.err

set -e

# Q1 (T-channels DirectUNet, obs-only) -- see
# config/experiment/Q1_direct_unet_tchannels_s0.yaml and PLAN.md's
# 2026-09-14 T-channels bench refresh section.
CACHE_DIR="/Odyssey/private/rfablet/Python/4dvarnet-fm-qg-100samples/reports/qg/outputs/qg_windows_1000_100_100/cache"
PY=/Odyssey/private/rfablet/miniforge3/envs/fdv-monai-proto/bin/python

$PY train_qg_neural.py \
    --model-type direct_unet_tchannels \
    --exp-id Q1_direct_unet_tchannels_s0 \
    --exp-dir experiments/Q1_direct_unet_tchannels_s0 \
    --nx 64 --epochs 200 \
    --num-train 1000 --num-val 100 --num-test 100 \
    --batch-size 2 --cache-dir "$CACHE_DIR"
