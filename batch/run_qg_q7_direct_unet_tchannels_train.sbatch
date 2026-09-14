#!/bin/bash
#SBATCH --job-name=qg_q7_tchannels_train
#SBATCH --partition=Odyssey_GPU
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=96G
#SBATCH --time=24:00:00
#SBATCH --output=batch/logs/qg_q7_tchannels_train_%j.out
#SBATCH --error=batch/logs/qg_q7_tchannels_train_%j.err

set -e

# Q7: same M-tier capacity + single-pass obs-only role as Q1, but merges
# T (days) into the channel axis (models.monai_unet_qg2d.
# MonaiDirectUNetQGChannelTime, built on Q6's own MonaiUNet2DQGSolver
# backbone) instead of Q1's per-day-independent batch-folding -- see
# config/experiment/Q7_direct_unet_tchannels_s0.yaml and PLAN.md's Q6/Q7
# sections for the full rationale.
CACHE_DIR="/Odyssey/private/rfablet/Python/4dvarnet-fm-qg-100samples/reports/qg/outputs/qg_windows_1000_100_100/cache"
PY=/Odyssey/private/rfablet/miniforge3/envs/fdv-monai-proto/bin/python

$PY train_qg_neural.py \
    --model-type direct_unet_tchannels --exp-id Q7_direct_unet_tchannels_s0 \
    --exp-dir experiments/Q7_direct_unet_tchannels_s0 \
    --nx 64 --epochs 200 \
    --num-train 1000 --num-val 100 --num-test 100 \
    --batch-size 2 --cache-dir "$CACHE_DIR"
