#!/bin/bash
#SBATCH --job-name=qg_q7_tchannels_noise05
#SBATCH --partition=Odyssey_GPU
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=96G
#SBATCH --time=24:00:00
#SBATCH --output=batch/logs/qg_q7_tchannels_noise05_%j.out
#SBATCH --error=batch/logs/qg_q7_tchannels_noise05_%j.err

set -e

# Q7-noise05: Q7 (T-channels DirectUNet, obs-only) retrained at the DA
# reference case's own lag=5.0d/noise=0.05, matching Q8/Q9/Q10's obs
# difficulty -- see config/experiment/Q7_direct_unet_tchannels_s0_noise05.yaml
# and PLAN.md's 2026-09-14 T-channels bench refresh section.
CACHE_DIR="/Odyssey/private/rfablet/Python/4dvarnet-fm-qg-100samples/reports/qg/outputs/qg_windows_1000_100_100/cache"
PY=/Odyssey/private/rfablet/miniforge3/envs/fdv-monai-proto/bin/python

$PY train_qg_neural.py \
    --model-type direct_unet_tchannels \
    --exp-id Q7_direct_unet_tchannels_s0_noise05 \
    --exp-dir experiments/Q7_direct_unet_tchannels_s0_noise05 \
    --nx 64 --epochs 200 \
    --num-train 1000 --num-val 100 --num-test 100 \
    --batch-size 2 --cache-dir "$CACHE_DIR"
