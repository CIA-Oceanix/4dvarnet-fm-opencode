#!/bin/bash
#SBATCH --job-name=qg_q5_gc1_train
#SBATCH --partition=Odyssey_GPU
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=96G
#SBATCH --time=24:00:00
#SBATCH --output=batch/logs/qg_q5_gc1_train_%j.out
#SBATCH --error=batch/logs/qg_q5_gc1_train_%j.err

set -e

# Q5 retrained with --gradient-clip-val 1.0 instead of the default 10.0.
# The original Q5 run (experiments/Q5_direct_unet_s1_noisy_ic_cond, job
# 53274) completed all 200 epochs cleanly at the default clip and is kept
# as-is. This rerun applies the tighter clip found to fix Q3-noise0.05's
# collapse (see config/experiment/Q3_direct_unet_s0_oracle_cond_noise05.yaml
# and CHANGELOG.d/2026-09-12-qg-q3-noise05-collapse-ablations.md) to Q5 too,
# since Q5 shares the same obs_noise_std_frac=0.05 as Q3-noise0.05 -- as a
# precaution / for consistency, not because Q5 itself collapsed.
#
# --exp-id is required: Q1/Q3/Q4/Q5 all share model_type=direct_unet.
CACHE_DIR="/Odyssey/private/rfablet/Python/4dvarnet-fm-qg-100samples/reports/qg/outputs/qg_windows_1000_100_100/cache"
PY=/Odyssey/private/rfablet/miniforge3/envs/fdv-monai-proto/bin/python

$PY train_qg_neural.py \
    --model-type direct_unet --exp-id Q5_direct_unet_s1_noisy_ic_cond \
    --exp-dir experiments/Q5_direct_unet_s1_noisy_ic_cond_gradclip1 \
    --nx 64 --epochs 200 --gradient-clip-val 1.0 \
    --num-train 1000 --num-val 100 --num-test 100 \
    --batch-size 2 --cache-dir "$CACHE_DIR"
