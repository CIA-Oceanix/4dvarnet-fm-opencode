#!/bin/bash
#SBATCH --job-name=qg_q3_lag5_train
#SBATCH --partition=Odyssey_GPU
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=96G
#SBATCH --time=24:00:00
#SBATCH --output=batch/logs/qg_q3_lag5_train_%j.out
#SBATCH --error=batch/logs/qg_q3_lag5_train_%j.err

set -e

# Q3-lag5: same oracle (true forcing/params) conditioning as
# Q3_direct_unet_s0_oracle_cond.yaml, retrained at the DA reference case's
# own lag=5.0d/noise=0.05 (see PLAN.md's 2026-09-11 Q5 section and
# config/experiment/Q3_direct_unet_s0_oracle_cond_lag5.yaml) -- paired with
# Q5 so the "oracle vs. IC-augmented" comparison isn't confounded by also
# comparing across two different training distributions.
#
# --exp-id is required: Q1/Q3/Q4/Q5 all share model_type=direct_unet.
#
# experiments/qg_psi_norm_stats.pt, qg_param_norm_stats.pt, and
# qg_forcing_norm_stats.pt must exist first (batch/run_qg_param_stats.sbatch).
CACHE_DIR="/Odyssey/private/rfablet/Python/4dvarnet-fm-qg-100samples/reports/qg/outputs/qg_windows_1000_100_100/cache"
PY=/Odyssey/private/rfablet/miniforge3/envs/fdv-monai-proto/bin/python

$PY train_qg_neural.py \
    --model-type direct_unet --exp-id Q3_direct_unet_s0_oracle_cond_lag5 \
    --exp-dir experiments/Q3_direct_unet_s0_oracle_cond_lag5 \
    --nx 64 --epochs 200 \
    --num-train 1000 --num-val 100 --num-test 100 \
    --batch-size 2 --cache-dir "$CACHE_DIR"
