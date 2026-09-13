#!/bin/bash
#SBATCH --job-name=qg_q5_train
#SBATCH --partition=Odyssey_GPU
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=96G
#SBATCH --time=24:00:00
#SBATCH --output=batch/logs/qg_q5_train_%j.out
#SBATCH --error=batch/logs/qg_q5_train_%j.err

set -e

# Q5: Q4's noisy forcing+param conditioning PLUS the raw initial-condition
# input, trained at the DA reference case's own lag=5.0d/noise=0.05 (see
# PLAN.md's 2026-09-11 Q5 section and
# config/experiment/Q5_direct_unet_s1_noisy_ic_cond.yaml). Same production
# cache / MONAI 2D circular arch / cosine-LR default as Q1 -- only the
# conditioning + obs/IC protocol differ.
#
# This script's original run (job 53274, default gradient_clip_val=10.0)
# completed cleanly (S0 psi EV=0.935/q EV=0.372) -- Q5 never collapsed.
# The YAML now sets training.gradient_clip_val: 1.0 anyway (2026-09-13),
# for consistency with Q3-noise0.05 where clip=1.0 fixed a real collapse
# at the same obs_noise_std_frac=0.05 -- confirmed no downside for Q5
# (gradclip1 rerun: psi EV=0.934/q EV=0.385, also 200 epochs clean; see
# experiments/Q5_direct_unet_s1_noisy_ic_cond_gradclip1/).
#
# --exp-id is required: Q1/Q3/Q4/Q5 all share model_type=direct_unet.
#
# experiments/qg_psi_norm_stats.pt, qg_param_norm_stats.pt, and
# qg_forcing_norm_stats.pt must exist first (batch/run_qg_param_stats.sbatch)
# -- reused unchanged from Q1/Q3/Q4 (they don't depend on obs_noise_std_frac/
# init_lag_days at all).
#
# --cpus-per-task=6 backs train_qg_neural.py's --num-workers default of 4
# (see run_qg_q4_train.sbatch's comment for the full rationale).
CACHE_DIR="/Odyssey/private/rfablet/Python/4dvarnet-fm-qg-100samples/reports/qg/outputs/qg_windows_1000_100_100/cache"
PY=/Odyssey/private/rfablet/miniforge3/envs/fdv-monai-proto/bin/python

$PY train_qg_neural.py \
    --model-type direct_unet --exp-id Q5_direct_unet_s1_noisy_ic_cond \
    --exp-dir experiments/Q5_direct_unet_s1_noisy_ic_cond \
    --nx 64 --epochs 200 \
    --num-train 1000 --num-val 100 --num-test 100 \
    --batch-size 2 --cache-dir "$CACHE_DIR"
