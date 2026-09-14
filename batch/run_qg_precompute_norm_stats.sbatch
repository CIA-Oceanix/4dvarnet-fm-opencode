#!/bin/bash
#SBATCH --job-name=qg_precompute_norm
#SBATCH --partition=Mee_Global_CPU
#SBATCH --account=mee
#SBATCH --qos=mee_short
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --time=02:00:00
#SBATCH --output=batch/logs/qg_precompute_norm_%j.out
#SBATCH --error=batch/logs/qg_precompute_norm_%j.err

set -e

# Precompute global psi mean/std + derived q_loss_weight for the Q1
# psi-only-normalization relaunch (see PLAN.md QG neural baseline section).
CACHE_DIR="/Odyssey/private/rfablet/Python/4dvarnet-fm-qg-100samples/reports/qg/outputs/qg_windows_1000_100_100/cache"
PY=/Odyssey/private/rfablet/miniforge3/envs/fdv/bin/python

$PY precompute_qg_norm_stats.py \
    --nx 64 --train-seed 42 --num-train 1000 \
    --cache-dir "$CACHE_DIR" \
    --output experiments/qg_psi_norm_stats.pt
