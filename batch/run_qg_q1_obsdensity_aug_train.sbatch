#!/bin/bash
#SBATCH --job-name=qg_q1_obsdensity_aug_train
#SBATCH --partition=Odyssey_GPU
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=96G
#SBATCH --time=24:00:00
#SBATCH --output=batch/logs/qg_q1_obsdensity_aug_train_%j.out
#SBATCH --error=batch/logs/qg_q1_obsdensity_aug_train_%j.err

set -e

# Q1-obsdensity: same obs-only DirectUNet as Q1, but the train split
# resamples cols_per_day ~ Uniform{4,...,12} every draw instead of a single
# fixed value (see config/experiment/Q1_direct_unet_s0_obsdensity_aug.yaml
# and data/qg_neural.py's cols_per_day_range docstring) -- mirrors L96's
# obs-density-augmented training so one checkpoint generalizes across
# observing-network densities. 12 (not the originally-planned 24) is the
# physical max at dt=7200s (steps_per_day=12) -- a smoke test with a wider
# range hung a real GPU job (job 53434) before this cap + a defensive
# ValueError were added, see PLAN.md's 2026-09-13 note.
#
# --exp-id is required: Q1/Q1-obsdensity/Q3/Q4/Q5 all share
# model_type=direct_unet.
#
# experiments/qg_psi_norm_stats.pt must exist first
# (batch/run_qg_param_stats.sbatch) -- reused unchanged from Q1.
#
# --cpus-per-task=6 backs train_qg_neural.py's --num-workers default of 4
# (see run_qg_q4_train.sbatch's comment for the full rationale).
CACHE_DIR="/Odyssey/private/rfablet/Python/4dvarnet-fm-qg-100samples/reports/qg/outputs/qg_windows_1000_100_100/cache"
PY=/Odyssey/private/rfablet/miniforge3/envs/fdv-monai-proto/bin/python

$PY train_qg_neural.py \
    --model-type direct_unet --exp-id Q1_direct_unet_s0_obsdensity_aug \
    --exp-dir experiments/Q1_direct_unet_s0_obsdensity_aug \
    --nx 64 --epochs 200 \
    --num-train 1000 --num-val 100 --num-test 100 \
    --batch-size 2 --cache-dir "$CACHE_DIR"
