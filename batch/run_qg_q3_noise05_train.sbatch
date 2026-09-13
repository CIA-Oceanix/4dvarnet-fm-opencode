#!/bin/bash
#SBATCH --job-name=qg_q3_noise05_train
#SBATCH --partition=Odyssey_GPU
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=96G
#SBATCH --time=24:00:00
#SBATCH --output=batch/logs/qg_q3_noise05_train_%j.out
#SBATCH --error=batch/logs/qg_q3_noise05_train_%j.err

set -e

# Q3-noise0.05: same oracle (true forcing/params) conditioning as
# Q3_direct_unet_s0_oracle_cond.yaml, retrained at obs_noise_std_frac=0.05
# (the DA reference case's own value) -- see
# config/experiment/Q3_direct_unet_s0_oracle_cond_noise05.yaml for the full
# collapse history/naming rationale.
#
# RESOLVED (2026-09-13): the run this script originally launched (job
# 53275) collapsed at epoch ~19-20 under the then-default
# gradient_clip_val=10.0. Two ablations (run_qg_q3_noise05_gradclip1_train.sbatch,
# run_qg_q3_noise05_seed_train.sbatch) confirmed the fix is tighter
# gradient clipping, not random seed -- the YAML now sets
# training.gradient_clip_val: 1.0 as its standing default, so running this
# script unchanged today reproduces the validated clean run (already
# available at experiments/Q3_direct_unet_s0_oracle_cond_noise05_gradclip1/,
# S0 psi EV=0.911/q EV=0.189, 200 epochs, no collapse) -- no need to rerun
# unless the checkpoint is needed under this script's original --exp-dir.
#
# --exp-id is required: Q1/Q3/Q4/Q5 all share model_type=direct_unet.
#
# experiments/qg_psi_norm_stats.pt, qg_param_norm_stats.pt, and
# qg_forcing_norm_stats.pt must exist first (batch/run_qg_param_stats.sbatch).
CACHE_DIR="/Odyssey/private/rfablet/Python/4dvarnet-fm-qg-100samples/reports/qg/outputs/qg_windows_1000_100_100/cache"
PY=/Odyssey/private/rfablet/miniforge3/envs/fdv-monai-proto/bin/python

$PY train_qg_neural.py \
    --model-type direct_unet --exp-id Q3_direct_unet_s0_oracle_cond_noise05 \
    --exp-dir experiments/Q3_direct_unet_s0_oracle_cond_noise05 \
    --nx 64 --epochs 200 \
    --num-train 1000 --num-val 100 --num-test 100 \
    --batch-size 2 --cache-dir "$CACHE_DIR"
