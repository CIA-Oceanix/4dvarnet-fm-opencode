#!/bin/bash
#SBATCH --job-name=qg_q3_n05_seed_train
#SBATCH --partition=Odyssey_GPU
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=96G
#SBATCH --time=24:00:00
#SBATCH --output=batch/logs/qg_q3_n05_seed_train_%j.out
#SBATCH --error=batch/logs/qg_q3_n05_seed_train_%j.err

set -e

# Ablation 2/2 for Q3-noise0.05's epoch-19-20 collapse (see
# config/experiment/Q3_direct_unet_s0_oracle_cond_noise05.yaml): identical
# config and gradient_clip_val (default 10.0, unchanged), but a different,
# explicit --seed -- tests whether the collapse recurs regardless of the
# specific random draw (model init + dataloader shuffling), or whether the
# original collapse was closer to "bad luck" tied to that particular
# (untracked -- the project had no seeding infrastructure before this)
# random state. train_qg_neural.py's --seed calls pl.seed_everything()
# once at startup.
CACHE_DIR="/Odyssey/private/rfablet/Python/4dvarnet-fm-qg-100samples/reports/qg/outputs/qg_windows_1000_100_100/cache"
PY=/Odyssey/private/rfablet/miniforge3/envs/fdv-monai-proto/bin/python

$PY train_qg_neural.py \
    --model-type direct_unet --exp-id Q3_direct_unet_s0_oracle_cond_noise05 \
    --exp-dir experiments/Q3_direct_unet_s0_oracle_cond_noise05_seed123 \
    --nx 64 --epochs 200 --seed 123 \
    --num-train 1000 --num-val 100 --num-test 100 \
    --batch-size 2 --cache-dir "$CACHE_DIR"
