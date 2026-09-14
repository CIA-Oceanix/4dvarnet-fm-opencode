#!/bin/bash
#SBATCH --job-name=qg_q3_n05_gc1_train
#SBATCH --partition=Odyssey_GPU
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=96G
#SBATCH --time=24:00:00
#SBATCH --output=batch/logs/qg_q3_n05_gc1_train_%j.out
#SBATCH --error=batch/logs/qg_q3_n05_gc1_train_%j.err

set -e

# Ablation 1/2 for Q3-noise0.05's epoch-19-20 collapse (see
# config/experiment/Q3_direct_unet_s0_oracle_cond_noise05.yaml): identical
# config, but --gradient-clip-val 1.0 instead of the default 10.0 --
# tests whether tighter global gradient-norm clipping alone prevents the
# collapse (working hypothesis: an occasional large gradient from the 5x
# higher obs noise isn't fully contained by the looser default clip).
# Note --seed is NOT set here (isolates the clip-value effect alone from
# --seed's random-draw effect, see run_qg_q3_noise05_seed_train.sbatch).
CACHE_DIR="/Odyssey/private/rfablet/Python/4dvarnet-fm-qg-100samples/reports/qg/outputs/qg_windows_1000_100_100/cache"
PY=/Odyssey/private/rfablet/miniforge3/envs/fdv-monai-proto/bin/python

$PY train_qg_neural.py \
    --model-type direct_unet --exp-id Q3_direct_unet_s0_oracle_cond_noise05 \
    --exp-dir experiments/Q3_direct_unet_s0_oracle_cond_noise05_gradclip1 \
    --nx 64 --epochs 200 --gradient-clip-val 1.0 \
    --num-train 1000 --num-val 100 --num-test 100 \
    --batch-size 2 --cache-dir "$CACHE_DIR"
