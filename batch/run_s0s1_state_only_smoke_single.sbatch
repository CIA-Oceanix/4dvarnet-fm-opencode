#!/bin/bash
#SBATCH --account=odyssey
#SBATCH --partition=Odyssey_GPU
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:15:00
#SBATCH --job-name=s0s1-smoke-s1
#SBATCH --output=sbatch_logs/s0s1_smoke_single_%j.out
#SBATCH --error=sbatch_logs/s0s1_smoke_single_%j.err

set -euo pipefail

cd /Odyssey/private/rfablet/Python/4dvarnet-fm-opencode

export PATH="/Odyssey/private/rfablet/miniforge3/envs/fdv/bin:$PATH"
export CONDA_DEFAULT_ENV=fdv
export PYTHONUNBUFFERED=1
export MPLCONFIGDIR="/tmp/mplconfig_${SLURM_JOB_ID}"
mkdir -p "$MPLCONFIGDIR" sbatch_logs experiments

python train.py \
  --config-name experiment/S1_direct_unet_s0s1 \
  training.stage1.epochs=1 \
  data.num_train_windows=10 \
  data.num_val_windows=2 \
  data.num_test_windows=5 \
  hydra.run.dir=. \
  hydra.output_subdir=null

echo "Done at $(date)"
