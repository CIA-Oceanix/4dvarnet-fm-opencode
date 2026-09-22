#!/bin/bash
#SBATCH --account=odyssey
#SBATCH --partition=Odyssey_GPU
#SBATCH --gres=gpu:rtx8000:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --array=0-10
#SBATCH --job-name=l96-da-obscount
#SBATCH --output=sbatch_logs/l96_da_obs_count_%A_%a.out
#SBATCH --error=sbatch_logs/l96_da_obs_count_%A_%a.err
# ETKF/EnKF (P1 settings, per-window fast_weights passed to the DA model) vs number of random obs times per window, S0+S1,
# 10 windows x N_DRAWS draws per cell. One array task per n_obs; reg30 is the
# cached regular-grid reference on the same windows. Then:
#   python reports/l96/generate_l96_da_obs_count_report.py --da-fast-weights
set -euo pipefail
cd /Odyssey/private/rfablet/Python/4dvarnet-fm-opencode/4dvarnet-fm-random-obs-times
export PATH="/Odyssey/private/rfablet/miniforge3/envs/fdv-monai-proto/bin:$PATH"
export CONDA_DEFAULT_ENV=fdv-monai-proto
export PYTHONUNBUFFERED=1
export MPLCONFIGDIR="/tmp/mplconfig_${SLURM_JOB_ID}"
mkdir -p "$MPLCONFIGDIR" sbatch_logs
CELLS=(5 10 15 20 30 45 60 100 150 300 reg30)
N_OBS="${CELLS[$SLURM_ARRAY_TASK_ID]}"
N_DRAWS="${N_DRAWS:-5}"
echo "===== n_obs=${N_OBS} draws=${N_DRAWS} | $(date) | $(nvidia-smi --query-gpu=name --format=csv,noheader -i 0 | tail -1) ====="
python eval_da_obs_count_l96.py --n-obs "${N_OBS}" --n-windows 10 --n-draws "${N_DRAWS}" --da-fast-weights
echo "Done at $(date)"
