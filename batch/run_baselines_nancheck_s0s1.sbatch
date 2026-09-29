#!/bin/bash
#SBATCH --account=odyssey
#SBATCH --partition=Odyssey_GPU
#SBATCH --gres=gpu:rtx8000:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --job-name=bl-nan-s0s1
#SBATCH --output=sbatch_logs/bl_nan_s0s1_%j.out
#SBATCH --error=sbatch_logs/bl_nan_s0s1_%j.err

set -euo pipefail

cd /Odyssey/private/rfablet/Python/4dvarnet-fm-opencode

export PATH="/Odyssey/private/rfablet/miniforge3/envs/fdv-monai-proto/bin:$PATH"
export CONDA_DEFAULT_ENV=fdv-monai-proto
export PYTHONUNBUFFERED=1
NODE_TMP="/SCRATCH/$USER/slurm_tmp/${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID:-0}"; mkdir -p "$NODE_TMP" 2>/dev/null || { NODE_TMP="/tmp/${USER}_slurm_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID:-0}"; mkdir -p "$NODE_TMP"; }
export TMPDIR="$NODE_TMP" MPLCONFIGDIR="$NODE_TMP/mplconfig"; trap 'rm -rf "$NODE_TMP"' EXIT
mkdir -p "$MPLCONFIGDIR" sbatch_logs experiments

echo "============================================"
echo " Baseline NaN Check S0/S1 | $(date)"
echo "  GPU: $(nvidia-smi --query-gpu=name --format=csv -i 0 2>/dev/null | tail -1)"
echo "  DWS=50, EnKF infl=2.0, ETKF infl=2.0"
echo "============================================"

python batch/_run_baselines_nancheck_s0s1.py

echo "============================================"
echo " Done at $(date)"
echo "============================================"