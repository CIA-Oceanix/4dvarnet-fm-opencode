#!/bin/bash
#SBATCH --account=odyssey
#SBATCH --partition=Odyssey_GPU
#SBATCH --gres=gpu:rtx8000:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --job-name=l96-sda-fdv1-sweep
#SBATCH --output=sbatch_logs/l96_sda_fdv1_sweep_%j.out
#SBATCH --error=sbatch_logs/l96_sda_fdv1_sweep_%j.err

set -euo pipefail

cd /Odyssey/private/rfablet/Python/4dvarnet-fm-opencode/4dvarnet-fm-fdv-monai

export PATH="/Odyssey/private/rfablet/miniforge3/envs/fdv-monai-proto/bin:$PATH"
export PYTHONUNBUFFERED=1
export MPLCONFIGDIR="/tmp/mplconfig_${SLURM_JOB_ID}"
mkdir -p "$MPLCONFIGDIR"

echo "FDV1(monai) + SDA3(monai) hybrid sweep | $(date)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv -i 0 2>/dev/null | tail -1)"
python sweep_sda_fdv1_hybrid_l96.py
echo "Sweep complete at $(date)"
