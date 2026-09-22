#!/bin/bash
#SBATCH --account=odyssey
#SBATCH --partition=Odyssey_GPU
#SBATCH --gres=gpu:a40:1
#SBATCH --nodelist=sl-mee-br-205
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=24:00:00
#SBATCH --job-name=l96-sda1_splus
#SBATCH --output=sbatch_logs/l96_sda1_splus_train_%j.out
#SBATCH --error=sbatch_logs/l96_sda1_splus_train_%j.err

set -euo pipefail
cd /Odyssey/private/rfablet/Python/4dvarnet-fm-opencode/4dvarnet-fm-fdv-tau-aware
export PATH="/Odyssey/private/rfablet/miniforge3/envs/fdv-monai-proto/bin:$PATH"
export CONDA_DEFAULT_ENV=fdv-monai-proto
export PYTHONUNBUFFERED=1
export MPLCONFIGDIR="/tmp/mplconfig_${SLURM_JOB_ID}"
mkdir -p "$MPLCONFIGDIR" sbatch_logs

EXPERIMENT="B4_sda1_monaiSplus_l96"
echo "=========================================="
echo " ${EXPERIMENT} | $(date)"
echo " GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader -i 0 2>/dev/null | tail -1)"
echo "=========================================="
python train.py --config-name experiment/${EXPERIMENT} hydra.run.dir=. hydra.output_subdir=null
echo " Done: ${EXPERIMENT} at $(date)"
