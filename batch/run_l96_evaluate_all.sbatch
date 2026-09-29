#!/bin/bash
#SBATCH --account=odyssey
#SBATCH --partition=Odyssey
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --job-name=l96-evaluate-all
#SBATCH --output=sbatch_logs/l96_evaluate_all_%j.out
#SBATCH --error=sbatch_logs/l96_evaluate_all_%j.err

set -euo pipefail

cd /Odyssey/private/rfablet/Python/4dvarnet-fm-opencode

export PATH="/Odyssey/private/rfablet/miniforge3/envs/fdv-monai-proto/bin:$PATH"
export CONDA_DEFAULT_ENV=fdv-monai-proto
export PYTHONUNBUFFERED=1
NODE_TMP="/SCRATCH/$USER/slurm_tmp/${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID:-0}"; mkdir -p "$NODE_TMP" 2>/dev/null || { NODE_TMP="/tmp/${USER}_slurm_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID:-0}"; mkdir -p "$NODE_TMP"; }
export TMPDIR="$NODE_TMP" MPLCONFIGDIR="$NODE_TMP/mplconfig"; trap 'rm -rf "$NODE_TMP"' EXIT
mkdir -p "$MPLCONFIGDIR"

echo "============================================"
echo " L96 consolidated benchmark report | $(date)"
echo "  (CPU-only over cached artifacts)"
echo "============================================"
echo ""

python reports/l96/generate_l96_consolidated_report.py

echo ""
echo " Done at: $(date)"
echo "============================================"
