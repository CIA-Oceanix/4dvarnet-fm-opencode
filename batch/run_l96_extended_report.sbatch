#!/bin/bash
#SBATCH --account=odyssey
#SBATCH --partition=Odyssey_GPU
#SBATCH --gres=gpu:rtx8000:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --time=04:00:00
#SBATCH --job-name=l96-ext-report
#SBATCH --output=sbatch_logs/l96_extended_report_%j.out
#SBATCH --error=sbatch_logs/l96_extended_report_%j.err
# Render reports/l96/outputs/l96_benchmark_extended.md (first pass scores the
# stored 30-member ensembles; needs more memory than the shared login node).
set -euo pipefail
cd /Odyssey/private/rfablet/Python/4dvarnet-fm-opencode/4dvarnet-fm-sda-nanfix
export PATH="/Odyssey/private/rfablet/miniforge3/envs/fdv-monai-proto/bin:$PATH"
export PYTHONUNBUFFERED=1
NODE_TMP="/SCRATCH/$USER/slurm_tmp/${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID:-0}"; mkdir -p "$NODE_TMP" 2>/dev/null || { NODE_TMP="/tmp/${USER}_slurm_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID:-0}"; mkdir -p "$NODE_TMP"; }
export TMPDIR="$NODE_TMP" MPLCONFIGDIR="$NODE_TMP/mplconfig"; trap 'rm -rf "$NODE_TMP"' EXIT
mkdir -p "$MPLCONFIGDIR"
python reports/l96/generate_l96_benchmark_extended_report.py
