#!/bin/bash
#SBATCH --account=odyssey
#SBATCH --partition=Odyssey_GPU
#SBATCH --gres=gpu:rtx8000:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=01:30:00
#SBATCH --job-name=qg-enks-tuning
#SBATCH --output=sbatch_logs/qg_enks_tuning_%A_%a.out
#SBATCH --error=sbatch_logs/qg_enks_tuning_%A_%a.err
# EnKS lag / time-taper tuning on val (evaluation/qg_specwind_enks_tuning.py): one task = one EnKS
# configuration (S0 or S1-tuned) over 20 val windows. rtx8000 only (val is regenerated from seeds).
#   sbatch --array=0-32 batch/run_qg_specwind_enks_tuning.sbatch   (13-26: ensemble size; 27-32: N=320 re-tune, use --time=05:00:00 --mem=96G)
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
PY=/Odyssey/private/rfablet/miniforge3/envs/${CONDA_ENV:-fdv-monai-proto}/bin/python
export OMP_NUM_THREADS=4
NODE_TMP="/SCRATCH/$USER/slurm_tmp/${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID:-0}"; mkdir -p "$NODE_TMP" 2>/dev/null || { NODE_TMP="/tmp/${USER}_slurm_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID:-0}"; mkdir -p "$NODE_TMP"; }
export TMPDIR="$NODE_TMP" MPLCONFIGDIR="$NODE_TMP/mplconfig"; trap 'rm -rf "$NODE_TMP"' EXIT
"$PY" -m evaluation.qg_specwind_enks_tuning --task "${SLURM_ARRAY_TASK_ID:?}" \
  --out-dir "${OUT_DIR:-experiments/qg_specwind_enks_tuning}" --root experiments/qg_datasets --device cuda
