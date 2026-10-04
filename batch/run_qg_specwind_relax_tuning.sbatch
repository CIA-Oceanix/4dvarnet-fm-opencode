#!/bin/bash
#SBATCH --account=odyssey
#SBATCH --partition=Odyssey_GPU
#SBATCH --gres=gpu:rtx8000:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --job-name=qg-relax-tuning
#SBATCH --output=sbatch_logs/qg_relax_tuning_%A_%a.out
#SBATCH --error=sbatch_logs/qg_relax_tuning_%A_%a.err
# RTPS/RTPP relaxation-inflation sweep on val (evaluation/qg_specwind_relax_tuning.py): one task = one
# ETKF or EnKF configuration over 20 val windows. rtx8000 only (val is regenerated from seeds).
#   sbatch --array=0-43%6 batch/run_qg_specwind_relax_tuning.sbatch
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
PY=/Odyssey/private/rfablet/miniforge3/envs/${CONDA_ENV:-fdv-monai-proto}/bin/python
export OMP_NUM_THREADS=4
NODE_TMP="/SCRATCH/$USER/slurm_tmp/${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID:-0}"; mkdir -p "$NODE_TMP" 2>/dev/null || { NODE_TMP="/tmp/${USER}_slurm_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID:-0}"; mkdir -p "$NODE_TMP"; }
export TMPDIR="$NODE_TMP" MPLCONFIGDIR="$NODE_TMP/mplconfig"; trap 'rm -rf "$NODE_TMP"' EXIT
"$PY" -m evaluation.qg_specwind_relax_tuning --task "${SLURM_ARRAY_TASK_ID:?}" \
  --out-dir "${OUT_DIR:-experiments/qg_specwind_relax_tuning}" --root experiments/qg_datasets --device cuda
