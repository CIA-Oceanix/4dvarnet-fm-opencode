#!/bin/bash
#SBATCH --account=odyssey
#SBATCH --partition=Odyssey_GPU
#SBATCH --gres=gpu:rtx8000:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=01:30:00
#SBATCH --job-name=qg-specwind-s1
#SBATCH --output=sbatch_logs/qg_specwind_s1_%A_%a.out
#SBATCH --error=sbatch_logs/qg_specwind_s1_%A_%a.err
# S1 calibration / attribution on val (evaluation/qg_specwind_s1_sweep.py): one array task = one
# (kappa, components) configuration over 20 val windows. rtx8000 only (val is regenerated from seeds).
#   PHASE=calib sbatch --array=0-18 batch/run_qg_specwind_s1.sbatch
#   PHASE=shapley KAPPA=2 sbatch --array=0-15 batch/run_qg_specwind_s1.sbatch
#   PHASE=realistic_calib sbatch --array=0-4 batch/run_qg_specwind_s1.sbatch
#   PHASE=realistic_shapley VARIANT=base sbatch --array=0-31 batch/run_qg_specwind_s1.sbatch
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
PY=/Odyssey/private/rfablet/miniforge3/envs/${CONDA_ENV:-fdv-monai-proto}/bin/python
export OMP_NUM_THREADS=4
NODE_TMP="/SCRATCH/$USER/slurm_tmp/${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID:-0}"; mkdir -p "$NODE_TMP" 2>/dev/null || { NODE_TMP="/tmp/${USER}_slurm_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID:-0}"; mkdir -p "$NODE_TMP"; }
export TMPDIR="$NODE_TMP" MPLCONFIGDIR="$NODE_TMP/mplconfig"; trap 'rm -rf "$NODE_TMP"' EXIT
EXTRA=()
[ -n "${KAPPA:-}" ] && EXTRA+=(--kappa "$KAPPA")
[ -n "${VARIANT:-}" ] && EXTRA+=(--variant "$VARIANT")
"$PY" -m evaluation.qg_specwind_s1_sweep --phase "${PHASE:-calib}" --task "${SLURM_ARRAY_TASK_ID:?}" \
  --disp-frac "${DISP:-1.0}" --out-dir "${OUT_DIR:-experiments/qg_specwind_s1_ensrf}" \
  --root experiments/qg_datasets --device cuda "${EXTRA[@]}"
