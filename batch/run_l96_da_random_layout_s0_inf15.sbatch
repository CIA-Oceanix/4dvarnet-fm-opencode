#!/bin/bash
#SBATCH --account=odyssey
#SBATCH --partition=Odyssey_GPU
#SBATCH --gres=gpu:rtx8000:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --array=0-25
#SBATCH --job-name=l96-da-rlayout-s0i15
#SBATCH --output=sbatch_logs/l96_da_rlayout_s0i15_%A_%a.out
#SBATCH --error=sbatch_logs/l96_da_rlayout_s0i15_%A_%a.err
# S0 ETKF/EnKF rerun of batch/run_l96_da_random_layout.sbatch at inflation 1.5
# (the inflation check found 1.5 best on S0 at n_obs 10 and 100; 2.0 is S1-tuned).
# Same layouts as the inflation-2.0 runs (seeds do not depend on inflation).
#   task 0      arm A : n_obs U{6..50},   k U{4..16}, 200 windows
#   task 1      arm A': n_obs U{10..100}, k U{4..16}, 200 windows
#   tasks 2-25  arm B : n_obs {10,20,30,50,75,100} x k {4,8,12,16}, 20 windows x 3 draws
set -euo pipefail
cd /Odyssey/private/rfablet/Python/4dvarnet-fm-opencode/4dvarnet-fm-da-random-layout
export PATH="/Odyssey/private/rfablet/miniforge3/envs/fdv-monai-proto/bin:$PATH"
export CONDA_DEFAULT_ENV=fdv-monai-proto
export PYTHONUNBUFFERED=1
export MPLCONFIGDIR="/tmp/mplconfig_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
mkdir -p "$MPLCONFIGDIR" sbatch_logs

TASKS=("6 50 4 16 200 1" "10 100 4 16 200 1")
for n in 10 20 30 50 75 100; do
  for k in 4 8 12 16; do
    TASKS+=("$n $n $k $k 20 3")
  done
done

read -r NLO NHI FLO FHI NWIN NDRAW <<< "${TASKS[$SLURM_ARRAY_TASK_ID]}"
echo "===== task ${SLURM_ARRAY_TASK_ID}: S0 inflation 1.5, n_obs ${NLO}-${NHI} k ${FLO}-${FHI} windows ${NWIN} draws ${NDRAW} | $(date) | $(nvidia-smi --query-gpu=name --format=csv,noheader -i 0 | tail -1) ====="
python eval_da_random_layout_l96.py --n-obs-range "$NLO" "$NHI" --fast-range "$FLO" "$FHI" \
  --n-windows "$NWIN" --n-draws "$NDRAW" --inflation 1.5 --da-window-steps 500 \
  --methods ETKF EnKF --cases s0
echo "Done at $(date)"
