#!/bin/bash
#SBATCH --account=odyssey
#SBATCH --partition=Odyssey_GPU
#SBATCH --gres=gpu:rtx8000:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --array=0-26
#SBATCH --job-name=l96-da-rlayout
#SBATCH --output=sbatch_logs/l96_da_rlayout_%A_%a.out
#SBATCH --error=sbatch_logs/l96_da_rlayout_%A_%a.err
# L96 DA baselines under the training random observing system (step 0 always
# observed, ring-interpolated step-0 fast channels), P1 settings + DA fast weights.
#   task 0      arm A: n_obs U{5..50}, k U{4..16}, 200 windows x 1 draw, ETKF/EnKF/Strong-4DVar
#   tasks 1-20  arm B: n_obs x k factorial grid, 20 windows x 3 draws, ETKF/EnKF/Strong-4DVar
#   tasks 21-26 inflation check: n_obs in {5,50}, k U{4..16}, inflation {1.5,2.0,2.5}, ETKF/EnKF
set -euo pipefail
cd /Odyssey/private/rfablet/Python/4dvarnet-fm-opencode/4dvarnet-fm-da-random-layout
export PATH="/Odyssey/private/rfablet/miniforge3/envs/fdv-monai-proto/bin:$PATH"
export CONDA_DEFAULT_ENV=fdv-monai-proto
export PYTHONUNBUFFERED=1
export MPLCONFIGDIR="/tmp/mplconfig_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
mkdir -p "$MPLCONFIGDIR" sbatch_logs

TASKS=("5 50 4 16 200 1 2.0 ETKF EnKF Strong-4DVar")
for n in 5 10 20 30 50; do
  for k in 4 8 12 16; do
    TASKS+=("$n $n $k $k 20 3 2.0 ETKF EnKF Strong-4DVar")
  done
done
for n in 5 50; do
  for inf in 1.5 2.0 2.5; do
    TASKS+=("$n $n 4 16 20 3 $inf ETKF EnKF")
  done
done

read -r NLO NHI FLO FHI NWIN NDRAW INF METHODS <<< "${TASKS[$SLURM_ARRAY_TASK_ID]}"
echo "===== task ${SLURM_ARRAY_TASK_ID}: n_obs ${NLO}-${NHI} k ${FLO}-${FHI} windows ${NWIN} draws ${NDRAW} inflation ${INF} [${METHODS}] | $(date) | $(nvidia-smi --query-gpu=name --format=csv,noheader -i 0 | tail -1) ====="
python eval_da_random_layout_l96.py --n-obs-range "$NLO" "$NHI" --fast-range "$FLO" "$FHI" \
  --n-windows "$NWIN" --n-draws "$NDRAW" --inflation "$INF" --methods $METHODS
echo "Done at $(date)"
