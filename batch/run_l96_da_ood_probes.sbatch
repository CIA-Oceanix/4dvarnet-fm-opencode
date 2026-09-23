#!/bin/bash
#SBATCH --account=odyssey
#SBATCH --partition=Odyssey_GPU
#SBATCH --gres=gpu:rtx8000:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --array=0-6%3
#SBATCH --job-name=l96-da-ood
#SBATCH --output=sbatch_logs/l96_da_ood_%A_%a.out
#SBATCH --error=sbatch_logs/l96_da_ood_%A_%a.err
# DA baselines on observing systems outside the learned models' training
# range (n_obs 10-300, k 4-16, R 0.5): sparse, slow-only / near slow-only,
# dense at and beyond the edge, and obs-noise shift (DA told the true R).
# 20 windows x 3 draws, benchmark inflation; layouts saved so the learned
# schemes are scored on exactly these obs.
set -euo pipefail
cd /Odyssey/private/rfablet/Python/4dvarnet-fm-opencode/4dvarnet-fm-sda-nanfix
export PATH="/Odyssey/private/rfablet/miniforge3/envs/fdv-monai-proto/bin:$PATH"
export PYTHONUNBUFFERED=1
export MPLCONFIGDIR="/tmp/mplconfig_${SLURM_JOB_ID}"
mkdir -p "$MPLCONFIGDIR" sbatch_logs
CELLS=("6 6 16 16 0.5" "30 30 0 0 0.5" "30 30 2 2 0.5" "300 300 16 16 0.5" "1000 1000 16 16 0.5" "30 30 16 16 0.25" "30 30 16 16 1.0")
read -r NLO NHI FLO FHI R <<< "${CELLS[$SLURM_ARRAY_TASK_ID]}"
echo "===== n ${NLO}-${NHI} k ${FLO}-${FHI} R ${R} | $(date) ====="
python eval_da_random_layout_l96.py --n-obs-range "$NLO" "$NHI" --fast-range "$FLO" "$FHI" --r-var "$R" \
  --n-windows 20 --n-draws 3 --da-window-steps 500 --methods ETKF EnKF Strong-4DVar \
  --data-cache experiments/l96_datasets_obsj2_int100_nwin200.pt
echo "Done at $(date)"
