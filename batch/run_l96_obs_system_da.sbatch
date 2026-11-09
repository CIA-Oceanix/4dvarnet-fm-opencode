#!/bin/bash
#SBATCH --account=mee
#SBATCH --partition=Mee_Global_CPU
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=48:00:00
#SBATCH --array=0-15%8
#SBATCH --job-name=l96-obs-da
#SBATCH --output=sbatch_logs/l96_obs_da_%A_%a.out
#SBATCH --error=sbatch_logs/l96_obs_da_%A_%a.err
# P1 observing-system figure (fig_obs_system): ETKF, EnKF, ETKS and Strong-4DVar on its 16 cells
# (n_obs at k=16, k at n_obs=30, R at n_obs=30/k=16; 20 windows x 3 draws), benchmark inflations
# (ETKF 1.15/2.5, EnKF 1.2/3.0, ETKS = ETKF's). Same layout seeds as the earlier DA cells, so the
# learned schemes' cell test sets (experiments/l96_testsets_{factorial,ood}) hold the same obs.
# Trajectories are kept (experiments/l96_baselines_trajectories_*rlayout*.npz) for the EV of the figure.
# CPU: no GPU needed for DA. Submit from this worktree's root.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
export PATH="/Odyssey/private/rfablet/miniforge3/envs/${CONDA_ENV:-fdv-monai-proto}/bin:$PATH"
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-16}
NODE_TMP="/tmp/${USER}_slurm_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID:-0}"; mkdir -p "$NODE_TMP"
export TMPDIR="$NODE_TMP" MPLCONFIGDIR="$NODE_TMP/mplconfig"; trap 'rm -rf "$NODE_TMP"' EXIT
mkdir -p "$MPLCONFIGDIR" sbatch_logs
CACHE=/Odyssey/private/rfablet/Python/4dvarnet-fm-opencode/experiments/l96_datasets_obsj2_int100_nwin200.pt
CELLS=()
for n in 6 10 20 30 50 75 100 300 1000; do CELLS+=("$n 16 0.5"); done
for k in 0 2 4 8 12; do CELLS+=("30 $k 0.5"); done
CELLS+=("30 16 0.25" "30 16 1.0")
read -r N K R <<< "${CELLS[$SLURM_ARRAY_TASK_ID]}"
echo "task $SLURM_ARRAY_TASK_ID: n $N k $K R $R | commit $(git rev-parse --short HEAD) | $(hostname) $(date +%T)"
python -u eval_da_random_layout_l96.py --n-obs-range "$N" "$N" --fast-range "$K" "$K" --r-var "$R" \
  --n-windows 20 --n-draws 3 --da-window-steps 500 --methods ETKF EnKF ETKS Strong-4DVar \
  --device cpu --data-cache "$CACHE"
echo "Done at $(date)"
