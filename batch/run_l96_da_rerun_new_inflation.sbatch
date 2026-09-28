#!/bin/bash
#SBATCH --account=odyssey
#SBATCH --partition=Odyssey_GPU
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=08:00:00
#SBATCH --array=0-33%10
#SBATCH --job-name=l96-da-rerun
#SBATCH --output=sbatch_logs/l96_da_rerun_%A_%a.out
#SBATCH --error=sbatch_logs/l96_da_rerun_%A_%a.err
# Every ETKF/EnKF row the L96 benchmark reports read, rerun at the per-method
# defaults of #295 (ETKF 1.15 / 2.5, EnKF 1.2 / 3.0) with the fixed ETKF (#291).
# Commands are those of the original runs with the inflation flags dropped;
# Strong-4DVar (no inflation) is not rerun.
#   task 0      regular 30-obs test set (batch/run_l96_da_crps.sbatch task 0)
#   task 1      regular 15-obs test set (batch/run_l96_da_marginal_value.sbatch task 0)
#   task 2      canonical random layout, 200 windows (batch/run_l96_da_crps.sbatch task 1)
#   tasks 3-26  factorial n_obs {10,20,30,50,75,100} x k {4,8,12,16}, 20 windows x 3 draws
#               (batch/run_l96_da_random_layout.sbatch arm B)
#   tasks 27-33 out-of-range probes (batch/run_l96_da_ood_probes.sbatch)
# Submit from this worktree's root.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
export PATH="/Odyssey/private/rfablet/miniforge3/envs/fdv-monai-proto/bin:$PATH"
export PYTHONUNBUFFERED=1
export MPLCONFIGDIR="/tmp/mplconfig_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
mkdir -p "$MPLCONFIGDIR" sbatch_logs
guard() { [ "$(sha256sum "$(readlink -f "experiments/$1")" | cut -d" " -f1)" = "$2" ] || { echo "cache $1 is not the expected file"; exit 1; }; }
guard l96_datasets_obsj2_int100_nwin200.pt 8eba1f87c73beb5449a66f17fd66c9a53b7900e69003eecd51fc991c3a823aae
RAND=(--da-window-steps 500 --methods ETKF EnKF)
TASKS=("regular30" "regular15" "rlayout 10 100 4 16 200 1 0.5")
for n in 10 20 30 50 75 100; do for k in 4 8 12 16; do TASKS+=("rlayout $n $n $k $k 20 3 0.5"); done; done
for c in "6 6 16 16" "30 30 0 0" "30 30 2 2" "300 300 16 16" "1000 1000 16 16"; do TASKS+=("rlayout_probe $c 20 3 0.5"); done
TASKS+=("rlayout_probe 30 30 16 16 20 3 0.25" "rlayout_probe 30 30 16 16 20 3 1.0")
read -r KIND NLO NHI FLO FHI NWIN NDRAW R _REST <<< "${TASKS[$SLURM_ARRAY_TASK_ID]} _ _ _ _ _ _ _"
echo "task $SLURM_ARRAY_TASK_ID: $KIND $NLO $NHI $FLO $FHI $NWIN $NDRAW $R | commit $(git rev-parse --short HEAD) | $(hostname) $(date +%T)"
S0C='{"F": {"randomized": true, "noise": 0.2, "biased": true, "bias": 0.1}, "c1": {"randomized": true, "noise": 0.2, "biased": true, "bias": 0.1}, "h": {"randomized": false, "noise": 0.2, "biased": false, "bias": 0.0}, "hx": {"randomized": true, "noise": 0.2, "biased": true, "bias": 0.1}, "eps": {"randomized": true, "noise": 0.2, "biased": true, "bias": 0.1}, "fast_weights": {"randomized": true, "noise": 0.2, "biased": true, "bias": 0.1}}'
case $KIND in
  regular30)
    python -u evaluate_all_l96.py --da-window-steps 500 --num-test-windows 200 --t-max 3.0 \
      --obs-j 2 --obs-interval 100 --suffix _s0c_crps --randomize "$S0C" --skip-weak --skip-strong --da-fast-weights ;;
  regular15)
    guard l96_datasets_obsj2_int200_nwin200.pt 1e276fdb1d516f128a77ae7afb78087eea6abe9ca14b143b33fce6b898774061
    python -u evaluate_all_l96.py --da-window-steps 500 --num-test-windows 200 --t-max 3.0 \
      --obs-j 2 --obs-interval 200 --suffix _s0c --randomize "$S0C" --skip-weak --skip-strong --da-fast-weights ;;
  rlayout)
    python -u eval_da_random_layout_l96.py --n-obs-range "$NLO" "$NHI" --fast-range "$FLO" "$FHI" \
      --n-windows "$NWIN" --n-draws "$NDRAW" "${RAND[@]}" --data-cache experiments/l96_datasets_obsj2_int100_nwin200.pt ;;
  rlayout_probe)
    python -u eval_da_random_layout_l96.py --n-obs-range "$NLO" "$NHI" --fast-range "$FLO" "$FHI" --r-var "$R" \
      --n-windows "$NWIN" --n-draws "$NDRAW" "${RAND[@]}" --data-cache experiments/l96_datasets_obsj2_int100_nwin200.pt ;;
esac
echo "Done at $(date)"
