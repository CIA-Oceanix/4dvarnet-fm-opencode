#!/bin/bash
#SBATCH --account=odyssey
#SBATCH --partition=Odyssey_GPU
#SBATCH --gres=gpu:rtx8000:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=10:00:00
#SBATCH --array=0-2
#SBATCH --job-name=l96-da-marginal
#SBATCH --output=sbatch_logs/l96_da_marginal_%A_%a.out
#SBATCH --error=sbatch_logs/l96_da_marginal_%A_%a.err
# Re-run of the marginal-value-of-observations DA experiment (15 vs 30 regular
# obs per window) behind the P1 paper's "6.2x / 1.9x" headline, with the DA
# forward model given each window's fast_weights (the 2026-08-19 run summed
# the fast variables unweighted, so its S0 was not perfect-model).
#   task 0: benchmark data (P1 test trajectories) at 15 obs, per-case
#           inflation S0 1.5 / S1 2.0 (the 30-obs corrected runs exist)
#   task 1/2: the original run's data configuration (legacy caches: h
#           randomized, fixed fast_weights) at 30 / 15 obs, inflation 2.0 as
#           originally -- only the fast_weights fix changes
# Each task refuses to run unless its cache hashes to the expected file.
set -euo pipefail
cd /Odyssey/private/rfablet/Python/4dvarnet-fm-opencode/4dvarnet-fm-sda-nanfix
export PATH="/Odyssey/private/rfablet/miniforge3/envs/fdv-monai-proto/bin:$PATH"
export PYTHONUNBUFFERED=1
export MPLCONFIGDIR="/tmp/mplconfig_${SLURM_JOB_ID}"
mkdir -p "$MPLCONFIGDIR" sbatch_logs
S0C='{"F": {"randomized": true, "noise": 0.2, "biased": true, "bias": 0.1}, "c1": {"randomized": true, "noise": 0.2, "biased": true, "bias": 0.1}, "h": {"randomized": false, "noise": 0.2, "biased": false, "bias": 0.0}, "hx": {"randomized": true, "noise": 0.2, "biased": true, "bias": 0.1}, "eps": {"randomized": true, "noise": 0.2, "biased": true, "bias": 0.1}, "fast_weights": {"randomized": true, "noise": 0.2, "biased": true, "bias": 0.1}}'
guard() { [ "$(sha256sum "$(readlink -f "experiments/$1")" | cut -d" " -f1)" = "$2" ] || { echo "cache $1 is not the expected file"; exit 1; }; }
COMMON=(--da-window-steps 500 --num-test-windows 200 --t-max 3.0 --obs-j 2 --skip-weak --da-fast-weights)
case $SLURM_ARRAY_TASK_ID in
  0) guard l96_datasets_obsj2_int200_nwin200.pt 1e276fdb1d516f128a77ae7afb78087eea6abe9ca14b143b33fce6b898774061
     python -u evaluate_all_l96.py "${COMMON[@]}" --obs-interval 200 --suffix _s0c --randomize "$S0C" ;;
  1) guard l96_datasets_obsj2_int100_nwin200_legacy_int100.pt 94855a1af6293f2a6a746186092ca98c81e96ccaf8d0ef9b941548dff6a8d63b
     python -u evaluate_all_l96.py "${COMMON[@]}" --obs-interval 100 --data-cache-tag _legacy_int100 \
       --suffix _legacy --enkf-inflation 2.0 --etkf-inflation 2.0 ;;
  2) guard l96_datasets_obsj2_int200_nwin200_legacy_int200.pt acf54af19e8107806b8ae59624d8242e56915d9b49b5423b8b7f674e0f9cf45d
     python -u evaluate_all_l96.py "${COMMON[@]}" --obs-interval 200 --data-cache-tag _legacy_int200 \
       --suffix _legacy --enkf-inflation 2.0 --etkf-inflation 2.0 ;;
esac
echo "Done at $(date)"
