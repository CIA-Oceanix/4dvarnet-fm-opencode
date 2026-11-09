#!/bin/bash
#SBATCH --account=odyssey
#SBATCH --partition=Odyssey_GPU
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=06:00:00
#SBATCH --array=0-20%5
#SBATCH --job-name=l96-s1full-test
#SBATCH --output=sbatch_logs/l96_s1full_test_%A_%a.out
#SBATCH --error=sbatch_logs/l96_s1full_test_%A_%a.err
# L96 S1-full benchmark rows: the S1 DA model keeps the full 40-variable model (--s1-j 4), with
# the biased parameters and corrupted forcing of S1, on the 200 regular P1 test windows (hash-guarded).
# Hyper-parameters selected on the validation sweep of run_l96_s1_full_model_val.sbatch (RMSE-optimal,
# regular validation windows): ETKF 1.3 (ETKS at the ETKF value), EnKF 1.5, Weak-4DVar q 0.03.
#   task 0:     ETKF, EnKF, ETKS and Strong-4DVar (trajectories, ensemble variance and CRPS kept)
#   tasks 1-20: Weak-4DVar (L-BFGS 40 it), 10 windows each, chunks kept in experiments/weak4dvar_s1full/
#               (assemble with scripts/assemble_weak4dvar_bench.py --cases s1=0.03 --name ...)
# Submit from the repository root.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
export PATH="/Odyssey/private/rfablet/miniforge3/envs/fdv-monai-proto/bin:$PATH"
export PYTHONUNBUFFERED=1
NODE_TMP="/SCRATCH/$USER/slurm_tmp/${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID:-0}"; mkdir -p "$NODE_TMP" 2>/dev/null || { NODE_TMP="/tmp/${USER}_slurm_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID:-0}"; mkdir -p "$NODE_TMP"; }
export TMPDIR="$NODE_TMP" MPLCONFIGDIR="$NODE_TMP/mplconfig"; trap 'rm -rf "$NODE_TMP"' EXIT
mkdir -p "$MPLCONFIGDIR" sbatch_logs
CACHE=/Odyssey/private/rfablet/Python/4dvarnet-fm-opencode/experiments/l96_datasets_obsj2_int100_nwin200.pt
[ "$(sha256sum "$(readlink -f "$CACHE")" | cut -d" " -f1)" = "8eba1f87c73beb5449a66f17fd66c9a53b7900e69003eecd51fc991c3a823aae" ] || { echo "test cache is not the P1 cache"; exit 1; }
RANDOMIZE='{"F": {"randomized": true, "noise": 0.2, "biased": true, "bias": 0.1}, "c1": {"randomized": true, "noise": 0.2, "biased": true, "bias": 0.1}, "h": {"randomized": false, "noise": 0.2, "biased": false, "bias": 0.0}, "hx": {"randomized": true, "noise": 0.2, "biased": true, "bias": 0.1}, "eps": {"randomized": true, "noise": 0.2, "biased": true, "bias": 0.1}, "fast_weights": {"randomized": true, "noise": 0.2, "biased": true, "bias": 0.1}}'
T=$SLURM_ARRAY_TASK_ID
echo "task $T | commit $(git rev-parse --short HEAD) | $(hostname) | $(date)"
if (( T == 0 )); then
  python -u evaluate_all_l96.py --data-cache "$CACHE" --cases s1 --s1-j 4 --da-window-steps 500 --num-test-windows 200 \
    --t-max 3.0 --obs-j 2 --obs-interval 100 --suffix _s1full_test --randomize "$RANDOMIZE" --da-fast-weights --skip-weak \
    --etkf-inflation "s0=1.15,s1=1.3" --enkf-inflation "s0=1.2,s1=1.5" --etks --etks-inflation "s0=1.15,s1=1.3"
else
  OUT=experiments/weak4dvar_s1full
  mkdir -p "$OUT"
  W0=$(( (T - 1) * 10 ))
  export W_START=$W0 CASES=s1 S1_J=4 KEEP_DIR=$OUT
  python -u scripts/weak4dvar_sweep_l96.py test_regular 10 weak 0.03 "$OUT/s1_q0.03_it40_w$(printf %03d $W0).json" 40
fi
echo "Done at $(date)"
