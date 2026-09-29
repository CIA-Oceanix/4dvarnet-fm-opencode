#!/bin/bash
#SBATCH --account=odyssey
#SBATCH --partition=Odyssey_GPU
#SBATCH --gres=gpu:rtx8000:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --array=0-3
#SBATCH --job-name=l96-etks-test
#SBATCH --output=sbatch_logs/l96_etks_test_%A_%a.out
#SBATCH --error=sbatch_logs/l96_etks_test_%A_%a.err
# ETKS PR 2 benchmark rows on the P1 test windows (hash-guarded), with the ETKF rerun
# alongside for paired comparisons. ETKS = `correct`, full window.
#   task 0: regular 30-obs grid, ETKS at the ETKF's per-case inflation (S0 1.5 / S1 2.0)
#   task 1: random layouts (n_obs 10-100, k 4-16), same inflation
#   task 2: regular grid, ETKF and ETKS at the validation-selected inflations
#   task 3: random layouts, same
# Tasks 2-3 need ETKF_INF and ETKS_INF (per case, e.g. "s0=1.3,s1=1.5") from
# evaluation/l96_etks_select_inflation.py; export them in the submitting shell and use
# --array=2-3 --export=ALL (--export=VAR=a,b would split the value at the comma)
set -euo pipefail
cd /Odyssey/private/rfablet/Python/4dvarnet-fm-opencode/4dvarnet-fm-l96-etks
export PATH="/Odyssey/private/rfablet/miniforge3/envs/fdv-monai-proto/bin:$PATH"
export PYTHONUNBUFFERED=1
NODE_TMP="/SCRATCH/$USER/slurm_tmp/${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID:-0}"; mkdir -p "$NODE_TMP" 2>/dev/null || { NODE_TMP="/tmp/${USER}_slurm_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID:-0}"; mkdir -p "$NODE_TMP"; }
export TMPDIR="$NODE_TMP" MPLCONFIGDIR="$NODE_TMP/mplconfig"; trap 'rm -rf "$NODE_TMP"' EXIT
mkdir -p "$MPLCONFIGDIR" sbatch_logs
CACHE=/Odyssey/private/rfablet/Python/4dvarnet-fm-opencode/experiments/l96_datasets_obsj2_int100_nwin200.pt
[ "$(sha256sum "$(readlink -f "$CACHE")" | cut -d" " -f1)" = "8eba1f87c73beb5449a66f17fd66c9a53b7900e69003eecd51fc991c3a823aae" ] || { echo "test cache is not the P1 cache"; exit 1; }
RANDOMIZE='{"F": {"randomized": true, "noise": 0.2, "biased": true, "bias": 0.1}, "c1": {"randomized": true, "noise": 0.2, "biased": true, "bias": 0.1}, "h": {"randomized": false, "noise": 0.2, "biased": false, "bias": 0.0}, "hx": {"randomized": true, "noise": 0.2, "biased": true, "bias": 0.1}, "eps": {"randomized": true, "noise": 0.2, "biased": true, "bias": 0.1}, "fast_weights": {"randomized": true, "noise": 0.2, "biased": true, "bias": 0.1}}'
case "$SLURM_ARRAY_TASK_ID" in
  0|1) ETKF_INF="s0=1.5,s1=2.0"; ETKS_INF="s0=1.5,s1=2.0" ;;
  *) : "${ETKF_INF:?set ETKF_INF}" "${ETKS_INF:?set ETKS_INF}" ;;
esac
if [ $((SLURM_ARRAY_TASK_ID % 2)) = 0 ]; then
  python -u evaluate_all_l96.py --data-cache "$CACHE" --da-window-steps 500 --num-test-windows 200 --t-max 3.0 \
    --obs-j 2 --obs-interval 100 --suffix _s0c_test --randomize "$RANDOMIZE" \
    --etkf-inflation "$ETKF_INF" --skip-weak --skip-strong --skip-enkf --da-fast-weights \
    --etks --etks-inflation "$ETKS_INF"
else
  python -u eval_da_random_layout_l96.py --data-cache "$CACHE" --n-obs-range 10 100 --fast-range 4 16 \
    --n-windows 200 --n-draws 1 --da-window-steps 500 --inflation "$ETKF_INF" \
    --methods ETKF ETKS --etks-inflation "$ETKS_INF"
fi
echo "Done at $(date)"
