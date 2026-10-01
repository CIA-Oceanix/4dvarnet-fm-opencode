#!/bin/bash
#SBATCH --account=odyssey
#SBATCH --partition=Odyssey_GPU
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=08:00:00
#SBATCH --array=0-1
#SBATCH --exclude=sl-mee-br-202,sl-mee-br-203,sl-mee-br-204,sl-mee-br-205,sl-mee-br-208,sl-mee-br-209,sl-mee-br-210,sl-mee-br-211,sl-mee-br-212,sl-mee-br-213,sl-mee-br-214,sl-mee-br-215,sl-mee-br-216
#SBATCH --job-name=l96-fmscore-da
#SBATCH --output=sbatch_logs/l96_fmscore_da_%A_%a.out
#SBATCH --error=sbatch_logs/l96_fmscore_da_%A_%a.err
# FM-operator score of the benchmark ETKF / ETKS (lambda S0 1.15 / S1 2.5, ETKS
# `correct`, full window) with their ensembles: the same commands as
# run_l96_etks_test.sbatch tasks 2-3 plus --save-members; the member dumps stay on
# node-local storage and are scored in this job (reports/l96/probe_fm_score_da.py).
#   task 0: regular 30-obs grid; task 1: canonical random layouts (n_obs 10-100, k 4-16).
# Submit from this worktree's root (its experiments/ receives the trajectory files,
# so the main checkout's benchmark files are never touched).
set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
export PATH="/Odyssey/private/rfablet/miniforge3/envs/fdv-monai-proto/bin:$PATH"
export PYTHONUNBUFFERED=1
NODE_TMP="/SCRATCH/$USER/slurm_tmp/${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID:-0}"; mkdir -p "$NODE_TMP" 2>/dev/null || { NODE_TMP="$SLURM_SUBMIT_DIR/sbatch_logs/tmp_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID:-0}"; mkdir -p "$NODE_TMP"; }
export TMPDIR="$NODE_TMP" MPLCONFIGDIR="$NODE_TMP/mplconfig"; trap 'rm -rf "$NODE_TMP"' EXIT
mkdir -p "$MPLCONFIGDIR" sbatch_logs
export PYTHONPATH=$PWD FDV_MEMBERS_TMP="$NODE_TMP/members"
CACHE=/Odyssey/private/rfablet/Python/4dvarnet-fm-opencode/experiments/l96_datasets_obsj2_int100_nwin200.pt
OUT=/Odyssey/private/rfablet/Python/4dvarnet-fm-opencode/experiments/l96_fm_score_current_v2
[ "$(sha256sum "$(readlink -f "$CACHE")" | cut -d" " -f1)" = "8eba1f87c73beb5449a66f17fd66c9a53b7900e69003eecd51fc991c3a823aae" ] || { echo "test cache is not the P1 cache"; exit 1; }
RANDOMIZE='{"F": {"randomized": true, "noise": 0.2, "biased": true, "bias": 0.1}, "c1": {"randomized": true, "noise": 0.2, "biased": true, "bias": 0.1}, "h": {"randomized": false, "noise": 0.2, "biased": false, "bias": 0.0}, "hx": {"randomized": true, "noise": 0.2, "biased": true, "bias": 0.1}, "eps": {"randomized": true, "noise": 0.2, "biased": true, "bias": 0.1}, "fast_weights": {"randomized": true, "noise": 0.2, "biased": true, "bias": 0.1}}'
INF="s0=1.15,s1=2.5"
echo "task $SLURM_ARRAY_TASK_ID | commit $(git rev-parse --short HEAD) | $(hostname) | $(date +%T)"
if [ "$SLURM_ARRAY_TASK_ID" = 0 ]; then
  python -u evaluate_all_l96.py --data-cache "$CACHE" --da-window-steps 500 --num-test-windows 200 --t-max 3.0 \
    --obs-j 2 --obs-interval 100 --suffix _s0c_test --randomize "$RANDOMIZE" \
    --etkf-inflation "$INF" --skip-weak --skip-strong --skip-enkf --da-fast-weights \
    --etks --etks-inflation "$INF" --save-members
  LAY=reg
else
  python -u eval_da_random_layout_l96.py --data-cache "$CACHE" --n-obs-range 10 100 --fast-range 4 16 \
    --n-windows 200 --n-draws 1 --da-window-steps 500 --inflation "$INF" \
    --methods ETKF ETKS --etks-inflation "$INF" --save-members
  LAY=rand
fi
du -sh "$FDV_MEMBERS_TMP"
python -u reports/l96/probe_fm_score_da.py "$FDV_MEMBERS_TMP" "$OUT" "da_${LAY}_"
echo "Done at $(date +%T)"
