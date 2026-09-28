#!/bin/bash
#SBATCH --account=odyssey
#SBATCH --partition=Odyssey_GPU
#SBATCH --gres=gpu:rtx8000:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=03:00:00
#SBATCH --array=0-17
#SBATCH --job-name=l96-etks-val
#SBATCH --output=sbatch_logs/l96_etks_val_%A_%a.out
#SBATCH --error=sbatch_logs/l96_etks_val_%A_%a.err
# ETKS PR 2: inflation tuning of the ETKF and the ETKS (`correct`, full window) on the
# 50-window L96 validation set (scripts/make_l96_validation_sets.py), same lambda for
# both cases per task. Tasks 0-8 regular 30-obs grid, 9-17 random layouts drawn from
# the same windows (n_obs 10-100, k 4-16). The grid was first 1.1-2.0 (tasks 0-4, 9-13);
# every selection landed on an edge, so 1.0/1.05/2.5/3.0 were added. Selection: docs/results/l96_etks_benchmark.md.
set -euo pipefail
cd /Odyssey/private/rfablet/Python/4dvarnet-fm-opencode/4dvarnet-fm-l96-etks
export PATH="/Odyssey/private/rfablet/miniforge3/envs/fdv-monai-proto/bin:$PATH"
export PYTHONUNBUFFERED=1
export MPLCONFIGDIR="/tmp/mplconfig_${SLURM_JOB_ID}"
mkdir -p "$MPLCONFIGDIR" sbatch_logs
VAL=/Odyssey/private/rfablet/Python/4dvarnet-fm-opencode/experiments/l96_valset_regular_w50.pt
[ "$(sha256sum "$(readlink -f "$VAL")" | cut -d" " -f1)" = "efd43ea5247072bccc245c236267d4a2e946899e427fa8b91886cefe7346617e" ] || { echo "validation set changed"; exit 1; }
LAMBDAS=(1.1 1.2 1.3 1.5 2.0 1.0 1.05 2.5 3.0)
NL=${#LAMBDAS[@]}
LAM=${LAMBDAS[$((SLURM_ARRAY_TASK_ID % NL))]}
RANDOMIZE='{"F": {"randomized": true, "noise": 0.2, "biased": true, "bias": 0.1}, "c1": {"randomized": true, "noise": 0.2, "biased": true, "bias": 0.1}, "h": {"randomized": false, "noise": 0.2, "biased": false, "bias": 0.0}, "hx": {"randomized": true, "noise": 0.2, "biased": true, "bias": 0.1}, "eps": {"randomized": true, "noise": 0.2, "biased": true, "bias": 0.1}, "fast_weights": {"randomized": true, "noise": 0.2, "biased": true, "bias": 0.1}}'
if [ "$SLURM_ARRAY_TASK_ID" -lt "$NL" ]; then
  python -u evaluate_all_l96.py --data-cache "$VAL" --da-window-steps 500 --num-test-windows 50 --t-max 3.0 \
    --obs-j 2 --obs-interval 100 --suffix _val --randomize "$RANDOMIZE" \
    --etkf-inflation "$LAM" --skip-weak --skip-strong --skip-enkf --da-fast-weights --etks
else
  python -u eval_da_random_layout_l96.py --data-cache "$VAL" --n-obs-range 10 100 --fast-range 4 16 \
    --n-windows 50 --n-draws 1 --da-window-steps 500 --inflation "$LAM" --methods ETKF ETKS
fi
echo "Done at $(date)"
