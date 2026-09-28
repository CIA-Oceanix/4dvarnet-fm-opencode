#!/bin/bash
#SBATCH --account=odyssey
#SBATCH --partition=Odyssey_GPU
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --array=0-19%8
#SBATCH --job-name=l96-etkf-infl
#SBATCH --output=sbatch_logs/l96_etkf_infl_%A_%a.out
#SBATCH --error=sbatch_logs/l96_etkf_infl_%A_%a.err
# ETKF inflation retuning after #291 on the L96 validation windows (50 per case,
# disjoint from train/val/test): {regular grid, canonical random layout} x
# lambda in {1.0 ... 3.0}, S0 and S1 per task. One JSON per task in
# experiments/l96_etkf_inflation_sweep/. Submit from this worktree's root.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
export PATH="/Odyssey/private/rfablet/miniforge3/envs/fdv-monai-proto/bin:$PATH"
export PYTHONUNBUFFERED=1
export MPLCONFIGDIR="/tmp/mplconfig_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
mkdir -p "$MPLCONFIGDIR" sbatch_logs experiments/l96_etkf_inflation_sweep
# CELLS overrides the default grid, e.g. CELLS="regular:3.5 rlayout:1.05" (set --array to match).
if [ -n "${CELLS:-}" ]; then
  read -r -a TASKS <<< "$CELLS"
else
  TASKS=()
  for V in regular rlayout; do for L in 1.0 1.1 1.2 1.3 1.4 1.5 1.7 2.0 2.5 3.0; do TASKS+=("$V:$L"); done; done
fi
i=$SLURM_ARRAY_TASK_ID
V=${TASKS[$i]%%:*}
L=${TASKS[$i]##*:}
echo "task $i: valset=$V lambda=$L commit=$(git rev-parse --short HEAD) $(hostname) $(date +%T)"
python -u scripts/sweep_l96_etkf_inflation.py --valset "$V" --inflation "$L" \
  --out "experiments/l96_etkf_inflation_sweep/etkf_${V}_lam${L}.json"
echo "done $V $L $(date +%T)"
