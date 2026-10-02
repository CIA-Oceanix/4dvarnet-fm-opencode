#!/bin/bash
#SBATCH --account=odyssey
#SBATCH --partition=Odyssey_GPU
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --array=0-3%2
#SBATCH --job-name=l96-tau0-1200
#SBATCH --output=sbatch_logs/l96_tau0_1200_%A_%a.out
#SBATCH --error=sbatch_logs/l96_tau0_1200_%A_%a.err
# tau=0 mean components (eval_mean_component.py, 30 draws) of the benchmark-default flows
# (VanillaCFM-M / PredictStateCFM-M, 1200 epochs, seeds 1-3) for the P1 report's
# benchmark-framework section 2: {regular, canonical random} test set x {S0, S1}.
# Submit from the repository root.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
export PATH="/Odyssey/private/rfablet/miniforge3/envs/fdv-monai-proto/bin:$PATH"
export PYTHONUNBUFFERED=1
NODE_TMP="/SCRATCH/$USER/slurm_tmp/${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID:-0}"; mkdir -p "$NODE_TMP" 2>/dev/null || { NODE_TMP="$SLURM_SUBMIT_DIR/sbatch_logs/tmp_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID:-0}"; mkdir -p "$NODE_TMP"; }
export TMPDIR="$NODE_TMP" MPLCONFIGDIR="$NODE_TMP/mplconfig"; trap 'rm -rf "$NODE_TMP"' EXIT
SHARED=/Odyssey/private/rfablet/Python/4dvarnet-fm-opencode/experiments
OUT=experiments/p1_tau0_1200
mkdir -p "$MPLCONFIGDIR" sbatch_logs "$OUT"
SETS=(regular regular rlayout rlayout); CASES=(s0 s1 s0 s1)
SET=${SETS[$SLURM_ARRAY_TASK_ID]}; CASE=${CASES[$SLURM_ARRAY_TASK_ID]}
if [ "$SET" = regular ]; then DS=$SHARED/l96_datasets_obsj2_int100_nwin200.pt; else DS=$SHARED/l96_testset_rlayout_n10-100_k4-16_w200_d1.pt; fi
RUNS=()
for fam in vanillacfm:VanillaCFM predictstatecfm:PredictStateCFM; do
  for s in 1 2 3; do RUNS+=(--run "${fam#*:} M seed$s" "$SHARED/l96/L96B_${fam%%:*}_monaiM_ep1200_seed$s/checkpoints/stage1_best.ckpt"); done
done
echo "task $SLURM_ARRAY_TASK_ID: $SET $CASE | commit $(git rev-parse --short HEAD) | $(hostname) | $(date)"
python -u reports/l96/eval_mean_component.py --dataset "$DS" --case "$CASE" --m-draws 30 "${RUNS[@]}" \
  --output "$OUT/tau0_${SET}_${CASE}.md" --json-output "$OUT/tau0_${SET}_${CASE}.json"
echo "Done at $(date)"
