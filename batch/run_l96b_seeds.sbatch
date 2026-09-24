#!/bin/bash
#SBATCH --account=odyssey
#SBATCH --partition=Odyssey_GPU
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --array=0-5
#SBATCH --job-name=l96b-seeds
#SBATCH --output=/Odyssey/private/rfablet/Python/4dvarnet-fm-opencode/4dvarnet-fm-bench-default/sbatch_logs/l96b_seeds_%A_%a.out
#SBATCH --error=/Odyssey/private/rfablet/Python/4dvarnet-fm-opencode/4dvarnet-fm-bench-default/sbatch_logs/l96b_seeds_%A_%a.err
# Benchmark-default DirectUNet-M / VanillaCFM-M, seeds 1-3 (task = 3*model + seed-1),
# each scored on the regular grid and on the 30-obs / 8-fast random set.
# Run from the 4dvarnet-fm-bench-default worktree (PR #242 branch).
set -euo pipefail
cd /Odyssey/private/rfablet/Python/4dvarnet-fm-opencode/4dvarnet-fm-bench-default
export PATH="/Odyssey/private/rfablet/miniforge3/envs/fdv-monai-proto/bin:$PATH"
export CONDA_DEFAULT_ENV=fdv-monai-proto
export PYTHONUNBUFFERED=1
export MPLCONFIGDIR="/tmp/mplconfig_${SLURM_JOB_ID}"
mkdir -p "$MPLCONFIGDIR"
i=$SLURM_ARRAY_TASK_ID
EXPS=(L96B_directunet_monaiM L96B_vanillacfm_monaiM)
MEMBERS=(1 30); OUTER=(1 20)
m=$((i / 3)); SEED=$((i % 3 + 1))
EXPERIMENT="${EXPS[$m]}"; SUB="ens${MEMBERS[$m]}_no${OUTER[$m]}"; DIR="experiments/${EXPERIMENT}_seed${SEED}"
echo "===== ${EXPERIMENT} seed ${SEED} | $(date) | $(nvidia-smi --query-gpu=name --format=csv,noheader -i 0 | tail -1) ====="
python train.py --config-name experiment/${EXPERIMENT} hydra.run.dir=. hydra.output_subdir=null training.seed=${SEED}
for TS in regular rand30_fast8; do
  if [ "$TS" = regular ]; then DS=/Odyssey/private/rfablet/Python/4dvarnet-fm-opencode/experiments/l96_datasets_obsj2_int100_nwin200.pt; OUT="${DIR}/${SUB}";
  else DS=experiments/l96_datasets_obsj2_rand30_fast8_nwin200.pt; OUT="experiments/eval_rand30_fast8/$(basename $DIR)/${SUB}"; fi
  mkdir -p "$OUT"
  python eval_neural_l96.py --checkpoint "${DIR}/checkpoints/stage1_best.ckpt" --dataset "$DS" \
    --num-windows 200 --n-members "${MEMBERS[$m]}" --n-outer "${OUTER[$m]}" --batch-size 25 --output "${OUT}/neural_eval.json"
done
echo "Done: ${EXPERIMENT} seed ${SEED} at $(date)"
