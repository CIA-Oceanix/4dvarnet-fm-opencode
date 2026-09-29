#!/bin/bash
#SBATCH --account=odyssey
#SBATCH --partition=Odyssey_GPU
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --array=0-11%4
#SBATCH --job-name=l96b-fdv-seeds
#SBATCH --output=sbatch_logs/l96b_fdv_seeds_%A_%a.out
#SBATCH --error=sbatch_logs/l96b_fdv_seeds_%A_%a.err
# docs/plans/analysis/l96_p2_fdv_benchmark.md Phase 1: FDV1 / FDV2 subgrad+state,
# M and S tiers, seeds 1-3, benchmark default (1200 epochs), then the deterministic
# eval on both benchmark test sets (n_outer = the solver's own N_outer).
set -euo pipefail
cd /Odyssey/private/rfablet/Python/4dvarnet-fm-opencode/4dvarnet-fm-p2-fdv
export PATH="/Odyssey/private/rfablet/miniforge3/envs/fdv-monai-proto/bin:$PATH"
export CONDA_DEFAULT_ENV=fdv-monai-proto
export PYTHONUNBUFFERED=1
NODE_TMP="/SCRATCH/$USER/slurm_tmp/${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID:-0}"; mkdir -p "$NODE_TMP" 2>/dev/null || { NODE_TMP="/tmp/${USER}_slurm_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID:-0}"; mkdir -p "$NODE_TMP"; }
export TMPDIR="$NODE_TMP" MPLCONFIGDIR="$NODE_TMP/mplconfig"; trap 'rm -rf "$NODE_TMP"' EXIT
mkdir -p "$MPLCONFIGDIR" sbatch_logs
R=/Odyssey/private/rfablet/Python/4dvarnet-fm-opencode
CFGS=(L96B_fdv1_monaiM L96B_fdv1_monaiS L96B_fdvsubgrad_monaiM L96B_fdvsubgrad_monaiS)
i=$SLURM_ARRAY_TASK_ID; C=${CFGS[$((i / 3))]}; SEED=$((i % 3 + 1))
NAME=${C}_seed${SEED}; SUB=ens1_no10
[ -f experiments/$NAME/checkpoints/stage1_best.ckpt ] && [ -f experiments/$NAME/results.json ] || \
  python train.py --config-name experiment/$C hydra.run.dir=. hydra.output_subdir=null training.seed=${SEED} \
    ++data.norm_stats_path=$R/experiments/l96_norm_stats_obsj2.pt \
    ++data.train_cache=$R/experiments/l96_trainval_cache/l96b_trainval_n1000_v100.pt
for TS in regular canonical; do
  if [ "$TS" = regular ]; then DS=$R/experiments/l96_datasets_obsj2_int100_nwin200.pt; OUT="experiments/${NAME}/${SUB}";
  else DS=$R/experiments/l96_testset_rlayout_n10-100_k4-16_w200_d1.pt; OUT="experiments/eval_rlayout_n10-100_k4-16_w200/${NAME}/${SUB}"; fi
  [ -f "$OUT/neural_eval.json" ] && continue; mkdir -p "$OUT"
  python eval_neural_l96.py --checkpoint "experiments/${NAME}/checkpoints/stage1_best.ckpt" --dataset "$DS" \
    --num-windows 200 --n-members 1 --batch-size 25 --members-file scores --output "$OUT/neural_eval.json"
done
echo "Done: ${NAME} at $(date)"
