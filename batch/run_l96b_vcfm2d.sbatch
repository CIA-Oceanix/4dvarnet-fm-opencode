#!/bin/bash
#SBATCH --account=odyssey
#SBATCH --partition=Odyssey_GPU
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --job-name=l96b-vcfm2d
#SBATCH --output=sbatch_logs/l96b_vcfm2d_%j.out
#SBATCH --error=sbatch_logs/l96b_vcfm2d_%j.err
# VanillaCFM with the 2D (time x sector) circular backbone, 1000 windows x 1200 epochs,
# then the same flow eval as L96B_vanillacfm_monaiM_ep1200 (batch/run_l96b_flows_ep1200.sbatch:
# ens30 x 20 steps, step power 0.5, scores only) on both test sets. Runs from the submitting
# worktree; resumes from checkpoints/stage1_last.ckpt if resubmitted. SEED (default 1).
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
export PATH="/Odyssey/private/rfablet/miniforge3/envs/${CONDA_ENV:-fdv-monai-proto}/bin:$PATH"
export PYTHONUNBUFFERED=1
NODE_TMP="/SCRATCH/$USER/slurm_tmp/${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID:-0}"; mkdir -p "$NODE_TMP" 2>/dev/null || { NODE_TMP="/tmp/${USER}_slurm_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID:-0}"; mkdir -p "$NODE_TMP"; }
export TMPDIR="$NODE_TMP" MPLCONFIGDIR="$NODE_TMP/mplconfig"; trap 'rm -rf "$NODE_TMP"' EXIT
mkdir -p "$MPLCONFIGDIR" sbatch_logs
R=/Odyssey/private/rfablet/Python/4dvarnet-fm-opencode
CONFIG=${CONFIG:-L96B_vanillacfm_monai2dM_ep1200}; SEED=${SEED:-1}; NAME=${CONFIG}_seed${SEED}
[ -f experiments/${NAME}/results.json ] || \
  python train.py --config-name experiment/$CONFIG hydra.run.dir=. hydra.output_subdir=null training.seed=$SEED \
    ++data.train_cache=$R/experiments/l96_trainval_cache/l96b_trainval_n1000_v100.pt \
    ++data.norm_stats_path=$R/experiments/l96_norm_stats_obsj2.pt
for TS in regular canonical; do
  if [ "$TS" = regular ]; then DS=$R/experiments/l96_datasets_obsj2_int100_nwin200.pt; OUT="experiments/${NAME}/ens30_no20";
  else DS=$R/experiments/l96_testset_rlayout_n10-100_k4-16_w200_d1.pt; OUT="experiments/eval_rlayout_n10-100_k4-16_w200/${NAME}/ens30_no20"; fi
  [ -f "$OUT/neural_eval.json" ] && continue; mkdir -p "$OUT"
  python eval_neural_l96.py --step-power 0.5 --checkpoint "experiments/${NAME}/checkpoints/stage1_best.ckpt" --dataset "$DS" \
    --num-windows 200 --n-members 30 --n-outer 20 --batch-size 25 --members-file scores --output "$OUT/neural_eval.json"
done
echo "Done: ${NAME} at $(date)"
