#!/bin/bash
#SBATCH --account=odyssey
#SBATCH --partition=Odyssey_GPU
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --array=1-3
#SBATCH --job-name=l96b-du2d
#SBATCH --output=sbatch_logs/l96b_du2d_%A_%a.out
#SBATCH --error=sbatch_logs/l96b_du2d_%A_%a.err
# DirectUNet with the 2D (time x sector) circular backbone, seeds 1-3, under the
# L96 benchmark default (1200 epochs), then the deterministic eval on both test
# sets -- the same protocol as L96B_directunet_monaiM_ep1200_seed{1,2,3}
# (batch/run_l96b_du_ep1200.sbatch). CONFIG=L96B_directunet2d_monaiMwide for the
# width-matched arm. Runs from the submitting worktree.
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
export PATH="/Odyssey/private/rfablet/miniforge3/envs/${CONDA_ENV:-fdv-monai-proto}/bin:$PATH"
export PYTHONUNBUFFERED=1
NODE_TMP="/SCRATCH/$USER/slurm_tmp/${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID:-0}"; mkdir -p "$NODE_TMP" 2>/dev/null || { NODE_TMP="/tmp/${USER}_slurm_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID:-0}"; mkdir -p "$NODE_TMP"; }
export TMPDIR="$NODE_TMP" MPLCONFIGDIR="$NODE_TMP/mplconfig"; trap 'rm -rf "$NODE_TMP"' EXIT
mkdir -p "$MPLCONFIGDIR" sbatch_logs
R=/Odyssey/private/rfablet/Python/4dvarnet-fm-opencode
CONFIG=${CONFIG:-L96B_directunet2d_monaiM}
S=$SLURM_ARRAY_TASK_ID
NAME=${CONFIG}_seed$S
python train.py --config-name experiment/$CONFIG hydra.run.dir=. hydra.output_subdir=null \
  training.seed=$S ++data.train_cache=$R/experiments/l96_trainval_cache/l96b_trainval_n1000_v100.pt \
  ++data.norm_stats_path=$R/experiments/l96_norm_stats_obsj2.pt
for TS in regular canonical; do
  if [ "$TS" = regular ]; then DS=$R/experiments/l96_datasets_obsj2_int100_nwin200.pt; OUT="experiments/${NAME}/ens1_no1";
  else DS=$R/experiments/l96_testset_rlayout_n10-100_k4-16_w200_d1.pt; OUT="experiments/eval_rlayout_n10-100_k4-16_w200/${NAME}/ens1_no1"; fi
  mkdir -p "$OUT"
  python eval_neural_l96.py --step-power 0.5 --checkpoint "experiments/${NAME}/checkpoints/stage1_best.ckpt" --dataset "$DS" \
    --num-windows 200 --n-members 1 --n-outer 1 --batch-size 25 --output "$OUT/neural_eval.json"
done
echo "Done at $(date)"
