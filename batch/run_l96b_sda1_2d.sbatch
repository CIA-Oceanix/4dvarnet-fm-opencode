#!/bin/bash
#SBATCH --account=odyssey
#SBATCH --partition=Odyssey_GPU
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --job-name=l96b-sda1-2d
#SBATCH --output=sbatch_logs/l96b_sda1_2d_%j.out
#SBATCH --error=sbatch_logs/l96b_sda1_2d_%j.err
# SDA1 prior with the 2D (time x sector) circular backbone, 1000 windows x 1200 epochs, then
# on both test sets (scores only):
#  - guided SDA as B4_sda1_monaiM_ep1200_l96 (batch/run_l96_sda_ep1200.sbatch: ens30, 10
#    steps, gw 25, r_var 0.5);
#  - the DirectUNet hybrid at the 1D validation-selected tau0 0.1 / gw 2, with
#    L96B_directunet2d_monaiM_seed1 (experiments/l96) as the mean.
# Runs from the submitting worktree; resumes from checkpoints/stage1_last.ckpt if
# resubmitted. CONFIG (default the M tier), SEED (default 1).
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
export PATH="/Odyssey/private/rfablet/miniforge3/envs/${CONDA_ENV:-fdv-monai-proto}/bin:$PATH"
export PYTHONUNBUFFERED=1
NODE_TMP="/SCRATCH/$USER/slurm_tmp/${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID:-0}"; mkdir -p "$NODE_TMP" 2>/dev/null || { NODE_TMP="/tmp/${USER}_slurm_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID:-0}"; mkdir -p "$NODE_TMP"; }
export TMPDIR="$NODE_TMP" MPLCONFIGDIR="$NODE_TMP/mplconfig"; trap 'rm -rf "$NODE_TMP"' EXIT
mkdir -p "$MPLCONFIGDIR" sbatch_logs
R=/Odyssey/private/rfablet/Python/4dvarnet-fm-opencode
NORM=$R/experiments/l96_norm_stats_obsj2.pt
CONFIG=${CONFIG:-L96B_sda1_monai2dM_ep1200}; SEED=${SEED:-1}; NAME=${CONFIG}_seed${SEED}
DU=$R/experiments/l96/L96B_directunet2d_monaiM_seed1
[ -f experiments/${NAME}/results.json ] || \
  python train.py --config-name experiment/$CONFIG hydra.run.dir=. hydra.output_subdir=null training.seed=$SEED \
    ++data.train_cache=$R/experiments/l96_trainval_cache/l96b_trainval_n1000_v100.pt \
    ++data.norm_stats_path=$NORM
CKPT=experiments/${NAME}/checkpoints/stage1_best.ckpt; CFG=experiments/${NAME}/resolved_config.yaml
for TS in regular canonical; do
  if [ "$TS" = regular ]; then DS=$R/experiments/l96_datasets_obsj2_int100_nwin200.pt; E=experiments/eval_regular;
  else DS=$R/experiments/l96_testset_rlayout_n10-100_k4-16_w200_d1.pt; E=experiments/eval_rlayout_n10-100_k4-16_w200; fi
  OUT=$E/${NAME}/ens30_gw25
  if [ ! -f "$OUT/sda_eval.json" ]; then mkdir -p "$OUT"
    python eval_sda_l96.py --checkpoint $CKPT --config $CFG --dataset "$DS" --num-windows 200 --batch-size 16 \
      --n-members 30 --n-outer 10 --r-var 0.5 --guidance-weight 25 \
      --normalize-stats $NORM --members-file scores --output "$OUT/sda_eval.json"
  fi
  OUT=$E/hybrid_DU2dMs1_${NAME}/tau0.1_gw2
  if [ ! -f "$OUT/hybrid_eval.json" ]; then mkdir -p "$OUT"
    python eval_sda_directunet_hybrid_l96.py \
      --mean-checkpoint $DU/checkpoints/stage1_best.ckpt --mean-config $DU/resolved_config.yaml \
      --sda-checkpoint $CKPT --sda-config $CFG --dataset "$DS" --num-windows 200 --batch-size 25 \
      --n-members 30 --n-outer 10 --tau0 0.1 --guidance-weight 2 \
      --normalize-stats $NORM --members-file scores --output "$OUT/hybrid_eval.json"
  fi
done
echo "Done: ${NAME} at $(date)"
