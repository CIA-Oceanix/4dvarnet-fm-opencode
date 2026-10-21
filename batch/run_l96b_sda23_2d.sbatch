#!/bin/bash
#SBATCH --account=odyssey
#SBATCH --partition=Odyssey_GPU
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --array=0-5%3
#SBATCH --job-name=l96b-sda23-2d
#SBATCH --output=sbatch_logs/l96b_sda23_2d_%A_%a.out
#SBATCH --error=sbatch_logs/l96b_sda23_2d_%A_%a.err
# SDA2 / SDA3-fix priors with the 2D (time x sector) circular backbone (L96B_sda{2,3fix}_monai2dM_ep1200:
# the 1D A3_*_ep1200_l96 recipe -- data generated with seed 42, the shared obsj2 normalization, 1200 epochs --
# with the backbone swapped), seeds 1-3, then as batch/run_l96_sda_ep1200.sbatch on both test sets
# (scores only): guided SDA (ens30, 10 steps, gw 25, r_var 0.5). For prior seed 1 only, the
# DirectUNet-2D hybrid (tau0 0.1, gw 1 and 2) from L96B_directunet2d_monaiM seeds 1-3, the benchmark's
# 3-seed hybrid convention. Task = variant * 3 + seed - 1. Resumes from stage1_last.ckpt if resubmitted.
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
export PATH="/Odyssey/private/rfablet/miniforge3/envs/${CONDA_ENV:-fdv-monai-proto}/bin:$PATH"
export PYTHONUNBUFFERED=1
NODE_TMP="/SCRATCH/$USER/slurm_tmp/${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID:-0}"; mkdir -p "$NODE_TMP" 2>/dev/null || { NODE_TMP="/tmp/${USER}_slurm_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID:-0}"; mkdir -p "$NODE_TMP"; }
export TMPDIR="$NODE_TMP" MPLCONFIGDIR="$NODE_TMP/mplconfig" FDV_MEMBERS_TMP="$NODE_TMP/members"; trap 'rm -rf "$NODE_TMP"' EXIT
mkdir -p "$MPLCONFIGDIR" sbatch_logs
R=/Odyssey/private/rfablet/Python/4dvarnet-fm-opencode
NORM=$R/experiments/l96_norm_stats_obsj2.pt
CONFIGS=(L96B_sda2_monai2dM_ep1200 L96B_sda3fix_monai2dM_ep1200)
i=$SLURM_ARRAY_TASK_ID
CONFIG=${CONFIGS[$((i / 3))]}; SEED=$((i % 3 + 1)); NAME=${CONFIG}_seed${SEED}
echo "===== ${NAME} | commit $(git rev-parse --short HEAD) | $(hostname) | $(date) ====="
[ -f experiments/${NAME}/results.json ] || \
  python train.py --config-name experiment/$CONFIG hydra.run.dir=. hydra.output_subdir=null ++training.seed=$SEED \
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
  [ "$SEED" = 1 ] || continue
  for DU_SEED in 1 2 3; do
    DU=$R/experiments/l96/L96B_directunet2d_monaiM_seed${DU_SEED}
    for GW in 1 2; do
      OUT=$E/hybrid_DU2dMs${DU_SEED}_${NAME}/tau0.1_gw$GW
      [ -f "$OUT/hybrid_eval.json" ] && continue; mkdir -p "$OUT"
      python eval_sda_directunet_hybrid_l96.py \
        --mean-checkpoint $DU/checkpoints/stage1_best.ckpt --mean-config $DU/resolved_config.yaml \
        --sda-checkpoint $CKPT --sda-config $CFG --dataset "$DS" --num-windows 200 --batch-size 25 \
        --n-members 30 --n-outer 10 --tau0 0.1 --guidance-weight $GW \
        --normalize-stats $NORM --members-file scores --output "$OUT/hybrid_eval.json"
    done
  done
done
echo "Done: ${NAME} at $(date)"
