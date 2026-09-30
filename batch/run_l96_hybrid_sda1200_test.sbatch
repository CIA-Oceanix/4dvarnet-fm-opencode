#!/bin/bash
#SBATCH --account=odyssey
#SBATCH --partition=Odyssey_GPU
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --array=0-5%4
#SBATCH --exclude=sl-mee-br-[212-213]
#SBATCH --job-name=l96-hyb1200-test
#SBATCH --output=sbatch_logs/l96_hyb1200_test_%A_%a.out
#SBATCH --error=sbatch_logs/l96_hyb1200_test_%A_%a.err
# DirectUNet-M(1200, seeds 1-3) -> SDA3-fix-M / SDA2-M (1200 epochs, seed 1) hybrid at the
# validation-selected tau0 0.1 / gw 2, regular grid and canonical random set, scores only
# (same convention as run_l96_hybrid_du1200.sbatch). Submit from this worktree's root.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
export PATH="/Odyssey/private/rfablet/miniforge3/envs/fdv-monai-proto/bin:$PATH"
export PYTHONUNBUFFERED=1
NODE_TMP="/SCRATCH/$USER/slurm_tmp/${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID:-0}"; mkdir -p "$NODE_TMP" 2>/dev/null || { NODE_TMP="$SLURM_SUBMIT_DIR/sbatch_logs/tmp_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID:-0}"; mkdir -p "$NODE_TMP"; }
export TMPDIR="$NODE_TMP" MPLCONFIGDIR="$NODE_TMP/mplconfig"; trap 'rm -rf "$NODE_TMP"' EXIT
mkdir -p "$MPLCONFIGDIR" sbatch_logs
R=/Odyssey/private/rfablet/Python/4dvarnet-fm-opencode
NORM=experiments/l96_norm_stats_obsj2.pt
PRIORS=(A3_sda3fix_monaiM_ep1200_l96 A3_sda2_monaiM_ep1200_l96)
i=$SLURM_ARRAY_TASK_ID; P=${PRIORS[$((i / 3))]}; S=$((i % 3 + 1))
DU=$R/experiments/l96/L96B_directunet_monaiM_ep1200_seed$S; SDA=experiments/${P}_seed1
NAME=hybrid_DU1200s${S}_${P}
echo "task $i: $NAME | commit $(git rev-parse --short HEAD) | $(hostname) | $(date)"
for TS in regular canonical; do
  if [ $TS = regular ]; then DS=experiments/l96_datasets_obsj2_int100_nwin200.pt; O=experiments/eval_regular/$NAME/tau0.1_gw2;
  else DS=experiments/l96_testset_rlayout_n10-100_k4-16_w200_d1.pt; O=experiments/eval_rlayout_n10-100_k4-16_w200/$NAME/tau0.1_gw2; fi
  [ -f $O/hybrid_eval.json ] && continue; mkdir -p $O
  python eval_sda_directunet_hybrid_l96.py --mean-checkpoint $DU/checkpoints/stage1_best.ckpt --mean-config $DU/resolved_config.yaml \
    --sda-checkpoint $SDA/checkpoints/stage1_best.ckpt --sda-config $SDA/resolved_config.yaml \
    --dataset $DS --num-windows 200 --batch-size 25 --n-members 30 --n-outer 10 --tau0 0.1 --guidance-weight 2 \
    --normalize-stats $NORM --members-file scores --output $O/hybrid_eval.json
done
echo "Done task $i at $(date)"
