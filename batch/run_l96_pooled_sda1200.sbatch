#!/bin/bash
#SBATCH --account=odyssey
#SBATCH --partition=Odyssey_GPU
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --array=0-29%3
#SBATCH --exclude=sl-mee-br-[212-213]
#SBATCH --job-name=l96-pooled-sda1200
#SBATCH --output=sbatch_logs/l96_pooled_sda1200_%A_%a.out
#SBATCH --error=sbatch_logs/l96_pooled_sda1200_%A_%a.err
# Pooled spread/RMSE (reports/l96/probe_pooled_calibration.py, members in memory) for the
# 1200-epoch SDA priors (gw 25) and the DU(1200) -> SDA(1200) hybrids (tau0 0.1, gw 2),
# seeds 1-3, regular grid and canonical random set. Submit from this worktree's root.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
export PATH="/Odyssey/private/rfablet/miniforge3/envs/fdv-monai-proto/bin:$PATH"
export PYTHONUNBUFFERED=1 PYTHONPATH="${SLURM_SUBMIT_DIR}"
NODE_TMP="/SCRATCH/$USER/slurm_tmp/${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID:-0}"; mkdir -p "$NODE_TMP" 2>/dev/null || { NODE_TMP="$SLURM_SUBMIT_DIR/sbatch_logs/tmp_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID:-0}"; mkdir -p "$NODE_TMP"; }
export TMPDIR="$NODE_TMP" MPLCONFIGDIR="$NODE_TMP/mplconfig"; trap 'rm -rf "$NODE_TMP"' EXIT
mkdir -p "$MPLCONFIGDIR" experiments/pooled_calibration_1200
R=/Odyssey/private/rfablet/Python/4dvarnet-fm-opencode
NORM=experiments/l96_norm_stats_obsj2.pt
declare -A DS=([reg]=experiments/l96_datasets_obsj2_int100_nwin200.pt [rand]=experiments/l96_testset_rlayout_n10-100_k4-16_w200_d1.pt)
TASKS=()
for P in B4_sda1_monaiM_ep1200_l96 A3_sda2_monaiM_ep1200_l96 A3_sda3fix_monaiM_ep1200_l96; do for S in 1 2 3; do for L in reg rand; do TASKS+=("sda $P $S $L"); done; done; done
for P in A3_sda3fix_monaiM_ep1200_l96 A3_sda2_monaiM_ep1200_l96; do for S in 1 2 3; do for L in reg rand; do TASKS+=("hyb $P $S $L"); done; done; done
read -r KIND P S L <<< "${TASKS[$SLURM_ARRAY_TASK_ID]}"
echo "task $SLURM_ARRAY_TASK_ID: $KIND $P seed $S $L | commit $(git rev-parse --short HEAD) | $(hostname) | $(date)"
if [ $KIND = sda ]; then
  NAME=${P}_seed$S; SDA=experiments/$NAME
  python -u reports/l96/probe_pooled_calibration.py eval_sda_l96 --checkpoint $SDA/checkpoints/stage1_best.ckpt --config $SDA/resolved_config.yaml \
    --dataset ${DS[$L]} --normalize-stats $NORM --n-members 30 --n-outer 10 --guidance-weight 25 --r-var 0.5 --seed 0 --batch-size 16 \
    --output $NODE_TMP/eval.json
else
  NAME=hybrid_DU1200s${S}_$P; DU=$R/experiments/l96/L96B_directunet_monaiM_ep1200_seed$S; SDA=experiments/${P}_seed1
  python -u reports/l96/probe_pooled_calibration.py eval_sda_directunet_hybrid_l96 --mean-checkpoint $DU/checkpoints/stage1_best.ckpt \
    --mean-config $DU/resolved_config.yaml --sda-checkpoint $SDA/checkpoints/stage1_best.ckpt --sda-config $SDA/resolved_config.yaml \
    --dataset ${DS[$L]} --normalize-stats $NORM --n-members 30 --n-outer 10 --tau0 0.1 --guidance-weight 2 --r-var 0.5 --seed 0 \
    --batch-size 16 --output $NODE_TMP/eval.json
fi
for c in s0 s1; do cp $NODE_TMP/pooled_$c.json experiments/pooled_calibration_1200/${NAME}_${L}_$c.json; done
echo "Done at $(date)"
