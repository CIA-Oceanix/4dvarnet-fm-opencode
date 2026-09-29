#!/bin/bash
#SBATCH --account=odyssey
#SBATCH --partition=Odyssey_GPU
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --array=0-8
#SBATCH --job-name=l96-sda-ep1200
#SBATCH --output=sbatch_logs/l96_sda_ep1200_%A_%a.out
#SBATCH --error=sbatch_logs/l96_sda_ep1200_%A_%a.err
# SDA1-M / SDA2-M / SDA3-fix-M priors at the 1200-epoch benchmark budget, seeds
# 1-3 (the *_ep1200_l96 configs change only stage-1 epochs), then the benchmark
# guided eval (ens30, 10 steps, gw 25, r_var 0.5) on the regular grid and the
# canonical random set, per-window scores only. Submit from this worktree's root.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
export PATH="/Odyssey/private/rfablet/miniforge3/envs/fdv-monai-proto/bin:$PATH"
export PYTHONUNBUFFERED=1
export MPLCONFIGDIR="/tmp/mplconfig_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
mkdir -p "$MPLCONFIGDIR" sbatch_logs
EXPS=(B4_sda1_monaiM_ep1200_l96 A3_sda2_monaiM_ep1200_l96 A3_sda3fix_monaiM_ep1200_l96)
i=$SLURM_ARRAY_TASK_ID
EXP="${EXPS[$((i / 3))]}"; SEED=$((i % 3 + 1)); NAME="${EXP}_seed${SEED}"
echo "===== ${NAME} | commit $(git rev-parse --short HEAD) | $(hostname) | $(nvidia-smi --query-gpu=name --format=csv,noheader -i 0 | tail -1) | $(date) ====="
[ -f "experiments/${NAME}/checkpoints/stage1_best.ckpt" ] || \
  python train.py --config-name experiment/${EXP} hydra.run.dir=. hydra.output_subdir=null ++training.seed=${SEED}
for TS in regular canonical; do
  if [ "$TS" = regular ]; then DS=experiments/l96_datasets_obsj2_int100_nwin200.pt; OUT="experiments/eval_regular/${NAME}/ens30_gw25";
  else DS=experiments/l96_testset_rlayout_n10-100_k4-16_w200_d1.pt; OUT="experiments/eval_rlayout_n10-100_k4-16_w200/${NAME}/ens30_gw25"; fi
  [ -f "$OUT/sda_eval.json" ] && continue; mkdir -p "$OUT"
  python eval_sda_l96.py --checkpoint "experiments/${NAME}/checkpoints/stage1_best.ckpt" \
    --config "experiments/${NAME}/resolved_config.yaml" --dataset "$DS" --num-windows 200 --batch-size 16 \
    --n-members 30 --n-outer 10 --r-var 0.5 --guidance-weight 25 \
    --normalize-stats experiments/l96_norm_stats_obsj2.pt --members-file scores --output "$OUT/sda_eval.json"
done
echo "Done: ${NAME} at $(date)"
