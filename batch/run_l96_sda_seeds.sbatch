#!/bin/bash
#SBATCH --account=odyssey
#SBATCH --partition=Odyssey_GPU
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --array=0-5%3
#SBATCH --job-name=l96-sda-seeds
#SBATCH --output=sbatch_logs/l96_sda_seeds_%A_%a.out
#SBATCH --error=sbatch_logs/l96_sda_seeds_%A_%a.err
# Extra training seeds (2, 3) for SDA1-M / SDA2-M / SDA3-M (P1 configs; the
# obs protocol does not enter SDA training), then the P1 guided eval (ens30,
# 10 steps, gw 20, r_var 0.5) on the regular grid and the canonical random
# set, saving per-window scores only.
set -euo pipefail
cd /Odyssey/private/rfablet/Python/4dvarnet-fm-opencode/4dvarnet-fm-sda-nanfix
export PATH="/Odyssey/private/rfablet/miniforge3/envs/fdv-monai-proto/bin:$PATH"
export CONDA_DEFAULT_ENV=fdv-monai-proto
export PYTHONUNBUFFERED=1
export MPLCONFIGDIR="/tmp/mplconfig_${SLURM_JOB_ID}"
mkdir -p "$MPLCONFIGDIR" sbatch_logs
R=/Odyssey/private/rfablet/Python/4dvarnet-fm-opencode
EXPS=(B4_sda1_monaiM_l96 A3_sda2_monaiM_l96 A3_sda3_monaiM_l96)
i=$SLURM_ARRAY_TASK_ID
EXP="${EXPS[$((i / 2))]}"; SEED=$((i % 2 + 2)); NAME="${EXP}_seed${SEED}"
echo "===== ${NAME} | $(date) | $(nvidia-smi --query-gpu=name --format=csv,noheader -i 0 | tail -1) ====="
python train.py --config-name experiment/${EXP} hydra.run.dir=. hydra.output_subdir=null ++training.seed=${SEED}
for TS in regular canonical; do
  if [ "$TS" = regular ]; then DS=$R/experiments/l96_datasets_obsj2_int100_nwin200.pt; OUT="experiments/${NAME}/ens30_gw20";
  else DS=$R/experiments/l96_testset_rlayout_n10-100_k4-16_w200_d1.pt; OUT="experiments/eval_rlayout_n10-100_k4-16_w200/${NAME}/ens30_gw20"; fi
  mkdir -p "$OUT"
  python eval_sda_l96.py --checkpoint "experiments/${NAME}/checkpoints/stage1_best.ckpt" \
    --config "experiments/${NAME}/resolved_config.yaml" --dataset "$DS" --num-windows 200 --batch-size 16 \
    --n-members 30 --n-outer 10 --r-var 0.5 --guidance-weight 20 \
    --normalize-stats experiments/l96_norm_stats_obsj2.pt --members-file scores --output "$OUT/sda_eval.json"
done
echo "Done: ${NAME} at $(date)"
