#!/bin/bash
#SBATCH --account=odyssey
#SBATCH --partition=Odyssey_GPU
#SBATCH --gres=gpu:a40:1
#SBATCH --nodelist=sl-mee-br-205
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=8:00:00
#SBATCH --job-name=l96-sda_rescore
#SBATCH --output=sbatch_logs/l96_sda_rescore_%j.out
#SBATCH --error=sbatch_logs/l96_sda_rescore_%j.err

export FDV_KEEP_MEMBERS=1  # report-read benchmark rows keep members on /Odyssey (evaluation/members_store.py)
set -euo pipefail
cd /Odyssey/private/rfablet/Python/4dvarnet-fm-opencode/4dvarnet-fm-fdv-tau-aware
export PATH="/Odyssey/private/rfablet/miniforge3/envs/fdv-monai-proto/bin:$PATH"
export PYTHONUNBUFFERED=1
export MPLCONFIGDIR="/tmp/mplconfig_${SLURM_JOB_ID}"
mkdir -p "$MPLCONFIGDIR" sbatch_logs
DATASET="/Odyssey/private/rfablet/Python/4dvarnet-fm-opencode/experiments/l96_datasets_obsj2_int100_nwin200.pt"
# Re-score the whole SDA family at gw=20. The sweep on SDA1-M showed the
# inherited gw=40 is 6.9% RMSE / 6.0% CRPS worse than gw=20, with a sharply
# non-monotone curve (10: 0.6815, 20: 0.5063, 40: 0.5439, 80: 0.7875), so every
# SDA row scored at 40 is understated. Eval only, no retraining.
for EXP in B4_sda1_monaiSplus_l96 B4_sda1_monaiM_l96 B4_sda1_monaiL_l96 A3_sda2_monaiM_l96 A3_sda3_monaiM_l96; do
  OUT="experiments/${EXP}/ens30_gw20"; mkdir -p "$OUT"
  echo "=== ${EXP} | gw=20 | $(date) ==="
  python eval_sda_l96.py \
    --checkpoint "experiments/${EXP}/checkpoints/stage1_best.ckpt" \
    --config "experiments/${EXP}/resolved_config.yaml" \
    --dataset "$DATASET" --num-windows 200 --batch-size 16 \
    --n-members 30 --n-outer 10 --r-var 0.5 --guidance-weight 20 \
    --normalize-stats experiments/l96_norm_stats_obsj2.pt \
    --output "${OUT}/sda_eval.json" || echo "FAILED: ${EXP}"
done
echo "Done: $(date)"
