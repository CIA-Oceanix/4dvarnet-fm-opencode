#!/bin/bash
#SBATCH --account=odyssey
#SBATCH --partition=Odyssey_GPU
#SBATCH --gres=gpu:rtx8000:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --job-name=l96-sda-fdv1-hybrid-ens30
#SBATCH --output=sbatch_logs/l96_sda_fdv1_hybrid_ens30_%j.out
#SBATCH --error=sbatch_logs/l96_sda_fdv1_hybrid_ens30_%j.err

set -euo pipefail

cd /Odyssey/private/rfablet/Python/4dvarnet-fm-opencode/4dvarnet-fm-fdv-monai

export PATH="/Odyssey/private/rfablet/miniforge3/envs/fdv-monai-proto/bin:$PATH"
export PYTHONUNBUFFERED=1
export MPLCONFIGDIR="/tmp/mplconfig_${SLURM_JOB_ID}"
mkdir -p "$MPLCONFIGDIR"

echo "FDV1(monai) + SDA3(monai) hybrid, ens30 S0/S1 | $(date)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv -i 0 2>/dev/null | tail -1)"

mkdir -p experiments/FDV1_SDA3_monai_hybrid
python eval_sda_mean_hybrid_l96.py \
  --mean-checkpoint experiments/FDV1_unrolled_monai_unet_l96/checkpoints/stage1_best.ckpt \
  --mean-config experiments/FDV1_unrolled_monai_unet_l96/resolved_config.yaml \
  --mean-n-outer 10 --no-mean-normalized \
  --sda-checkpoint experiments/SDA3_monai_cond_noisy_l96_norm/checkpoints/stage1_best.ckpt \
  --sda-config experiments/SDA3_monai_cond_noisy_l96_norm/resolved_config.yaml \
  --dataset experiments/l96_datasets_obsj2_int100_nwin200.pt \
  --n-members 30 --n-outer 10 --tau0 0.3 --guidance-weight 2.0 --r-var 0.5 \
  --normalize-stats experiments/l96_norm_stats_obsj2.pt \
  --batch-size 16 --cases s0 s1 \
  --output experiments/FDV1_SDA3_monai_hybrid/fdv1_sda3_hybrid_ens30.json

echo "Done at $(date)"
