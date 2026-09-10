#!/bin/bash
#SBATCH --account=odyssey
#SBATCH --partition=Odyssey_GPU
#SBATCH --gres=gpu:rtx8000:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --job-name=l96-sda-fdv1-hybrids-sda12
#SBATCH --output=sbatch_logs/l96_sda_fdv1_hybrids_sda12_%j.out
#SBATCH --error=sbatch_logs/l96_sda_fdv1_hybrids_sda12_%j.err

set -euo pipefail

cd /Odyssey/private/rfablet/Python/4dvarnet-fm-opencode/4dvarnet-fm-fdv-monai

export PATH="/Odyssey/private/rfablet/miniforge3/envs/fdv-monai-proto/bin:$PATH"
export PYTHONUNBUFFERED=1
export MPLCONFIGDIR="/tmp/mplconfig_${SLURM_JOB_ID}"
mkdir -p "$MPLCONFIGDIR"

# FDV1(monai) + SDA1/SDA2(monai) hybrids, ens30 S0/S1, for coherence with the
# already-run FDV1+SDA3 hybrid. Same tau0=0.3/guidance_weight=2.0 as
# FDV1+SDA3 -- NOT re-swept per variant, since three independent sweeps this
# session (DirectUNet+SDA1/2/3's joint sweep, and FDV1+SDA3's own sweep) all
# converged on this exact point regardless of which SDA variant or mean
# model was used.
MEAN_CKPT=experiments/FDV1_unrolled_monai_unet_l96/checkpoints/stage1_best.ckpt
MEAN_CFG=experiments/FDV1_unrolled_monai_unet_l96/resolved_config.yaml

for SDA in SDA1_monai_prior_l96_norm SDA2_monai_cond_mixed_l96_norm; do
  OUT_DIR="experiments/FDV1_${SDA%%_monai*}_monai_hybrid"
  echo ""
  echo "=== FDV1(monai) + ${SDA} hybrid, ens30 | $(date) ==="
  echo "GPU: $(nvidia-smi --query-gpu=name --format=csv -i 0 2>/dev/null | tail -1)"
  mkdir -p "$OUT_DIR"
  python eval_sda_mean_hybrid_l96.py \
    --mean-checkpoint "$MEAN_CKPT" \
    --mean-config "$MEAN_CFG" \
    --mean-n-outer 10 --no-mean-normalized \
    --sda-checkpoint experiments/${SDA}/checkpoints/stage1_best.ckpt \
    --sda-config experiments/${SDA}/resolved_config.yaml \
    --dataset experiments/l96_datasets_obsj2_int100_nwin200.pt \
    --n-members 30 --n-outer 10 --tau0 0.3 --guidance-weight 2.0 --r-var 0.5 \
    --normalize-stats experiments/l96_norm_stats_obsj2.pt \
    --batch-size 16 --cases s0 s1 \
    --output "${OUT_DIR}/hybrid_ens30.json"
done

echo ""
echo "All FDV1+SDA1/SDA2 hybrid ens30 evals complete at $(date)"
