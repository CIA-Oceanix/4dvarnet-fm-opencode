#!/bin/bash
#SBATCH --account=odyssey
#SBATCH --partition=Odyssey_GPU
#SBATCH --gres=gpu:rtx8000:1
#SBATCH --nodelist=sl-mee-br-203
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=03:00:00
#SBATCH --job-name=l96-sda-directunet-ens30
#SBATCH --output=sbatch_logs/l96_neural_%j.out
#SBATCH --error=sbatch_logs/l96_neural_%j.err

set -euo pipefail

cd /Odyssey/private/rfablet/Python/4dvarnet-fm-opencode/4dvarnet-fm-l96-eval-config-persist

export PATH="/Odyssey/private/rfablet/miniforge3/envs/fdv-monai-proto/bin:$PATH"
export CONDA_DEFAULT_ENV=fdv-monai-proto
export PYTHONUNBUFFERED=1
export MPLCONFIGDIR="/tmp/mplconfig_${SLURM_JOB_ID}"
mkdir -p "$MPLCONFIGDIR"

# Full ens30 S0/S1 eval for DirectUNet-M + SDA{1,2,3} hybrids at the swept
# best hyperparameters (tau0=0.3, guidance_weight=2.0, all three variants
# converged on the same point -- see sda_directunet_hybrid_sweep_results.json).
MEAN_CKPT=experiments/L1b_monai_unet_s0s1_norm/checkpoints/stage1_best.ckpt
MEAN_CFG=experiments/L1b_monai_unet_s0s1_norm/resolved_config.yaml

for SDA in SDA1_monai_prior_l96_norm SDA2_monai_cond_mixed_l96_norm SDA3_monai_cond_noisy_l96_norm; do
  echo ""
  echo "=== DirectUNet-M + ${SDA} hybrid, ens30 | $(date) ==="
  mkdir -p "experiments/${SDA}/directunet_hybrid"
  python eval_sda_directunet_hybrid_l96.py \
    --mean-checkpoint "$MEAN_CKPT" \
    --mean-config "$MEAN_CFG" \
    --sda-checkpoint experiments/${SDA}/checkpoints/stage1_best.ckpt \
    --sda-config experiments/${SDA}/resolved_config.yaml \
    --dataset experiments/l96_datasets_obsj2_int100_nwin200.pt \
    --num-windows 200 --batch-size 16 \
    --n-members 30 --n-outer 10 --tau0 0.3 --guidance-weight 2.0 --r-var 0.5 \
    --normalize-stats experiments/l96_norm_stats_obsj2.pt \
    --output experiments/${SDA}/directunet_hybrid/directunet_hybrid_ens30.json
done

echo ""
echo "All DirectUNet+SDA hybrid ens30 evals complete at $(date)"
