#!/bin/bash
#SBATCH --account=odyssey
#SBATCH --partition=Odyssey_GPU
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=24:00:00
#SBATCH --job-name=l96-fdv2-subgtrue
#SBATCH --output=sbatch_logs/l96_fdv2_subgrad_trueprior_monai_perfectmodel_train_%j.out
#SBATCH --error=sbatch_logs/l96_fdv2_subgrad_trueprior_monai_perfectmodel_train_%j.err

set -euo pipefail

cd /Odyssey/private/rfablet/Python/4dvarnet-fm-opencode/4dvarnet-fm-obs-density-gen

export PATH="/Odyssey/private/rfablet/miniforge3/envs/fdv-monai-proto/bin:$PATH"
export CONDA_DEFAULT_ENV=fdv-monai-proto
export PYTHONUNBUFFERED=1
export MPLCONFIGDIR="/tmp/mplconfig_${SLURM_JOB_ID}"

mkdir -p "$MPLCONFIGDIR" sbatch_logs

EXPERIMENT="FDV2_subgrad_trueprior_monai_l96_perfectmodel"
echo "============================================"
echo " L96 FDV2(monai) subgrad+trueprior (NO raw-state channel), perfect-model S0 | $(date)"
echo "  Experiment: ${EXPERIMENT}"
echo "  Same true-ODE prior as subgrad+state+trueprior (job 53564), but the"
echo "  update UNet is fed cat([g_obs, g_prior]) only -- NOT cat([g_obs,"
echo "  g_prior, x]). Ablation: do the two residuals alone carry enough"
echo "  information, or does the explicit raw-state channel matter, mirroring"
echo "  the existing grad-only vs grad+state comparison for the real-"
echo "  autograd-gradient family."
echo "  state_dim=40 (full physical state); Phi always reads the window's"
echo "  TRUE forcing/params (perfect-model, S0-targeted, same as job 53564)."
echo "  GPU: $(nvidia-smi --query-gpu=name --format=csv -i 0 2>/dev/null | tail -1)"
echo "============================================"
echo

python train.py \
  --config-name experiment/${EXPERIMENT} \
  hydra.run.dir=. \
  hydra.output_subdir=null \
  +fresh=true

echo ""
echo " Done: ${EXPERIMENT} at $(date)"
echo ""
