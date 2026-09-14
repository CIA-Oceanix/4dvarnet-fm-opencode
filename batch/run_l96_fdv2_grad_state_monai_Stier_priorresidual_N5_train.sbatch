#!/bin/bash
#SBATCH --account=odyssey
#SBATCH --partition=Odyssey_GPU
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=24:00:00
#SBATCH --job-name=l96-fdv2-grad-Stier-priorres-N5
#SBATCH --output=sbatch_logs/l96_fdv2_grad_state_monai_Stier_priorresidual_N5_train_%j.out
#SBATCH --error=sbatch_logs/l96_fdv2_grad_state_monai_Stier_priorresidual_N5_train_%j.err

set -euo pipefail

cd /Odyssey/private/rfablet/Python/4dvarnet-fm-opencode/4dvarnet-fm-obs-density-gen

export PATH="/Odyssey/private/rfablet/miniforge3/envs/fdv-monai-proto/bin:$PATH"
export CONDA_DEFAULT_ENV=fdv-monai-proto
export PYTHONUNBUFFERED=1
export MPLCONFIGDIR="/tmp/mplconfig_${SLURM_JOB_ID}"

mkdir -p "$MPLCONFIGDIR" sbatch_logs

TEST_CACHE="/Odyssey/private/rfablet/Python/4dvarnet-fm-opencode/experiments/l96_datasets_obsj2_int100_nwin200.pt"

EXPERIMENT="FDV2_grad_state_monai_l96_Stier_priorresidual_N5"
echo "============================================"
echo " L96 FDV2(monai) grad+state, S-tier, prior_residual=true, N_outer=5 | $(date)"
echo "  Experiment: ${EXPERIMENT}"
echo "  Combined single-gradient mode (torch.autograd.grad of the summed"
echo "  var_cost), trainable_prior_weight=true, prior_residual=true"
echo "  (Phi(x) = x + prior_unet(x, tau)), prior_dropout=0.0 (decoupled from"
echo "  the main solver unet's dropout=0.1 -- avoids injecting extra mask"
echo "  noise into the double-backward g_prior computation), N_outer=5 (down"
echo "  from 10, faster + shorter double-backward chain)."
echo "  Closer to the ocean4dvarnet/4dvarnet-global-mapping (ronan_devs)"
echo "  setup that previously trained a real-gradient 4DVarNet solver"
echo "  successfully with an OpenAI guided-diffusion UNet backbone, at"
echo "  steepest-descent quality at least matching the proxy (subgrad+state)."
echo "  GPU: $(nvidia-smi --query-gpu=name --format=csv -i 0 2>/dev/null | tail -1)"
echo "============================================"
echo

python train.py \
  --config-name experiment/${EXPERIMENT} \
  hydra.run.dir=. \
  hydra.output_subdir=null \
  ++data.test_cache=${TEST_CACHE} \
  +fresh=true

echo ""
echo " Done: ${EXPERIMENT} at $(date)"
echo ""
