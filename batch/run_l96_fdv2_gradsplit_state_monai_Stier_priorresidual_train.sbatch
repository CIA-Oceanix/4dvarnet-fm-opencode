#!/bin/bash
#SBATCH --account=odyssey
#SBATCH --partition=Odyssey_GPU
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=24:00:00
#SBATCH --job-name=l96-fdv2-gradsplit-Stier-priorres
#SBATCH --output=sbatch_logs/l96_fdv2_gradsplit_state_monai_Stier_priorresidual_train_%j.out
#SBATCH --error=sbatch_logs/l96_fdv2_gradsplit_state_monai_Stier_priorresidual_train_%j.err

set -euo pipefail

cd /Odyssey/private/rfablet/Python/4dvarnet-fm-opencode/4dvarnet-fm-obs-density-gen

export PATH="/Odyssey/private/rfablet/miniforge3/envs/fdv-monai-proto/bin:$PATH"
export CONDA_DEFAULT_ENV=fdv-monai-proto
export PYTHONUNBUFFERED=1
export MPLCONFIGDIR="/tmp/mplconfig_${SLURM_JOB_ID}"

mkdir -p "$MPLCONFIGDIR" sbatch_logs

TEST_CACHE="/Odyssey/private/rfablet/Python/4dvarnet-fm-opencode/experiments/l96_datasets_obsj2_int100_nwin200.pt"

EXPERIMENT="FDV2_gradsplit_state_monai_l96_Stier_priorresidual"
echo "============================================"
echo " L96 FDV2(monai) gradsplit+state, S-tier, prior_residual=true | $(date)"
echo "  Experiment: ${EXPERIMENT}"
echo "  Same config as FDV2_gradsplit_state_monai_l96_Stier (job 53447,"
echo "  permanently plateaued at val_loss ~3.4-3.6 through 250 epochs), with"
echo "  ONE change: Phi(x) = x + prior_unet(x, tau) instead of the bare"
echo "  backbone output. Tests whether an explicit, permanent identity"
echo "  anchor around Phi resolves the plateau -- a Jacobian decomposition"
echo "  of 53447's trained prior_unet found the true prior_cost gradient's"
echo "  Jacobian term dominating the useful residual term by 11-53x in RMS"
echo "  and nearly orthogonal to it (cos~0.04-0.13), consistent with"
echo "  MonaiUNet1D's zero-init (confirmed present, matches OpenAI"
echo "  guided-diffusion's own zero_module() convention) only anchoring Phi"
echo "  near identity AT INITIALIZATION, not throughout training."
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
