#!/bin/bash
#SBATCH --account=odyssey
#SBATCH --partition=Odyssey_GPU
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=24:00:00
#SBATCH --job-name=l96-fdv2-subgrad-Stier-priorres-nzinit
#SBATCH --output=sbatch_logs/l96_fdv2_subgrad_state_monai_Stier_priorresidual_nonzeroinit_train_%j.out
#SBATCH --error=sbatch_logs/l96_fdv2_subgrad_state_monai_Stier_priorresidual_nonzeroinit_train_%j.err

set -euo pipefail

cd /Odyssey/private/rfablet/Python/4dvarnet-fm-opencode/4dvarnet-fm-obs-density-gen

export PATH="/Odyssey/private/rfablet/miniforge3/envs/fdv-monai-proto/bin:$PATH"
export CONDA_DEFAULT_ENV=fdv-monai-proto
export PYTHONUNBUFFERED=1
export MPLCONFIGDIR="/tmp/mplconfig_${SLURM_JOB_ID}"

mkdir -p "$MPLCONFIGDIR" sbatch_logs

TEST_CACHE="/Odyssey/private/rfablet/Python/4dvarnet-fm-opencode/experiments/l96_datasets_obsj2_int100_nwin200.pt"

EXPERIMENT="FDV2_subgrad_state_monai_l96_Stier_priorresidual_nonzeroinit"
echo "============================================"
echo " L96 FDV2(monai) subgrad+state, S-tier, prior_residual=true + prior_output_init_std=0.1 | $(date)"
echo "  Experiment: ${EXPERIMENT}"
echo "  Same base as subgrad+state-Stier(monai), the best standalone neural"
echo "  mean-estimate model in this whole investigation, plus"
echo "  prior_residual=true + prior_output_init_std=0.1 (breaks MONAI's"
echo "  zero_module() exact-zero init on prior_unet's final conv -- see the"
echo "  sibling grad+state run's header for the full derivation)."
echo "  ABLATION, not a fix for a known problem: subgrad+state-Stier(monai)'s"
echo "  own prior_unet was checked directly and is NOT stuck (final conv"
echo "  weight norm 5.30, clearly trained) -- subgrad+state's g_prior=x-"
echo "  Phi(x) proxy is LINEAR in Phi(x), never structurally vanishing"
echo "  regardless of Phi(x)'s magnitude, and prior_residual=false (its"
echo "  default) means prior_cost never collapses to a self-referential"
echo "  ||f(x)||^2 form in the first place. Tests whether prior_residual's"
echo "  Jacobian-anchor benefit ALSO helps a scheme that was never broken."
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
