#!/bin/bash
#SBATCH --account=odyssey
#SBATCH --partition=Odyssey_GPU
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=24:00:00
#SBATCH --job-name=l96-fdv2-grad-Stier-priorres-detach
#SBATCH --output=sbatch_logs/l96_fdv2_grad_state_monai_Stier_priorresidual_detachvarcost_train_%j.out
#SBATCH --error=sbatch_logs/l96_fdv2_grad_state_monai_Stier_priorresidual_detachvarcost_train_%j.err

set -euo pipefail

cd /Odyssey/private/rfablet/Python/4dvarnet-fm-opencode/4dvarnet-fm-obs-density-gen

export PATH="/Odyssey/private/rfablet/miniforge3/envs/fdv-monai-proto/bin:$PATH"
export CONDA_DEFAULT_ENV=fdv-monai-proto
export PYTHONUNBUFFERED=1
export MPLCONFIGDIR="/tmp/mplconfig_${SLURM_JOB_ID}"

mkdir -p "$MPLCONFIGDIR" sbatch_logs

TEST_CACHE="/Odyssey/private/rfablet/Python/4dvarnet-fm-opencode/experiments/l96_datasets_obsj2_int100_nwin200.pt"

EXPERIMENT="FDV2_grad_state_monai_l96_Stier_priorresidual_detachvarcost"
echo "============================================"
echo " L96 FDV2(monai) grad+state, S-tier, prior_residual + detach_var_cost_grad=true | $(date)"
echo "  Experiment: ${EXPERIMENT}"
echo "  Same as FDV2_grad_state_monai_l96_Stier_priorresidual_nonzeroinit"
echo "  (job 53542), plus ONE change: detach_var_cost_grad=true."
echo "  Isolates 'does the solver benefit from consuming the TRUE"
echo "  differentiated-prior gradient' from 'does jointly training the"
echo "  prior through the per-iteration double-backward help' -- the fed"
echo "  signal is IDENTICAL to 53542's (still the real gradient, Jacobian"
echo "  included), only create_graph=False now on that autograd.grad call,"
echo "  so prior_unet trains ONLY via the aux prior_cost loss, not the"
echo "  deep double-backward through all 10 unrolled iterations."
echo "  trainable_prior_weight=false (prior_weight has no other gradient"
echo "  source than this now-severed pathway -- would be permanently stuck"
echo "  otherwise)."
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
