#!/bin/bash
#SBATCH --account=odyssey
#SBATCH --partition=Odyssey_GPU
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=24:00:00
#SBATCH --job-name=l96-fdv2-grad-Stier-priorres-tbptt2x5
#SBATCH --output=sbatch_logs/l96_fdv2_grad_state_monai_Stier_priorresidual_tbptt2x5_train_%j.out
#SBATCH --error=sbatch_logs/l96_fdv2_grad_state_monai_Stier_priorresidual_tbptt2x5_train_%j.err

set -euo pipefail

cd /Odyssey/private/rfablet/Python/4dvarnet-fm-opencode/4dvarnet-fm-obs-density-gen

export PATH="/Odyssey/private/rfablet/miniforge3/envs/fdv-monai-proto/bin:$PATH"
export CONDA_DEFAULT_ENV=fdv-monai-proto
export PYTHONUNBUFFERED=1
export MPLCONFIGDIR="/tmp/mplconfig_${SLURM_JOB_ID}"

mkdir -p "$MPLCONFIGDIR" sbatch_logs

TEST_CACHE="/Odyssey/private/rfablet/Python/4dvarnet-fm-opencode/experiments/l96_datasets_obsj2_int100_nwin200.pt"

EXPERIMENT="FDV2_grad_state_monai_l96_Stier_priorresidual_tbptt2x5"
echo "============================================"
echo " L96 FDV2(monai) grad+state, S-tier, prior_residual=true, N_outer=10 as 2x5 truncated-BPTT | $(date)"
echo "  Experiment: ${EXPERIMENT}"
echo "  Same as FDV2_grad_state_monai_l96_Stier_priorresidual_N5 (job 53501:"
echo "  trainable_prior_weight=true, prior_residual=true, prior_dropout=0.0"
echo "  for prior_unet, dropout=0.1 for the main solver unet only), but with"
echo "  the full N_outer=10 unroll restored, split into tbptt_n_blocks=2"
echo "  blocks of tbptt_block_size=5 (graph detached at k=5) instead of a"
echo "  single N_outer=5 block -- isolates the double-backward-chain-length"
echo "  effect from the iteration-count effect."
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
