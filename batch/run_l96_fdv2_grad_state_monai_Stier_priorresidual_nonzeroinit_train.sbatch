#!/bin/bash
#SBATCH --account=odyssey
#SBATCH --partition=Odyssey_GPU
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=24:00:00
#SBATCH --job-name=l96-fdv2-grad-Stier-priorres-nzinit
#SBATCH --output=sbatch_logs/l96_fdv2_grad_state_monai_Stier_priorresidual_nonzeroinit_train_%j.out
#SBATCH --error=sbatch_logs/l96_fdv2_grad_state_monai_Stier_priorresidual_nonzeroinit_train_%j.err

set -euo pipefail

cd /Odyssey/private/rfablet/Python/4dvarnet-fm-opencode/4dvarnet-fm-obs-density-gen

export PATH="/Odyssey/private/rfablet/miniforge3/envs/fdv-monai-proto/bin:$PATH"
export CONDA_DEFAULT_ENV=fdv-monai-proto
export PYTHONUNBUFFERED=1
export MPLCONFIGDIR="/tmp/mplconfig_${SLURM_JOB_ID}"

mkdir -p "$MPLCONFIGDIR" sbatch_logs

TEST_CACHE="/Odyssey/private/rfablet/Python/4dvarnet-fm-opencode/experiments/l96_datasets_obsj2_int100_nwin200.pt"

EXPERIMENT="FDV2_grad_state_monai_l96_Stier_priorresidual_nonzeroinit"
echo "============================================"
echo " L96 FDV2(monai) grad+state, S-tier, prior_residual=true + prior_output_init_std=0.1 | $(date)"
echo "  Experiment: ${EXPERIMENT}"
echo "  Full N_outer=10 (no truncated BPTT), trainable_prior_weight=true,"
echo "  prior_residual=true (Phi(x)=x+prior_unet(x,tau)), prior_dropout=0.0"
echo "  (decoupled from main solver's dropout=0.1), plus the new fix:"
echo "  prior_output_init_std=0.1 -- overrides prior_unet's final output"
echo "  conv's MONAI zero_module() init (exact 0) with N(0,0.1)."
echo "  Motivation: a checkpoint diagnostic on job 53509 (the tbptt2x5"
echo "  variant, 300+ epochs trained) found prior_unet's final conv weight"
echo "  STILL bit-for-bit 0.0 -- f(x) identically zero for every input the"
echo "  whole time. Root cause: prior_residual=true makes"
echo "  prior_cost(x)=||x-Phi(x)||^2 collapse to ||f(x)||^2 (x cancels"
echo "  algebraically), a pure self-penalty on the network's own output"
echo "  with an exact critical point at f(x)=0 -- since f(x) and its"
echo "  Jacobian both start at exactly 0 (zero_module), EVERY gradient path"
echo "  that could move the final layer away from 0 is provably also 0"
echo "  there (aux prior_cost loss, per-iteration g_prior, AND the deep"
echo "  double-backward path from the outer supervised loss) -- a"
echo "  permanent structural fixed point, confirmed empirically, not a"
echo "  slow-training issue. prior_output_init_std=0.1 breaks it: f(x) is"
echo "  then nonzero from step 1, restoring a real gradient."
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
