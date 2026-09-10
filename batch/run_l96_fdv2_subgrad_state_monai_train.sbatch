#!/bin/bash
#SBATCH --account=odyssey
#SBATCH --partition=Odyssey_GPU
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=24:00:00
#SBATCH --job-name=l96-fdv2-subgrad-monai
#SBATCH --output=sbatch_logs/l96_fdv2_subgrad_state_monai_train_%j.out
#SBATCH --error=sbatch_logs/l96_fdv2_subgrad_state_monai_train_%j.err

set -euo pipefail

# Deliberately run from THIS worktree, not 4dvarnet-fm-fdv-monai -- that
# worktree has an active training job (FDV2_grad_state_monai_l96, job
# 52926) and per this session's standing rule it is never touched, even
# just to `cd` into for an unrelated launch.
cd /Odyssey/private/rfablet/Python/4dvarnet-fm-opencode/4dvarnet-fm-obs-density-gen

export PATH="/Odyssey/private/rfablet/miniforge3/envs/fdv-monai-proto/bin:$PATH"
export CONDA_DEFAULT_ENV=fdv-monai-proto
export PYTHONUNBUFFERED=1
export MPLCONFIGDIR="/tmp/mplconfig_${SLURM_JOB_ID}"

mkdir -p "$MPLCONFIGDIR" sbatch_logs

# Reuse the canonical 200-window S0/S1 test splits instead of regenerating
# them (slow per-window path). Absolute path to the master worktree cache
# (shared filesystem; every worktree sees it).
TEST_CACHE="/Odyssey/private/rfablet/Python/4dvarnet-fm-opencode/experiments/l96_datasets_obsj2_int100_nwin200.pt"

EXPERIMENT="FDV2_subgrad_state_monai_l96"
echo "============================================"
echo " L96 FDV2(monai): subgrad+state (two-residual proxy gradient), MonaiUNet1D backbone | $(date)"
echo "  Experiment: ${EXPERIMENT}"
echo "  Exact copy of FDV2_grad_state_monai_l96 (job 52926) with only"
echo "  update_input swapped to subgrad+state -- never calls"
echo "  torch.autograd.grad/create_graph (obs residual + prior-autoencoder"
echo "  residual proxy instead), architecturally immune to the double-"
echo "  backward instability grad+state needed a stability recipe for."
echo "  Previously trained only with the plain UNet1D backbone (job 52358,"
echo "  2026-09-07), reached 37/400 epochs training cleanly before an"
echo "  unrelated shared-env torch corruption killed it; never resumed,"
echo "  never ported to monai, never benchmarked. This is that port."
echo "  GPU: $(nvidia-smi --query-gpu=name --format=csv -i 0 2>/dev/null | tail -1)"
echo "============================================"
echo

python train.py \
  --config-name experiment/${EXPERIMENT} \
  hydra.run.dir=. \
  hydra.output_subdir=null \
  ++data.test_cache=${TEST_CACHE}

echo ""
echo " Done: ${EXPERIMENT} at $(date)"
echo ""
