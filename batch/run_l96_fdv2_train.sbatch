#!/bin/bash
#SBATCH --account=odyssey
#SBATCH --partition=Odyssey_GPU
#SBATCH --gres=gpu:rtx8000:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=24:00:00
#SBATCH --job-name=l96-fdv2
#SBATCH --output=sbatch_logs/l96_fdv2_train_%j.out
#SBATCH --error=sbatch_logs/l96_fdv2_train_%j.err

set -euo pipefail

cd /Odyssey/private/rfablet/Python/4dvarnet-fm-opencode/4dvarnet-fm-fdv2

export PATH="/Odyssey/private/rfablet/miniforge3/envs/fdv/bin:$PATH"
export CONDA_DEFAULT_ENV=fdv
export PYTHONUNBUFFERED=1
export MPLCONFIGDIR="/tmp/mplconfig_${SLURM_JOB_ID}"

mkdir -p "$MPLCONFIGDIR" sbatch_logs

# Reuse the canonical 200-window S0/S1 test splits instead of regenerating
# them (slow per-window path). Absolute path to the master worktree cache
# (shared filesystem; every worktree sees it).
TEST_CACHE="/Odyssey/private/rfablet/Python/4dvarnet-fm-opencode/experiments/l96_datasets_obsj2_int100_nwin200.pt"

EXPERIMENT="FDV2_grad_state_l96"
echo "============================================"
echo " L96 FDV2: unrolled 4DVarNet-style solver (update_input=grad+state) | $(date)"
echo "  Experiment: ${EXPERIMENT}"
echo "  Config: 24D state, 400 epochs, single-stage, deterministic (no ensemble)"
echo "  Gradient-conditioned mode: real autograd.grad of prior_cost+obs_cost"
echo "  w.r.t. state each iteration (ported from ocean4dvarnet's GradSolver),"
echo "  concatenated with state -- 2 forwards + 1 backward per iteration,"
echo "  vs. obs+state's single forward."
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
