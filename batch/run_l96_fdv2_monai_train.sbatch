#!/bin/bash
#SBATCH --account=odyssey
#SBATCH --partition=Odyssey_GPU
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=24:00:00
#SBATCH --job-name=l96-fdv2-monai
#SBATCH --output=sbatch_logs/l96_fdv2_monai_train_%j.out
#SBATCH --error=sbatch_logs/l96_fdv2_monai_train_%j.err

set -euo pipefail

cd /Odyssey/private/rfablet/Python/4dvarnet-fm-opencode/4dvarnet-fm-fdv-monai

export PATH="/Odyssey/private/rfablet/miniforge3/envs/fdv-monai-proto/bin:$PATH"
export CONDA_DEFAULT_ENV=fdv-monai-proto
export PYTHONUNBUFFERED=1
export MPLCONFIGDIR="/tmp/mplconfig_${SLURM_JOB_ID}"

mkdir -p "$MPLCONFIGDIR" sbatch_logs

# Reuse the canonical 200-window S0/S1 test splits instead of regenerating
# them (slow per-window path). Absolute path to the master worktree cache
# (shared filesystem; every worktree sees it).
TEST_CACHE="/Odyssey/private/rfablet/Python/4dvarnet-fm-opencode/experiments/l96_datasets_obsj2_int100_nwin200.pt"

EXPERIMENT="FDV2_grad_state_monai_l96"
echo "============================================"
echo " L96 FDV2: unrolled 4DVarNet-style solver (update_input=grad+state), MonaiUNet1D backbone | $(date)"
echo "  Experiment: ${EXPERIMENT}"
echo "  Config: 24D state, 400 epochs, single-stage, deterministic (no ensemble)"
echo "  Backbone-only swap vs FDV2_grad_state_l96 (job 52305): both self.unet"
echo "  and self.prior_unet are MONAI's DiffusionModelUNet instead of UNet1D --"
echo "  identical trainable-prior_weight/aux_var_cost_weight/cosine-scheduler"
echo "  recipe otherwise. prior_unet is always called with tau=None (see"
echo "  models/fourdvarnet.py::_build_backbone_unet's docstring for the one"
echo "  known semantic gap vs UNet1D(time_emb_dim=0)'s literal no-conditioning)."
echo "  Requires an H200 (not the RTX8000 used by every other L96 sbatch job):"
echo "  grad+state needs create_graph=True for a double-backward chained across"
echo "  all N_outer=10 unrolled iterations, and MONAI's backbone has ~3x more"
echo "  parameters than UNet1D at matching hidden_channels -- confirmed CUDA OOM"
echo "  on an RTX8000 (46GB) at batch_size=16 (and even at batch_size=1), and"
echo "  ALSO on an H100 (93GB) at the real batch_size=16 (only fit on H100 at a"
echo "  reduced batch_size=8 smoke test -- do not be misled by that). Confirmed"
echo "  working on an H200 (141GB) at the real batch_size=16."
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
