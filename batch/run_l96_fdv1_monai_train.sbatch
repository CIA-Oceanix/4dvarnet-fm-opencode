#!/bin/bash
#SBATCH --account=odyssey
#SBATCH --partition=Odyssey_GPU
#SBATCH --gres=gpu:rtx8000:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=24:00:00
#SBATCH --job-name=l96-fdv1-monai
#SBATCH --output=sbatch_logs/l96_fdv1_monai_train_%j.out
#SBATCH --error=sbatch_logs/l96_fdv1_monai_train_%j.err

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

EXPERIMENT="FDV1_unrolled_monai_unet_l96"
echo "============================================"
echo " L96 FDV1: unrolled 4DVarNet-style solver (update_input=obs+state), MonaiUNet1D backbone | $(date)"
echo "  Experiment: ${EXPERIMENT}"
echo "  Config: 24D state, 400 epochs, single-stage, deterministic (no ensemble)"
echo "  Backbone-only swap vs FDV1_unrolled_unet_l96: self.unet is MONAI's"
echo "  DiffusionModelUNet (models/monai_unet_adapter.py::MonaiUNet1D) instead"
echo "  of UNet1D -- identical data pipeline, no normalization (isolates the"
echo "  backbone variable, see PLAN.md / CHANGELOG for the follow-up decision"
echo "  on bundling per-channel normalization)."
echo "  GPU: $(nvidia-smi --query-gpu=name --format=csv -i 0 2>/dev/null | tail -1)"
echo "============================================"
echo

rm -rf "experiments/${EXPERIMENT}"
python train.py \
  --config-name experiment/${EXPERIMENT} \
  hydra.run.dir=. \
  hydra.output_subdir=null \
  ++data.test_cache=${TEST_CACHE}

echo ""
echo " Done: ${EXPERIMENT} at $(date)"
echo ""
