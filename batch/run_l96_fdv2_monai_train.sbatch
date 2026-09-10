#!/bin/bash
#SBATCH --account=odyssey
#SBATCH --partition=Odyssey_GPU
#SBATCH --gres=gpu:h100:1
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
echo "  Was H200-only pre-PR#172 (grad+state's create_graph=True double-backward"
echo "  OOM'd on RTX8000/H100 without gradient checkpointing). PR #172's"
echo "  checkpointing removed that N_outer-linear activation graph; job 52672"
echo "  (the fixed-prior_weight FDV2_grad_state_monai_l96_fixedw sibling config,"
echo "  identical architecture/memory footprint) ran 190 epochs on an H100"
echo "  batch_size=16 with no OOM (it died from a NaN loss divergence, now"
echo "  fixed by the models/fourdvarnet.py::_normalize_channels ceiling clamp"
echo "  -- see CHANGELOG.md 2026-09-10), so H100 is confirmed sufficient here too."
echo "  GPU: $(nvidia-smi --query-gpu=name --format=csv -i 0 2>/dev/null | tail -1)"
echo "============================================"
echo

python train.py \
  --config-name experiment/${EXPERIMENT} \
  hydra.run.dir=. \
  hydra.output_subdir=null \
  ++data.test_cache=${TEST_CACHE}

echo ""
echo " Training done: ${EXPERIMENT} at $(date)"
echo "============================================"

echo ""
echo " Verifying checkpoint/config persistence..."
ls -la "experiments/${EXPERIMENT}/resolved_config.yaml" "experiments/${EXPERIMENT}/checkpoints/"

echo ""
echo " Running full neural_eval.json evaluation (via evaluation/neural_inference.py's"
echo " model_factory path -- required for unet_backbone=monai). --n-outer 10 is"
echo " REQUIRED here: eval_neural_l96.py's default (1) is correct for tau0-only CFM"
echo " but silently produces garbage for FourDVarNetSolver's zero-init N_outer=10"
echo " unrolled refinement (see CHANGELOG.md -- this exact trap already bit"
echo " FDV1_unrolled_monai_unet_l96 twice)."
python eval_neural_l96.py \
  --checkpoint experiments/${EXPERIMENT}/checkpoints/stage1_best.ckpt \
  --config experiments/${EXPERIMENT}/resolved_config.yaml \
  --dataset ${TEST_CACHE} \
  --num-windows 200 \
  --batch-size 16 \
  --n-outer 10 \
  --output experiments/${EXPERIMENT}/neural_eval.json

echo ""
echo " Done: ${EXPERIMENT} at $(date)"
echo "============================================"
echo ""
