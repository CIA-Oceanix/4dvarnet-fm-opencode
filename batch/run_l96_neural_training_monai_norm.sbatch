#!/bin/bash
#SBATCH --account=odyssey
#SBATCH --partition=Odyssey_GPU
#SBATCH --gres=gpu:rtx8000:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=24:00:00
#SBATCH --job-name=l96-monai-norm
#SBATCH --output=sbatch_logs/l96_neural_%j.out
#SBATCH --error=sbatch_logs/l96_neural_%j.err

set -euo pipefail

cd /Odyssey/private/rfablet/Python/4dvarnet-fm-opencode/4dvarnet-fm-monai-unet-norm

export PATH="/Odyssey/private/rfablet/miniforge3/envs/fdv-monai-proto/bin:$PATH"
export CONDA_DEFAULT_ENV=fdv-monai-proto
export PYTHONUNBUFFERED=1
export MPLCONFIGDIR="/tmp/mplconfig_${SLURM_JOB_ID}"
mkdir -p "$MPLCONFIGDIR"

EXP=L1b_monai_unet_s0s1_norm

echo "============================================"
echo " L96 Neural Training (MonaiDirectUNet, data.normalize=true) | $(date)"
echo "  Experiment: ${EXP} -- MonaiUNet1D-backed analogue of"
echo "  L1b_direct_unet_s0s1_norm (same lr/epochs/normalization), swapping"
echo "  models.unet.UNet1D for MONAI's DiffusionModelUNet (FiLM timestep"
echo "  conditioning) to see if it avoids L1b/DirectUNet's collapse under"
echo "  normalization (see reports/l96/outputs/l96_normalization_ablation.md)."
echo "  GPU: $(nvidia-smi --query-gpu=name --format=csv -i 0 2>/dev/null | tail -1)"
echo "============================================"
echo ""

rm -rf "experiments/${EXP}"

python train.py \
  --config-name experiment/${EXP} \
  hydra.run.dir=. \
  hydra.output_subdir=null

echo ""
echo " Training done: ${EXP} at $(date)"
echo "============================================"

echo ""
echo " Running full neural_eval.json evaluation (matches L1b's report schema)..."
python eval_monai_l96.py \
  --checkpoint experiments/${EXP}/checkpoints/stage1.pt \
  --config config/experiment/${EXP}.yaml \
  --dataset experiments/l96_datasets_obsj2_int100_nwin200.pt \
  --num-windows 200 \
  --output experiments/${EXP}/neural_eval.json

echo ""
echo " Done: ${EXP} at $(date)"
echo "============================================"
