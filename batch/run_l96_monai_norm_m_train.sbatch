#!/bin/bash
#SBATCH --account=odyssey
#SBATCH --partition=Odyssey_GPU
#SBATCH --gres=gpu:rtx8000:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=24:00:00
#SBATCH --job-name=l96-monai-norm-m
#SBATCH --output=sbatch_logs/l96_neural_%j.out
#SBATCH --error=sbatch_logs/l96_neural_%j.err

set -euo pipefail

cd /Odyssey/private/rfablet/Python/4dvarnet-fm-opencode/4dvarnet-fm-l96-eval-config-persist

export PATH="/Odyssey/private/rfablet/miniforge3/envs/fdv-monai-proto/bin:$PATH"
export CONDA_DEFAULT_ENV=fdv-monai-proto
export PYTHONUNBUFFERED=1
export MPLCONFIGDIR="/tmp/mplconfig_${SLURM_JOB_ID}"
mkdir -p "$MPLCONFIGDIR"

EXP=L1b_monai_unet_s0s1_norm

echo "============================================"
echo " L96 Neural Training (MonaiDirectUNet, M tier, data.normalize=true) | $(date)"
echo "  Experiment: ${EXP} -- M tier ([64,128,256], num_res_blocks=2,"
echo "  5,889,048 params). Fresh retrain (post-PR#171 codebase) for a clean"
echo "  apples-to-apples S+/M/L reference-architecture selection experiment"
echo "  for the L96 case study -- S+ tested 2026-09-08 (job 52529: S0"
echo "  0.532/S1 0.531). See memory project_l96_monai_unet_complexity_tiers."
echo "  GPU: $(nvidia-smi --query-gpu=name --format=csv -i 0 2>/dev/null | tail -1)"
echo "============================================"
echo ""


python train.py \
  --config-name experiment/${EXP} \
  hydra.run.dir=. \
  hydra.output_subdir=null

echo ""
echo " Training done: ${EXP} at $(date)"
echo "============================================"

echo ""
echo " Verifying checkpoint/config persistence..."
ls -la "experiments/${EXP}/resolved_config.yaml" "experiments/${EXP}/checkpoints/"

echo ""
echo " Running full neural_eval.json evaluation (matches L1b's report schema)..."
python eval_monai_l96.py \
  --checkpoint experiments/${EXP}/checkpoints/stage1.pt \
  --config experiments/${EXP}/resolved_config.yaml \
  --dataset experiments/l96_datasets_obsj2_int100_nwin200.pt \
  --num-windows 200 \
  --output experiments/${EXP}/neural_eval.json

echo ""
echo " Done: ${EXP} at $(date)"
echo "============================================"
