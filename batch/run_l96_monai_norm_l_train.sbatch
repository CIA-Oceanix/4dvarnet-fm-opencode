#!/bin/bash
#SBATCH --account=odyssey
#SBATCH --partition=Odyssey_GPU
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=24:00:00
#SBATCH --job-name=l96-monai-norm-l
#SBATCH --output=sbatch_logs/l96_neural_%j.out
#SBATCH --error=sbatch_logs/l96_neural_%j.err

set -euo pipefail

cd /Odyssey/private/rfablet/Python/4dvarnet-fm-opencode/4dvarnet-fm-l96-eval-config-persist

export PATH="/Odyssey/private/rfablet/miniforge3/envs/fdv-monai-proto/bin:$PATH"
export CONDA_DEFAULT_ENV=fdv-monai-proto
export PYTHONUNBUFFERED=1
export MPLCONFIGDIR="/tmp/mplconfig_${SLURM_JOB_ID}"
mkdir -p "$MPLCONFIGDIR"

EXP=L1b_monai_unet_s0s1_norm_l

echo "============================================"
echo " L96 Neural Training (MonaiDirectUNet, L tier, data.normalize=true) | $(date)"
echo "  Experiment: ${EXP} -- L tier ([128,256,512], num_res_blocks=2,"
echo "  23,476,248 params, ~4x the M tier). New reference-architecture"
echo "  candidate for the S+/M/L L96 case-study selection experiment -- S+"
echo "  tested 2026-09-08 (job 52529: S0 0.532/S1 0.531). See memory"
echo "  project_l96_monai_unet_complexity_tiers."
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
