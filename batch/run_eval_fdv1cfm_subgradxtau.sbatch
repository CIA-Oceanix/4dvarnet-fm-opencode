#!/bin/bash
#SBATCH --account=odyssey
#SBATCH --partition=Odyssey_GPU
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=01:00:00
#SBATCH --job-name=eval-fdv1cfm-xtau
#SBATCH --output=sbatch_logs/eval_fdv1cfm_subgradxtau_%j.out
#SBATCH --error=sbatch_logs/eval_fdv1cfm_subgradxtau_%j.err

set -euo pipefail

cd /Odyssey/private/rfablet/Python/4dvarnet-fm-opencode/4dvarnet-fm-obs-density-gen

export PATH="/Odyssey/private/rfablet/miniforge3/envs/fdv-monai-proto/bin:$PATH"
export CONDA_DEFAULT_ENV=fdv-monai-proto
export PYTHONUNBUFFERED=1

mkdir -p sbatch_logs experiments/FDV1CFM_predict_state_l96_subgradxtau/eval

python eval_neural_l96.py \
  --checkpoint experiments/FDV1CFM_predict_state_l96_subgradxtau/checkpoints/stage1_best.ckpt \
  --config experiments/FDV1CFM_predict_state_l96_subgradxtau/resolved_config.yaml \
  --dataset /Odyssey/private/rfablet/Python/4dvarnet-fm-opencode/experiments/l96_datasets_obsj2_int100_nwin200.pt \
  --n-outer 10 --n-members 30 --batch-size 25 \
  --output experiments/FDV1CFM_predict_state_l96_subgradxtau/eval/neural_eval_results.json

echo "Done: eval_fdv1cfm_subgradxtau at $(date)"
