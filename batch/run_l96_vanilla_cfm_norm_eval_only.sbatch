#!/bin/bash
#SBATCH --account=odyssey
#SBATCH --partition=Odyssey_GPU
#SBATCH --gres=gpu:rtx8000:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=00:30:00
#SBATCH --job-name=l96-cfm-norm-eval
#SBATCH --output=sbatch_logs/l96_neural_%j.out
#SBATCH --error=sbatch_logs/l96_neural_%j.err

set -euo pipefail

cd /Odyssey/private/rfablet/Python/4dvarnet-fm-opencode/4dvarnet-fm-l96-eval-config-persist

# EXP is passed in via --export=EXP=<name> at submission time. Both
# L2b_vanilla_cfm_s0s1_norm and L2b_monai_vanilla_cfm_s0s1_norm were trained
# correctly but their sbatch scripts pointed the eval step at checkpoints/
# stage1.pt (the plain torch.save(model.state_dict()) format) instead of
# stage1_best.ckpt (the Lightning-wrapped {"state_dict":...} format
# evaluation.neural_inference.load_checkpoint actually requires) -- fixed in
# the training sbatch scripts; this reruns just the eval step against the
# already-trained checkpoint, no retraining needed.
: "${EXP:?Set EXP via --export=EXP=<experiment_name>}"

# ENV_NAME selects fdv (plain VanillaCFM) vs fdv-monai-proto (Monai variant).
ENV_NAME="${ENV_NAME:-fdv}"
export PATH="/Odyssey/private/rfablet/miniforge3/envs/${ENV_NAME}/bin:$PATH"
export CONDA_DEFAULT_ENV="${ENV_NAME}"
export PYTHONUNBUFFERED=1
export MPLCONFIGDIR="/tmp/mplconfig_${SLURM_JOB_ID}"
mkdir -p "$MPLCONFIGDIR"

echo "============================================"
echo " Eval-only rerun (corrected checkpoint path) | $(date)"
echo "  Experiment: ${EXP}  env: ${ENV_NAME}"
echo "============================================"

python eval_neural_l96.py \
  --checkpoint experiments/${EXP}/checkpoints/stage1_best.ckpt \
  --config experiments/${EXP}/resolved_config.yaml \
  --dataset experiments/l96_datasets_obsj2_int100_nwin200.pt \
  --num-windows 200 \
  --normalize-stats experiments/l96_norm_stats_obsj2.pt \
  --output experiments/${EXP}/neural_eval.json

echo ""
echo " Done: ${EXP} eval at $(date)"
echo "============================================"
