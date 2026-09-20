#!/bin/bash
#SBATCH --account=odyssey
#SBATCH --partition=Odyssey_GPU
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=24:00:00
#SBATCH --job-name=l96-fdv2-varcost
#SBATCH --output=sbatch_logs/l96_fdv2_subgrad_trueprior_monai_varcost_train_%j.out
#SBATCH --error=sbatch_logs/l96_fdv2_subgrad_trueprior_monai_varcost_train_%j.err

set -euo pipefail

cd /Odyssey/private/rfablet/Python/4dvarnet-fm-opencode/4dvarnet-fm-obs-density-gen

export PATH="/Odyssey/private/rfablet/miniforge3/envs/fdv-monai-proto/bin:$PATH"
export CONDA_DEFAULT_ENV=fdv-monai-proto
export PYTHONUNBUFFERED=1
export MPLCONFIGDIR="/tmp/mplconfig_${SLURM_JOB_ID}"

mkdir -p "$MPLCONFIGDIR" sbatch_logs

EXPERIMENT="FDV2_subgrad_trueprior_monai_l96_varcost"
echo "============================================"
echo " L96 FDV2(monai) subgrad+trueprior, weak-constraint-4DVar VAR-COST loss | $(date)"
echo "  Experiment: ${EXPERIMENT}"
echo "  Same architecture/update_input as job 53595 (subgrad+trueprior,"
echo "  no raw-state channel) -- ONLY the training loss differs: instead of"
echo "  MSE(x_final, true_state), trains on loss_type=var_cost:"
echo "  obs_cost/R_var + prior_cost/var_cost_Q_var, evaluated on the"
echo "  solver's OWN output, NO ground-truth supervision at all."
echo "  R_var=0.5/var_cost_Q_var=0.05 reuse evaluation/baselines.py's"
echo "  Weak4DVar class defaults exactly (its J_o/R_var, J_q/Q_var terms --"
echo "  Q_var's q[t]=x[t]-dynamics.step(x[t-1]) is the identical quantity"
echo "  this model's own g_prior/prior_cost measures), i.e. the dynamics"
echo "  are trusted 10x tighter than the raw observations per unit squared"
echo "  error, the same ratio the classical weak-4DVar baseline uses."
echo "  lr/gradient_clip_val unchanged from the sibling MSE config -- NOT"
echo "  independently re-tuned for this loss."
echo "  GPU: $(nvidia-smi --query-gpu=name --format=csv -i 0 2>/dev/null | tail -1)"
echo "============================================"
echo

python train.py \
  --config-name experiment/${EXPERIMENT} \
  hydra.run.dir=. \
  hydra.output_subdir=null \
  +fresh=true

echo ""
echo " Done: ${EXPERIMENT} at $(date)"
echo ""
