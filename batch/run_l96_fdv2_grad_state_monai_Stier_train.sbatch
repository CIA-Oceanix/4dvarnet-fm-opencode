#!/bin/bash
#SBATCH --account=odyssey
#SBATCH --partition=Odyssey_GPU
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=24:00:00
#SBATCH --job-name=l96-fdv2-grad-Stier
#SBATCH --output=sbatch_logs/l96_fdv2_grad_state_monai_Stier_train_%j.out
#SBATCH --error=sbatch_logs/l96_fdv2_grad_state_monai_Stier_train_%j.err

set -euo pipefail

cd /Odyssey/private/rfablet/Python/4dvarnet-fm-opencode/4dvarnet-fm-obs-density-gen

export PATH="/Odyssey/private/rfablet/miniforge3/envs/fdv-monai-proto/bin:$PATH"
export CONDA_DEFAULT_ENV=fdv-monai-proto
export PYTHONUNBUFFERED=1
export MPLCONFIGDIR="/tmp/mplconfig_${SLURM_JOB_ID}"

mkdir -p "$MPLCONFIGDIR" sbatch_logs

TEST_CACHE="/Odyssey/private/rfablet/Python/4dvarnet-fm-opencode/experiments/l96_datasets_obsj2_int100_nwin200.pt"

EXPERIMENT="FDV2_grad_state_monai_l96_Stier"
echo "============================================"
echo " L96 FDV2(monai) grad+state, S-tier + random init + corrected aux loss 0.01 | $(date)"
echo "  Experiment: ${EXPERIMENT}"
echo "  First grad+state retrain since PR #196 landed the normalization/"
echo "  soft-clip fix, the corrected aux-loss formula (no prior_weight/"
echo "  obs_cost mixed in), init_state_var, and monai_num_res_blocks."
echo "  Every earlier grad+state attempt this session (jobs 52926, 53104,"
echo "  53258) was at M-tier only. Rationale: FDV1's own obs+state"
echo "  architecture showed the aux-loss term severely degrading val_loss"
echo "  at M-tier (aux_var_cost_weight=0.01 -> ~0.38 permanent plateau,"
echo "  job 53289) but NOT at all at S-tier (job 53315, matched its own"
echo "  clean control exactly) -- this tests whether that same tier-"
echo "  dependence holds for grad+state's real-autograd-gradient update"
echo "  rule, on hidden_channels=[32,64,128]/monai_num_res_blocks=1"
echo "  (1,055,544 params) instead of the M tier's 5,889,048."
echo "  GPU: $(nvidia-smi --query-gpu=name --format=csv -i 0 2>/dev/null | tail -1)"
echo "============================================"
echo

python train.py \
  --config-name experiment/${EXPERIMENT} \
  hydra.run.dir=. \
  hydra.output_subdir=null \
  ++data.test_cache=${TEST_CACHE} \
  +fresh=true

echo ""
echo " Done: ${EXPERIMENT} at $(date)"
echo ""
