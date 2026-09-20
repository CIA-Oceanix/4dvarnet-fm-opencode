#!/bin/bash
#SBATCH --account=odyssey
#SBATCH --partition=Odyssey_GPU
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=24:00:00
#SBATCH --job-name=l96-fdv2-subgrad-Stier
#SBATCH --output=sbatch_logs/l96_fdv2_subgrad_state_monai_Stier_train_%j.out
#SBATCH --error=sbatch_logs/l96_fdv2_subgrad_state_monai_Stier_train_%j.err

set -euo pipefail

cd /Odyssey/private/rfablet/Python/4dvarnet-fm-opencode/4dvarnet-fm-obs-density-gen

export PATH="/Odyssey/private/rfablet/miniforge3/envs/fdv-monai-proto/bin:$PATH"
export CONDA_DEFAULT_ENV=fdv-monai-proto
export PYTHONUNBUFFERED=1
export MPLCONFIGDIR="/tmp/mplconfig_${SLURM_JOB_ID}"

mkdir -p "$MPLCONFIGDIR" sbatch_logs

TEST_CACHE="/Odyssey/private/rfablet/Python/4dvarnet-fm-opencode/experiments/l96_datasets_obsj2_int100_nwin200.pt"

EXPERIMENT="FDV2_subgrad_state_monai_l96_Stier"
echo "============================================"
echo " L96 FDV2(monai) subgrad+state, S-tier + random init + corrected aux loss 0.01 | $(date)"
echo "  Experiment: ${EXPERIMENT}"
echo "  First subgrad+state retrain since PR #196 landed the normalization"
echo "  fix (g_obs/g_prior never normalized), the corrected aux-loss formula"
echo "  (no prior_weight/obs_cost mixed in), init_state_var, and"
echo "  monai_num_res_blocks. Every earlier subgrad+state attempt this"
echo "  session (jobs 53011, 53045, 53077, 53078, 53254, 53259) was at"
echo "  M-tier or the priorS+tbptt variant, never at this S tier."
echo "  Rationale: FDV1's own obs+state architecture showed the aux-loss"
echo "  term severely degrading val_loss at M-tier but not at all at S-tier"
echo "  -- this tests whether that same tier-dependence holds for"
echo "  subgrad+state's two-residual proxy-gradient update rule, on"
echo "  hidden_channels=[32,64,128]/monai_num_res_blocks=1 (1,055,544"
echo "  params) instead of the M tier's 5,889,048."
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
