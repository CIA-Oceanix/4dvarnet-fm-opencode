#!/bin/bash
#SBATCH --account=odyssey
#SBATCH --partition=Odyssey_GPU
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=24:00:00
#SBATCH --job-name=l96-fdv2-gradsplit-Stier
#SBATCH --output=sbatch_logs/l96_fdv2_gradsplit_state_monai_Stier_train_%j.out
#SBATCH --error=sbatch_logs/l96_fdv2_gradsplit_state_monai_Stier_train_%j.err

set -euo pipefail

cd /Odyssey/private/rfablet/Python/4dvarnet-fm-opencode/4dvarnet-fm-obs-density-gen

export PATH="/Odyssey/private/rfablet/miniforge3/envs/fdv-monai-proto/bin:$PATH"
export CONDA_DEFAULT_ENV=fdv-monai-proto
export PYTHONUNBUFFERED=1
export MPLCONFIGDIR="/tmp/mplconfig_${SLURM_JOB_ID}"

mkdir -p "$MPLCONFIGDIR" sbatch_logs

TEST_CACHE="/Odyssey/private/rfablet/Python/4dvarnet-fm-opencode/experiments/l96_datasets_obsj2_int100_nwin200.pt"

EXPERIMENT="FDV2_gradsplit_state_monai_l96_Stier"
echo "============================================"
echo " L96 FDV2(monai) NEW gradsplit+state, S-tier + random init + aux loss 0.01 | $(date)"
echo "  Experiment: ${EXPERIMENT}"
echo "  New update_input mode: same two-residual-plus-state input shape as"
echo "  subgrad+state (g_obs, g_prior, x), but g_obs/g_prior are each a real,"
echo "  SEPARATE torch.autograd.grad of their own cost term"
echo "  (obs_weight*obs_cost(x,obs) / prior_weight*prior_cost(x)), not"
echo "  grad-only/grad+state's single combined gradient of the summed"
echo "  var_cost, and not subgrad+state's cheap hand-written proxy residuals."
echo "  g_prior is normalized/soft-clipped (dense, real gradient of an"
echo "  unbounded cost term); g_obs is NOT (architecturally sparse, masked to"
echo "  zero outside observation times, same normalization-dilution"
echo "  vulnerability as subgrad+state's own g_obs proxy regardless of being"
echo "  a true gradient here). hidden_channels=[32,64,128],"
echo "  monai_num_res_blocks=1 (S tier, 1,055,544 main-solver params),"
echo "  init_state_var=0.1, aux_var_cost_weight=0.01 -- same combination"
echo "  that kept grad+state's prior_weight stable (jobs 53428) and let"
echo "  subgrad+state match/beat FDV1's own best run at this tier (job"
echo "  53429)."
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
