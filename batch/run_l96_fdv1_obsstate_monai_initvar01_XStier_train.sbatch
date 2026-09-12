#!/bin/bash
#SBATCH --account=odyssey
#SBATCH --partition=Odyssey_GPU
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=24:00:00
#SBATCH --job-name=l96-fdv1-obsstate-XStier-clean
#SBATCH --output=sbatch_logs/l96_fdv1_obsstate_monai_initvar01_XStier_train_%j.out
#SBATCH --error=sbatch_logs/l96_fdv1_obsstate_monai_initvar01_XStier_train_%j.err

set -euo pipefail

cd /Odyssey/private/rfablet/Python/4dvarnet-fm-opencode/4dvarnet-fm-obs-density-gen

export PATH="/Odyssey/private/rfablet/miniforge3/envs/fdv-monai-proto/bin:$PATH"
export CONDA_DEFAULT_ENV=fdv-monai-proto
export PYTHONUNBUFFERED=1
export MPLCONFIGDIR="/tmp/mplconfig_${SLURM_JOB_ID}"

mkdir -p "$MPLCONFIGDIR" sbatch_logs

TEST_CACHE="/Odyssey/private/rfablet/Python/4dvarnet-fm-opencode/experiments/l96_datasets_obsj2_int100_nwin200.pt"

EXPERIMENT="FDV1_obsstate_monai_l96_initvar01_XStier"
echo "============================================"
echo " L96 CONTROL: FDV1 (obs+state), ~half-S-tier UNet + random init, NO aux loss | $(date)"
echo "  Experiment: ${EXPERIMENT}"
echo "  Clean control for FDV1_obsstate_monai_l96_initvar01_XStier_auxpriorcost01"
echo "  (job 53316): same hidden_channels=[24,48,96], monai_num_res_blocks=1"
echo "  (596,784 params, ~56.5% of S-tier), init_state_var=0.1, but"
echo "  aux_var_cost_weight=0.0 (default -- no prior_unet, no aux loss)."
echo "  Job 53316 is tracking meaningfully behind job 53315 (S-tier + same"
echo "  aux weight 0.01) at matched epochs -- this isolates whether that"
echo "  gap is the aux-loss degradation reappearing at this smaller"
echo "  capacity, or just this tier's own natural convergence pace."
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
