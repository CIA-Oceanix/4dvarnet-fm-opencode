#!/bin/bash
#SBATCH --account=odyssey
#SBATCH --partition=Odyssey_GPU
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=24:00:00
#SBATCH --job-name=l96-fdv1-obsstate-XStier-auxpc01
#SBATCH --output=sbatch_logs/l96_fdv1_obsstate_monai_initvar01_XStier_auxpriorcost01_train_%j.out
#SBATCH --error=sbatch_logs/l96_fdv1_obsstate_monai_initvar01_XStier_auxpriorcost01_train_%j.err

set -euo pipefail

cd /Odyssey/private/rfablet/Python/4dvarnet-fm-opencode/4dvarnet-fm-obs-density-gen

export PATH="/Odyssey/private/rfablet/miniforge3/envs/fdv-monai-proto/bin:$PATH"
export CONDA_DEFAULT_ENV=fdv-monai-proto
export PYTHONUNBUFFERED=1
export MPLCONFIGDIR="/tmp/mplconfig_${SLURM_JOB_ID}"

mkdir -p "$MPLCONFIGDIR" sbatch_logs

TEST_CACHE="/Odyssey/private/rfablet/Python/4dvarnet-fm-opencode/experiments/l96_datasets_obsj2_int100_nwin200.pt"

EXPERIMENT="FDV1_obsstate_monai_l96_initvar01_XStier_auxpriorcost01"
echo "============================================"
echo " L96 ABLATION: FDV1 (obs+state), ~half-S-tier UNet + random init + aux prior-cost 0.01 | $(date)"
echo "  Experiment: ${EXPERIMENT}"
echo "  hidden_channels=[24,48,96], monai_num_res_blocks=1,"
echo "  monai_norm_num_groups=8 -- 596,784 params, ~56.5% of the S tier's"
echo "  1,055,544 (closest achievable ratio to 'half of S' among candidates"
echo "  tried: [16,32,64]x1=267,944 (25%), [20,40,80]x1=416,124 (39%),"
echo "  [24,48,96]x1=596,784 (57%), [28,56,112]x1=809,924 (77%))."
echo "  init_state_var=0.1, aux_var_cost_weight=0.01 -- same as jobs 53289"
echo "  (M-tier: permanently plateaued at val_loss ~0.384, severe lasting"
echo "  degradation) and 53315 (S-tier: converged cleanly, well below"
echo "  53289's plateau by epoch ~50). Tests whether the aux-loss"
echo "  degradation keeps shrinking as capacity shrinks further, or there's"
echo "  a floor below which even a small network can't absorb the aux"
echo "  term's gradient interference either."
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
