#!/bin/bash
#SBATCH --account=odyssey
#SBATCH --partition=Odyssey_GPU
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=24:00:00
#SBATCH --job-name=l96-fdv1-obsstate-Stier-auxpc01
#SBATCH --output=sbatch_logs/l96_fdv1_obsstate_monai_initvar01_Stier_auxpriorcost01_train_%j.out
#SBATCH --error=sbatch_logs/l96_fdv1_obsstate_monai_initvar01_Stier_auxpriorcost01_train_%j.err

set -euo pipefail

cd /Odyssey/private/rfablet/Python/4dvarnet-fm-opencode/4dvarnet-fm-obs-density-gen

export PATH="/Odyssey/private/rfablet/miniforge3/envs/fdv-monai-proto/bin:$PATH"
export CONDA_DEFAULT_ENV=fdv-monai-proto
export PYTHONUNBUFFERED=1
export MPLCONFIGDIR="/tmp/mplconfig_${SLURM_JOB_ID}"

mkdir -p "$MPLCONFIGDIR" sbatch_logs

TEST_CACHE="/Odyssey/private/rfablet/Python/4dvarnet-fm-opencode/experiments/l96_datasets_obsj2_int100_nwin200.pt"

EXPERIMENT="FDV1_obsstate_monai_l96_initvar01_Stier_auxpriorcost01"
echo "============================================"
echo " L96 ABLATION: FDV1 (obs+state), S-tier UNet + random init + aux prior-cost weight 0.01 | $(date)"
echo "  Experiment: ${EXPERIMENT}"
echo "  Base: FDV1_obsstate_monai_l96_initvar01_Stier (job 53292, S-tier"
echo "  MonaiUNet1D, 1,055,544 params, init_state_var=0.1) -- full eval"
echo "  confirmed this reproduces FDV1's healthy fast-Y reconstruction"
echo "  (variance ratio 0.975/0.976 vs FDV1 baseline's 0.926, correlation"
echo "  0.951/0.952 vs 0.948), and actually beats FDV1's own val_loss"
echo "  (0.197 vs 0.205)."
echo "  Adds: aux_var_cost_weight=0.01, the same weight used in the M-tier"
echo "  ablation (job 53289), which plateaued at val_loss ~0.39 by epoch"
echo "  323 -- a real, substantial, lasting degradation vs every aux-loss-"
echo "  free FDV1 variant (0.20-0.29 range). Tests whether the same"
echo "  degradation shows up at S-tier too, or is specific to M-tier's"
echo "  larger prior_unet/solver-unet combination."
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
