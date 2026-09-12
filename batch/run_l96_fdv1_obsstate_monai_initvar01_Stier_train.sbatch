#!/bin/bash
#SBATCH --account=odyssey
#SBATCH --partition=Odyssey_GPU
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=24:00:00
#SBATCH --job-name=l96-fdv1-obsstate-initvar-Stier
#SBATCH --output=sbatch_logs/l96_fdv1_obsstate_monai_initvar01_Stier_train_%j.out
#SBATCH --error=sbatch_logs/l96_fdv1_obsstate_monai_initvar01_Stier_train_%j.err

set -euo pipefail

cd /Odyssey/private/rfablet/Python/4dvarnet-fm-opencode/4dvarnet-fm-obs-density-gen

export PATH="/Odyssey/private/rfablet/miniforge3/envs/fdv-monai-proto/bin:$PATH"
export CONDA_DEFAULT_ENV=fdv-monai-proto
export PYTHONUNBUFFERED=1
export MPLCONFIGDIR="/tmp/mplconfig_${SLURM_JOB_ID}"

mkdir -p "$MPLCONFIGDIR" sbatch_logs

TEST_CACHE="/Odyssey/private/rfablet/Python/4dvarnet-fm-opencode/experiments/l96_datasets_obsj2_int100_nwin200.pt"

EXPERIMENT="FDV1_obsstate_monai_l96_initvar01_Stier"
echo "============================================"
echo " L96 FDV1 (obs+state) + random init, on the S-tier MonaiUNet1D | $(date)"
echo "  Experiment: ${EXPERIMENT}"
echo "  Same as FDV1_obsstate_monai_l96_initvar01 (job 53288 -- random init"
echo "  confirmed harmless to FDV1's convergence) but hidden_channels="
echo "  [32,64,128], monai_num_res_blocks=1 -- the 'S' capacity tier"
echo "  (1,055,544 params) instead of the default 'M' tier"
echo "  (hidden_channels=[64,128,256], monai_num_res_blocks=2, 5,889,048"
echo "  params) every other FDV1/FDV2 config uses. First time"
echo "  monai_num_res_blocks (new FourDVarNetSolver param, default 2,"
echo "  unchanged) is actually exercised at a non-default value."
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
