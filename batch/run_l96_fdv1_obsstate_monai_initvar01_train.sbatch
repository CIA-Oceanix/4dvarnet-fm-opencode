#!/bin/bash
#SBATCH --account=odyssey
#SBATCH --partition=Odyssey_GPU
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=24:00:00
#SBATCH --job-name=l96-fdv1-obsstate-initvar01
#SBATCH --output=sbatch_logs/l96_fdv1_obsstate_monai_initvar01_train_%j.out
#SBATCH --error=sbatch_logs/l96_fdv1_obsstate_monai_initvar01_train_%j.err

set -euo pipefail

cd /Odyssey/private/rfablet/Python/4dvarnet-fm-opencode/4dvarnet-fm-obs-density-gen

export PATH="/Odyssey/private/rfablet/miniforge3/envs/fdv-monai-proto/bin:$PATH"
export CONDA_DEFAULT_ENV=fdv-monai-proto
export PYTHONUNBUFFERED=1
export MPLCONFIGDIR="/tmp/mplconfig_${SLURM_JOB_ID}"

mkdir -p "$MPLCONFIGDIR" sbatch_logs

TEST_CACHE="/Odyssey/private/rfablet/Python/4dvarnet-fm-opencode/experiments/l96_datasets_obsj2_int100_nwin200.pt"

EXPERIMENT="FDV1_obsstate_monai_l96_initvar01"
echo "============================================"
echo " L96 ABLATION: FDV1 (obs+state) with a random, non-zero initial condition | $(date)"
echo "  Experiment: ${EXPERIMENT}"
echo "  Every other FDV config (FDV1 and FDV2 alike) starts the N_outer=10"
echo "  unroll from x_0=0 (all-zeros). This run instead samples"
echo "  x_0 ~ N(0, init_state_var=0.1) -- a VARIANCE, not a std (std~0.316"
echo "  in this normalized state space) -- fresh on every forward() call."
echo "  update_input stays exactly obs+state, everything else identical to"
echo "  FDV1_unrolled_monai_unet_l96. Sanity/robustness check: does FDV1's"
echo "  own healthy fast-Y reconstruction (variance ratio ~0.93, correlation"
echo "  ~0.95) hold up starting from a random init, or does it depend on"
echo "  the deterministic zero start in some way."
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
