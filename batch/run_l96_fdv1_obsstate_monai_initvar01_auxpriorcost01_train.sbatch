#!/bin/bash
#SBATCH --account=odyssey
#SBATCH --partition=Odyssey_GPU
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=24:00:00
#SBATCH --job-name=l96-fdv1-obsstate-initvar-auxpc01
#SBATCH --output=sbatch_logs/l96_fdv1_obsstate_monai_initvar01_auxpriorcost01_train_%j.out
#SBATCH --error=sbatch_logs/l96_fdv1_obsstate_monai_initvar01_auxpriorcost01_train_%j.err

set -euo pipefail

cd /Odyssey/private/rfablet/Python/4dvarnet-fm-opencode/4dvarnet-fm-obs-density-gen

export PATH="/Odyssey/private/rfablet/miniforge3/envs/fdv-monai-proto/bin:$PATH"
export CONDA_DEFAULT_ENV=fdv-monai-proto
export PYTHONUNBUFFERED=1
export MPLCONFIGDIR="/tmp/mplconfig_${SLURM_JOB_ID}"

mkdir -p "$MPLCONFIGDIR" sbatch_logs

TEST_CACHE="/Odyssey/private/rfablet/Python/4dvarnet-fm-opencode/experiments/l96_datasets_obsj2_int100_nwin200.pt"

EXPERIMENT="FDV1_obsstate_monai_l96_initvar01_auxpriorcost01"
echo "============================================"
echo " L96 ABLATION: FDV1 (obs+state) + random init + aux_var_cost_weight/10 | $(date)"
echo "  Experiment: ${EXPERIMENT}"
echo "  Combines two FDV1 ablations:"
echo "  1. init_state_var=0.1 (random x_0 ~ N(0,0.1), not the usual x_0=0) --"
echo "     job 53288 confirmed this alone does NOT hurt FDV1's convergence"
echo "     (tracked/beat plain FDV1's own pace)."
echo "  2. aux_var_cost_weight=0.01, i.e. 1/10th the weight used in the"
echo "     killed job 53287 (aux_var_cost_weight=0.1), which plateaued at"
echo "     val_loss ~1.22-1.32 by epoch 104-122 vs plain FDV1's own"
echo "     ~0.86-0.95 at the same epochs -- a real degradation directly"
echo "     attributable to the aux prior-consistency loss term."
echo "  Tests whether a 10x lighter aux weight still degrades FDV1, combined"
echo "  with the random init already shown harmless. update_input stays"
echo "  exactly obs+state throughout."
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
