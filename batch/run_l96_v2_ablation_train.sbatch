#!/bin/bash
#SBATCH --account=odyssey
#SBATCH --partition=Odyssey_GPU
#SBATCH --gres=gpu:rtx8000:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=24:00:00
#SBATCH --job-name=l96-v2-ablation
#SBATCH --array=0-2
#SBATCH --output=sbatch_logs/l96_v2_ablation_train_%j_%a.out
#SBATCH --error=sbatch_logs/l96_v2_ablation_train_%j_%a.err

set -euo pipefail

cd /Odyssey/private/rfablet/Python/4dvarnet-fm-opencode/4dvarnet-fm-cfm-v2v3

export PATH="/Odyssey/private/rfablet/miniforge3/envs/fdv/bin:$PATH"
export CONDA_DEFAULT_ENV=fdv
export PYTHONUNBUFFERED=1
export MPLCONFIGDIR="/tmp/mplconfig_${SLURM_JOB_ID}"

mkdir -p "$MPLCONFIGDIR"

TEST_CACHE="/Odyssey/private/rfablet/Python/4dvarnet-fm-opencode/experiments/l96_datasets_obsj2_int100_nwin200.pt"
TEST_CACHE_OPTS="++data.test_cache=${TEST_CACHE}"

case "${SLURM_ARRAY_TASK_ID}" in
  0)
    EXPERIMENT="V2_tweedie_cfm_l96_rerun"
    DESC="rerun (K_inner=5, sigma_prior=0.5) — sanity check the Group A fix didn't perturb training"
    EXTRA_OPTS=""
    ;;
  1)
    EXPERIMENT="V2_tweedie_cfm_l96_s0p2"
    DESC="sigma_prior=0.2 ablation (#7)"
    EXTRA_OPTS=""
    ;;
  2)
    EXPERIMENT="V2_tweedie_cfm_l96_kinner1"
    DESC="K_inner=1 ablation (#4)"
    EXTRA_OPTS=""
    ;;
  *)
    echo "Invalid task ID: ${SLURM_ARRAY_TASK_ID}"
    exit 1
    ;;
esac

echo "============================================"
echo " L96 V2 TweedieCFM Ablation: ${EXPERIMENT} | $(date)"
echo "  Task: ${SLURM_ARRAY_TASK_ID} / ${DESC}"
echo "  GPU: $(nvidia-smi --query-gpu=name --format=csv -i 0 2>/dev/null | tail -1)"
echo "============================================"
echo

rm -rf "experiments/${EXPERIMENT}"
python train.py \
  --config-name experiment/${EXPERIMENT} \
  hydra.run.dir=. \
  hydra.output_subdir=null \
  ${TEST_CACHE_OPTS} \
  ${EXTRA_OPTS}

echo ""
echo " Done: ${EXPERIMENT} at $(date)"
echo ""
