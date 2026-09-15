#!/bin/bash
#SBATCH --account=odyssey
#SBATCH --partition=Odyssey_GPU
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=24:00:00
#SBATCH --job-name=l96-fdv1cfm-subgradxtau
#SBATCH --output=sbatch_logs/l96_fdv1cfm_subgradxtau_train_%j.out
#SBATCH --error=sbatch_logs/l96_fdv1cfm_subgradxtau_train_%j.err

set -euo pipefail

cd /Odyssey/private/rfablet/Python/4dvarnet-fm-opencode/4dvarnet-fm-obs-density-gen

export PATH="/Odyssey/private/rfablet/miniforge3/envs/fdv/bin:$PATH"
export CONDA_DEFAULT_ENV=fdv
export PYTHONUNBUFFERED=1
export MPLCONFIGDIR="/tmp/mplconfig_${SLURM_JOB_ID}"

mkdir -p "$MPLCONFIGDIR" sbatch_logs

TEST_CACHE="/Odyssey/private/rfablet/Python/4dvarnet-fm-opencode/experiments/l96_datasets_obsj2_int100_nwin200.pt"

EXPERIMENT="FDV1CFM_predict_state_l96_subgradxtau"
echo "============================================"
echo " L96 FDV1-CFM: subgrad+state+xtau update_input (new) | $(date)"
echo "  Experiment: ${EXPERIMENT}"
echo "  Same as FDV1CFM_predict_state_l96 (M-tier UNet1D, K_inner=5,"
echo "  N_outer=10) but update_input=subgrad+state+xtau instead of"
echo "  obs+state -- extends subgrad+state's [obs-x, x-Phi(x), x] with a"
echo "  4th channel g_flow = x - beta_tau*x_tau, the un-prefactored core of"
echo "  reports/notes_4dvarnet_fm.pdf's 'energy-informed parameterization'"
echo "  (Sec 3.3) third residual term for the CFM inner-refinement operator."
echo "  x_tau = the outer flow-time's own conditioning value, held fixed"
echo "  across the whole K_inner unroll; beta_tau = interpolant.beta(tau)"
echo "  using the OUTER tau, not the inner iteration's own tau_k."
echo "  NOTE: FourDVarNetPredictStateCFM has no monai backbone option today"
echo "  -- this run uses UNet1D (M-tier), same as the existing FDV1CFM"
echo "  baseline, NOT comparable to this investigation's S-tier MonaiUNet1D"
echo "  runs -- only to FDV1CFM_predict_state_l96 itself."
echo "  GPU: $(nvidia-smi --query-gpu=name --format=csv -i 0 2>/dev/null | tail -1)"
echo "============================================"
echo

python train.py \
  --config-name experiment/${EXPERIMENT} \
  hydra.run.dir=. \
  hydra.output_subdir=null \
  ++data.test_cache=${TEST_CACHE}

echo ""
echo " Done: ${EXPERIMENT} at $(date)"
echo ""
