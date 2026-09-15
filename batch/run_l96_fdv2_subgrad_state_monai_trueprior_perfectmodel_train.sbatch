#!/bin/bash
#SBATCH --account=odyssey
#SBATCH --partition=Odyssey_GPU
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=24:00:00
#SBATCH --job-name=l96-fdv2-trueprior
#SBATCH --output=sbatch_logs/l96_fdv2_subgrad_state_monai_trueprior_perfectmodel_train_%j.out
#SBATCH --error=sbatch_logs/l96_fdv2_subgrad_state_monai_trueprior_perfectmodel_train_%j.err

set -euo pipefail

cd /Odyssey/private/rfablet/Python/4dvarnet-fm-opencode/4dvarnet-fm-obs-density-gen

export PATH="/Odyssey/private/rfablet/miniforge3/envs/fdv-monai-proto/bin:$PATH"
export CONDA_DEFAULT_ENV=fdv-monai-proto
export PYTHONUNBUFFERED=1
export MPLCONFIGDIR="/tmp/mplconfig_${SLURM_JOB_ID}"

mkdir -p "$MPLCONFIGDIR" sbatch_logs

EXPERIMENT="FDV2_subgrad_state_monai_l96_trueprior_perfectmodel"
echo "============================================"
echo " L96 FDV2(monai) subgrad+state+trueprior, perfect-model S0 control | $(date)"
echo "  Experiment: ${EXPERIMENT}"
echo "  Phi = the TRUE L96 ODE (Lorenz96Dynamics), not a learned prior_unet."
echo "  g_prior = time-shifted x[t+1]-Phi_ode(x[t]) (weak-constraint-4DVar"
echo "  style). state_dim=40 (full physical state, NOT the 24-dim observed"
echo "  subspace every other FourDVarNetSolver config uses) -- the true ODE"
echo "  needs the full state to be well-defined."
echo "  'Perfect model': Phi always reads the window's TRUE forcing/params"
echo "  (batch.true_forcing/batch.true_params), targeting the S0 (true"
echo "  params/forcing at eval) config only -- goal: match/beat the DA"
echo "  baselines on S0. S1 (DA-biased at eval) and an 'exposure' training"
echo "  variant (noisy_da_bias mixed into Phi's own inputs during training)"
echo "  are explicit follow-ups, not covered by this run."
echo "  No learned prior_unet exists for this mode (0 extra parameters vs"
echo "  the main solver unet) -- not directly comparable to subgrad+state's"
echo "  own MonaiUNet1D param count, only to its RMSE/EV performance."
echo "  GPU: $(nvidia-smi --query-gpu=name --format=csv -i 0 2>/dev/null | tail -1)"
echo "============================================"
echo

python train.py \
  --config-name experiment/${EXPERIMENT} \
  hydra.run.dir=. \
  hydra.output_subdir=null \
  +fresh=true

echo ""
echo " Done: ${EXPERIMENT} at $(date)"
echo ""
