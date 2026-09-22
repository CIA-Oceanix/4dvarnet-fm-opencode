#!/bin/bash
#SBATCH --account=odyssey
#SBATCH --partition=Odyssey_GPU
#SBATCH --gres=gpu:a100:1
#SBATCH --nodelist=sl-mee-br-206
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=8:00:00
#SBATCH --job-name=l96-mean_comp
#SBATCH --output=sbatch_logs/l96_mean_comp_%j.out
#SBATCH --error=sbatch_logs/l96_mean_comp_%j.err
set -euo pipefail
cd /Odyssey/private/rfablet/Python/4dvarnet-fm-opencode/4dvarnet-fm-fdv-tau-aware
export PATH="/Odyssey/private/rfablet/miniforge3/envs/fdv-monai-proto/bin:$PATH"
export PYTHONUNBUFFERED=1
export MPLCONFIGDIR="/tmp/mplconfig_${SLURM_JOB_ID}"
mkdir -p "$MPLCONFIGDIR" sbatch_logs
# Psi_mean alone: mu(x0, tau=0, y) averaged over 30 draws, matching the ens30
# member budget. At tau=0, x_tau=x0 is independent of x1, so E[x1|x_tau,y] =
# E[x1|y] exactly -- the only error is Monte-Carlo over the 30 draws, which
# mc_floor converts into a residual floor. psi_of handles VanillaCFM's velocity
# target via x1 = x_tau + (1-tau)v; without that branch a velocity would be read
# as if it were mu.
python reports/l96/eval_mean_component.py \
  --dataset "/Odyssey/private/rfablet/Python/4dvarnet-fm-opencode/experiments/l96_datasets_obsj2_int100_nwin200.pt" \
  --case s0 --group all_obs --batch-size 25 --m-draws 30 \
  --output reports/l96/outputs/p1_mean_component.md \
  --json-output reports/l96/outputs/p1_mean_component.json
echo "Done: $(date)"
