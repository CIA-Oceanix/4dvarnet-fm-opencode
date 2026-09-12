#!/bin/bash
#SBATCH --account=odyssey
#SBATCH --partition=Odyssey_GPU
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=24:00:00
#SBATCH --job-name=l96-fdv1-obsstate-auxpriorcost
#SBATCH --output=sbatch_logs/l96_fdv1_obsstate_monai_auxpriorcost_train_%j.out
#SBATCH --error=sbatch_logs/l96_fdv1_obsstate_monai_auxpriorcost_train_%j.err

set -euo pipefail

cd /Odyssey/private/rfablet/Python/4dvarnet-fm-opencode/4dvarnet-fm-obs-density-gen

export PATH="/Odyssey/private/rfablet/miniforge3/envs/fdv-monai-proto/bin:$PATH"
export CONDA_DEFAULT_ENV=fdv-monai-proto
export PYTHONUNBUFFERED=1
export MPLCONFIGDIR="/tmp/mplconfig_${SLURM_JOB_ID}"

mkdir -p "$MPLCONFIGDIR" sbatch_logs

TEST_CACHE="/Odyssey/private/rfablet/Python/4dvarnet-fm-opencode/experiments/l96_datasets_obsj2_int100_nwin200.pt"

EXPERIMENT="FDV1_obsstate_monai_l96_auxpriorcost"
echo "============================================"
echo " L96 ABLATION: FDV1 (update_input=obs+state) + FDV2's prior_cost aux loss | $(date)"
echo "  Experiment: ${EXPERIMENT}"
echo "  Isolates whether the auxiliary prior-consistency loss term"
echo "  (aux_var_cost_weight=0.1, prior_cost(x_final)+prior_cost(states),"
echo "  no prior_weight/obs_cost mixed in -- see the 2026-09-11"
echo "  fdv-aux-prior-loss-drop-prior-weight-obs-cost fix) degrades fast-Y"
echo "  reconstruction on its own, independent of FDV2's gradient-"
echo "  conditioned update-input construction (grad+state/subgrad+state)."
echo "  update_input stays exactly obs+state -- _build_update_input still"
echo "  returns cat([x, obs_clean]) only, completely untouched. A prior_unet"
echo "  is built purely to supply this loss term (FourDVarNetSolver.__init__"
echo "  now builds one whenever aux_var_cost_weight>0, not only when"
echo "  update_input is in _PRIOR_MODES)."
echo "  All three FDV2 retrains under this same aux loss term still show a"
echo "  collapsed fast-Y variance ratio (0.48-0.53 vs FDV1's healthy 0.926)"
echo "  despite the term's actual weight in their total loss ranging only"
echo "  0.2%-9.5% and NOT correlating with collapse severity -- this run"
echo "  tests the term's effect directly on FDV1's own healthy config."
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
