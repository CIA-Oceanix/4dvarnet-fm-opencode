#!/bin/bash
#SBATCH --account=odyssey
#SBATCH --partition=Odyssey_GPU
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --job-name=l96-sda-monai-train
#SBATCH --output=sbatch_logs/l96_neural_%j.out
#SBATCH --error=sbatch_logs/l96_neural_%j.err

set -euo pipefail

cd /Odyssey/private/rfablet/Python/4dvarnet-fm-opencode/4dvarnet-fm-l96-eval-config-persist

export PATH="/Odyssey/private/rfablet/miniforge3/envs/fdv-monai-proto/bin:$PATH"
export CONDA_DEFAULT_ENV=fdv-monai-proto
export PYTHONUNBUFFERED=1
export MPLCONFIGDIR="/tmp/mplconfig_${SLURM_JOB_ID}"
mkdir -p "$MPLCONFIGDIR"

# SDA1/SDA2/SDA3 (score-based DA flow-matching priors), MonaiUNet1D M-tier
# backbone (hidden_channels [64,128,256], num_res_blocks=2 -- same tier as
# the DirectUNet/VanillaCFM monai M-tier baselines), data.normalize=true.
# Consolidates the three individually-drafted single-experiment sbatch
# scripts (run_l96_sda{1,2,3}_monai_norm_train.sbatch, which requested
# rtx8000/h100/h100 respectively) into one sequential job on one GPU, so as
# not to load more than one training job onto the server at a time.
#
# TRAINING ONLY, no eval step here -- eval_sda_l96.py does not yet support
# --normalize-stats (needs both obs/state normalization AND rescaling the
# DPS guidance cost's r_var to normalized units); eval is a separate
# follow-up once that's fixed. Do not confuse this with the DirectUNet/CFM
# tier sweeps' combined train+eval sbatch scripts -- this one stops after
# checkpoint/config persistence.
#
# GPU: l40s. models/sda.py's forward(x_t, batch, tau) -- both
# UnconditionalPriorCFM and ConditionalPriorCFM -- is a single UNet call per
# training step, with train_tau_0_only=False (a random tau sampled each
# step) but otherwise the exact same per-step cost shape as VanillaCFM's
# training forward; N_outer's unrolled multi-step loop only runs inside
# sample() at DPS-guided inference time, never during training. So none of
# FDV2's cost drivers (unrolled solver, double-backward/create_graph,
# checkpointing recompute) apply here -- rtx8000/a40/l40s would all work
# (the already-running CFM monai tier sweep, job 52654, confirms rtx8000 is
# fine for this backbone family). l40s picked over rtx8000/a40 based on
# real-time cluster state at launch (2026-09-08 ~19:40): 6 free GPUs
# (3 each on sl-mee-br-208/209, gres-level check via `scontrol show node`,
# not just CPU-based sinfo) vs 4 free rtx8000 and only 1 free a40, plus l40s
# (Ada Lovelace) being the newest/fastest architecture of the three.
STAGES="SDA1_monai_prior_l96_norm SDA2_monai_cond_mixed_l96_norm SDA3_monai_cond_noisy_l96_norm"

for EXP in $STAGES; do
  echo "============================================"
  echo " L96 SDA Training (MonaiUNet1D backbone, M tier, data.normalize=true) | $(date)"
  echo "  Experiment: ${EXP}"
  echo "  GPU: $(nvidia-smi --query-gpu=name --format=csv -i 0 2>/dev/null | tail -1)"
  echo "============================================"
  echo ""


  python train.py \
    --config-name experiment/${EXP} \
    hydra.run.dir=. \
    hydra.output_subdir=null

  echo ""
  echo " Training done: ${EXP} at $(date)"
  echo "============================================"

  echo ""
  echo " Verifying checkpoint/config persistence..."
  ls -la "experiments/${EXP}/resolved_config.yaml" "experiments/${EXP}/checkpoints/"
  echo ""
  echo " (No eval run -- see header comment. Run eval_sda_l96.py separately"
  echo "  once its --normalize-stats support lands.)"
  echo ""
done

echo "All three SDA monai-backbone training runs complete at $(date)"
