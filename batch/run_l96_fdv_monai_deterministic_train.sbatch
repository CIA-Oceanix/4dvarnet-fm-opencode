#!/bin/bash
#SBATCH --account=odyssey
#SBATCH --partition=Odyssey_GPU
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=96:00:00
#SBATCH --job-name=l96-fdv-monai-det
#SBATCH --output=sbatch_logs/l96_fdv_%j.out
#SBATCH --error=sbatch_logs/l96_fdv_%j.err

set -euo pipefail

cd /Odyssey/private/rfablet/Python/4dvarnet-fm-opencode/4dvarnet-fm-fdv-monai

export PATH="/Odyssey/private/rfablet/miniforge3/envs/fdv-monai-proto/bin:$PATH"
export CONDA_DEFAULT_ENV=fdv-monai-proto
export PYTHONUNBUFFERED=1
export MPLCONFIGDIR="/tmp/mplconfig_${SLURM_JOB_ID}"
mkdir -p "$MPLCONFIGDIR"

# FDV1 (obs+state) and FDV2 (grad+state), MonaiUNet1D backbone, DETERMINISTIC
# recipes only (no trainable prior_weight / cosine scheduler -- that's the
# separate FDV2_grad_state_monai_l96.yaml trainable-weight variant, not run
# here). Single sbatch, one GPU, run strictly sequentially so as not to load
# more than one training job onto the server at a time.
#
# Rebased this branch (feature/l96-fdv-monai-backbone) onto origin/master
# before preparing this run specifically to pick up PR #172 (gradient
# checkpointing for the unrolled solver loop, unconditional, no config flag)
# -- FDV2+monai grad+state previously CUDA-OOM'd on RTX8000/H100 at this exact
# production config (N_outer=10, hidden_channels=[64,128,256], batch_size=16)
# and needed an H200; checkpointing fixes that (confirmed by a real smoke
# test on the RTX8000: peaked at 19.5GB, comfortably under 46GB, vs OOMing at
# batch_size=1 before) by removing the N_outer-linear retained-activation
# graph that was the documented root cause -- no longer H200-only.
#
# GPU CHOICE: --gres=gpu:h100:1 (not rtx8000). A live GPU-utilization sample
# during a real production-scale (batch=16) training step on the RTX8000
# showed ~100% SM / 60-87% memory utilization throughout -- this workload is
# compute/bandwidth-bound, not launch-latency-bound, so a faster GPU should
# give a close-to-proportional wall-clock speedup (not eaten by Python/
# kernel-launch overhead). H100 (Hopper) has much higher TF32 tensor-core
# throughput than RTX8000's plain-FP32-only Turing cores (no code change
# needed -- PyTorch enables TF32 for conv/matmul automatically on Ampere+)
# plus far higher memory bandwidth, so it's the better default over A100 when
# available. NOT YET VERIFIED with an actual H100 run (see the wall-clock
# estimate below, which is RTX8000-measured) -- this is an informed
# expectation, not a benchmarked number.
#
# AVAILABILITY: chosen based on cluster state at prep time (2026-09-08
# ~18:20) -- sl-mee-br-211 (gpu:h100:2) was fully idle. Cluster state changes;
# RE-CHECK before actually launching (`sinfo -p Odyssey_GPU -o "%N %G %C %t"`)
# and swap this line to --gres=gpu:a100:1 if no H100 is free (a100 nodes
# sl-mee-br-206/207 had partial free capacity at prep time).
#
# WALL-CLOCK WARNING (measured on RTX8000, not yet on H100): a real 2-step/
# 1-epoch smoke run of each config, batch_size=16 (production), gave ~2.4s/
# step for FDV1 (obs+state) and ~10.5s/step for FDV2-fixedw (grad+state) --
# FDV2's double-backward (create_graph=True) plus the checkpointing recompute
# on top of the monai backbone's attention blocks is markedly more expensive
# than the #173 microbenchmark's checkpointing-only overhead (1.55-1.66x,
# measured on the plain UNet1D backbone) would suggest alone. Extrapolated to
# the full 1000-window/batch=16 production config (63 steps/epoch, 400
# epochs) on RTX8000: FDV1 ~151s/epoch =~ 17h total; FDV2-fixedw ~660s/epoch
# =~ 73h (~3 days) total -- combined ~4 DAYS on RTX8000. H100's compute/
# bandwidth advantage (see above) should cut this substantially given the
# workload is compute-bound, plausibly to somewhere in the 1-2 day range
# combined, but that has not been measured -- consider a short real per-epoch
# timing check on the actual H100/A100 node before committing to the full
# 400-epoch run. Time limit kept at 96h (partition MaxTime is unlimited) as a
# conservative ceiling even if the H100 speedup is smaller than expected.
STAGES="FDV1_unrolled_monai_unet_l96:obs+state FDV2_grad_state_monai_l96_fixedw:grad+state"

for STAGE in $STAGES; do
  EXP="${STAGE%%:*}"
  MODE="${STAGE##*:}"

  echo "============================================"
  echo " L96 FDV Training (MonaiUNet1D backbone, deterministic, ${MODE}) | $(date)"
  echo "  Experiment: ${EXP}"
  echo "  GPU: $(nvidia-smi --query-gpu=name --format=csv -i 0 2>/dev/null | tail -1)"
  echo "============================================"
  echo ""

  rm -rf "experiments/${EXP}"

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
  echo " Running full neural_eval.json evaluation (via evaluation/neural_inference.py's"
  echo " model_factory path -- required for unet_backbone=monai, shape-inference cannot"
  echo " recover a monai backbone's config; see the committed evaluation/neural_inference.py fix)."
  echo " --n-outer 10 is REQUIRED here: eval_neural_l96.py's default (1) is correct for"
  echo " tau0-only CFM but silently produces garbage for FourDVarNetSolver's zero-init"
  echo " N_outer=10 unrolled refinement (confirmed: RMSE 3.47/EV -3.75 at n_outer=1 vs"
  echo " train.py's own in-training eval of the same checkpoint, RMSE ~0.42)..."
  python eval_neural_l96.py \
    --checkpoint experiments/${EXP}/checkpoints/stage1_best.ckpt \
    --config experiments/${EXP}/resolved_config.yaml \
    --dataset experiments/l96_datasets_obsj2_int100_nwin200.pt \
    --num-windows 200 \
    --batch-size 16 \
    --n-outer 10 \
    --output experiments/${EXP}/neural_eval.json

  echo ""
  echo " Done: ${EXP} at $(date)"
  echo "============================================"
  echo ""
done

echo "Both deterministic FDV1/FDV2 monai-backbone runs complete at $(date)"
