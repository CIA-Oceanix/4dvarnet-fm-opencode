#!/bin/bash
#SBATCH --account=odyssey
#SBATCH --partition=Odyssey_GPU
#SBATCH --gres=gpu:rtx8000:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=06:00:00
#SBATCH --job-name=l96-cfm-sda-ens30-eval
#SBATCH --output=sbatch_logs/l96_neural_%j.out
#SBATCH --error=sbatch_logs/l96_neural_%j.err

set -euo pipefail

cd /Odyssey/private/rfablet/Python/4dvarnet-fm-opencode/4dvarnet-fm-l96-eval-config-persist

export PATH="/Odyssey/private/rfablet/miniforge3/envs/fdv-monai-proto/bin:$PATH"
export CONDA_DEFAULT_ENV=fdv-monai-proto
export PYTHONUNBUFFERED=1
export MPLCONFIGDIR="/tmp/mplconfig_${SLURM_JOB_ID}"
mkdir -p "$MPLCONFIGDIR"

# ens30 (n_members=30) ensemble evaluation for the trained monai-backbone CFM
# and SDA schemes. Submitted with --dependency=afterok:52654 so it waits for
# the still-running CFM cosine tier sweep (job 52654) to produce the L-tier
# checkpoint before starting.
#
# SDA fix included: eval_sda_l96.py had two real bugs fixed this session --
# (1) no --normalize-stats support at all (obs/state normalization now wired
# through prepare_dataset + a denormalize step on the returned trajectories/
# members, mirroring eval_neural_l96.py's existing pattern; --r-var is
# deliberately NOT rescaled -- sda_guided_sample's guidance step normalizes
# its own gradient by its norm, which exactly cancels any positive R_var
# scale factor, verified empirically); (2) --batch-size was silently ignored
# (never forwarded to prepare_dataset, which defaulted to batch_size=200) --
# guided sampling's real backward pass through all 200 windows at once
# reproducibly OOM'd (13.41GiB alloc failure) even for MonaiUnconditionalPriorCFM's
# M-tier backbone; fixed by threading args.batch_size through, confirmed
# working at batch_size=16.
echo "============================================"
echo " ens30 eval starting | $(date)"
echo "  GPU: $(nvidia-smi --query-gpu=name --format=csv -i 0 2>/dev/null | tail -1)"
echo "============================================"

# --- CFM (VanillaCFM monai, tau=0 -> --n-outer default 1 is correct: the
#     model is trained to predict directly at tau=0, no multi-step Euler
#     integration needed) ---
CFM_EXPS="L2b_monai_vanilla_cfm_s0s1_norm L2b_monai_vanilla_cfm_s0s1_norm_splus_cosine L2b_monai_vanilla_cfm_s0s1_norm_cosine L2b_monai_vanilla_cfm_s0s1_norm_l_cosine"
for EXP in $CFM_EXPS; do
  echo ""
  echo "=== CFM ens30: ${EXP} | $(date) ==="
  python eval_neural_l96.py \
    --checkpoint experiments/${EXP}/checkpoints/stage1_best.ckpt \
    --config experiments/${EXP}/resolved_config.yaml \
    --dataset experiments/l96_datasets_obsj2_int100_nwin200.pt \
    --num-windows 200 \
    --batch-size 16 \
    --n-members 30 \
    --normalize-stats experiments/l96_norm_stats_obsj2.pt \
    --output experiments/${EXP}/neural_eval_ens30.json
done

# --- SDA (monai M-tier, guided sampling). guidance_weight=40, r_var=0.5,
#     n_outer=10 match the established pure-noise-start (not warm-started)
#     SDA1/SDA2 convention from the original non-monai, non-normalized
#     benchmark (CHANGELOG: "guidance_weight=40 (picked by an S0 RMSE sweep
#     over {0.3..400})" -- an initial smoke test at the CLI's default
#     guidance_weight=1.0 gave poor RMSE/EV, which is this mismatch, not a
#     guidance failure). SDA3 (noisy-params, new variant) has no dedicated
#     sweep of its own -- carries over SDA2's value since it's architecturally
#     identical, just a different conditioning signal.
SDA_EXPS="SDA1_monai_prior_l96_norm SDA2_monai_cond_mixed_l96_norm SDA3_monai_cond_noisy_l96_norm"
for EXP in $SDA_EXPS; do
  echo ""
  echo "=== SDA ens30: ${EXP} | $(date) ==="
  python eval_sda_l96.py \
    --checkpoint experiments/${EXP}/checkpoints/stage1_best.ckpt \
    --config experiments/${EXP}/resolved_config.yaml \
    --dataset experiments/l96_datasets_obsj2_int100_nwin200.pt \
    --num-windows 200 \
    --batch-size 16 \
    --n-members 30 \
    --n-outer 10 \
    --r-var 0.5 \
    --guidance-weight 40 \
    --normalize-stats experiments/l96_norm_stats_obsj2.pt \
    --output experiments/${EXP}/sda_eval_ens30.json
done

echo ""
echo "All ens30 evals complete at $(date)"
