#!/bin/bash
#SBATCH --job-name=qg_q6_fourdvarnet_train
#SBATCH --partition=Odyssey_GPU
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=96G
#SBATCH --time=24:00:00
#SBATCH --output=batch/logs/qg_q6_fourdvarnet_train_%j.out
#SBATCH --error=batch/logs/qg_q6_fourdvarnet_train_%j.err

set -e

# Q6: FourDVarNetSolver (FDV1) obs-only QG scheme, "S"-tier +
# aux_var_cost_weight=0.01 (L96's current winning config, adapted to QG's
# state_dim) -- see config/experiment/Q6_fourdvarnet_s0.yaml for the full
# design rationale and the obs_mask correctness fix this depended on.
#
# --exp-id is required (model_type=fourdvarnet is unique to Q6, but every
# other QG config also always passes --exp-id explicitly).
CACHE_DIR="/Odyssey/private/rfablet/Python/4dvarnet-fm-qg-100samples/reports/qg/outputs/qg_windows_1000_100_100/cache"
PY=/Odyssey/private/rfablet/miniforge3/envs/fdv-monai-proto/bin/python

$PY train_qg_neural.py \
    --model-type fourdvarnet --exp-id Q6_fourdvarnet_s0 \
    --exp-dir experiments/Q6_fourdvarnet_s0 \
    --nx 64 --epochs 400 \
    --num-train 1000 --num-val 100 --num-test 100 \
    --batch-size 2 --cache-dir "$CACHE_DIR"
