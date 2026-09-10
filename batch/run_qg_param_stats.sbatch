#!/bin/bash
#SBATCH --job-name=qg_param_stats
#SBATCH --partition=Mee_Global_CPU
#SBATCH --account=mee
#SBATCH --qos=mee_short
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --time=02:00:00
#SBATCH --output=batch/logs/qg_param_stats_%j.out
#SBATCH --error=batch/logs/qg_param_stats_%j.err

set -e

# Precompute [U1,rd,rek,beta] global mean/std (Q3/Q4 forcing+param
# conditioning, see PLAN.md's 2026-09-10 QG Q3/Q4 section) alongside the
# psi stats (--output-params reuses the same loaded 1000-window train cache
# precompute_qg_norm_stats.py already loads for psi -- the psi output is
# recomputed too as a harmless deterministic side effect, not the expensive
# part of this job; the cache load dominates either way).
CACHE_DIR="/Odyssey/private/rfablet/Python/4dvarnet-fm-qg-100samples/reports/qg/outputs/qg_windows_1000_100_100/cache"
PY=/Odyssey/private/rfablet/miniforge3/envs/fdv/bin/python

$PY precompute_qg_norm_stats.py \
    --nx 64 --train-seed 42 --num-train 1000 \
    --cache-dir "$CACHE_DIR" \
    --output experiments/qg_psi_norm_stats.pt \
    --output-params experiments/qg_param_norm_stats.pt
