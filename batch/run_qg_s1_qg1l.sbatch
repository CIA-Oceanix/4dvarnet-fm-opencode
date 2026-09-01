#!/bin/bash
#SBATCH --job-name=qg_s1_qg1l
#SBATCH --partition=Odyssey_GPU
#SBATCH --gres=gpu:a40:1
#SBATCH --time=04:00:00
#SBATCH --output=qgs1qg1l_%A_%a.out
#SBATCH --error=qgs1qg1l_%A_%a.err

set -e

# S1 structural-model-error case: reduced-gravity single-layer (qg1l) DA model
# at FULL resolution (da_nx=64), psi-obs, cols=4, 1% obs noise, lag 1.0.
# Same param bias (rd,rek *0.85) + corrupted wind as test_s1, but the DA model
# is structurally different (1-layer, no baroclinic mode) - the closest QG
# analog to the L63/L96 "wrong dynamics" case. S0 run alongside as same-run anchor.
# NOTE: dataset cache key differs from the 2-layer S1 (different da_model), so a
# fresh ~5 min CPU spinup is expected on first run.
LAG=1.0

/Odyssey/private/rfablet/miniforge3/envs/fdv/bin/python evaluation/sweep_qg_baselines.py \
    --nx 64 \
    --window-days 30 \
    --num-windows 5 \
    --spinup-years 2.0 \
    --ensemble 80 \
    --inflation-list 1.0 \
    --loc-list 6 \
    --method-list etkf \
    --init lagged \
    --geometry random_columns \
    --cols-per-day 4 \
    --obs-noise-frac-list 0.01 \
    --init-lag-days-list ${LAG} \
    --band 0.25 \
    --obs-var psi \
    --scenarios test_s0,test_s1_qg1l \
    --outdir reports/outputs/qg_s1_qg1l_lag${LAG//./p} \
    --tag s1_qg1l
