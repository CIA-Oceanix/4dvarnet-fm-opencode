#!/bin/bash
#SBATCH --job-name=qg_s1_qg1l_noparam_nowind
#SBATCH --partition=Odyssey_GPU
#SBATCH --gres=gpu:a40:1
#SBATCH --time=04:00:00
#SBATCH --output=qgs1qg1lnpnw_%A_%a.out
#SBATCH --error=qgs1qg1lnpnw_%A_%a.err

set -e

# S1-QG1L ablation: structural 1-layer mismatch only, NO param corruption and NO
# wind corruption. Isolates the pure structural 1-layer model-error contribution
# (analogous to the S1 "resolution-only" off/off cell, but structural not grids).
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
    --s1-param-bias 0 \
    --s1-amp-bias 0 \
    --s1-loc-sigma-frac 0 \
    --s1-sigma-eta-frac 0 \
    --scenarios test_s0,test_s1_qg1l \
    --outdir reports/qg/outputs/qg_s1_qg1l_noparam_nowind_lag${LAG//./p} \
    --tag s1_qg1l_noparam_nowind
