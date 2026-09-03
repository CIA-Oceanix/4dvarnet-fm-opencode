#!/bin/bash
#SBATCH --job-name=qg_s1_nores
#SBATCH --partition=Odyssey_GPU
#SBATCH --gres=gpu:a40:1
#SBATCH --time=04:00:00
#SBATCH --output=qgs1nr_%A_%a.out
#SBATCH --error=qgs1nr_%A_%a.err

set -e

# S1 sensitivity ablation C (lag 1.0): REMOVE the resolution-mismatch component
# (da_nx=64 == truth nx -> no cross-resolution; the DA 2-layer model runs at full
# res), keeping the param-bias and corrupted-wind ON. `da_model` stays
# "qg2l_lores" (label-only; _build_dyn treats it as a 2-layer QG at nx_da, so at
# da_nx=64 it is functionally full-res). Mirrors batch/run_qg_s1_da32.sbatch
# (the reference) otherwise.
/Odyssey/private/rfablet/miniforge3/envs/fdv/bin/python evaluation/sweep_qg_baselines.py \
    --nx 64 \
    --da-nx 64 \
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
    --init-lag-days-list 1.0 \
    --band 0.25 \
    --obs-var psi \
    --scenarios test_s1 \
    --outdir reports/qg/outputs/qg_s1_nores_lag1p0 \
    --tag s1_nores