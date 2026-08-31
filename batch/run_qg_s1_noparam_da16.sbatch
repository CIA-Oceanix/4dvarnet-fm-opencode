#!/bin/bash
#SBATCH --job-name=qg_s1_np_16
#SBATCH --partition=Odyssey_GPU
#SBATCH --gres=gpu:a40:1
#SBATCH --time=04:00:00
#SBATCH --output=qgs1np16_%A_%a.out
#SBATCH --error=qgs1np16_%A_%a.err

set -e

# S1 factorial run @ da_nx=16: no param corruption (--s1-param-bias 0) at the
# 4:1 mismatch (DA 16x16 vs truth 64x64), keeping the wind corruption ON.
# lag 1.0, psi-obs, cols=4, 1% noise.
/Odyssey/private/rfablet/miniforge3/envs/fdv/bin/python evaluation/sweep_qg_baselines.py \
    --nx 64 \
    --da-nx 16 \
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
    --s1-param-bias 0.0 \
    --scenarios test_s1 \
    --outdir reports/outputs/qg_s1_noparam_da16_lag1p0 \
    --tag s1_noparam_da16