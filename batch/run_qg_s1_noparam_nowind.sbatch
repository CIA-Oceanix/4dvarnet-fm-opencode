#!/bin/bash
#SBATCH --job-name=qg_s1_np_nw
#SBATCH --partition=Odyssey_GPU
#SBATCH --gres=gpu:a40:1
#SBATCH --time=04:00:00
#SBATCH --output=qgs1npnw_%A_%a.out
#SBATCH --error=qgs1npnw_%A_%a.err

set -e

# S1 factorial run #4: no param corruption AND no wind corruption at da_nx=32
# (--s1-param-bias 0 --s1-amp-bias 0 --s1-loc-sigma-frac 0 --s1-sigma-eta-frac 0),
# keeping only the resolution mismatch (da_nx=32 vs truth 64). lag 1.0, psi-obs,
# cols=4, 1% noise. Mirrors batch/run_qg_s1_da32.sbatch otherwise.
/Odyssey/private/rfablet/miniforge3/envs/fdv/bin/python evaluation/sweep_qg_baselines.py \
    --nx 64 \
    --da-nx 32 \
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
    --s1-amp-bias 0.0 \
    --s1-loc-sigma-frac 0.0 \
    --s1-sigma-eta-frac 0.0 \
    --scenarios test_s1 \
    --outdir reports/outputs/qg_s1_noparam_nowind_lag1p0 \
    --tag s1_noparam_nowind