#!/bin/bash
#SBATCH --job-name=qg_s1_nowind
#SBATCH --partition=Odyssey_GPU
#SBATCH --gres=gpu:a40:1
#SBATCH --time=04:00:00
#SBATCH --output=qgs1nw_%A_%a.out
#SBATCH --error=qgs1nw_%A_%a.err

set -e

# S1 sensitivity ablation B (lag 1.0): REMOVE the forcing-noise component
# (wind corruption OFF: s1_amp_bias/loc_sigma_frac/sigma_eta_frac = 0 -> corrupted
# wind == true wind), keeping da_nx=32 cross-resolution and the param-bias ON.
# Mirrors batch/run_qg_s1_da32.sbatch (the reference) otherwise.
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
    --s1-amp-bias 0.0 \
    --s1-loc-sigma-frac 0.0 \
    --s1-sigma-eta-frac 0.0 \
    --scenarios test_s1 \
    --outdir reports/qg/outputs/qg_s1_nowind_lag1p0 \
    --tag s1_nowind