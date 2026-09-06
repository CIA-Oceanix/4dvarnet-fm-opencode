#!/bin/bash
#SBATCH --job-name=qg_w4dvar_s1ref
#SBATCH --partition=Odyssey_GPU
#SBATCH --gres=gpu:a40:1
#SBATCH --time=02:30:00
#SBATCH --output=%A.out
#SBATCH --error=%A.err
#SBATCH --nodes=1

set -e

# Weak-4DVar on the S1 (model-error) dataset at the reference-case settings,
# matching the S0 weak-4DVar reference: nx=64, c4, random_columns, psi-obs,
# lags 1.0 & 2.0, da_nx=64 (no resolution mismatch -> the S1 analogue of the
# full-res S0 reference). Same LBFGS / 5-day-window config + q-var-scale=0.1
# tuned on S0. QVAR sets the model-error penalty (default 0.1).
QVAR=${QVAR:-0.1}

/Odyssey/private/rfablet/miniforge3/envs/fdv/bin/python evaluation/sweep_qg_baselines.py \
    --nx 64 \
    --da-nx 64 \
    --window-days 30 \
    --num-windows 5 \
    --spinup-years 2.0 \
    --ensemble 80 \
    --inflation-list 1.0 \
    --loc-list 6 \
    --method-list weak4dvar \
    --init lagged \
    --geometry random_columns \
    --cols-per-day 4 \
    --obs-noise-frac-list 0.01 \
    --init-lag-days-list 1.0,2.0 \
    --band 0.25 \
    --obs-var psi \
    --scenarios test_s1 \
    --outdir reports/qg/outputs/qg_s1_weak4dvar_nores \
    --tag s1_weak4dvar_nores \
    --da-window-steps 60 \
    --fourdvar-optimizer lbfgs \
    --fourdvar-max-iter 60 \
    --fourdvar-lr 1.0 \
    --b-var-scale 1.0 \
    --q-var-scale ${QVAR} \
    --fourdvar-grad-clip 1000
