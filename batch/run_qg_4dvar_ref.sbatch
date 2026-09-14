#!/bin/bash
#SBATCH --job-name=qg_4dvar_ref
#SBATCH --partition=Odyssey_GPU
#SBATCH --gres=gpu:a40:1
#SBATCH --time=02:30:00
#SBATCH --output=%A.out
#SBATCH --error=%A.err
#SBATCH --nodes=1

set -e

# Strong-4DVar at the reference-case settings (mirror of run_qg_matrix_metrics
# and run_qg_enkf_ref): nx=64, c4, random_columns, psi-obs, lags 1.0 & 2.0.
# Uses the LBFGS / 5-day-window config that beat the free forecast in tuning
# (improv 1.14, EV 0.85 on the single-window probe).

/Odyssey/private/rfablet/miniforge3/envs/fdv/bin/python evaluation/sweep_qg_baselines.py \
    --nx 64 \
    --window-days 30 \
    --num-windows 5 \
    --spinup-years 2.0 \
    --ensemble 80 \
    --inflation-list 1.0 \
    --loc-list 6 \
    --method-list strong4dvar \
    --init lagged \
    --geometry random_columns \
    --cols-per-day 4 \
    --obs-noise-frac-list 0.01 \
    --init-lag-days-list 1.0,2.0 \
    --band 0.25 \
    --obs-var psi \
    --scenarios test_s0 \
    --outdir reports/qg/outputs/qg_matrix_c4_psi \
    --tag qgmatrix \
    --da-window-steps 60 \
    --fourdvar-optimizer lbfgs \
    --fourdvar-max-iter 60 \
    --fourdvar-lr 1.0 \
    --b-var-scale 1.0 \
    --fourdvar-grad-clip 1000
