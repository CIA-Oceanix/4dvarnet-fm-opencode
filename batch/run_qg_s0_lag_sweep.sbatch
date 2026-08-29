#!/bin/bash
#SBATCH --job-name=qg_s0_lag
#SBATCH --partition=Odyssey_GPU
#SBATCH --gres=gpu:a40:1
#SBATCH --time=02:00:00
#SBATCH --array=0-1
#SBATCH --output=%A_%a.out
#SBATCH --error=%A_%a.err

set -e

# array task 0 = q-obs, task 1 = psi-obs
OBSVARS=(q psi)
OV=${OBSVARS[$SLURM_ARRAY_TASK_ID]}

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
    --init-lag-days-list 0.5,1.0,2.0,4.0,6.0 \
    --obs-var ${OV} \
    --scenarios test_s0 \
    --outdir reports/outputs/qg_s0_lag_sweep_${OV} \
    --tag s0_lag
