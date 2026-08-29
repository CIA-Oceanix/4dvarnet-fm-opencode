#!/bin/bash
#SBATCH --job-name=qg_anticollapse
#SBATCH --partition=Odyssey_GPU
#SBATCH --gres=gpu:a40:1
#SBATCH --time=02:30:00
#SBATCH --array=0-1
#SBATCH --output=%A_%a.out
#SBATCH --error=%A_%a.err

set -e

# array task 0 = q-obs, task 1 = psi-obs
OBSVARS=(q psi)
OV=${OBSVARS[$SLURM_ARRAY_TASK_ID]}

# Anti-collapse tuning: sweep ETKF ridge/additive x inflation x loc at the
# lags where EV is still high (0.5, 1.0), targeting improv>=1.0 AND EV>=0.8.
# Reports both expvar_full and forecast_improvement per cell.
/Odyssey/private/rfablet/miniforge3/envs/fdv/bin/python evaluation/sweep_qg_baselines.py \
    --nx 64 \
    --window-days 30 \
    --num-windows 5 \
    --spinup-years 2.0 \
    --ensemble-list 80 \
    --inflation-list 1.0,1.2 \
    --loc-list 6,12 \
    --method-list etkf \
    --init lagged \
    --geometry random_columns \
    --cols-per-day 4 \
    --init-lag-days-list 0.5,1.0 \
    --obs-var ${OV} \
    --scenarios test_s0 \
    --etkf-ridge-list 0,0.1,1.0 \
    --etkf-additive-list 0,1e-9,1e-7 \
    --outdir reports/outputs/qg_anticollapse_${OV} \
    --tag ac
