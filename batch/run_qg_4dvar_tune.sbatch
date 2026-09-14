#!/bin/bash
#SBATCH --job-name=qg_4dvar_tune
#SBATCH --partition=Odyssey_GPU
#SBATCH --gres=gpu:a40:1
#SBATCH --time=00:40:00
#SBATCH --array=0-3
#SBATCH --output=%A_%a.out
#SBATCH --error=%A_%a.err
#SBATCH --nodes=1

set -e

# Strong-4DVar tuning at the reference scale (nx=64, c4, psi, lag1.0) on a
# single 30-day eval window. Goal: find a config that beats the free forecast
# (improv > 1) / high EV. Each array task is one (window,opt-steps,lr,B) combo.
mkdir -p reports/qg/outputs/qg_tune_4dvar

W=(12 12 60 12)
S=(300 300 150 300)
LR=(0.5 0.5 0.5 0.2)
B=(1 10 10 0.1)

T=${SLURM_ARRAY_TASK_ID}
OUT=reports/qg/outputs/qg_tune_4dvar/strong_w${W[$T]}_s${S[$T]}_lr${LR[$T]}_b${B[$T]}.json

/Odyssey/private/rfablet/miniforge3/envs/fdv/bin/python evaluation/run_qg_baselines.py \
    --method-list strong4dvar \
    --nx 64 \
    --num-windows 1 \
    --window-days 30 \
    --spinup-years 2.0 \
    --geometry random_columns \
    --cols-per-day 4 \
    --obs-var psi \
    --init-lag-days 1.0 \
    --band 0.25 \
    --scenarios test_s0 \
    --device cuda \
    --da-window-steps ${W[$T]} \
    --fourdvar-opt-steps ${S[$T]} \
    --fourdvar-lr ${LR[$T]} \
    --b-var-scale ${B[$T]} \
    --out ${OUT}
