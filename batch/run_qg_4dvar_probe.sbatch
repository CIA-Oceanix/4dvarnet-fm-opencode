#!/bin/bash
#SBATCH --job-name=qg_4dvar_probe
#SBATCH --partition=Odyssey_GPU
#SBATCH --gres=gpu:a40:1
#SBATCH --time=01:00:00
#SBATCH --output=%A.out
#SBATCH --error=%A.err
#SBATCH --nodes=1

set -e

# Sizing probe: time one strong4dvar solve at the reference-case scale
# (nx=64, c4, psi, lag1.0) on a single eval window with reduced Adam steps,
# to choose the full Win reference configuration.
start=$(date +%s)
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
    --da-window-steps 12 \
    --fourdvar-optimizer adam \
    --fourdvar-opt-steps 60 \
    --fourdvar-lr 0.05 \
    --b-var-scale 1.0 \
    --scenarios test_s0 \
    --device cuda
end=$(date +%s)
echo "PROBE_WALL_SECONDS=$((end - start))"
