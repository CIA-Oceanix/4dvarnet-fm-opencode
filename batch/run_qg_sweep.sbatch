#!/bin/bash
#SBATCH --job-name=qg_sweep
#SBATCH --partition=Odyssey_GPU
#SBATCH --gres=gpu:a40:1
#SBATCH --time=48:00:00
#SBATCH --array=0-3
#SBATCH --output=%A_%a.out
#SBATCH --error=%A_%a.err

set -e

/Odyssey/private/rfablet/miniforge3/envs/fdv/bin/python evaluation/sweep_qg_baselines.py --nx 64 --window-days 60 --num-windows 5 --spinup-years 2.0 --ensemble 80 --inflation-list 1.0,1.15,1.3 --loc-list 6,10,14 --method-list etkf,enkf --init lagged --geometry random_columns --cols-per-day 4 --scenarios test_s0,test_s1a --outdir reports/outputs/qg_sweep_results --device cuda --tag full_sweep_array${SLURM_ARRAY_TASK_ID}
