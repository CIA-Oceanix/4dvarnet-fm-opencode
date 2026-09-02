#!/bin/bash
#SBATCH --job-name=qg_lag_full
#SBATCH --partition=Odyssey_GPU
#SBATCH --gres=gpu:a40:1
#SBATCH --time=02:00:00
#SBATCH --array=0-2
#SBATCH --output=%A_%a.out
#SBATCH --error=%A_%a.err

set -e

/Odyssey/private/rfablet/miniforge3/envs/fdv/bin/python evaluation/run_qg_baselines.py --method etkf --nx 64 --window-days 30 --num-windows 5 --spinup-years 2.0 --ensemble 80 --init-lag-days-list 1.0,2.0,5.0 --geometry random_columns --cols-per-day 4 --scenarios test_s0 --outdir reports/qg/outputs/qg_lag_results --tag lag${SLURM_ARRAY_TASK_ID}
