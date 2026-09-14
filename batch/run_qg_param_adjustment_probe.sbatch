#!/bin/bash
#SBATCH --job-name=qg_adjust_probe
#SBATCH --partition=Odyssey_GPU
#SBATCH --gres=gpu:a40:1
#SBATCH --time=01:30:00
#SBATCH --output=%j.out
#SBATCH --error=%j.err

set -e

/Odyssey/private/rfablet/miniforge3/envs/fdv/bin/python reports/qg/probe_param_adjustment_time.py
