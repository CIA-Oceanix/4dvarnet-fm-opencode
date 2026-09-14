#!/bin/bash
#SBATCH --job-name=qg_devfix_timing
#SBATCH --partition=Odyssey_GPU
#SBATCH --gres=gpu:a40:1
#SBATCH --time=00:45:00
#SBATCH --output=%j.out
#SBATCH --error=%j.err

set -e

/Odyssey/private/rfablet/miniforge3/envs/fdv/bin/python reports/qg/probe_device_fix_timing.py
