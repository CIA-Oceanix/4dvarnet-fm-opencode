#!/bin/bash
#SBATCH --job-name=qg_assemble_train
#SBATCH --partition=Odyssey_GPU
#SBATCH --gres=gpu:a40:1
#SBATCH --mem=96G
#SBATCH --time=00:20:00
#SBATCH --output=%j.out
#SBATCH --error=%j.err

set -e

WIN_DIR="$PWD/reports/qg/outputs/qg_windows_1000_100_100"
CACHE_DIR="$WIN_DIR/cache"

/Odyssey/private/rfablet/miniforge3/envs/fdv/bin/python \
    reports/qg/assemble_qg_windows.py --split train --windows-dir "$WIN_DIR" --cache-dir "$CACHE_DIR"
