#!/bin/bash
#SBATCH --time=00:30:00
#SBATCH --partition=Odyssey_GPU
#SBATCH --gres=gpu:rtx8000:1
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --chdir=/Odyssey/private/rfablet/Python/4dvarnet-fm-opencode
#SBATCH --output=experiments/gen_reconstruction_fig_%j.out
#SBATCH --error=experiments/gen_reconstruction_fig_%j.err

set -euo pipefail

PYTHON="/Odyssey/private/rfablet/miniforge3/envs/fdv/bin/python3"
$PYTHON -u reports/generate_l96_reconstruction_figures.py