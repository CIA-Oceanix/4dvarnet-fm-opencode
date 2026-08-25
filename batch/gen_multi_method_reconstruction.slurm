#!/bin/bash
#SBATCH --job-name=gen_multi_rec
#SBATCH --output=sbatch_logs/gen_multi_rec_%j.out
#SBATCH --error=sbatch_logs/gen_multi_rec_%j.err
#SBATCH --time=1:00:00
#SBATCH --gres=gpu:rtx8000:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --partition=Odyssey_GPU

cd /Odyssey/private/rfablet/Python/4dvarnet-fm-opencode

source /Odyssey/private/rfablet/miniforge3/etc/profile.d/conda.sh
conda activate fdv

python -u reports/generate_l96_multi_method_reconstruction.py