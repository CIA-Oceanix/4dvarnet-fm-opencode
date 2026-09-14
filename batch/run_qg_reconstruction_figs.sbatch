#!/bin/bash
#SBATCH --job-name=qg_recon_figs
#SBATCH --partition=Odyssey_GPU
#SBATCH --gres=gpu:h100:1
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=batch/logs/qg_recon_figs_%j.out
#SBATCH --error=batch/logs/qg_recon_figs_%j.err

set -e
PY=/Odyssey/private/rfablet/miniforge3/envs/fdv/bin/python
export PYTHONPATH=/Odyssey/private/rfablet/Python/4dvarnet-fm-qg
export PYTHONUNBUFFERED=1

$PY reports/qg/generate_qg_reconstruction_figs.py \
    --etkf-json reports/qg/outputs/qg_repro_validation/etkf.json \
    --out-dir reports/qg/outputs/figs
