#!/bin/bash
#SBATCH --job-name=qg_figs
#SBATCH --partition=Odyssey_GPU
#SBATCH --gres=gpu:a40:1
#SBATCH --time=04:00:00
#SBATCH --output=qg_figs_%A.out
#SBATCH --error=qg_figs_%A.err

set -e

# Regenerate the S0 / S1-QG2L da_nx=32 illustration figures + DA-cycle GIFs
# under the per-column independent intra-day obs-timing geometry
# (obs (T,ny), OBS_GEOMETRY_VERSION=2). Production config: nx=64, psi-obs,
# cols=4, 1% noise, lag 1.0, ens=80, both scenarios.
# Writes into reports/qg/outputs/figs/ (the report section 8 default).
/Odyssey/private/rfablet/miniforge3/envs/fdv/bin/python reports/qg/generate_qg_s0s1_figs.py \
    --scenario both \
    --cols-per-day 4 \
    --ensemble 80 \
    --inflation 1.0 \
    --loc-radius 6.0 \
    --init-lag-days 1.0 \
    --band 0.25
