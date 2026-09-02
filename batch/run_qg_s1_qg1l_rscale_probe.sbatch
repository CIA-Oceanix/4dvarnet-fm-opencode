#!/bin/bash
#SBATCH --job-name=qg_s1_qg1l_rs
#SBATCH --partition=Odyssey_GPU
#SBATCH --gres=gpu:a40:1
#SBATCH --time=04:00:00
#SBATCH --array=0-3
#SBATCH --output=/tmp/opencode/qgs1qg1lrs_%A_%a.out
#SBATCH --error=/tmp/opencode/qgs1qg1lrs_%A_%a.err

set -e

# S1 structural-model-error probe: does model-error-aware observation covariance
# (obs_var_r_scale) move the 1-layer (qg1l) DA model toward the free forecast at
# nx=64? Under the 1-layer-vs-2-layer mismatch the nonlocal psi obs are
# over-trusted (DA worse than the free forecast, improv~0.39 at default R).
# Inflating R models the unmodelled structural error. Production result: improv
# rises monotonically 0.39 -> 0.81 (rs 1/100/1e4) toward the free-forecast limit
# (1.0) but does NOT cross it at nx=64; the S0 (error-free) anchor is unaffected.
# A scale-relative background additive was probed separately and provably
# over-disperses the localized ETKF even at a regime-relative scale at nx=64
# (it diverges the error-free S0), so it is NOT a shipped lever here.
#
# Cells (R-scale over obs-noise variance on q- and psi-obs paths):
#   0: psi  rs=1    (reference: DA worse than free forecast)
#   1: psi  rs=100  (model-error-aware R: improves toward free)
#   2: psi  rs=1e4  (near free-forecast limit)
#   3: q    rs=1    (local PV obs)
LAG=1.0
OUT=reports/qg/outputs/qg_s1_qg1l_rscale

case $SLURM_ARRAY_TASK_ID in
  0) OV=psi; RS=1;    TAG=s1_qg1l_psi_rs1;;
  1) OV=psi; RS=100;  TAG=s1_qg1l_psi_rs100;;
  2) OV=psi; RS=1e4;  TAG=s1_qg1l_psi_rs1e4;;
  3) OV=q;   RS=1;    TAG=s1_qg1l_q_rs1;;
esac

/Odyssey/private/rfablet/miniforge3/envs/fdv/bin/python evaluation/sweep_qg_baselines.py \
    --nx 64 \
    --window-days 30 \
    --num-windows 5 \
    --spinup-years 2.0 \
    --ensemble 80 \
    --inflation-list 1.0 \
    --loc-list 6 \
    --method-list etkf \
    --init lagged \
    --geometry random_columns \
    --cols-per-day 4 \
    --obs-noise-frac-list 0.01 \
    --init-lag-days-list ${LAG} \
    --band 0.25 \
    --obs-var ${OV} \
    --obs-var-r-scale-list ${RS} \
    --scenarios test_s0,test_s1_qg1l \
    --outdir ${OUT} \
    --tag ${TAG}
