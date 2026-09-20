#!/bin/bash
#SBATCH --job-name=qg_neural_s0s1_eval
#SBATCH --partition=Odyssey_GPU
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=96G
#SBATCH --time=02:00:00
#SBATCH --output=batch/logs/qg_neural_s0s1_eval_%j.out
#SBATCH --error=batch/logs/qg_neural_s0s1_eval_%j.err

set -e

# Regenerates the neural half of reports/qg/outputs/qg_neural_report.md's
# benchmark table (Q1-Q4, S0 and S1) from the archived checkpoints. This is the
# run that makes the published table reproducible rather than historical: it
# resolves checkpoints, architectures and normalization stats through
# evaluation/archive.py, so it does not depend on the worktree that trained
# them.
#
#   sbatch batch/run_qg_neural_s0s1_eval.sbatch [OUTPUT_JSON]
#
# With no argument it overwrites the tracked
# reports/qg/outputs/qg_neural_s0_s1_cross_scenario/results_lag5_noise0.05_bias0.1.json
# (the report's own input); pass a path to write elsewhere and diff first.
CACHE_DIR="${CACHE_DIR:-/Odyssey/private/rfablet/Python/4dvarnet-fm-qg-100samples/reports/qg/outputs/qg_windows_1000_100_100/cache}"
PY="${PY:-/Odyssey/private/rfablet/miniforge3/envs/fdv-monai-proto/bin/python}"

OUT_ARG=()
if [ -n "$1" ]; then OUT_ARG=(--output "$1"); fi

# --s1-param-bias/--s1-amp-bias 0.1 match the DA S1 reference campaign
# (qg_da_s1_scratch.py), deliberately milder than QGConfig's own 0.15 default --
# without them the neural rows would not be apples-to-apples with the DA rows
# they are tabulated against.
$PY eval_qg_neural_s0_s1.py \
    --schemes Q1 Q2 Q3 Q4 \
    --s1-param-bias 0.1 --s1-amp-bias 0.1 \
    --cache-dir "$CACHE_DIR" "${OUT_ARG[@]}"
