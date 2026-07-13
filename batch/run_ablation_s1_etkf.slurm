#!/bin/bash
#SBATCH --time=00:15:00
#SBATCH --partition=Odyssey_GPU
#SBATCH --gres=gpu:rtx8000:1
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --chdir=/Odyssey/private/rfablet/Python/4dvarnet-fm-opencode
#SBATCH --output=experiments/ablation_%A_%a.out
#SBATCH --error=experiments/ablation_%A_%a.err
#SBATCH --array=0-24

set -euo pipefail

PYTHON="/Odyssey/private/rfablet/miniforge3/envs/fdv/bin/python3"

# Five factors, each with N values (total 25 array jobs, 0-24)
# Factor 0: s1_j (J order of S1 DA model)
# Factor 1: truth_fast_weights (weight for unobserved Y3/Y4 fast vars)
# Factor 2: inflation
# Factor 3: ensemble_size (N)
# Factor 4: param_bias (F bias)

S1_J_VALS=(1 2 3 4 4)
WEIGHT_VALS=(1.0 0.5 0.1 0.01 0.01)
INFL_VALS=(1.0 1.05 1.1 1.2 1.5)
ENS_VALS=(10 20 30 50 100)
BIAS_VALS=(0.0 0.05 0.1 0.2 0.3)

idx=$SLURM_ARRAY_TASK_ID
factor=$(( idx / 5 ))
val_idx=$(( idx % 5 ))

LABEL_PREFIX=""
EXTRA=""

case $factor in
    0)
        val="${S1_J_VALS[$val_idx]}"
        LABEL_PREFIX="ablation_s1j_${val}"
        EXTRA="--s1-j ${val}"
        ;;
    1)
        val="${WEIGHT_VALS[$val_idx]}"
        LABEL_PREFIX="ablation_weight_${val}"
        EXTRA="--truth-fast-weights-unobserved ${val}"
        ;;
    2)
        val="${INFL_VALS[$val_idx]}"
        LABEL_PREFIX="ablation_inf_${val}"
        EXTRA="--inflation ${val}"
        ;;
    3)
        val="${ENS_VALS[$val_idx]}"
        LABEL_PREFIX="ablation_ens_${val}"
        EXTRA="--ensemble-size ${val}"
        ;;
    4)
        val="${BIAS_VALS[$val_idx]}"
        LABEL_PREFIX="ablation_bias_${val}"
        EXTRA="--param-bias ${val}"
        ;;
esac

echo "Factor $factor, value $val_idx: $LABEL_PREFIX"

$PYTHON -u evaluation/run_l96_sweep2.py \
    --label "${LABEL_PREFIX}" \
    --param-bias 0.0 --forcing-state-bias 0.0 \
    --num-windows 5 --da-window-steps 500 \
    --obs-j 2 --s1-j 2 --obs-interval 5 \
    --ensemble-size 30 --inflation 1.1 \
    --truth-fast-weights-unobserved 0.1 \
    --skip-weak --skip-strong \
    ${EXTRA}