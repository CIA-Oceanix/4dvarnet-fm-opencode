#!/bin/bash
#SBATCH --account=odyssey
#SBATCH --partition=Odyssey_GPU
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --job-name=4dvarnet-cs3cs4-eval
#SBATCH --output=sbatch_logs/cs3cs4_eval_%j.out
#SBATCH --error=sbatch_logs/cs3cs4_eval_%j.err

set -euo pipefail

cd /Odyssey/private/rfablet/Python/4dvarnet-fm-opencode

export PATH="/Odyssey/private/rfablet/miniforge3/envs/fdv/bin:$PATH"
export CONDA_DEFAULT_ENV=fdv
export PYTHONUNBUFFERED=1

echo "============================================"
echo " 4DVarNet-FM CS3/CS4 Evaluation | $(date)"
echo "============================================"
echo "  GPU: $(nvidia-smi --query-gpu=name --format=csv -i 0 2>/dev/null | tail -1)"
echo ""

F3_DIR=experiments/F3_vanilla_cfm_rand

if [ ! -f "$F3_DIR/checkpoints/stage2.pt" ]; then
    echo "Copying stage1.pt → checkpoints/stage2.pt for evaluate_all.py compatibility..."
    cp "$F3_DIR/checkpoints/stage1.pt" "$F3_DIR/checkpoints/stage2.pt"
fi

echo "Running evaluate_all.py for F3 on CS1–CS4..."
python evaluate_all.py --experiments F3_vanilla_cfm_rand

echo ""
echo "============================================"
echo " Done at $(date)"
echo "============================================"