#!/bin/bash
#SBATCH --account=odyssey
#SBATCH --partition=Odyssey_GPU
#SBATCH --gres=gpu:rtx8000:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --job-name=maooam_comparison
#SBATCH --output=sbatch_logs/maooam_comparison_%j.out
#SBATCH --error=sbatch_logs/maooam_comparison_%j.err

cd /Odyssey/private/rfablet/Python/4dvarnet-fm-opencode || exit 1

# Use fdv env Python directly
PYTHON=/Odyssey/private/rfablet/miniforge3/envs/fdv/bin/python

echo "=== MAOOAM Comparison Dataset Generation ==="
echo "Host: $(hostname)"
echo "Date: $(date)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)"
echo ""

for cfg in ddv2016 vspd2019 hamilton2023; do
    echo "--- Generating $cfg ---"
    $PYTHON batch/gen_maooam_comparison.py --config "$cfg" --n-trajs 5 --device cuda --no-compile 2>&1
    echo ""
done

echo "=== All done ==="
date