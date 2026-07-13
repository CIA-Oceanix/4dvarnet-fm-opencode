#!/bin/bash
#SBATCH --account=odyssey
#SBATCH --partition=Odyssey_GPU
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --job-name=sw-dynamics
#SBATCH --output=sbatch_logs/sw_dynamics_%j.out
#SBATCH --error=sbatch_logs/sw_dynamics_%j.err

set -euo pipefail
cd /Odyssey/private/rfablet/Python/4dvarnet-fm-opencode
export PATH="/Odyssey/private/rfablet/miniforge3/envs/fdv/bin:$PATH"
export PYTHONUNBUFFERED=1

echo "============================================"
echo " SW Dynamics Smoke Test | $(date)"
echo "============================================"
echo "  GPU: $(nvidia-smi --query-gpu=name --format=csv -i 0 2>/dev/null | tail -1)"

python -c "
import torch
print(f'CUDA: {torch.cuda.is_available()} | {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"CPU\"}')
from models.shallow_water_dynamics import ShallowWaterDynamics
dyn = ShallowWaterDynamics(Nx=64, Ny=64)
traj, forcing = dyn.generate_full_trajectory(3000, seed=42)
print(f'traj shape: {traj.shape}, finite: {traj.isfinite().all().item()}')
print(f'h1 range: [{traj[:, :4096].min():.4f}, {traj[:, :4096].max():.4f}]')
print(f'|u1| range: [{traj[:, 4096:8192].abs().min():.4f}, {traj[:, 4096:8192].abs().max():.4f}]')
print('PASSED')
"

echo "============================================"
echo " Done: $(date)"
echo "============================================"
