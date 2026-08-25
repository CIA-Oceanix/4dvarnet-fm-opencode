#SBATCH --account=odyssey
#SBATCH --partition=Odyssey_GPU
#SBATCH --gres=gpu:rtx8000:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --job-name=l96-joint-eval-ens30
#SBATCH --array=0-3

set -euo pipefail

cd /Odyssey/private/rfablet/Python/4dvarnet-joint-eval

export PATH="/Odyssey/private/rfablet/miniforge3/envs/fdv/bin:$PATH"
export PYTHONUNBUFFERED=1

DATASET="experiments/l96_datasets_obsj2_int100_nwin200.pt"

# case | n_members | n_outer | extra_flags (tau0 flag required for tau=0-trained joint CFM)
TASKS=(
  "L7_joint_cfm_s0s1|30|1|--train-tau0-only"
  "L7_joint_cfm_s0s1|30|10|--train-tau0-only"
  "L8_joint_direct_unet_s0s1|30|1|"
  "L8_joint_direct_unet_s0s1|30|10|"
)

SPEC=${TASKS[$SLURM_ARRAY_TASK_ID]}
EXP=${SPEC%%|*}
NM=${SPEC#*|}
NM=${NM%%|*}
NO=${SPEC##*|}
NO=${NO%%|*}
FLAGS=${SPEC##*|}

echo "============================================"
echo " L96 Joint Neural Eval (ens30) | $(date)"
echo "  Task ${SLURM_ARRAY_TASK_ID} / Experiment: ${EXP} |"
echo "  Members: ${NM} | Outer: ${NO}"
echo "  GPU: $(nvidia-smi --query-gpu=name --format=csv -i 0 2>/dev/null | tail -1)"
echo "============================================"

python eval_joint_neural_l96.py \
  --checkpoint "experiments/${EXP}/checkpoints/stage1_best.ckpt" \
  --dataset "${DATASET}" \
  --num-windows 200 \
  --obs-interval 100 \
  --obs-j 2 \
  --device cuda \
  --n-members ${NM} \
  --n-outer ${NO} \
  ${FLAGS} \
  --output "experiments/${EXP}/joint_neural_eval_ens30.json"

echo ""
echo " Done: ${EXP} ens30 (M=${NM}, k=${NO}) at $(date)"
