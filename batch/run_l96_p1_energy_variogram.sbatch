#!/bin/bash
#SBATCH --account=odyssey
#SBATCH --partition=Odyssey_GPU
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=10:00:00
#SBATCH --array=0-3
#SBATCH --exclude=sl-mee-br-205,sl-mee-br-206,sl-mee-br-208,sl-mee-br-209,sl-mee-br-210,sl-mee-br-211,sl-mee-br-212,sl-mee-br-213,sl-mee-br-214,sl-mee-br-215,sl-mee-br-216
#SBATCH --job-name=l96-p1-ev
#SBATCH --output=sbatch_logs/l96_p1_ev_%A_%a.out
#SBATCH --error=sbatch_logs/l96_p1_ev_%A_%a.err
# Energy and variogram scores of the P1 L96 ensembles (regular test set, S0 and S1), scored in the
# job from node-local member dumps (reports/l96/p1_energy_variogram.py score); only the per-window
# scores are kept, in experiments/l96/p1_energy_variogram/.
#   task 0: ETKF / EnKF / ETKS, 30 members, benchmark inflations (ETKF/ETKS 1.15 / 2.5, EnKF 1.2 / 3.0,
#           ETKS `correct`, full window): the run_l96_fm_score_da.sbatch command with the EnKF added;
#   task 1: VanillaCFM-2D-M seed 1; 2: SDA1-2D-M seed 1 (gw 25); 3: DirectUNet-2D-M -> SDA1-2D-M
#           (tau0 0.1, gw 1): the run_l96b_2d_variance_eval.sbatch commands with --members-file full.
# Submit from this worktree's root (task 0 writes its trajectory files to ./experiments).
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
export PATH="/Odyssey/private/rfablet/miniforge3/envs/${CONDA_ENV:-fdv-monai-proto}/bin:$PATH"
export PYTHONUNBUFFERED=1 PYTHONPATH=$PWD
NODE_TMP="/tmp/${USER}_slurm_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID:-0}"; mkdir -p "$NODE_TMP"
export TMPDIR="$NODE_TMP" MPLCONFIGDIR="$NODE_TMP/mplconfig" FDV_MEMBERS_TMP="$NODE_TMP/members"
trap 'rm -rf "$NODE_TMP"' EXIT
mkdir -p "$MPLCONFIGDIR" sbatch_logs
R=/Odyssey/private/rfablet/Python/4dvarnet-fm-opencode
L96=$R/experiments/l96
NORM=$R/experiments/l96_norm_stats_obsj2.pt
DS=$R/experiments/l96_datasets_obsj2_int100_nwin200.pt
SCORES=$L96/p1_energy_variogram
mkdir -p "$SCORES"
CFM=$L96/L96B_vanillacfm_monai2dM_ep1200_seed1
SDA=$L96/L96B_sda1_monai2dM_ep1200_seed1
DU=$L96/L96B_directunet2d_monaiM_seed1
EV="$NODE_TMP/eval"; mkdir -p "$EV"
echo "task ${SLURM_ARRAY_TASK_ID:-0} | commit $(git rev-parse --short HEAD) | $(hostname) | $(date +%T)"
case ${SLURM_ARRAY_TASK_ID:-0} in
  0) RANDOMIZE='{"F": {"randomized": true, "noise": 0.2, "biased": true, "bias": 0.1}, "c1": {"randomized": true, "noise": 0.2, "biased": true, "bias": 0.1}, "h": {"randomized": false, "noise": 0.2, "biased": false, "bias": 0.0}, "hx": {"randomized": true, "noise": 0.2, "biased": true, "bias": 0.1}, "eps": {"randomized": true, "noise": 0.2, "biased": true, "bias": 0.1}, "fast_weights": {"randomized": true, "noise": 0.2, "biased": true, "bias": 0.1}}'
     INF="s0=1.15,s1=2.5"
     if ls experiments/l96_baselines_dws500_*_s0c_ev*.json >/dev/null 2>&1; then
       echo "results cache in $PWD/experiments: the evaluators would skip every method; move it aside"; exit 1
     fi
     python -u evaluate_all_l96.py --data-cache "$DS" --da-window-steps 500 --num-test-windows 200 --t-max 3.0 \
       --obs-j 2 --obs-interval 100 --suffix _s0c_ev --randomize "$RANDOMIZE" \
       --etkf-inflation "$INF" --enkf-inflation "s0=1.2,s1=3.0" --skip-weak --skip-strong --da-fast-weights \
       --etks --etks-inflation "$INF" --save-members
     LABEL="" ;;
  1) python eval_neural_l96.py --step-power 0.5 --checkpoint $CFM/checkpoints/stage1_best.ckpt --dataset $DS \
       --num-windows 200 --n-members 30 --n-outer 20 --batch-size 25 --members-file full \
       --output "$EV/neural_eval.json"
     LABEL="--label CFM" ;;
  2) python eval_sda_l96.py --checkpoint $SDA/checkpoints/stage1_best.ckpt --config $SDA/resolved_config.yaml \
       --dataset $DS --num-windows 200 --batch-size 16 --n-members 30 --n-outer 10 --r-var 0.5 \
       --guidance-weight 25 --normalize-stats $NORM --members-file full --output "$EV/sda_eval.json"
     LABEL="--label SDA" ;;
  3) python eval_sda_directunet_hybrid_l96.py \
       --mean-checkpoint $DU/checkpoints/stage1_best.ckpt --mean-config $DU/resolved_config.yaml \
       --sda-checkpoint $SDA/checkpoints/stage1_best.ckpt --sda-config $SDA/resolved_config.yaml \
       --dataset $DS --num-windows 200 --batch-size 25 --n-members 30 --n-outer 10 --tau0 0.1 \
       --guidance-weight 1 --normalize-stats $NORM --members-file full --output "$EV/hybrid_eval.json"
     LABEL="--label D-Unet+SDA" ;;
esac
du -sh "$FDV_MEMBERS_TMP"
python -u reports/l96/p1_energy_variogram.py score "$FDV_MEMBERS_TMP" "$SCORES" $LABEL
echo "Done: task ${SLURM_ARRAY_TASK_ID:-0} at $(date +%T)"
