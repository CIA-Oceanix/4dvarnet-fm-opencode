#!/bin/bash
#SBATCH --account=odyssey
#SBATCH --partition=Odyssey_GPU
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=08:00:00
#SBATCH --array=0-1
#SBATCH --job-name=l96-rankhist
#SBATCH --output=/Odyssey/private/rfablet/Python/4dvarnet-fm-opencode/experiments/l96_rank_hist/rh_%a_%A.out
#SBATCH --error=/Odyssey/private/rfablet/Python/4dvarnet-fm-opencode/experiments/l96_rank_hist/rh_%a_%A.err
# Rank histograms (P1 paper C4). All large intermediates (DA trajectories and members)
# live on the node-local /tmp and are deleted on exit; only JSON/PNG come back to Odyssey.
# Task 0: P1 protocol (DA inflation 2.0) vs P1 flows + SDA; task 1: benchmark default
# (DA S0 inflation 1.5, DA fast weights) vs the benchmark-default flows.
# Submit from the repository root after `mkdir -p experiments/l96_rank_hist`; the code copied
# to /tmp is the submitting checkout.
set -euo pipefail
export PATH="/Odyssey/private/rfablet/miniforge3/envs/fdv-monai-proto/bin:$PATH"
export PYTHONUNBUFFERED=1
i=$SLURM_ARRAY_TASK_ID
T=/tmp/rankhist_${SLURM_JOB_ID}_${i}
trap 'rm -rf "$T"' EXIT
FREE=$(df -BG --output=avail /tmp | tail -1 | tr -dc 0-9)
[ "$FREE" -ge 20 ] || { echo "only ${FREE}G free on /tmp"; exit 1; }
mkdir -p $T/code/experiments $T/mpl; export MPLCONFIGDIR=$T/mpl
rsync -a --exclude experiments --exclude .git --exclude 'reports/l96/outputs' "${SLURM_SUBMIT_DIR}"/ $T/code/
ln -s /Odyssey/private/rfablet/Python/4dvarnet-fm-opencode/experiments/l96_datasets_obsj2_int100_nwin200.pt $T/code/experiments/
cd $T/code
if [ $i -eq 0 ]; then
  INF=2.0; TAG=p1; EXTRA=""
else
  INF=1.5; TAG=bench; EXTRA="--da-fast-weights"
fi
python -u evaluate_all_l96.py --da-window-steps 500 --num-test-windows 200 --t-max 3.0   --enkf-inflation $INF --etkf-inflation $INF --obs-j 2 --obs-interval 100 --suffix _s0c   --randomize '{"F": {"randomized": true, "noise": 0.2, "biased": true, "bias": 0.1}, "c1": {"randomized": true, "noise": 0.2, "biased": true, "bias": 0.1}, "h": {"randomized": false, "noise": 0.2, "biased": false, "bias": 0.0}, "hx": {"randomized": true, "noise": 0.2, "biased": true, "bias": 0.1}, "eps": {"randomized": true, "noise": 0.2, "biased": true, "bias": 0.1}, "fast_weights": {"randomized": true, "noise": 0.2, "biased": true, "bias": 0.1}}' --skip-weak --skip-strong --cases s0 --save-members $EXTRA
ls -la experiments/
ETKF=$(ls experiments/*ETKF*_members.npz 2>/dev/null | head -1 || true)
ENKF=$(ls experiments/*EnKF*_members.npz 2>/dev/null | head -1 || true)
[ -z "$ETKF" ] && ETKF=$(ls experiments/*s0*etkf*members.npz 2>/dev/null | head -1)
echo "DA members: $ETKF | $ENKF"
cp experiments/*.json /Odyssey/private/rfablet/Python/4dvarnet-fm-opencode/experiments/l96_rank_hist/ 2>/dev/null || true
if [ $i -eq 0 ]; then
  IN="--input ETKF=$ETKF --input EnKF=$ENKF
      --input VanillaCFM-M=/Odyssey/private/rfablet/Python/4dvarnet-fm-opencode/4dvarnet-fm-fdv-tau-aware/experiments/A1_vanillacfm_monaiM_l96/ens30_no20/members_s0.npz
      --input PredictStateCFM-M=/Odyssey/private/rfablet/Python/4dvarnet-fm-opencode/4dvarnet-fm-fdv-tau-aware/experiments/A2_predictstatecfm_monaiM_l96/ens30_no20/members_s0.npz
      --input VanillaCFM-M_uniform10=/Odyssey/private/rfablet/Python/4dvarnet-fm-opencode/4dvarnet-fm-fdv-tau-aware/experiments/A1_vanillacfm_monaiM_l96/ens30_no10/members_s0.npz
      --input PredictStateCFM-M_uniform10=/Odyssey/private/rfablet/Python/4dvarnet-fm-opencode/4dvarnet-fm-fdv-tau-aware/experiments/A2_predictstatecfm_monaiM_l96/ens30_no10/members_s0.npz
      --input SDA1-M=/Odyssey/private/rfablet/Python/4dvarnet-fm-opencode/4dvarnet-fm-fdv-tau-aware/experiments/B4_sda1_monaiM_l96/ens30_gw20/members_s0.npz
      --input SDA2-M=/Odyssey/private/rfablet/Python/4dvarnet-fm-opencode/4dvarnet-fm-fdv-tau-aware/experiments/A3_sda2_monaiM_l96/ens30_gw20/members_s0.npz"
else
  IN="--input ETKF=$ETKF --input EnKF=$ENKF"
  for s in 1 2 3; do
    IN="$IN --input PredictStateCFM-M_s$s=/Odyssey/private/rfablet/Python/4dvarnet-fm-opencode/4dvarnet-fm-sda-nanfix/experiments/L96B_predictstatecfm_monaiM_seed$s/ens30_no20/members_s0.npz"
    IN="$IN --input VanillaCFM-M_s$s=/Odyssey/private/rfablet/Python/4dvarnet-fm-opencode/4dvarnet-fm-bench-default/experiments/L96B_vanillacfm_monaiM_seed$s/ens30_no20/members_s0.npz"
  done
fi
python reports/l96/probe_rank_histograms.py $IN --out $T/rank_hist_${TAG}_s0.json --png $T/rank_hist_${TAG}_s0.png
cp $T/rank_hist_${TAG}_s0.json $T/rank_hist_${TAG}_s0.png /Odyssey/private/rfablet/Python/4dvarnet-fm-opencode/experiments/l96_rank_hist/
echo done
