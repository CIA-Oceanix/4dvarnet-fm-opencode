#!/bin/bash
#SBATCH --account=odyssey
#SBATCH --partition=Odyssey_GPU
#SBATCH --gres=gpu:rtx8000:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=140G
#SBATCH --time=06:00:00
#SBATCH --job-name=l96-report-bundles
#SBATCH --output=sbatch_logs/l96_report_bundles_%j.out
#SBATCH --error=sbatch_logs/l96_report_bundles_%j.err
# Build and verify the L96 benchmark report input bundles (scripts/bundle_report_inputs.py).
#   record: run each generator against the topic worktrees; its outputs must equal the checked-in ones
#   link:   hard-link what it read into experiments/l96/report_inputs/
#   run:    regenerate from the bundles only; outputs must again equal the checked-in ones
# PHASES selects phases (default "record link run"); RECORD_REPORTS / RUN_REPORTS limit record / run.
# No git calls: compute-node Kerberos tickets can expire mid-job, so outputs are compared with diff -r.
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
PY=/Odyssey/private/rfablet/miniforge3/envs/fdv-monai-proto/bin/python
export MPLCONFIGDIR="/tmp/mplconfig_${SLURM_JOB_ID}"; mkdir -p "$MPLCONFIGDIR"
export OMP_NUM_THREADS=4
PHASES="${PHASES:-record link run}"
LOGS="experiments/report_inputs_logs"; mkdir -p "$LOGS"
OUTS="reports/l96/outputs"
REF="$LOGS/outputs_checked_in_${SLURM_JOB_ID}"
cp -a "$OUTS" "$REF"
reset() { rm -f experiments/l96_benchmark_default_metrics.json experiments/l96_benchmark_extended_metrics.json; }
compare() {
  local log="$LOGS/diff_after_$1_${SLURM_JOB_ID}.txt"
  if diff -rq "$REF" "$OUTS" > "$log"; then echo "$1: outputs identical"
  else echo "$1: outputs DIFFER"; cat "$log"; diff -ru "$REF" "$OUTS" --exclude='*.png' > "$log.patch" || true; fi
  rm -rf "$OUTS"; cp -a "$REF" "$OUTS"
}

for phase in $PHASES; do
  echo "===== $phase $(date)"
  case $phase in
    record) reset; $PY scripts/bundle_report_inputs.py record ${RECORD_REPORTS:-} --workdir "$LOGS"; compare record ;;
    link)   $PY scripts/bundle_report_inputs.py link --apply --workdir "$LOGS" ;;
    run)    reset; $PY scripts/bundle_report_inputs.py run ${RUN_REPORTS:-}; compare run ;;
  esac
done
echo "===== done $(date)"
