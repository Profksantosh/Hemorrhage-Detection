#!/bin/bash
# Addendum to run_bhsd_excluded_v2.sh (2026-09-11 BHSD/RSNA patient-overlap
# remediation). Per Santosh's (PI) direct confirmation, two more conditions
# trained on the original contaminated RSNA pool also need retraining on
# the patient-excluded fold, for consistency with the other 7:
#   - baseline4c (condition=random_window_single): the WICL comparison
#     target used throughout the manuscript as the decisive-check baseline.
#   - baseline5  (condition=semi_supervised): self-training teacher-student.
# Same protocol/hyperparameters as every other v2 run for comparability:
# densenet121, 12 epochs, 20k samples, fold 0, lr 1e-4, seed 0.
# Exclusion is the DEFAULT behavior of ich_gen.train (--bhsd-exclusion,
# default True) -- no extra flag needed, but each run's exclusion is
# grep-verified from its log before being declared done, same as the
# parent v2 driver does.
#
# This script does NOT run concurrently with run_bhsd_excluded_v2.sh: it
# polls until that script's process has exited AND its driver log shows
# the ALL_V2_BHSD_EXCLUDED_DONE sentinel, then runs baseline4c and
# baseline5 sequentially on the same (single, shared) GPU. If the parent
# script's process disappears WITHOUT that sentinel (crash/kill), this
# script aborts rather than guessing about state.
set -e
cd /home/fcse.santoshkumar/ich_windowing
source /opt/anaconda3/etc/profile.d/conda.sh
conda activate ich_windowing

PARENT_LOG=run_bhsd_excluded_v2_driver.log
DONE_SENTINEL='ALL_V2_BHSD_EXCLUDED_DONE'

echo "[$(date -Is)] addendum driver waiting for run_bhsd_excluded_v2.sh to finish..."
while pgrep -f 'run_bhsd_excluded_v2\.sh' > /dev/null 2>&1; do
  sleep 60
done

if ! grep -q "$DONE_SENTINEL" "$PARENT_LOG" 2>/dev/null; then
  echo "[$(date -Is)] FATAL: run_bhsd_excluded_v2.sh process ended but $DONE_SENTINEL not found in $PARENT_LOG -- it may have failed/aborted. Refusing to start baseline4c/baseline5 to avoid running on top of an unknown/contended state." >&2
  exit 1
fi
echo "[$(date -Is)] parent 7-condition run confirmed complete ($DONE_SENTINEL found). Starting baseline4c + baseline5 on the excluded fold."

OUT=runs/baselines_bhsd_excluded
mkdir -p "$OUT"

COMMON="--rsna-root data/rsna/rsna-intracranial-hemorrhage-detection   --backbone densenet121 --pretrained   --epochs 12 --batch-size 32 --lr 1e-4 --weight-decay 1e-5   --fold 0 --n-folds 5 --max-samples 20000 --seed 0 --device cuda"

run() {
  local condition=$1
  local outdir=$2
  local logfile=$3
  shift 3
  echo "=== [$(date -Is)] training condition=$condition -> $outdir ==="
  python -m ich_gen.train --condition "$condition" $COMMON     --out-dir "$outdir" "$@" > "$logfile" 2>&1
  if ! grep -q "loaded 191 RSNA patient id(s) to exclude" "$logfile"; then
    echo "FATAL: $logfile does not show BHSD exclusion being applied -- aborting." >&2
    exit 1
  fi
  echo "=== [$(date -Is)] done condition=$condition ==="
}

run random_window_single "$OUT/baseline4c_random_window_single" baseline4c_random_window_single_train_bhsd_excluded.log
run semi_supervised        "$OUT/baseline5_selftraining"          baseline5_selftraining_train_bhsd_excluded.log

echo ALL_V2_ADDENDUM_BHSD_EXCLUDED_DONE
