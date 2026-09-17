#!/bin/bash
# Retrain the 7 Santosh-confirmed conditions (2026-09-11 BHSD/RSNA
# patient-overlap remediation) on the patient-excluded RSNA fold-0 pool.
# Same hyperparameters as every prior baselines_v1/decisive_v2 run for
# comparability: densenet121, 12 epochs, 20k samples, fold 0, lr 1e-4,
# seed 0. Exclusion is now the DEFAULT behavior of ich_gen.train (see
# build_argparser's --bhsd-exclusion, default True) -- no extra flag
# needed, but nothing here relies on that default silently: each run
# logs "loaded 191 RSNA patient id(s) to exclude" at start, which we
# grep-verify below before declaring ALL_V2_DONE.
#
# NOT included (per task scope -- see agent report for why):
#   - baseline4c (random_window_single): NOT in Santosh's confirmed
#     list of 7, but IS needed for the "WICL vs baseline4c" decisive
#     comparison referenced in step 5 of the task. Flagged, not
#     unilaterally retrained.
#   - baseline5 (semi_supervised): explicitly excluded per task
#     instructions -- open question for the project lead.
set -e
cd /home/fcse.santoshkumar/ich_windowing
source /opt/anaconda3/etc/profile.d/conda.sh
conda activate ich_windowing

OUT=runs/baselines_bhsd_excluded
mkdir -p "$OUT"

COMMON="--rsna-root data/rsna/rsna-intracranial-hemorrhage-detection \
  --backbone densenet121 --pretrained \
  --epochs 12 --batch-size 32 --lr 1e-4 --weight-decay 1e-5 \
  --fold 0 --n-folds 5 --max-samples 20000 --seed 0 --device cuda"

run() {
  local condition=$1
  local outdir=$2
  local logfile=$3
  shift 3
  echo "=== [$(date -Is)] training condition=$condition -> $outdir ==="
  python -m ich_gen.train --condition "$condition" $COMMON \
    --out-dir "$outdir" "$@" > "$logfile" 2>&1
  if ! grep -q "loaded 191 RSNA patient id(s) to exclude" "$logfile"; then
    echo "FATAL: $logfile does not show BHSD exclusion being applied -- aborting." >&2
    exit 1
  fi
  echo "=== [$(date -Is)] done condition=$condition ==="
}

run fixed_single_window   "$OUT/baseline1_fixed_single"    baseline1_fixed_single_train_bhsd_excluded.log
run fixed_three_window    "$OUT/baseline2_fixed_three"     baseline2_fixed_three_train_bhsd_excluded.log
run adaptive_hrt           "$OUT/baseline3a_hrt"            baseline3a_hrt_train_bhsd_excluded.log
run adaptive_wem           "$OUT/baseline3b_wem"            baseline3b_wem_train_bhsd_excluded.log
run dg_augment              "$OUT/baseline4a_augmentation"   baseline4a_augmentation_train_bhsd_excluded.log
run dg_coral                 "$OUT/baseline4b_coral"          baseline4b_coral_train_bhsd_excluded.log --coral-weight 1000
run wicl                      "$OUT/decisive_wicl"             decisive_wicl_train_bhsd_excluded.log

echo ALL_V2_BHSD_EXCLUDED_DONE
