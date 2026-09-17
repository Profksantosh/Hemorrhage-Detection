#!/bin/bash
# New experiments requested for the manuscript revision (2026-09-14):
#   1. WICL component ablation (pred-only, embed-only) on the BHSD-excluded
#      fold-0 protocol, matched to every other condition (densenet121, 12
#      epochs, 20k samples, fold 0, lr 1e-4, seed 0).
#   2. Second-fold H1 replication: baseline2 (fixed_three_window) retrained
#      on fold 1 instead of fold 0, same protocol otherwise, to check
#      whether the primary H1 CQ500 result is fold-specific.
# Full WICL (lambda_c=1.0, lambda_e=0.5) already exists at
# runs/baselines_bhsd_excluded/decisive_wicl -- not rerun here.
set -e
cd /home/fcse.santoshkumar/ich_windowing
source /opt/anaconda3/etc/profile.d/conda.sh
conda activate ich_windowing

OUT=runs/baselines_bhsd_excluded
mkdir -p "$OUT"

COMMON="--rsna-root data/rsna/rsna-intracranial-hemorrhage-detection \
  --backbone densenet121 --pretrained \
  --epochs 12 --batch-size 32 --lr 1e-4 --weight-decay 1e-5 \
  --n-folds 5 --max-samples 20000 --seed 0 --device cuda"

run() {
  local condition=$1
  local outdir=$2
  local logfile=$3
  shift 3
  echo "=== [$(date -Is)] training condition=$condition -> $outdir $@ ==="
  python -m ich_gen.train --condition "$condition" $COMMON \
    --out-dir "$outdir" "$@" > "$logfile" 2>&1
  if ! grep -q "loaded 191 RSNA patient id(s) to exclude" "$logfile"; then
    echo "FATAL: $logfile does not show BHSD exclusion being applied -- aborting." >&2
    exit 1
  fi
  echo "=== [$(date -Is)] done condition=$condition ==="
}

run wicl "$OUT/wicl_ablation_pred_only"  wicl_ablation_pred_only_train_bhsd_excluded.log  --fold 0 --lambda-consistency 1.0 --lambda-embedding 0.0
run wicl "$OUT/wicl_ablation_embed_only" wicl_ablation_embed_only_train_bhsd_excluded.log --fold 0 --lambda-consistency 0.0 --lambda-embedding 0.5
run fixed_three_window "$OUT/baseline2_fixed_three_fold1" baseline2_fixed_three_fold1_train_bhsd_excluded.log --fold 1

echo ALL_ABLATION_AND_FOLD1_DONE
