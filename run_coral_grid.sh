#!/bin/bash
set -e
cd /home/fcse.santoshkumar/ich_windowing
source /opt/anaconda3/etc/profile.d/conda.sh
conda activate ich_windowing
for W in 1 10 50 200 1000; do
  echo "=== training dg_coral coral_weight=$W ==="
  python -m ich_gen.train --condition dg_coral     --rsna-root data/rsna/rsna-intracranial-hemorrhage-detection     --backbone densenet121 --pretrained     --epochs 12 --batch-size 32 --lr 1e-4 --weight-decay 1e-5     --fold 0 --n-folds 5 --max-samples 20000 --seed 0 --device cuda     --coral-weight $W     --out-dir runs/baselines_v1/baseline4b_coral_tuning/w$W     > coral_grid_w${W}_train.log 2>&1
  echo "=== done w=$W ==="
done
echo ALL_CORAL_GRID_DONE
