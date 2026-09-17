#!/bin/bash
set -e
cd /home/fcse.santoshkumar/ich_windowing
source /opt/anaconda3/etc/profile.d/conda.sh
conda activate ich_windowing
python -m ich_gen.train --condition semi_supervised   --rsna-root data/rsna/rsna-intracranial-hemorrhage-detection   --backbone densenet121 --pretrained   --epochs 12 --batch-size 32 --lr 1e-4 --weight-decay 1e-5   --fold 0 --n-folds 5 --max-samples 20000 --seed 0 --device cuda   --out-dir runs/baselines_v1/baseline5_selftraining   > baseline5_selftraining_train.log 2>&1
echo BASELINE5_DONE
