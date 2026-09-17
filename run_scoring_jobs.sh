#!/bin/bash
set -e
cd /home/fcse.santoshkumar/ich_windowing
source /opt/anaconda3/etc/profile.d/conda.sh
conda activate ich_windowing

RD=runs/baselines_bhsd_excluded

echo "=== [$(date -Is)] H4 WICL ablation scoring ==="
python h4_wicl_ablation.py --runs-dir "$RD" \
  --wicl-dir "$RD/decisive_wicl" \
  --pred-only-dir "$RD/wicl_ablation_pred_only" \
  --embed-only-dir "$RD/wicl_ablation_embed_only" \
  --rsna-root data/rsna/rsna-intracranial-hemorrhage-detection \
  > h4_wicl_ablation.log 2>&1
echo "=== [$(date -Is)] H4 ablation done ==="

echo "=== [$(date -Is)] H1 fold-1 replication scoring ==="
python h1_fold1_replication.py --runs-dir "$RD" \
  --b2-subdir baseline2_fixed_three_fold1 \
  --cq500-root data/cq500 \
  > h1_fold1_replication.log 2>&1
echo "=== [$(date -Is)] H1 fold-1 done ==="

echo ALL_SCORING_JOBS_DONE
