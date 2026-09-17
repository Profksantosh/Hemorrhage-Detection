#!/bin/bash
cd ~/ich_windowing
RD=runs/baselines_bhsd_excluded
CONDA="/opt/anaconda3/bin/conda run -n ich_windowing python -u"
echo "START $(date)"
$CONDA score_baselines_external.py --runs-dir $RD > score_baselines_external_bhsd_excluded.log 2>&1
echo "b1b2 exit=$? $(date)"
$CONDA score_baselines3ab_external.py --runs-dir $RD --wicl-dir $RD/decisive_wicl > score_baselines3ab_external_bhsd_excluded.log 2>&1
echo "b3ab exit=$? $(date)"
$CONDA score_baselines4ab_external.py --runs-dir $RD > score_baselines4ab_external_bhsd_excluded.log 2>&1
echo "b4ab exit=$? $(date)"
echo "ALL_EXTERNALS_DONE $(date)"
