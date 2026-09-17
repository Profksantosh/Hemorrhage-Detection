#!/bin/bash
cd ~/ich_windowing
RD=runs/baselines_bhsd_excluded
W=$RD/decisive_wicl
C="/opt/anaconda3/bin/conda run -n ich_windowing python -u"
echo "START $(date)"
$C score_baseline5_internal.py --runs-dir $RD --wicl-dir $W > score_baseline5_internal_bhsd_excluded.log 2>&1
echo "b5_internal exit=$? $(date)"
$C score_baseline5_external.py --runs-dir $RD --wicl-dir $W > score_baseline5_external_bhsd_excluded.log 2>&1
echo "b5_external exit=$? $(date)"
$C score_external_bhsd_excluded.py --runs-dir $RD > score_external_decisive_bhsd_excluded.log 2>&1
echo "decisive_cq500 exit=$? $(date)"
$C score_physionet_ich_external_bhsd_excluded.py --runs-dir $RD --decisive-dir $RD --physionet-root data/physionet_ich > score_physionet_bhsd_excluded.log 2>&1
echo "physionet exit=$? $(date)"
$C h1_baseline2_degradation_test_bhsd_excluded.py --runs-dir $RD > h1_bhsd_excluded.log 2>&1
echo "h1 exit=$? $(date)"
$C h3_calibration_ece_test_bhsd_excluded.py --runs-dir $RD --wicl-dir $W > h3_bhsd_excluded.log 2>&1
echo "h3 exit=$? $(date)"
$C h4_hu_recalibration_stress_test_bhsd_excluded.py --runs-dir $RD --wicl-dir $W > h4_bhsd_excluded.log 2>&1
echo "h4 exit=$? $(date)"
echo "ALL_REMAINING_DONE $(date)"
