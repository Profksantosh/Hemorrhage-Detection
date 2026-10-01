#!/bin/bash
ulimit -n 65536
cd /home/fcse.santoshkumar/ich_windowing
source /opt/anaconda3/etc/profile.d/conda.sh
conda activate ich_windowing
export PYTHONPATH=$PWD:$PYTHONPATH
BIG=runs/g1_100k
BM="--model BIG_B2_seed0=$BIG/b2_seed0:plain --model BIG_B4c_seed0=$BIG/b4c_seed0:plain --model BIG_WICL_seed0=$BIG/wicl_seed0:wicl"
echo "=== $(date -Is) rsna per-patient (100k)"
python g1_rsna_patient_infer.py --out-dir runs/g1_rsna_patient $BM > logs_g1/score_100k_rsna_retry.log 2>&1; echo "rsna EXIT=$?"
echo "=== $(date -Is) harmonized (100k)"
python g1_thickness_harmonize_infer.py --out-dir runs/g1_cq500_harmonized $BM > logs_g1/harmonize_100k.log 2>&1; echo "harm EXIT=$?"
echo "ALL_100K_RERUN_DONE $(date -Is)"
