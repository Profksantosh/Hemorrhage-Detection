#!/bin/bash
# G1 revision, round 2 (2026-10-01), sequential on hp100. Every step skips outputs that already exist.
#   0. CQ500 slice-thickness harmonization test for the 9 existing DenseNet models (inference only)
#   1. ConvNeXt-Tiny: B2 / random window / WICL, seeds 0-2, 20,000-slice budget (same protocol otherwise)
#   2. Score ConvNeXt models: CQ500 + PhysioNet (upright), RSNA per-patient, H4, harmonized CQ500
#   3. DenseNet-121 with a 100,000-slice budget: B2 / random window / WICL, seed 0
#   4. Score 100k models: CQ500 + PhysioNet (upright), RSNA per-patient, harmonized CQ500 (no H4: its held-out
#      subset lies inside the 100k pool)
set -e
cd /home/fcse.santoshkumar/ich_windowing
source /opt/anaconda3/etc/profile.d/conda.sh
conda activate ich_windowing
export PYTHONPATH=$PWD:$PYTHONPATH
mkdir -p logs_g1 runs/g1_convnext runs/g1_100k runs/g1_ext_more runs/g1_cq500_harmonized
stamp() { echo "=== [$(date -Is)] $*"; }
S0=runs/baselines_bhsd_excluded; SD=runs/g1_seeds; CX=runs/g1_convnext; BIG=runs/g1_100k

stamp "step 0: thickness harmonization, existing DenseNet models"
python g1_thickness_harmonize_infer.py --out-dir runs/g1_cq500_harmonized \
  --model B2_seed0=$S0/baseline2_fixed_three:plain --model B4c_seed0=$S0/baseline4c_random_window_single:plain \
  --model WICL_seed0=$S0/decisive_wicl:wicl \
  --model B2_seed1=$SD/b2_seed1:plain --model B4c_seed1=$SD/b4c_seed1:plain --model WICL_seed1=$SD/wicl_seed1:wicl \
  --model B2_seed2=$SD/b2_seed2:plain --model B4c_seed2=$SD/b4c_seed2:plain --model WICL_seed2=$SD/wicl_seed2:wicl \
  > logs_g1/harmonize_densenet.log 2>&1

COMMON="--rsna-root data/rsna/rsna-intracranial-hemorrhage-detection --pretrained \
  --epochs 12 --batch-size 32 --lr 1e-4 --weight-decay 1e-5 --fold 0 --n-folds 5 --device cuda"
train() {
  local condition=$1 out=$2 log=$3; shift 3
  if [ -f $out/checkpoint_last.pt ] && grep -q '"epoch": 11' $out/history.json 2>/dev/null; then
    stamp "skip $out (done)"; return; fi
  stamp "training $condition -> $out"
  python -m ich_gen.train --condition $condition $COMMON --out-dir $out "$@" > $log 2>&1
  grep -q "loaded 191 RSNA patient id(s) to exclude" $log || { echo "FATAL: BHSD exclusion missing in $log"; exit 1; }
}

for s in 0 1 2; do
  train fixed_three_window   $CX/b2_seed$s   logs_g1/train_cnx_b2_seed$s.log   --backbone convnext_tiny --max-samples 20000 --seed $s
  train random_window_single $CX/b4c_seed$s  logs_g1/train_cnx_b4c_seed$s.log  --backbone convnext_tiny --max-samples 20000 --seed $s
  train wicl                 $CX/wicl_seed$s logs_g1/train_cnx_wicl_seed$s.log --backbone convnext_tiny --max-samples 20000 --seed $s
done

CXM=""; for s in 0 1 2; do CXM="$CXM --model CNX_B2_seed$s=$CX/b2_seed$s:plain --model CNX_B4c_seed$s=$CX/b4c_seed$s:plain --model CNX_WICL_seed$s=$CX/wicl_seed$s:wicl"; done
stamp "step 2: scoring ConvNeXt models"
python g1_score_models.py --out-dir runs/g1_ext_more $CXM > logs_g1/score_cnx_external.log 2>&1
python g1_rsna_patient_infer.py --out-dir runs/g1_rsna_patient $CXM > logs_g1/score_cnx_rsna.log 2>&1
python g1_thickness_harmonize_infer.py --out-dir runs/g1_cq500_harmonized $CXM > logs_g1/harmonize_cnx.log 2>&1
for s in 0 1 2; do
  [ -f $CX/h4_seed${s}_result.json ] || python g1_h4_models.py --out $CX/h4_seed${s}_result.json \
    --model baseline2=$CX/b2_seed$s:plain --model b4c=$CX/b4c_seed$s:plain --model wicl=$CX/wicl_seed$s:wicl \
    > logs_g1/h4_cnx_seed$s.log 2>&1
done

train fixed_three_window   $BIG/b2_seed0   logs_g1/train_100k_b2.log   --backbone densenet121 --max-samples 100000 --seed 0
train random_window_single $BIG/b4c_seed0  logs_g1/train_100k_b4c.log  --backbone densenet121 --max-samples 100000 --seed 0
train wicl                 $BIG/wicl_seed0 logs_g1/train_100k_wicl.log --backbone densenet121 --max-samples 100000 --seed 0

BM="--model BIG_B2_seed0=$BIG/b2_seed0:plain --model BIG_B4c_seed0=$BIG/b4c_seed0:plain --model BIG_WICL_seed0=$BIG/wicl_seed0:wicl"
stamp "step 4: scoring 100k models"
python g1_score_models.py --out-dir runs/g1_ext_more $BM > logs_g1/score_100k_external.log 2>&1
python g1_rsna_patient_infer.py --out-dir runs/g1_rsna_patient $BM > logs_g1/score_100k_rsna.log 2>&1
python g1_thickness_harmonize_infer.py --out-dir runs/g1_cq500_harmonized $BM > logs_g1/harmonize_100k.log 2>&1

stamp "ALL_G1_IMPROVE_DONE"
