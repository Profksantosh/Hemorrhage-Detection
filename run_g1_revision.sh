#!/bin/bash
# G1 revision jobs (2026-10-01), run sequentially on hp100:
#   1. H4 HU-corner test with baseline 4c added (seed-0 checkpoints, inference only)
#   2. RSNA internal per-patient scoring for seed-0 B2 / 4c / WICL (inference only)
#   3. Train B2, 4c, WICL with seeds 1 and 2 (same protocol as the seed-0 runs; seed also sets the split)
#   4. Score the seed replicates: CQ500 + PhysioNet, RSNA per-patient, H4
# Each step skips work whose output already exists, so the script can be re-run after an interruption.
set -e
cd /home/fcse.santoshkumar/ich_windowing
source /opt/anaconda3/etc/profile.d/conda.sh
conda activate ich_windowing
export PYTHONPATH=$PWD:$PYTHONPATH

S0=runs/baselines_bhsd_excluded
SEEDS=runs/g1_seeds
mkdir -p "$SEEDS" runs/g1_rsna_patient logs_g1
stamp() { echo "=== [$(date -Is)] $*"; }

stamp "step 1: H4 with 4c (seed 0)"
[ -f $S0/h4_4c_result.json ] || python g1_h4_models.py --out $S0/h4_4c_result.json \
  --model baseline2=$S0/baseline2_fixed_three:plain \
  --model b4c=$S0/baseline4c_random_window_single:plain \
  --model wicl=$S0/decisive_wicl:wicl > logs_g1/h4_4c_seed0.log 2>&1

stamp "step 2: RSNA per-patient (seed 0)"
python g1_rsna_patient_infer.py --out-dir runs/g1_rsna_patient \
  --model B2_seed0=$S0/baseline2_fixed_three:plain \
  --model B4c_seed0=$S0/baseline4c_random_window_single:plain \
  --model WICL_seed0=$S0/decisive_wicl:wicl > logs_g1/rsna_patient_seed0.log 2>&1

COMMON="--rsna-root data/rsna/rsna-intracranial-hemorrhage-detection --backbone densenet121 --pretrained \
  --epochs 12 --batch-size 32 --lr 1e-4 --weight-decay 1e-5 --fold 0 --n-folds 5 --max-samples 20000 --device cuda"
train() {
  local condition=$1 name=$2 seed=$3
  local out=$SEEDS/${name}_seed${seed} log=logs_g1/train_${name}_seed${seed}.log
  if [ -f $out/checkpoint_last.pt ] && grep -q '"epoch": 11\|"epoch": 12' $out/history.json 2>/dev/null; then
    stamp "skip training $name seed $seed (done)"; return; fi
  stamp "step 3: training $condition seed $seed -> $out"
  python -m ich_gen.train --condition $condition $COMMON --seed $seed --out-dir $out > $log 2>&1
  grep -q "loaded 191 RSNA patient id(s) to exclude" $log || { echo "FATAL: BHSD exclusion missing in $log"; exit 1; }
}
for seed in 1 2; do
  train fixed_three_window b2 $seed
  train random_window_single b4c $seed
  train wicl wicl $seed
done

stamp "step 4a: external scoring of seed replicates"
python g1_seed_external_infer.py --seeds 1 2 > logs_g1/seed_external.log 2>&1

stamp "step 4b: RSNA per-patient for seed replicates"
python g1_rsna_patient_infer.py --out-dir runs/g1_rsna_patient \
  --model B2_seed1=$SEEDS/b2_seed1:plain --model B4c_seed1=$SEEDS/b4c_seed1:plain --model WICL_seed1=$SEEDS/wicl_seed1:wicl \
  --model B2_seed2=$SEEDS/b2_seed2:plain --model B4c_seed2=$SEEDS/b4c_seed2:plain --model WICL_seed2=$SEEDS/wicl_seed2:wicl \
  > logs_g1/rsna_patient_seeds.log 2>&1

for seed in 1 2; do
  stamp "step 4c: H4 seed $seed"
  [ -f $SEEDS/h4_seed${seed}_result.json ] || python g1_h4_models.py --out $SEEDS/h4_seed${seed}_result.json \
    --model baseline2=$SEEDS/b2_seed${seed}:plain --model b4c=$SEEDS/b4c_seed${seed}:plain \
    --model wicl=$SEEDS/wicl_seed${seed}:wicl > logs_g1/h4_seed${seed}.log 2>&1
done

stamp "ALL_G1_REVISION_DONE"
