# Reproducing the G1 revision analyses

Evaluation of RSNA-trained intracranial-hemorrhage detectors (fixed three-window B2, random-window RW = `B4c`, WICL) on
CQ500 and PhysioNet-ICH. All metrics are per scan; threshold 0.5 is fixed on RSNA. Raw data and checkpoints are not included.

Order of execution (hp100 scripts first, then CPU analysis):

1. Training: `ich_gen/train.py` (DenseNet-121 20k slices, seeds 0-2: `run_g1_revision.sh`; ConvNeXt-Tiny 20k slices and
   DenseNet-121 100k slices: `run_g1_improve.sh`).
2. Scoring: `g1_score_models.py` (CQ500, PhysioNet-ICH), `g1_physionet_reoriented_infer.py` (PhysioNet-ICH rotated to
   radiological orientation), `g1_rsna_patient_infer.py` (every slice of the validation patients), `g1_seed_external_infer.py`,
   `g1_thickness_harmonize_infer.py` (CQ500 slice-thickness harmonization), `g1_h4_models.py` (synthetic HU-shift test).
   `g1_rerun_100k.sh` reruns the 100k RSNA and harmonization scoring with a raised open-file limit (`ulimit -n`).
3. Analysis: `g1_final_analysis.py` -> `runs/g1_final_analysis_{max,top5mean}.json`; `g1_recalibration.py`;
   `g1_tables.py` -> `runs/g1_tables.json`; `g1_extra_analysis.py` -> `runs/g1_extra_analysis.json` (ConvNeXt, 100k, harmonization);
   `g1_figures.py` (figures; labels and layout revised 2026-10-03, data unchanged).
4. Sampling counts quoted in the Methods: `g1_pool_counts.py` (run on the training host; patient and slice counts of the 20,000-slice pool and fold 0) -> `runs/g1_pool_counts.json`.

Notes
- Seeds share one patient split (the grouped k-fold ignores the seed); seeds vary head initialization, data order and augmentation.
- `runs/g1_rsna_patient/split_seed*_fold0.json` describe the 20k models; `split_100k_seed0_fold0.json` describes the 100k models.
- RSNA patient identifiers in all result files are hashed. `g1_study_level_reanalysis.py` (rotated PhysioNet-ICH) is superseded and not included.
- `runs/g1_rsna_patient/split_100k_seed0_fold0.json` describes the 100,000-slice models (3,720 held-out patients, 147,421 slices).
