"""Independent from-scratch re-derivation of the BHSD/RSNA patient-ID
overlap (see G1 task instructions, 2026-09-11). Reads every RSNA DICOM
file's PatientID and StudyInstanceUID tags directly (no CSV shortcut,
no reuse of any pre-existing patient_index_cache.parquet), and compares
against BHSD's 192 labeled-volume filenames.
"""
import json
import os
import re
import sys
import time
from datetime import datetime, timezone

import pydicom

RSNA_DICOM_DIR = "/home/fcse.santoshkumar/ich_windowing/data/rsna/rsna-intracranial-hemorrhage-detection/stage_2_train"
BHSD_IMAGES_DIR = "/home/fcse.santoshkumar/ich_windowing/data/bhsd/label_192/images"
OUT_JSON = "/home/fcse.santoshkumar/ich_windowing/data/bhsd_rsna_overlap_patient_ids.json"
OUT_RAW_CSV = "/home/fcse.santoshkumar/ich_windowing/runs/audit_rederive/rsna_dicom_patient_study_index_fresh.csv"

BHSD_FNAME_RE = re.compile(r"^(ID_[0-9a-f]+)_(ID_[0-9a-f]+)\.nii\.gz$")


def main():
    t_start = time.time()

    # --- Step A: enumerate BHSD's 192 labeled volumes, parse filename tokens ---
    bhsd_files = sorted(f for f in os.listdir(BHSD_IMAGES_DIR) if f.endswith(".nii.gz"))
    bhsd_records = []
    for fname in bhsd_files:
        m = BHSD_FNAME_RE.match(fname)
        if not m:
            raise ValueError(f"BHSD filename does not match expected pattern: {fname}")
        bhsd_records.append({
            "filename": fname,
            "patient_token": m.group(1),
            "study_token": m.group(2),
        })
    n_bhsd = len(bhsd_records)
    print(f"[A] BHSD labeled volumes found: {n_bhsd}", flush=True)

    # --- Step B: read PatientID + StudyInstanceUID directly from every RSNA
    #     training DICOM file (fresh read, no cache reuse) ---
    dcm_files = sorted(f for f in os.listdir(RSNA_DICOM_DIR) if f.endswith(".dcm"))
    n_dicom = len(dcm_files)
    print(f"[B] RSNA stage_2_train DICOM files found: {n_dicom}", flush=True)

    rsna_patient_ids = set()
    # patient_id -> set of study_uids seen for that patient (for compound-id check)
    patient_to_studies = {}
    n_read = 0
    t0 = time.time()
    with open(OUT_RAW_CSV, "w") as fout:
        fout.write("sop_uid,patient_id,study_uid\n")
        for fname in dcm_files:
            path = os.path.join(RSNA_DICOM_DIR, fname)
            ds = pydicom.dcmread(path, stop_before_pixels=True)
            pid = str(ds.PatientID)
            study = str(ds.StudyInstanceUID)
            sop_uid = os.path.splitext(fname)[0]
            rsna_patient_ids.add(pid)
            patient_to_studies.setdefault(pid, set()).add(study)
            fout.write(f"{sop_uid},{pid},{study}\n")
            n_read += 1
            if n_read % 100000 == 0:
                print(f"    read {n_read}/{n_dicom} DICOM headers "
                      f"({time.time()-t0:.1f}s elapsed)", flush=True)
    t1 = time.time()
    print(f"[B] done reading {n_read} DICOM headers in {t1-t0:.1f}s; "
          f"{len(rsna_patient_ids)} unique RSNA PatientIDs found", flush=True)

    # --- Step C: cross-match BHSD patient tokens against RSNA PatientID set ---
    matched = []
    unmatched = []
    study_confirmed = []
    study_mismatched = []
    for rec in bhsd_records:
        pid = rec["patient_token"]
        study = rec["study_token"]
        if pid in rsna_patient_ids:
            matched.append(rec)
            if study in patient_to_studies.get(pid, set()):
                study_confirmed.append(rec)
            else:
                study_mismatched.append(rec)
        else:
            unmatched.append(rec)

    print(f"[C] BHSD patient-token matches against RSNA PatientID: "
          f"{len(matched)}/{n_bhsd} (unmatched: {len(unmatched)})", flush=True)
    print(f"[C] of matched, StudyInstanceUID compound-id confirmed: "
          f"{len(study_confirmed)}/{len(matched)} "
          f"(study-mismatched: {len(study_mismatched)})", flush=True)

    if unmatched:
        print("UNMATCHED BHSD RECORDS:", unmatched, flush=True)
    if study_mismatched:
        print("STUDY-MISMATCHED RECORDS:", study_mismatched, flush=True)

    # --- Step D: persist exclusion-list artifact ---
    exclude_ids = sorted({rec["patient_token"] for rec in matched})

    out = {
        "description": (
            "RSNA PatientID values (raw, WITHOUT the 'RSNA_' prefix that "
            "ich_gen.datasets.rsna.build_samples() adds internally) "
            "confirmed to overlap with BHSD's 192 labeled training volumes. "
            "Independently re-derived from scratch for the G1 remediation "
            "task; NOT a reuse of repro-auditor's reported numbers."
        ),
        "date_derived": datetime.now(timezone.utc).isoformat(),
        "method": (
            "1) Enumerated all BHSD label_192/images/*.nii.gz filenames "
            "(pattern <patient_token>_<study_token>.nii.gz). "
            "2) Read PatientID and StudyInstanceUID DICOM tags directly "
            "(pydicom, stop_before_pixels=True) from every file in "
            "RSNA stage_2_train/ (no CSV shortcut -- stage_2_train.csv "
            "has no PatientID column). "
            "3) Set-intersected BHSD's first filename token against the "
            "RSNA PatientID set. "
            "4) For every matched patient, additionally checked whether "
            "the BHSD filename's second token appears among the "
            "StudyInstanceUID values observed for that RSNA PatientID "
            "(compound patient+study identifier confirmation, not a "
            "hash/token collision check)."
        ),
        "source_counts": {
            "bhsd_labeled_volumes": n_bhsd,
            "rsna_train_dicom_files": n_dicom,
            "rsna_unique_patient_ids": len(rsna_patient_ids),
        },
        "result": {
            "n_bhsd_patient_token_matched": len(matched),
            "n_bhsd_patient_token_unmatched": len(unmatched),
            "n_study_uid_compound_confirmed": len(study_confirmed),
            "n_study_uid_compound_mismatched": len(study_mismatched),
        },
        "n_excluded_patient_ids": len(exclude_ids),
        "excluded_patient_ids": exclude_ids,
    }

    os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
    with open(OUT_JSON, "w") as f:
        json.dump(out, f, indent=2)

    print(f"[D] wrote exclusion artifact ({len(exclude_ids)} patient ids) "
          f"to {OUT_JSON}", flush=True)
    print(f"total elapsed: {time.time()-t_start:.1f}s", flush=True)


if __name__ == "__main__":
    main()
