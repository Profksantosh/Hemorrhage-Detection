from pathlib import Path

ROOT = Path("/home/fcse.santoshkumar/ich_windowing")
src = ROOT / "h4_hu_recalibration_stress_test.py"
dst = ROOT / "h4_hu_recalibration_stress_test_bhsd_excluded.py"
text = src.read_text()

NEEDLE = '''    print(f"loading RSNA pool (first {TRAIN_POOL_MAX_SAMPLES} rows, the same "
          f"prefix baseline2/WICL were trained+validated on) to determine "
          f"its patient set ...")
    pool_samples = build_rsna_samples(rsna_root, max_samples=TRAIN_POOL_MAX_SAMPLES)
    pool_patients = {s.patient_id for s in pool_samples}
    print(f"  pool: {len(pool_samples)} slices, {len(pool_patients)} patients")

    print(f"loading RSNA rows [{TRAIN_POOL_MAX_SAMPLES}:"
          f"{TRAIN_POOL_MAX_SAMPLES + HOLDOUT_RAW_N}] as the candidate "
          f"held-out subset ...")
    raw_all = build_rsna_samples(rsna_root, max_samples=TRAIN_POOL_MAX_SAMPLES + HOLDOUT_RAW_N)
    candidate = raw_all[TRAIN_POOL_MAX_SAMPLES:]
'''

if text.count(NEEDLE) != 1:
    raise SystemExit(f"NEEDLE match count = {text.count(NEEDLE)}, expected 1")

PATCH = '''    # BHSD/RSNA overlap remediation (2026-09-11): baseline2_bhsd_excluded
    # and WICL_bhsd_excluded were trained on a pool that excludes the 191
    # BHSD-overlapping RSNA patients BEFORE max_samples truncation (see
    # ich_gen.datasets.rsna.build_samples docstring) -- so "the same
    # prefix baseline2/WICL were trained+validated on" now means this
    # exclusion-then-truncate pool, not a raw CSV-row prefix. Passing the
    # same exclude_patient_ids here is required for pool_patients (and
    # therefore the disjointness filter below) to actually match what
    # those checkpoints were trained on; otherwise this function would
    # silently compute disjointness against the WRONG (pre-remediation)
    # pool definition.
    from ich_gen.datasets.rsna import load_bhsd_overlap_exclusion_ids
    from ich_gen.train import DEFAULT_BHSD_EXCLUSION_JSON
    exclude_ids = load_bhsd_overlap_exclusion_ids(DEFAULT_BHSD_EXCLUSION_JSON)
    print(f"loaded {len(exclude_ids)} BHSD-overlap RSNA patient id(s) to "
          f"exclude, matching the bhsd_excluded training protocol")

    print(f"loading RSNA pool (first {TRAIN_POOL_MAX_SAMPLES} rows AFTER "
          f"BHSD-overlap exclusion, the same pool baseline2/WICL were "
          f"trained+validated on) to determine its patient set ...")
    pool_samples = build_rsna_samples(rsna_root, max_samples=TRAIN_POOL_MAX_SAMPLES,
                                       exclude_patient_ids=exclude_ids)
    pool_patients = {s.patient_id for s in pool_samples}
    print(f"  pool: {len(pool_samples)} slices, {len(pool_patients)} patients")

    print(f"loading RSNA rows [{TRAIN_POOL_MAX_SAMPLES}:"
          f"{TRAIN_POOL_MAX_SAMPLES + HOLDOUT_RAW_N}] (post-exclusion "
          f"ordering) as the candidate held-out subset ...")
    raw_all = build_rsna_samples(rsna_root, max_samples=TRAIN_POOL_MAX_SAMPLES + HOLDOUT_RAW_N,
                                  exclude_patient_ids=exclude_ids)
    candidate = raw_all[TRAIN_POOL_MAX_SAMPLES:]
'''

text = text.replace(NEEDLE, PATCH, 1)

header = ('# AUTO-GENERATED bhsd_excluded variant of '
          'h4_hu_recalibration_stress_test.py,\n'
          '# created 2026-09-11 for the BHSD/RSNA patient-overlap '
          'remediation task.\n'
          '# Functional change from the original: select_holdout_samples() '
          'now builds\n'
          '# its training-pool patient set (used to filter the held-out '
          'subset for\n'
          '# patient-disjointness) with the same BHSD-overlap '
          'exclude_patient_ids\n'
          '# applied to baseline2_bhsd_excluded/WICL_bhsd_excluded\'s '
          'actual training\n'
          '# pool -- see inline comment at select_holdout_samples() for '
          'why this\n'
          '# matters (using the unexcluded pool definition here would '
          'silently check\n'
          '# disjointness against the WRONG pool). Point --runs-dir / '
          '--wicl-dir at\n'
          '# the bhsd_excluded run directories via CLI flags when '
          'invoking. See the\n'
          '# original h4_hu_recalibration_stress_test.py for full '
          'methodology docs.\n')

dst.write_text(header + text)
print(f"wrote {dst}")
