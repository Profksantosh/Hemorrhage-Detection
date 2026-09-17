"""Create '_bhsd_excluded' copies of the internal-fold scoring scripts
(and h1/h3 hypothesis tests) that call ich_gen.train.build_folds via a
hand-built argparse.Namespace, patching that Namespace to explicitly set
bhsd_exclusion=True (see ich_gen/train.py's build_folds docstring for why
this must be explicit, not automatic, for any pre-existing Namespace-
builder). External-site scoring scripts (score_external.py,
score_baselines_external.py, score_baselines3ab_external.py,
score_baselines4ab_external.py, score_physionet_ich_external.py) are
NOT copied here -- they never call build_folds (no RSNA fold
reconstruction needed for external-site inference), so they can be reused
UNCHANGED just by pointing --runs-dir/--wicl-dir/--decisive-dir at the
new bhsd_excluded checkpoint directory via CLI flags.
"""
import re
from pathlib import Path

ROOT = Path("/home/fcse.santoshkumar/ich_windowing")

FILES = [
    "score_baselines_internal.py",
    "score_baselines3ab_internal.py",
    "score_baselines4ab_internal.py",
    "score_decisive.py",
    "h1_baseline2_degradation_test.py",
    "h3_calibration_ece_test.py",
]

NEEDLE = '        seed=config["seed"],\n    )\n'
PATCH = ('        seed=config["seed"],\n'
         '        bhsd_exclusion=True,  # BHSD/RSNA overlap remediation '
         '(2026-09-11) -- see\n'
         '                              # ich_gen/train.py build_folds() '
         'docstring: this\n'
         '                              # must be explicit, the default '
         'for a hand-built\n'
         '                              # Namespace (as opposed to one '
         'from build_argparser())\n'
         '                              # is False, to keep OLD '
         '(pre-remediation) scoring\n'
         '                              # scripts reconstructing their '
         'checkpoints\' exact\n'
         '                              # original (unexcluded) folds if '
         'ever rerun.\n'
         '    )\n')

for fname in FILES:
    src = ROOT / fname
    dst = ROOT / fname.replace(".py", "_bhsd_excluded.py")
    text = src.read_text()
    n = text.count(NEEDLE)
    if n != 1:
        raise SystemExit(f"{fname}: expected exactly 1 match of the "
                          f"Namespace-closing pattern, found {n} -- "
                          f"inspect manually, refusing to guess.")
    patched = text.replace(NEEDLE, PATCH, 1)
    header = (
        f"# AUTO-GENERATED bhsd_excluded variant of {fname}, created\n"
        f"# 2026-09-11 for the BHSD/RSNA patient-overlap remediation task.\n"
        f"# Only functional change from the original: load_val_split()'s\n"
        f"# Namespace now sets bhsd_exclusion=True so it reconstructs the\n"
        f"# SAME (patient-excluded) RSNA fold split used to train the\n"
        f"# runs/baselines_bhsd_excluded / runs/decisive_bhsd_excluded\n"
        f"# checkpoints this script is meant to score. Point --runs-dir\n"
        f"# (and --wicl-dir / --baseline2-dir / --decisive-dir, if this\n"
        f"# script has one) at the bhsd_excluded run directories via CLI\n"
        f"# flags when invoking. See {fname} (the unmodified original)\n"
        f"# for full methodology documentation -- not duplicated here.\n"
    )
    dst.write_text(header + patched)
    print(f"wrote {dst} (patched {n} Namespace block)")
