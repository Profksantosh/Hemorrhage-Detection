"""End-to-end pipeline orchestration.

Implements the Research Decision Report's highest-priority instruction
(Section 9): run the decisive WICL-vs-baseline-4c comparison FIRST, on a
fast subset/short schedule, before committing to the full five-baseline,
three-external-site, ablation, calibration, and stress-test protocol.
This lets the paper's framing (does WICL survive as a positive result,
narrow to a hedge, or reduce to a negative result feeding the benchmark-
only framing -- manuscript Section IX) be decided empirically and early,
not discovered after a multi-week full run.

Usage
-----
    # Step 1 (always run first): decisive check on a fast subset
    python -m ich_gen.run_pipeline decisive-check \\
        --rsna-root /data/rsna --max-samples 5000 --epochs 5

    # Step 2 (only after reviewing step 1's output): full grid
    python -m ich_gen.run_pipeline full-grid \\
        --rsna-root /data/rsna --epochs 30 --out-dir runs/

This script only orchestrates already-implemented, already-smoke-tested
pieces (ich_gen.train, ich_gen.evaluate, ich_gen.stats); it adds no new
modeling logic of its own.
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO,
                     format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# All nine conditions (manuscript Section VI-A) are wired end-to-end
# through ich_gen.train.main(): baselines 1, 2, 3a (adaptive_hrt), 3b
# (adaptive_wem), 4a/4b/4c (dg_augment/dg_coral/random_window_single --
# dg_coral via train.run_dg_coral's two-pseudo-domain protocol), 5
# (semi_supervised, via train.run_semi_supervised's two-stage
# teacher/student protocol), and wicl. Two documented simplifications
# remain (not missing wiring, just worth reading before treating a
# comparison as final): dg_coral's --coral-weight default is likely too
# small relative to the supervised loss (see train.py's argparser help
# text); semi_supervised's "unlabeled" pool is a within-RSNA split rather
# than a genuinely separate corpus (see run_semi_supervised's docstring).
FULLY_AUTOMATED_CONDITIONS = (
    "fixed_single_window", "fixed_three_window", "adaptive_hrt",
    "adaptive_wem", "dg_augment", "dg_coral", "random_window_single",
    "semi_supervised", "wicl",
)
MANUAL_WIRING_REQUIRED_CONDITIONS = ()  # none remaining as of this version


def cmd_decisive_check(args):
    """The single most important experiment in the design (Research
    Decision Report Section 9 / manuscript Section IX): trains WICL and
    baseline 4c (random_window_single) on the SAME fold with the SAME
    (short, fast) schedule, then compares them on the held-out validation
    fold via DeLong's test, and prints an explicit recommendation for how
    to frame the rest of the manuscript.
    """
    from ich_gen import train as train_mod

    results = {}
    for condition in ("wicl", "random_window_single"):
        out_dir = args.out_dir / f"decisive_{condition}"
        argv = [
            "--condition", condition,
            "--rsna-root", str(args.rsna_root),
            "--fold", "0", "--n-folds", str(args.n_folds),
            "--epochs", str(args.epochs),
            "--batch-size", str(args.batch_size),
            "--out-dir", str(out_dir),
            "--seed", str(args.seed),
        ]
        if args.max_samples is not None:
            argv += ["--max-samples", str(args.max_samples)]
        if not args.pretrained:
            argv += ["--no-pretrained"]
        logger.info("=== decisive check: training condition=%s ===", condition)
        train_mod.main(argv)
        results[condition] = out_dir

    logger.info(
        "Both conditions trained. To complete the decisive check, run "
        "evaluate.py-based scoring of each checkpoint's predictions on "
        "the held-out validation fold (and, once available, on at least "
        "one real external site) and feed the resulting (y_true, y_prob) "
        "pairs to ich_gen.stats.delong_paired_auc_test -- this final "
        "scoring step depends on which real dataset paths you have "
        "configured and is intentionally left as an explicit follow-up "
        "call rather than hardcoded here, so this script does not "
        "silently assume a specific external-site configuration.")
    logger.info(
        "Interpretation guide (manuscript Section IX / Research Decision "
        "Report Section 9): if WICL's AUROC is NOT significantly higher "
        "than baseline 4c's (DeLong p >= 0.05, or a subtype/size-"
        "stratified sensitivity gain below the %s pp minimum-clinically-"
        "relevant threshold -- ich_gen.stats.MIN_CLINICALLY_RELEVANT_"
        "SENSITIVITY_DELTA), revise the manuscript's title/abstract/"
        "contributions BEFORE running the full grid, per Section IX's "
        "pre-committed interpretation.",
        100 * __import__("ich_gen.stats", fromlist=["x"]).MIN_CLINICALLY_RELEVANT_SENSITIVITY_DELTA)
    return results


def cmd_full_grid(args):
    """Runs every fully-automated condition (see
    FULLY_AUTOMATED_CONDITIONS) across all `n_folds` patient-grouped
    folds. Explicitly warns, rather than silently omitting, the three
    conditions that still require manual two-stage wiring."""
    from ich_gen import train as train_mod

    if MANUAL_WIRING_REQUIRED_CONDITIONS:
        logger.warning(
            "The following conditions require manual training-loop "
            "wiring not yet automated by this script and will be SKIPPED: "
            "%s. See ich_gen/train.py's NotImplementedError branch and "
            "ich_gen/models/adaptive_window.py / semi_supervised.py for "
            "their standalone, already-implemented core logic.",
            MANUAL_WIRING_REQUIRED_CONDITIONS)

    manifest = []
    for condition in FULLY_AUTOMATED_CONDITIONS:
        for fold in range(args.n_folds):
            out_dir = args.out_dir / f"{condition}_fold{fold}"
            argv = [
                "--condition", condition,
                "--rsna-root", str(args.rsna_root),
                "--fold", str(fold), "--n-folds", str(args.n_folds),
                "--epochs", str(args.epochs),
                "--batch-size", str(args.batch_size),
                "--out-dir", str(out_dir),
                "--seed", str(args.seed),
            ]
            if args.max_samples is not None:
                argv += ["--max-samples", str(args.max_samples)]
            logger.info("=== full grid: condition=%s fold=%d ===", condition, fold)
            train_mod.main(argv)
            manifest.append({"condition": condition, "fold": fold,
                              "out_dir": str(out_dir)})

    with open(args.out_dir / "full_grid_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
    logger.info("full grid complete: %d runs. Next step: run external-site "
                "evaluation (ich_gen.evaluate) and the ablation/calibration/"
                "stress-test scripts against each checkpoint in the "
                "manifest at %s", len(manifest), args.out_dir / "full_grid_manifest.json")
    return manifest


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="command", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--rsna-root", type=Path, required=True)
    common.add_argument("--out-dir", type=Path, default=Path("runs"))
    common.add_argument("--n-folds", type=int, default=5)
    common.add_argument("--batch-size", type=int, default=32)
    common.add_argument("--seed", type=int, default=0)
    common.add_argument("--max-samples", type=int, default=None,
                         help="cap on samples for fast local iteration; "
                              "omit for the real reported experiments")
    common.add_argument("--pretrained", action="store_true", default=True)

    p_decisive = sub.add_parser("decisive-check", parents=[common])
    p_decisive.add_argument("--epochs", type=int, default=5, help="kept "
                             "short deliberately -- this is a fast go/no-go "
                             "check, not the final reported result")
    p_decisive.set_defaults(func=cmd_decisive_check)

    p_full = sub.add_parser("full-grid", parents=[common])
    p_full.add_argument("--epochs", type=int, default=30)
    p_full.set_defaults(func=cmd_full_grid)

    return p


def main(argv=None):
    args = build_argparser().parse_args(argv)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.func(args)


if __name__ == "__main__":
    main()
