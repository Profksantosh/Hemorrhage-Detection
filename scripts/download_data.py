#!/usr/bin/env python
"""Data-acquisition helper for the four datasets specified in manuscript
Section IV. Cross-platform (works wherever `kaggle` CLI works, whether
that's this machine or the GPU environment you run training on).

WHAT THIS SCRIPT DOES: downloads and unzips RSNA / CQ500 / BHSD via the
Kaggle API. It does NOT and CANNOT fetch PhysioNet-ICH -- that dataset is
restricted-access and requires you to register a PhysioNet account and
sign their Restricted Health Data Use Agreement (v1.5.0) yourself; see
ich_gen/datasets/physionet_ich.py's docstring. START THAT APPROVAL
PROCESS AS EARLY AS POSSIBLE, since it is a review step, not an instant
download -- it is very plausibly the slowest item in your data-acquisition
timeline, not the GPU.

SETUP (one-time, on whichever machine runs this script):
    pip install kaggle
    # Get an API token from https://www.kaggle.com/settings -> "Create New Token"
    # This downloads kaggle.json -- place it at:
    #   Linux/Mac: ~/.kaggle/kaggle.json
    #   Windows:   C:\\Users\\<you>\\.kaggle\\kaggle.json
    # This script never asks for or handles your credentials directly --
    # the kaggle CLI reads them from that file itself.

USAGE:
    python scripts/download_data.py --dest /data/ich_gen --datasets rsna cq500 bhsd
    python scripts/download_data.py --dest /data/ich_gen --datasets all   # rsna+cq500+bhsd
                                                                            # (never physionet_ich)

For RSNA specifically: it is a COMPETITION download, not a plain dataset,
so you must first accept the competition rules at
https://www.kaggle.com/competitions/rsna-intracranial-hemorrhage-detection/rules
(one click, logged in) before the API download below will succeed --
Kaggle returns a 403 with a clear message if you skip this step.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import zipfile
from pathlib import Path

# Kaggle identifiers located via web search for this manuscript's
# companion code (2026-08); NOT independently verified against a real
# download by this repository's authors -- if a slug below 404s or the
# download otherwise fails, search kaggle.com directly for the current
# canonical source, since dataset slugs/ownership occasionally change.
KAGGLE_SOURCES = {
    "rsna": {
        "kind": "competition",
        "id": "rsna-intracranial-hemorrhage-detection",
        "note": "Official RSNA 2019 challenge data (manuscript ref [27]). "
                "Requires accepting the competition rules on kaggle.com "
                "first (one-time, logged-in click) before the API download "
                "works. ~450GB uncompressed -- budget disk space and time.",
    },
    "cq500": {
        "kind": "dataset",
        "id": "crawford/qureai-headct",
        "note": "Community Kaggle mirror of Qure.ai's CQ500 release "
                "(manuscript ref [28]). Cross-check against the original "
                "at http://headctstudy.qure.ai/dataset if anything looks "
                "off -- that is the authoritative source. Also worth "
                "checking: kaggle.com/datasets/fereshtej/"
                "rsna-and-cq500-data-frames, which may already contain "
                "pre-aggregated label data frames useful for "
                "ich_gen/datasets/cq500.py's expected aggregated-CSV "
                "schema -- inspect its columns before writing your own "
                "prepare_cq500_labels.py aggregation step.",
    },
    "bhsd": {
        "kind": "dataset",
        "id": "stevezeyuzhang/bhsd-dataset",
        "note": "Kaggle mirror of Wu et al.'s BHSD (manuscript ref [30]). "
                "Cross-check against the original GitHub release "
                "(github.com/White65534/BHSD) or the Hugging Face mirror "
                "(huggingface.co/datasets/Wendy-Fly/BHSD) if the Kaggle "
                "copy's directory layout doesn't match what "
                "ich_gen/datasets/bhsd.py's manifest-building step expects.",
    },
}


def _run(cmd: list[str]) -> None:
    print(f"$ {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


def download_one(name: str, dest: Path) -> None:
    src = KAGGLE_SOURCES[name]
    target_dir = dest / name
    target_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n=== {name} ===\n{src['note']}\n")

    if src["kind"] == "competition":
        _run(["kaggle", "competitions", "download", "-c", src["id"],
              "-p", str(target_dir)])
    else:
        _run(["kaggle", "datasets", "download", "-d", src["id"],
              "-p", str(target_dir)])

    zips = list(target_dir.glob("*.zip"))
    if not zips:
        print(f"  no .zip found in {target_dir} -- Kaggle may have "
              f"delivered files directly, or the download failed silently; "
              f"check the output above.")
        return
    for z in zips:
        print(f"  unzipping {z.name} ...")
        with zipfile.ZipFile(z) as zf:
            zf.extractall(target_dir)
        print(f"  done -- leaving {z.name} in place in case you need to "
              f"re-extract; delete it manually once you've verified the "
              f"extracted contents.")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dest", type=Path, required=True,
                    help="root directory to download into; one subfolder "
                         "per dataset will be created")
    p.add_argument("--datasets", nargs="+",
                    choices=list(KAGGLE_SOURCES) + ["all"], default=["all"])
    args = p.parse_args(argv)

    try:
        subprocess.run(["kaggle", "--version"], check=True,
                        capture_output=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("ERROR: the `kaggle` CLI is not available or not "
              "authenticated. Run `pip install kaggle` and place your API "
              "token at ~/.kaggle/kaggle.json (see this script's "
              "docstring), then retry.", file=sys.stderr)
        sys.exit(1)

    names = list(KAGGLE_SOURCES) if "all" in args.datasets else args.datasets
    args.dest.mkdir(parents=True, exist_ok=True)

    for name in names:
        download_one(name, args.dest)

    print("\n=== PhysioNet-ICH (manuscript ref [29]) is NOT downloaded by "
          "this script ===")
    print("It is restricted-access: register at https://physionet.org, "
          "sign the Restricted Health Data Use Agreement (v1.5.0), then "
          "download manually from "
          "https://physionet.org/content/ct-ich/1.3.1/ into "
          f"{args.dest / 'physionet_ich'}. Start this today if you haven't "
          "already -- the approval step, not the download itself, is what "
          "takes time.")

    print(f"\nNext: point each ich_gen loader at {args.dest}/<name> and "
          f"read that loader's 'SCHEMA ASSUMPTION' docstring before "
          f"trusting its output against what you actually downloaded "
          f"(see code/README.md's 'Getting the data' section).")


if __name__ == "__main__":
    main()
