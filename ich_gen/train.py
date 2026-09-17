"""Unified training entrypoint for all nine conditions of manuscript
Section VI-A (baselines 1-5 with sub-variants, and WICL). Selecting the
condition via `--condition` is the ONLY thing that differs between runs
of this script; backbone, optimizer, LR schedule, batch size, and the
patient-grouped data split are held identical across conditions (Section
VI-B), which is enforced here by routing every condition through the
same argument parser and the same fold-construction code path.

Usage (once real data paths are configured, see README.md):

    python -m ich_gen.train --condition wicl \
        --rsna-root /data/rsna --fold 0 --epochs 30 \
        --backbone densenet121 --out-dir runs/wicl_fold0

    python -m ich_gen.train --condition random_window_single \
        --rsna-root /data/rsna --fold 0 --epochs 30 \
        --backbone densenet121 --out-dir runs/baseline4c_fold0

Per the Research Decision Report's highest-priority recommendation, run
`--condition wicl` and `--condition random_window_single` FIRST (the
decisive WICL-vs-baseline-4c comparison) before launching the full
five-baseline grid -- see run_pipeline.py, which encodes this ordering.
"""
from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from ich_gen.datasets.common import MultiSiteICHDataset
from ich_gen.datasets.splits import (patient_grouped_kfold,
                                       verify_no_patient_overlap)
from ich_gen.models.backbone import build_backbone, EmbeddingBackbone, DEFAULT_BACKBONE
from ich_gen.models.wicl import build_wicl_model, WICLLossWeights
from ich_gen.models.augmentation_baselines import (
    CTAppropriateRandAugment, coral_loss, random_window_single_view_loss)
from ich_gen.models.adaptive_window import WEMClassifier
from ich_gen.models.semi_supervised import (
    PseudoLabelConfig, generate_pseudo_labels, student_training_loss)

logging.basicConfig(level=logging.INFO,
                     format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

CONDITIONS = (
    "fixed_single_window",     # baseline 1
    "fixed_three_window",      # baseline 2
    "adaptive_hrt",            # baseline 3a  (see models/adaptive_window.py)
    "adaptive_wem",            # baseline 3b
    "dg_augment",               # baseline 4a
    "dg_coral",                  # baseline 4b
    "random_window_single",     # baseline 4c -- reimplements [31]/[32]
    "semi_supervised",           # baseline 5  -- reimplements [34]
    "wicl",                       # proposed method (Section V)
)

# view_mode for MultiSiteICHDataset per condition, per manuscript Section
# VI-A / Section V-E. "adaptive_wem" is handled specially: it needs the
# raw_hu_for_wem view mode (see common.py) since its windowing happens
# inside the training loop, not at dataset time.
CONDITION_VIEW_MODE = {
    "fixed_single_window": "fixed_single_window",
    "fixed_three_window": "fixed_three_window",
    "dg_augment": "fixed_three_window",   # RandAugment applied on top
    "dg_coral": "fixed_three_window",
    "random_window_single": "random_window_single",
    "semi_supervised": "fixed_three_window",
    "wicl": "random_window_dual",
    "adaptive_hrt": "adaptive_hrt",
    "adaptive_wem": "raw_hu_for_wem",
}


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--condition", required=True, choices=CONDITIONS)
    p.add_argument("--rsna-root", type=Path, required=True)
    p.add_argument("--fold", type=int, default=0, help="which of the 5 "
                    "patient-grouped folds (manuscript Section IV-A) to "
                    "hold out as validation for this run")
    p.add_argument("--n-folds", type=int, default=5)
    p.add_argument("--bhsd-exclusion-json", type=Path,
                    default=Path(__file__).resolve().parent.parent / "data" /
                    "bhsd_rsna_overlap_patient_ids.json",
                    help="path to the BHSD/RSNA patient-ID overlap "
                         "exclusion list (see "
                         "code/scripts/derive_bhsd_rsna_overlap.py); RSNA "
                         "patients in this list are dropped from every "
                         "training fold before sampling, since BHSD is "
                         "used as external validation elsewhere in this "
                         "pipeline. Required -- pass "
                         "--no-bhsd-exclusion to explicitly reproduce the "
                         "pre-remediation (contaminated) protocol.")
    p.add_argument("--no-bhsd-exclusion", dest="bhsd_exclusion",
                    action="store_false", default=True,
                    help="DANGEROUS: skip BHSD/RSNA patient exclusion "
                         "entirely. Only for intentionally reproducing "
                         "the pre-2026-09-11 contaminated-fold runs "
                         "(runs/decisive*, runs/baselines_v1/*) -- never "
                         "use this for new results.")
    p.add_argument("--backbone", default=DEFAULT_BACKBONE)
    p.add_argument("--pretrained", action="store_true", default=True)
    p.add_argument("--no-pretrained", dest="pretrained", action="store_false")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=1e-5)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--max-samples", type=int, default=None,
                    help="cap on training samples, for fast local "
                         "smoke-testing only -- omit for real runs")
    p.add_argument("--lambda-consistency", type=float, default=1.0,
                    help="WICL only (Section V-C)")
    p.add_argument("--lambda-embedding", type=float, default=0.5,
                    help="WICL only (Section V-C)")
    p.add_argument("--sampling-breadth", default="moderate",
                    choices=("narrow", "moderate", "wide"),
                    help="WICL / random_window_single only (Section VI-D "
                         "ablation over delta_level/delta_width breadth)")
    p.add_argument("--unlabeled-fraction", type=float, default=0.5,
                    help="semi_supervised only: fraction of this fold's "
                         "training patients whose labels are WITHHELD to "
                         "simulate an unlabeled pool for pseudo-labeling. "
                         "See run_semi_supervised()'s docstring for why "
                         "this is a documented simplification rather than "
                         "a genuinely separate unlabeled corpus.")
    p.add_argument("--teacher-epochs", type=int, default=None,
                    help="semi_supervised only: epochs for the teacher "
                         "stage; defaults to --epochs if unset")
    p.add_argument("--coral-weight", type=float, default=1.0,
                    help="dg_coral only: weight of the cross-domain "
                         "covariance-alignment term relative to the "
                         "supervised loss (Section VI-A). NOTE: CORAL's "
                         "raw term (normalized by 4*d^2 over the "
                         "embedding dimension d) is typically several "
                         "orders of magnitude smaller than the supervised "
                         "BCE loss -- the default of 1.0 will barely "
                         "register; tune this up (commonly O(10)-O(1000) "
                         "in published CORAL implementations) on the "
                         "patient-grouped validation fold before treating "
                         "dg_coral's comparison to WICL as meaningful.")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p


DEFAULT_BHSD_EXCLUSION_JSON = (Path(__file__).resolve().parent.parent / "data" /
                                "bhsd_rsna_overlap_patient_ids.json")


def build_folds(args) -> tuple:
    """Build the RSNA sample pool and patient-grouped folds.

    `args` is normally produced by build_argparser() (which defaults
    `bhsd_exclusion=True` for any NEW training run), but several
    pre-existing scoring scripts (score_*.py, h1/h3/h4_*.py) construct
    their own argparse.Namespace directly to reconstruct the SAME fold
    split used at training time for their (pre-remediation) checkpoints,
    without going through build_argparser and without ever setting
    `bhsd_exclusion` -- so here the getattr() fallback for a MISSING
    `bhsd_exclusion` attribute is deliberately False, not True: this is
    the only way to guarantee those old scripts keep reconstructing the
    exact (unexcluded) folds their v1 checkpoints were actually trained
    on, if they are ever rerun, without editing every one of them. Any
    NEW v2 (post-remediation) scoring script MUST explicitly set
    `ns.bhsd_exclusion = True` on the Namespace it builds -- this is not
    automatic just by calling build_folds().
    """
    from ich_gen.datasets.rsna import build_samples, load_bhsd_overlap_exclusion_ids

    exclude_patient_ids = None
    if getattr(args, "bhsd_exclusion", False):
        json_path = getattr(args, "bhsd_exclusion_json", DEFAULT_BHSD_EXCLUSION_JSON)
        if not json_path.exists():
            raise FileNotFoundError(
                f"BHSD/RSNA exclusion list not found at {json_path}. This "
                f"is required (see code/scripts/derive_bhsd_rsna_overlap.py "
                f"and the 2026-09-11 BHSD/RSNA patient-overlap remediation) "
                f"-- refusing to silently train on the un-excluded, "
                f"BHSD-contaminated RSNA pool. Pass --no-bhsd-exclusion "
                f"if you are intentionally reproducing a pre-remediation "
                f"run.")
        exclude_patient_ids = load_bhsd_overlap_exclusion_ids(json_path)
        logger.info("loaded %d RSNA patient id(s) to exclude (BHSD overlap, "
                    "see %s)", len(exclude_patient_ids), json_path)
    else:
        logger.warning("bhsd_exclusion is False (either --no-bhsd-exclusion "
                        "was passed, or this Namespace never set "
                        "bhsd_exclusion at all -- e.g. a pre-remediation "
                        "scoring script): building RSNA folds WITHOUT "
                        "excluding the BHSD-overlapping patients. This "
                        "reproduces the pre-2026-09-11 contaminated "
                        "protocol; only correct for intentionally "
                        "reconstructing a v1 checkpoint's original folds.")

    logger.info("loading RSNA sample index (max_samples=%s)...", args.max_samples)
    samples = build_samples(args.rsna_root, max_samples=args.max_samples,
                             exclude_patient_ids=exclude_patient_ids)
    patient_ids = np.array([s.patient_id for s in samples])
    folds = patient_grouped_kfold(patient_ids, n_splits=args.n_folds, seed=args.seed)
    verify_no_patient_overlap(folds)  # Section IV-F control (i), enforced
                                        # every run, not just in tests
    logger.info("patient-grouped %d-fold split verified leakage-free "
                "(%d total patients)", args.n_folds, len(set(patient_ids.tolist())))
    return samples, folds


def build_model(condition: str, args):
    if condition == "wicl":
        return build_wicl_model(args.backbone, args.pretrained,
                                 args.lambda_consistency, args.lambda_embedding)
    if condition == "adaptive_wem":
        return WEMClassifier(args.backbone, args.pretrained)
    if condition == "dg_coral":
        # needs penultimate-layer embeddings (not just logits) to compute
        # the cross-domain covariance-alignment term -- see run_dg_coral
        return EmbeddingBackbone(args.backbone, args.pretrained)
    # every other condition (including semi_supervised's teacher AND
    # student, which are each just fixed_three_window-style classifiers)
    # uses a plain classifier head; WICL, adaptive_wem, and dg_coral are
    # the only conditions needing a non-standard module
    return build_backbone(args.backbone, args.pretrained)


def train_one_epoch(condition: str, model, loader, optimizer, device, args) -> dict:
    """Handles all conditions EXCEPT semi_supervised and dg_coral, which
    have their own dedicated orchestration (run_semi_supervised,
    run_dg_coral, below) since each needs more than one plain
    (model, single_loader) training step: semi_supervised trains two
    separate models (teacher, then student); dg_coral needs TWO
    simultaneous pseudo-domain batches per step to compute its
    cross-domain covariance-alignment term, which a single `loader`
    argument cannot provide."""
    model.train()
    running = {}
    n_batches = 0
    randaug = CTAppropriateRandAugment(seed=args.seed) if condition == "dg_augment" else None

    for batch in loader:
        optimizer.zero_grad()

        if condition == "wicl":
            x1 = batch["image_1"].to(device)
            x2 = batch["image_2"].to(device)
            y = batch["labels"].to(device)
            out = model.training_step(x1, x2, y)
            loss = out["loss"]
        elif condition == "adaptive_wem":
            hu_for_wem = batch["hu_for_wem"].to(device)
            hu_raw = batch["hu_raw"].to(device)
            y = batch["labels"].to(device)
            logits = model(hu_for_wem, hu_raw)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, y)
        else:
            x = batch["image"].to(device)
            y = batch["labels"].to(device)
            if condition == "dg_augment" and randaug is not None:
                x = torch.stack([randaug(xi) for xi in x.cpu()]).to(device)
            if condition == "random_window_single":
                logits = model(x)
                loss = random_window_single_view_loss(logits, y)
            else:
                logits = model(x)
                loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, y)

        loss.backward()
        optimizer.step()

        running["loss"] = running.get("loss", 0.0) + float(loss.detach())
        n_batches += 1

    return {k: v / max(n_batches, 1) for k, v in running.items()}


def _split_labeled_unlabeled(train_samples: list, unlabeled_fraction: float,
                              seed: int) -> tuple[list, list]:
    """manuscript Section VI-A, baseline 5: split this fold's PATIENTS
    (not slices, to avoid leaking a patient's slices across the
    labeled/unlabeled boundary) into a labeled subset (for teacher
    training) and an unlabeled subset (whose labels are withheld and
    replaced by the teacher's pseudo-labels for student training)."""
    rng = np.random.RandomState(seed)
    patient_ids = sorted({s.patient_id for s in train_samples})
    rng.shuffle(patient_ids)
    n_unlabeled = int(len(patient_ids) * unlabeled_fraction)
    unlabeled_patients = set(patient_ids[:n_unlabeled])
    labeled_samples = [s for s in train_samples if s.patient_id not in unlabeled_patients]
    unlabeled_samples = [s for s in train_samples if s.patient_id in unlabeled_patients]
    return labeled_samples, unlabeled_samples


def _split_two_pseudo_domains(train_samples: list, seed: int) -> tuple[list, list]:
    """manuscript Section VI-A, baseline 4b: split this fold's PATIENTS
    (patient-grouped, so no patient's slices cross the domain boundary --
    the same leakage-avoidance discipline as _split_labeled_unlabeled)
    into two roughly equal pseudo-domains for CORAL's cross-domain
    covariance-alignment term. Unlike _split_labeled_unlabeled, BOTH
    domains keep their TRUE labels: CORAL aligns feature-covariance
    statistics across domains, it does not require or use unlabeled data,
    so there is no labeled/unlabeled asymmetry here."""
    rng = np.random.RandomState(seed)
    patient_ids = sorted({s.patient_id for s in train_samples})
    rng.shuffle(patient_ids)
    half = len(patient_ids) // 2
    domain_a_patients = set(patient_ids[:half])
    domain_a = [s for s in train_samples if s.patient_id in domain_a_patients]
    domain_b = [s for s in train_samples if s.patient_id not in domain_a_patients]
    return domain_a, domain_b


def train_one_epoch_coral(model, loader_a: DataLoader, loader_b: DataLoader,
                           optimizer, device: str, coral_weight: float = 1.0
                           ) -> dict:
    """Baseline 4b's actual training step (manuscript Section VI-A): pulls
    ONE batch from each pseudo-domain per step, computes the standard
    supervised loss on both (so the model remains a valid ICH classifier,
    not just a domain-invariance-only model), plus
    `coral_weight * coral_loss(embedding_a, embedding_b)`
    (models.augmentation_baselines.coral_loss) aligning the two domains'
    penultimate-layer feature covariances. `loader_b` is cycled if shorter
    than `loader_a` so one epoch is defined by exhausting domain A."""
    model.train()
    running = {"loss": 0.0, "supervised_loss": 0.0, "coral_loss": 0.0}
    n_batches = 0
    iter_b = iter(loader_b)

    for batch_a in loader_a:
        try:
            batch_b = next(iter_b)
        except StopIteration:
            iter_b = iter(loader_b)
            batch_b = next(iter_b)

        x_a = batch_a["image"].to(device)
        y_a = batch_a["labels"].to(device)
        x_b = batch_b["image"].to(device)
        y_b = batch_b["labels"].to(device)

        optimizer.zero_grad()
        logits_a, emb_a = model(x_a)
        logits_b, emb_b = model(x_b)

        bce = torch.nn.functional.binary_cross_entropy_with_logits
        supervised_loss = 0.5 * (bce(logits_a, y_a) + bce(logits_b, y_b))
        c_loss = coral_loss(emb_a, emb_b)
        loss = supervised_loss + coral_weight * c_loss

        loss.backward()
        optimizer.step()

        running["loss"] += float(loss.detach())
        running["supervised_loss"] += float(supervised_loss.detach())
        running["coral_loss"] += float(c_loss.detach())
        n_batches += 1

    return {k: v / max(n_batches, 1) for k, v in running.items()}


def run_dg_coral(args, train_samples: list, device: str) -> torch.nn.Module:
    """Baseline 4b (manuscript Section VI-A), CORAL feature-alignment
    regularization, applied across two patient-grouped pseudo-domains
    carved out of this fold's own training data (Section VI-A: "no
    external-domain data is available at training time in a zero-shot
    protocol"). Trains one EmbeddingBackbone model for `args.epochs`
    epochs via train_one_epoch_coral, mirroring the epoch/checkpoint
    structure of every other single-model condition in main()."""
    domain_a, domain_b = _split_two_pseudo_domains(train_samples, args.seed)
    logger.info("dg_coral: domain A %d patients' slices (%d), "
                "domain B %d patients' slices (%d)",
                len({s.patient_id for s in domain_a}), len(domain_a),
                len({s.patient_id for s in domain_b}), len(domain_b))

    ds_a = MultiSiteICHDataset(domain_a, view_mode="fixed_three_window", seed=args.seed)
    ds_b = MultiSiteICHDataset(domain_b, view_mode="fixed_three_window", seed=args.seed)
    loader_a = DataLoader(ds_a, batch_size=args.batch_size, shuffle=True,
                           num_workers=args.num_workers, drop_last=True)
    loader_b = DataLoader(ds_b, batch_size=args.batch_size, shuffle=True,
                           num_workers=args.num_workers, drop_last=True)

    model = EmbeddingBackbone(args.backbone, args.pretrained).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr,
                                  weight_decay=args.weight_decay)

    history = []
    for epoch in range(args.epochs):
        t0 = time.time()
        stats = train_one_epoch_coral(model, loader_a, loader_b, optimizer,
                                       device, args.coral_weight)
        stats["epoch"] = epoch
        stats["elapsed_sec"] = time.time() - t0
        history.append(stats)
        logger.info("[dg_coral] epoch %d/%d loss=%.4f (supervised=%.4f coral=%.4f) (%.1fs)",
                    epoch + 1, args.epochs, stats["loss"], stats["supervised_loss"],
                    stats["coral_loss"], stats["elapsed_sec"])

    with open(args.out_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2)
    return model


def run_semi_supervised(args, train_samples: list, device: str) -> torch.nn.Module:
    """Baseline 5 (manuscript Section VI-A), reimplementing Lin & Yuh's
    teacher-student protocol [34]:

      Stage 1 (teacher): train a standard fixed_three_window classifier
      on the LABELED subset of this fold's training patients.

      Stage 2 (pseudo-labeling): run the teacher on the UNLABELED subset
      (their true labels withheld) to produce confidence-thresholded
      pseudo-labels (models/semi_supervised.generate_pseudo_labels).

      Stage 3 (student): train a classifier WARM-STARTED from the
      teacher's weights (verified against [34]'s full text -- their
      student "was initialized based on the weights of the teacher
      model") on the labeled data (true labels) PLUS the unlabeled data
      (pseudo-labels), combined per training step via
      models/semi_supervised.student_training_loss.

    DOCUMENTED SIMPLIFICATION (corrected after full-text verification
    against [34], per models/semi_supervised.py's module docstring):
    [34]'s own "unlabeled" pool is concretely the RSNA 2019 challenge
    dataset itself (25,000 exams), whose image-level labels are present
    in the RSNA release but deliberately DISCARDED for their SSL step --
    it was never a separate never-labeled corpus, so treating a
    patient-disjoint subset of RSNA's OWN labels as "withheld" here
    (`--unlabeled-fraction`) is actually a reasonably faithful analogue of
    what [34] does on the unlabeled side. The real, remaining difference
    is on the LABELED/teacher side: [34]'s teacher trains on a genuinely
    separate, single-institution cohort (457 pixel-labeled scans from one
    US institution, not part of RSNA), whereas this function's teacher
    trains on a patient-disjoint subset of RSNA itself. A run with access
    to a genuinely separate labeled institutional cohort for the teacher
    stage would more faithfully reproduce [34]'s data regime; absent that,
    this difference should be stated explicitly wherever baseline 5's
    results are reported, per the manuscript's Section VI-A commitment to
    document such deviations.
    """
    teacher_epochs = args.teacher_epochs or args.epochs
    labeled, unlabeled = _split_labeled_unlabeled(
        train_samples, args.unlabeled_fraction, args.seed)
    logger.info("semi_supervised: %d labeled patients' slices (%d), "
                "%d unlabeled patients' slices (%d)",
                len({s.patient_id for s in labeled}), len(labeled),
                len({s.patient_id for s in unlabeled}), len(unlabeled))

    # --- Stage 1: teacher ---
    teacher_ds = MultiSiteICHDataset(labeled, view_mode="fixed_three_window",
                                      seed=args.seed)
    teacher_loader = DataLoader(teacher_ds, batch_size=args.batch_size,
                                 shuffle=True, num_workers=args.num_workers,
                                 drop_last=True)
    teacher = build_backbone(args.backbone, args.pretrained).to(device)
    teacher_opt = torch.optim.Adam(teacher.parameters(), lr=args.lr,
                                    weight_decay=args.weight_decay)
    for epoch in range(teacher_epochs):
        stats = train_one_epoch("fixed_three_window", teacher, teacher_loader,
                                 teacher_opt, device, args)
        logger.info("[teacher] epoch %d/%d loss=%.4f",
                    epoch + 1, teacher_epochs, stats["loss"])

    # --- Stage 2: pseudo-labeling ---
    # Rank-based thresholding (PseudoLabelConfig's default, matching [34]'s
    # verified "top 10% by probability" rule) requires ranking over the
    # FULL unlabeled pool at once -- teacher logits are therefore collected
    # across every batch FIRST, and generate_pseudo_labels is called ONCE
    # on the concatenated tensor, not per mini-batch (calling it per-batch
    # would rank within each small batch independently, which is wrong for
    # a pool-relative percentile rule and was a bug in an earlier version
    # of this function).
    teacher.eval()
    unlabeled_ds = MultiSiteICHDataset(unlabeled, view_mode="fixed_three_window",
                                        seed=args.seed)
    unlabeled_loader = DataLoader(unlabeled_ds, batch_size=args.batch_size,
                                   shuffle=False, num_workers=args.num_workers)
    pseudo_config = PseudoLabelConfig()
    all_logits, all_images = [], []
    with torch.no_grad():
        for batch in unlabeled_loader:
            x = batch["image"].to(device)
            all_logits.append(teacher(x).cpu())
            all_images.append(x.cpu())
    all_logits_cat = torch.cat(all_logits)
    pseudo_labels, pseudo_masks = generate_pseudo_labels(all_logits_cat, pseudo_config)
    logger.info("pseudo-label coverage (non-ambiguous class fraction): %.3f",
                float(pseudo_masks.mean()))

    # --- Stage 3: student ---
    # warm-started from the teacher's weights, matching [34]'s verified
    # protocol ("student was initialized based on the weights of the
    # teacher model") rather than a fresh random initialization -- an
    # earlier version of this function trained the student from scratch,
    # corrected here after full-text verification against [34].
    student = build_backbone(args.backbone, args.pretrained).to(device)
    student.load_state_dict(teacher.state_dict())
    student_opt = torch.optim.Adam(student.parameters(), lr=args.lr,
                                    weight_decay=args.weight_decay)
    labeled_loader = DataLoader(teacher_ds, batch_size=args.batch_size,
                                 shuffle=True, num_workers=args.num_workers,
                                 drop_last=True)

    n_unlabeled = pseudo_labels.shape[0]
    if n_unlabeled == 0:
        raise ValueError(
            "unlabeled pool is empty -- increase --unlabeled-fraction or "
            "the number of training patients in this fold")
    all_images_cat = torch.cat(all_images)  # concatenated ONCE, reused every
                                              # batch/epoch below rather than
                                              # re-concatenated per step
    rng = np.random.RandomState(args.seed)
    for epoch in range(args.epochs):
        student.train()
        running_loss, n_batches = 0.0, 0
        for batch in labeled_loader:
            x_labeled = batch["image"].to(device)
            y_labeled = batch["labels"].to(device)

            bs = x_labeled.shape[0]
            # sampled WITH replacement so this is correct regardless of
            # how the unlabeled pool size compares to the batch size --
            # important for small/synthetic smoke-testing runs where the
            # unlabeled pool can be smaller than one batch
            idx = rng.choice(n_unlabeled, size=bs, replace=(n_unlabeled < bs))
            # NOTE: images for the unlabeled/pseudo-labeled batch were
            # collected once (Stage 2, no augmentation) rather than
            # re-sampled with fresh augmentation each epoch; a full
            # "Noisy Student" implementation applies stronger noise to the
            # unlabeled branch specifically -- CTAppropriateRandAugment
            # could be applied to x_unlabeled here for closer fidelity to
            # [34]'s "noisy" student framing, left as a follow-up rather
            # than assumed equivalent.
            x_unlabeled = all_images_cat[idx].to(device)
            pl_batch = pseudo_labels[idx].to(device)
            mask_batch = pseudo_masks[idx].to(device)

            student_opt.zero_grad()
            logits_labeled = student(x_labeled)
            logits_unlabeled = student(x_unlabeled)
            out = student_training_loss(logits_labeled, y_labeled,
                                         logits_unlabeled, pl_batch, mask_batch)
            out["loss"].backward()
            student_opt.step()

            running_loss += float(out["loss"].detach())
            n_batches += 1
        logger.info("[student] epoch %d/%d loss=%.4f",
                    epoch + 1, args.epochs, running_loss / max(n_batches, 1))

    return student


def main(argv=None):
    args = build_argparser().parse_args(argv)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    with open(args.out_dir / "config.json", "w") as f:
        json.dump(vars(args), f, indent=2, default=str)

    samples, folds = build_folds(args)
    fold = folds[args.fold]

    train_samples = [samples[i] for i in fold.train_idx]
    val_samples = [samples[i] for i in fold.val_idx]
    val_ds = MultiSiteICHDataset(val_samples, view_mode="fixed_three_window",
                                  seed=args.seed)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.num_workers)

    logger.info("training condition=%s fold=%d train_n=%d val_n=%d device=%s",
                args.condition, args.fold, len(train_samples), len(val_samples),
                args.device)

    if args.condition == "semi_supervised":
        model = run_semi_supervised(args, train_samples, args.device)
        torch.save({"model_state": model.state_dict(), "args": vars(args)},
                   args.out_dir / "checkpoint_last.pt")
        logger.info("done. student checkpoint written to %s", args.out_dir)
        return

    if args.condition == "dg_coral":
        model = run_dg_coral(args, train_samples, args.device)
        torch.save({"model_state": model.state_dict(), "args": vars(args)},
                   args.out_dir / "checkpoint_last.pt")
        logger.info("done. checkpoint + history written to %s", args.out_dir)
        return

    view_mode = CONDITION_VIEW_MODE[args.condition]
    train_ds = MultiSiteICHDataset(train_samples, view_mode=view_mode,
                                    sampling_breadth=args.sampling_breadth,
                                    seed=args.seed)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                               num_workers=args.num_workers, drop_last=True)

    model = build_model(args.condition, args).to(args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr,
                                  weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    history = []
    for epoch in range(args.epochs):
        t0 = time.time()
        stats = train_one_epoch(args.condition, model, train_loader,
                                 optimizer, args.device, args)
        scheduler.step()
        stats["epoch"] = epoch
        stats["elapsed_sec"] = time.time() - t0
        history.append(stats)
        logger.info("epoch %d/%d loss=%.4f (%.1fs)",
                    epoch + 1, args.epochs, stats["loss"], stats["elapsed_sec"])

        torch.save({"model_state": model.state_dict(), "args": vars(args),
                    "epoch": epoch}, args.out_dir / "checkpoint_last.pt")

    with open(args.out_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2)
    logger.info("done. checkpoint + history written to %s", args.out_dir)


if __name__ == "__main__":
    main()
