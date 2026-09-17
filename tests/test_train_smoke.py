"""End-to-end smoke test: every fully-wired condition (manuscript Section
VI-A/VI-B) must actually train for one epoch on synthetic data without
NaN losses, and model weights must actually update -- catching the class
of bug where a condition silently no-ops (e.g. an unused branch, a
detached loss, wrong optimizer target)."""
import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from ich_gen.datasets.common import Sample, MultiSiteICHDataset
from ich_gen.train import (build_model, train_one_epoch, CONDITION_VIEW_MODE,
                            run_semi_supervised, _split_labeled_unlabeled,
                            run_dg_coral, _split_two_pseudo_domains,
                            train_one_epoch_coral)

FULLY_WIRED_CONDITIONS = (
    "fixed_single_window", "fixed_three_window", "adaptive_hrt",
    "adaptive_wem", "dg_augment", "random_window_single", "wicl",
)
# dg_coral is intentionally excluded from this parametrized list: it now
# uses its own dedicated two-loader training path (train_one_epoch_coral /
# run_dg_coral, tested separately below), not the single-loader
# train_one_epoch this parametrized test drives -- same reason
# semi_supervised is excluded (its own two-stage run_semi_supervised path).


def _synthetic_samples(n=16, seed=0):
    rng = np.random.RandomState(seed)

    def make_loader():
        arr = rng.uniform(-200, 300, size=(64, 64)).astype(np.float32)
        return lambda a=arr: a

    samples = []
    for i in range(n):
        labels = np.zeros(6, dtype=np.float32)
        if rng.rand() < 0.5:
            labels[0] = 1
            labels[rng.randint(1, 6)] = 1
        samples.append(Sample(f"s{i}", f"p{i // 2}", "synthetic",
                               make_loader(), labels, None))
    return samples


class _Args:
    backbone = "densenet121"
    pretrained = False
    lambda_consistency = 1.0
    lambda_embedding = 0.5
    sampling_breadth = "moderate"
    seed = 0


@pytest.mark.parametrize("condition", FULLY_WIRED_CONDITIONS)
def test_condition_trains_and_updates_weights(condition):
    samples = _synthetic_samples()
    view_mode = CONDITION_VIEW_MODE.get(condition, "fixed_three_window")
    ds = MultiSiteICHDataset(samples, view_mode=view_mode, seed=0)
    loader = DataLoader(ds, batch_size=4, shuffle=True, num_workers=0, drop_last=True)

    args = _Args()
    args.condition = condition
    model = build_model(condition, args)
    param_before = next(model.parameters()).clone()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    stats = train_one_epoch(condition, model, loader, optimizer, "cpu", args)

    assert stats["loss"] == stats["loss"], f"{condition}: loss is NaN"
    assert stats["loss"] > 0
    param_after = next(model.parameters())
    assert not torch.allclose(param_before, param_after), \
        f"{condition}: model weights did not update"


def test_all_nine_conditions_are_recognized_by_the_cli():
    """Every condition in ich_gen.train.CONDITIONS (manuscript Section
    VI-A's nine conditions) must be accepted by the argument parser --
    catches the case where a condition is documented/tested but not
    actually reachable via the CLI."""
    from ich_gen.train import build_argparser, CONDITIONS
    parser = build_argparser()
    assert len(CONDITIONS) == 9
    for condition in CONDITIONS:
        args = parser.parse_args(["--condition", condition,
                                   "--rsna-root", "/nonexistent",
                                   "--out-dir", "/nonexistent/out"])
        assert args.condition == condition


def test_split_labeled_unlabeled_is_patient_grouped():
    """manuscript Section VI-A, baseline 5: the labeled/unlabeled split
    used for semi-supervised pseudo-labeling must not split a single
    patient's slices across the boundary -- otherwise the 'unlabeled'
    pseudo-label stage would trivially leak that patient's true labels
    via the teacher having seen other slices from the same patient."""
    samples = _synthetic_samples(n=40)
    labeled, unlabeled = _split_labeled_unlabeled(samples, unlabeled_fraction=0.5, seed=0)

    labeled_patients = {s.patient_id for s in labeled}
    unlabeled_patients = {s.patient_id for s in unlabeled}
    assert not (labeled_patients & unlabeled_patients), \
        "a patient's slices must not appear in both the labeled and unlabeled subsets"
    assert len(labeled) + len(unlabeled) == len(samples)
    assert len(unlabeled_patients) > 0 and len(labeled_patients) > 0


def test_split_labeled_unlabeled_fraction_respected_approximately():
    samples = _synthetic_samples(n=100, seed=1)
    labeled, unlabeled = _split_labeled_unlabeled(samples, unlabeled_fraction=0.3, seed=1)
    total_patients = len({s.patient_id for s in samples})
    unlabeled_patients = len({s.patient_id for s in unlabeled})
    # patient-level fraction, not slice-level -- should be close to 0.3
    assert abs(unlabeled_patients / total_patients - 0.3) < 0.15


def test_split_two_pseudo_domains_is_patient_grouped():
    """manuscript Section VI-A, baseline 4b: the two CORAL pseudo-domains
    must not split a single patient's slices across the domain boundary,
    the same leakage discipline as the semi-supervised labeled/unlabeled
    split -- otherwise the 'cross-domain' covariance term would partly be
    comparing a patient against themselves."""
    samples = _synthetic_samples(n=40)
    domain_a, domain_b = _split_two_pseudo_domains(samples, seed=0)

    patients_a = {s.patient_id for s in domain_a}
    patients_b = {s.patient_id for s in domain_b}
    assert not (patients_a & patients_b)
    assert len(domain_a) + len(domain_b) == len(samples)
    assert len(patients_a) > 0 and len(patients_b) > 0
    # both domains retain TRUE labels (unlike semi-supervised's unlabeled
    # side) -- every sample in both domains keeps its original label array
    original_by_id = {s.sample_id: s.labels for s in samples}
    for s in domain_a + domain_b:
        assert np.array_equal(s.labels, original_by_id[s.sample_id])


def test_train_one_epoch_coral_updates_weights_and_finite_loss():
    samples = _synthetic_samples(n=32, seed=1)
    domain_a, domain_b = _split_two_pseudo_domains(samples, seed=1)

    from torch.utils.data import DataLoader
    ds_a = MultiSiteICHDataset(domain_a, view_mode="fixed_three_window", seed=0)
    ds_b = MultiSiteICHDataset(domain_b, view_mode="fixed_three_window", seed=0)
    loader_a = DataLoader(ds_a, batch_size=4, shuffle=True, num_workers=0, drop_last=True)
    loader_b = DataLoader(ds_b, batch_size=4, shuffle=True, num_workers=0, drop_last=True)

    args = _Args()
    model = build_model("dg_coral", args)
    param_before = next(model.parameters()).clone()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    stats = train_one_epoch_coral(model, loader_a, loader_b, optimizer, "cpu",
                                   coral_weight=1.0)

    assert stats["loss"] == stats["loss"], "loss is NaN"
    assert stats["supervised_loss"] == stats["supervised_loss"]
    assert stats["coral_loss"] == stats["coral_loss"]
    assert stats["coral_loss"] >= 0.0  # CORAL term is a squared distance
    param_after = next(model.parameters())
    assert not torch.allclose(param_before, param_after), \
        "dg_coral: model weights did not update"


def test_loader_b_cycles_when_shorter_than_loader_a():
    """domain B being smaller than domain A must not truncate the epoch
    early or crash -- loader_b should cycle."""
    rng = np.random.RandomState(2)

    def make_loader():
        arr = rng.uniform(-200, 300, size=(64, 64)).astype(np.float32)
        return lambda a=arr: a

    # 8 patients x 2 slices for domain A-ish, but force a small domain B
    samples = [Sample(f"s{i}", f"p{i}", "synthetic", make_loader(),
                       np.zeros(6, dtype=np.float32), None) for i in range(20)]
    from torch.utils.data import DataLoader
    ds_a = MultiSiteICHDataset(samples[:16], view_mode="fixed_three_window", seed=0)
    ds_b = MultiSiteICHDataset(samples[16:], view_mode="fixed_three_window", seed=0)  # only 4 samples
    loader_a = DataLoader(ds_a, batch_size=4, shuffle=True, num_workers=0, drop_last=True)
    loader_b = DataLoader(ds_b, batch_size=4, shuffle=True, num_workers=0, drop_last=True)

    args = _Args()
    model = build_model("dg_coral", args)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    # loader_a has 4 batches, loader_b has only 1 -- must cycle 4 times, not crash
    stats = train_one_epoch_coral(model, loader_a, loader_b, optimizer, "cpu")
    assert stats["loss"] == stats["loss"]


def test_run_dg_coral_end_to_end():
    import tempfile
    from pathlib import Path

    class _CoralArgs(_Args):
        condition = "dg_coral"
        batch_size = 4
        lr = 1e-3
        weight_decay = 1e-5
        num_workers = 0
        epochs = 2
        coral_weight = 1.0
        device = "cpu"

    samples = _synthetic_samples(n=32, seed=3)
    with tempfile.TemporaryDirectory() as tmp:
        args = _CoralArgs()
        args.out_dir = Path(tmp)
        model = run_dg_coral(args, samples, "cpu")
        assert sum(p.numel() for p in model.parameters()) > 0
        assert (args.out_dir / "history.json").exists()


def test_semi_supervised_two_stage_training_runs_end_to_end():
    """The full teacher -> pseudo-label -> student pipeline (manuscript
    Section VI-A, baseline 5, reimplementing Lin & Yuh [34]) must run
    without error and produce a student model with real, finite
    parameters -- the most complex single piece of wiring in this
    codebase, so it gets its own dedicated (slower) test rather than
    relying only on the parametrized fast check above."""

    class _SSArgs(_Args):
        condition = "semi_supervised"
        batch_size = 4
        lr = 1e-3
        weight_decay = 1e-5
        num_workers = 0
        epochs = 1
        teacher_epochs = 1
        unlabeled_fraction = 0.5
        device = "cpu"

    samples = _synthetic_samples(n=32, seed=2)
    student = run_semi_supervised(_SSArgs(), samples, "cpu")

    params = list(student.parameters())
    assert len(params) > 0
    assert all(torch.isfinite(p).all() for p in params)


def test_semi_supervised_student_warm_start_does_not_crash_with_zero_student_epochs():
    """manuscript Section VI-A / models/semi_supervised.py: the student is
    warm-started via student.load_state_dict(teacher.state_dict()) --
    verified against [34]'s full text. With args.epochs=0 the Stage-3
    training loop never executes, so the returned student IS exactly the
    warm-started (teacher-initialized) state; if load_state_dict had a
    key/shape mismatch (e.g. from a future architecture change breaking
    the teacher/student parity this warm-start silently assumes), this
    would raise immediately rather than surface only in a slow real run."""

    class _SSArgsZeroEpochs(_Args):
        condition = "semi_supervised"
        batch_size = 4
        lr = 1e-3
        weight_decay = 1e-5
        num_workers = 0
        epochs = 0
        teacher_epochs = 1
        unlabeled_fraction = 0.5
        device = "cpu"

    samples = _synthetic_samples(n=32, seed=4)
    student = run_semi_supervised(_SSArgsZeroEpochs(), samples, "cpu")
    assert all(torch.isfinite(p).all() for p in student.parameters())
