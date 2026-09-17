"""Tests for the WICL model's forward/backward pass and loss ablation
(manuscript Section V-C, Section VI-D)."""
import torch

from ich_gen.models.wicl import build_wicl_model, WICLLossWeights, WICLModel


def _synthetic_batch(seed=0, batch=4):
    torch.manual_seed(seed)
    x1 = torch.randn(batch, 3, 224, 224)
    x2 = torch.randn(batch, 3, 224, 224)
    y = torch.randint(0, 2, (batch, 6)).float()
    return x1, x2, y


def test_training_step_produces_finite_loss():
    model = build_wicl_model(pretrained=False)
    x1, x2, y = _synthetic_batch()
    out = model.training_step(x1, x2, y)
    assert torch.isfinite(out["loss"])
    assert torch.isfinite(out["supervised_loss"])
    assert torch.isfinite(out["consistency_loss"])
    assert torch.isfinite(out["embedding_loss"])


def test_gradients_flow_to_all_parameters():
    model = build_wicl_model(pretrained=False)
    x1, x2, y = _synthetic_batch()
    out = model.training_step(x1, x2, y)
    out["loss"].backward()
    grads = [p.grad for p in model.parameters() if p.requires_grad]
    assert len(grads) > 0
    assert all(g is not None for g in grads)
    assert all(torch.isfinite(g).all() for g in grads)


def test_zero_weight_ablation_reduces_to_supervised_only():
    """manuscript Section VI-D ablation: lambda_consistency = lambda_embedding
    = 0 must reduce WICL's loss EXACTLY to the plain supervised term."""
    model = WICLModel(pretrained=False,
                       weights=WICLLossWeights(lambda_consistency=0.0,
                                                lambda_embedding=0.0))
    x1, x2, y = _synthetic_batch()
    out = model.training_step(x1, x2, y)
    assert abs(float(out["loss"].detach()) - float(out["supervised_loss"])) < 1e-5


def test_prediction_only_ablation():
    model = WICLModel(pretrained=False,
                       weights=WICLLossWeights(lambda_consistency=1.0,
                                                lambda_embedding=0.0))
    x1, x2, y = _synthetic_batch()
    out = model.training_step(x1, x2, y)
    expected = out["supervised_loss"] + 1.0 * out["consistency_loss"]
    assert abs(float(out["loss"].detach()) - float(expected)) < 1e-4


def test_inference_path_matches_expected_shape():
    model = build_wicl_model(pretrained=False)
    x1, _, _ = _synthetic_batch()
    logits = model.forward_single(x1)
    assert logits.shape == (4, 6)


def test_identical_views_give_near_zero_consistency_loss():
    """Sanity check: if the two 'independent' views happen to be identical,
    the KL and cosine consistency terms should collapse to ~0 (no
    disagreement to penalize), confirming the loss terms measure what
    they claim to measure."""
    model = build_wicl_model(pretrained=False)
    x1, _, y = _synthetic_batch()
    out = model.training_step(x1, x1.clone(), y)
    assert float(out["consistency_loss"]) < 1e-4
    assert float(out["embedding_loss"]) < 1e-4
