"""Tests for per-sample validation loss computation."""

import torch
import torch.nn.functional as F

from training.metrics import per_sample_nll


def test_per_sample_nll_matches_mean_cross_entropy():
    torch.manual_seed(0)
    logits = torch.randn(2, 6, 11)
    labels = torch.tensor([
        [1, 2, 3, -100, -100, -100],
        [4, 5, 6, 7, 8, 9],
    ])

    got = per_sample_nll(logits, labels)
    expected = torch.stack([
        F.cross_entropy(logits[i, :-1], labels[i, 1:], ignore_index=-100)
        for i in range(labels.size(0))
    ])

    assert torch.allclose(got, expected, atol=1e-6)


def test_per_sample_nll_handles_fully_masked_sample():
    logits = torch.randn(2, 4, 7)
    labels = torch.tensor([
        [-100, -100, -100, -100],
        [1, 2, 3, 4],
    ])

    got = per_sample_nll(logits, labels)

    # The fully masked sample has no contributing token and gets a zero loss.
    assert got[0].item() == 0.0
    assert got[1].item() > 0.0
