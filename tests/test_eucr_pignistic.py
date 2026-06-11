"""Unit tests for the pignistic (BetP + NLL) EUCR head.

Fast, CPU-only: checks the pignistic transform is a valid distribution and that
``PignisticNLLLoss`` is the negative log-likelihood of the true class.
"""

from __future__ import annotations

# ruff: noqa: E402

import math
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from model.evidential_modules import PignisticNLLLoss, pignistic_probability


def test_pignistic_is_a_distribution() -> None:
    # mass = [beliefs (C), omega]; beliefs sum to 1 - omega.
    mass = torch.tensor([[0.3, 0.2, 0.1, 0.4], [0.0, 0.0, 0.0, 1.0]])
    betp = pignistic_probability(mass)
    assert betp.shape == (2, 3)
    assert torch.allclose(betp.sum(dim=-1), torch.ones(2), atol=1e-5)
    assert (betp >= 0).all()


def test_pignistic_splits_omega_uniformly() -> None:
    # Total ignorance -> uniform; omega is divided equally over C classes.
    mass = torch.tensor([[0.0, 0.0, 0.0, 1.0]])
    betp = pignistic_probability(mass)
    assert torch.allclose(betp, torch.full((1, 3), 1.0 / 3.0), atol=1e-5)

    mass2 = torch.tensor([[0.6, 0.0, 0.0, 0.4]])  # belief 0.6 on class 0, omega 0.4
    betp2 = pignistic_probability(mass2)
    expected = torch.tensor([[0.6 + 0.4 / 3, 0.4 / 3, 0.4 / 3]])
    assert torch.allclose(betp2, expected, atol=1e-5)


def test_pignistic_nll_matches_negative_log_prob() -> None:
    loss_fn = PignisticNLLLoss(num_classes=3)
    probs = torch.tensor([[0.7, 0.2, 0.1], [0.1, 0.8, 0.1]])
    targets = torch.tensor([0, 1])
    loss = loss_fn(probs, targets)
    expected = -(math.log(0.7) + math.log(0.8)) / 2
    assert math.isclose(loss.item(), expected, rel_tol=1e-4)


def test_pignistic_nll_ignores_beliefs_and_epoch() -> None:
    loss_fn = PignisticNLLLoss(num_classes=3)
    probs = torch.tensor([[0.7, 0.2, 0.1]])
    targets = torch.tensor([0])
    assert torch.allclose(
        loss_fn(probs, targets), loss_fn(probs, targets, beliefs=None, epoch=7)
    )
