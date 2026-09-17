"""ResNet1D normalisation must follow train/eval mode rather than forcing train.

The backbone used to default ``bn_training=True`` and force train mode for every
call, so evaluation normalised each batch with its own statistics. That made eval
metrics depend on test batch composition and hid any hyperparameter that only
shapes running statistics.
"""

from __future__ import annotations

import sys
from types import SimpleNamespace
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from model.resnet1d import ResNet1D  # noqa: E402


def _backbone(norm_type: str = "batchnorm") -> ResNet1D:
    torch.manual_seed(0)
    args = SimpleNamespace(norm_type=norm_type, n_tasks=3, kappa=1.0)
    return ResNet1D(num_classes=4, args=args)


def _running_means(net: ResNet1D) -> list[torch.Tensor]:
    return [
        module.running_mean.detach().clone()
        for module in net.modules()
        if hasattr(module, "running_mean") and module.running_mean is not None
    ]


def _unchanged(before: list[torch.Tensor], after: list[torch.Tensor]) -> bool:
    return all(torch.allclose(a, b) for a, b in zip(before, after))


@pytest.mark.parametrize("norm_type", ["batchnorm", "adab1n"])
def test_eval_does_not_update_running_stats(norm_type: str) -> None:
    net = _backbone(norm_type)
    net.train()
    net(torch.randn(8, 2, 32))

    net.eval()
    before = _running_means(net)
    with torch.no_grad():
        net(torch.randn(8, 2, 32) + 5.0)
    assert _unchanged(before, _running_means(net))


@pytest.mark.parametrize("norm_type", ["batchnorm", "adab1n"])
def test_train_still_updates_running_stats(norm_type: str) -> None:
    net = _backbone(norm_type)
    net.train()
    before = _running_means(net)
    net(torch.randn(8, 2, 32) + 5.0)
    assert not _unchanged(before, _running_means(net))


@pytest.mark.parametrize("norm_type", ["batchnorm", "adab1n"])
def test_eval_output_is_independent_of_other_rows_in_the_batch(norm_type: str) -> None:
    """No transductive leakage: a row's eval logits must not depend on its batch."""
    net = _backbone(norm_type)
    net.train()
    for _ in range(3):
        net(torch.randn(16, 2, 32))

    torch.manual_seed(5)
    target = torch.randn(2, 2, 32)
    distractors = torch.randn(14, 2, 32) * 20.0 + 50.0

    net.eval()
    with torch.no_grad():
        alone = net(target)
        together = net(torch.cat([target, distractors], dim=0))[:2]
    assert torch.allclose(alone, together, atol=1e-5)


def test_bn_training_override_still_works() -> None:
    """Meta-learning inner loops can still force the mode explicitly."""
    net = _backbone()
    net.train()
    net(torch.randn(8, 2, 32))

    # Forced update while in eval mode.
    net.eval()
    before = _running_means(net)
    net(torch.randn(8, 2, 32) + 5.0, bn_training=True)
    assert not _unchanged(before, _running_means(net))

    # Forced suppression while in train mode.
    net.train()
    before = _running_means(net)
    net(torch.randn(8, 2, 32) + 5.0, bn_training=False)
    assert _unchanged(before, _running_means(net))
