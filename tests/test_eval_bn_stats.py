"""Evaluation BatchNorm policy for shared-BN task-incremental runs.

Since ResNet1D stopped forcing train mode at evaluation (47f143c4), shared-BN TIL
runs normalized every task with running statistics that only describe the most
recently trained task, so old tasks looked collapsed (LwF old-task recall 0.44
vs 0.66 on the same weights). ``--eval_bn_stats batch`` restores batch
statistics for those evaluation forwards without writing any buffer.
"""

from __future__ import annotations

# ruff: noqa: E402

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from model import task_bn
from model.adab1n import AdaB1N
from utils.training_forward import model_forward_for_metric_loop


def _args(**overrides) -> SimpleNamespace:
    values = {
        "loader": "task_incremental_loader",
        "bn_mode": "shared",
        "eval_bn_stats": "batch",
        "model": "lwf",
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _buffers(module: nn.Module) -> dict:
    return {k: v.detach().clone() for k, v in module.state_dict().items()}


def _skewed_layer(
    layer: nn.modules.batchnorm._BatchNorm,
) -> nn.modules.batchnorm._BatchNorm:
    """Give ``layer`` running stats far from any unit-scale batch."""
    with torch.no_grad():
        layer.running_mean.fill_(5.0)
        layer.running_var.fill_(9.0)
    return layer


@pytest.mark.parametrize(
    "factory",
    [
        lambda: nn.BatchNorm1d(4),
        lambda: AdaB1N(num_features=4, num_tasks=1),
    ],
    ids=["batchnorm", "adab1n"],
)
def test_batch_statistics_normalizes_with_batch_and_writes_nothing(factory) -> None:
    torch.manual_seed(0)
    layer = _skewed_layer(factory()).eval()
    x = torch.randn(6, 4, 10)
    before = _buffers(layer)

    with task_bn.batch_statistics(layer):
        out = layer(x)

    expected = nn.functional.batch_norm(
        x, None, None, layer.weight, layer.bias, True, 0.0, layer.eps
    )
    torch.testing.assert_close(out, expected)
    for key, value in _buffers(layer).items():
        torch.testing.assert_close(value, before[key])
    assert not layer.training
    assert "forward" not in layer.__dict__


def test_batch_statistics_covers_task_specific_rows() -> None:
    torch.manual_seed(0)
    layer = task_bn.TaskSpecificBatchNorm1d(num_features=4, num_tasks=2)
    layer.set_task(1)
    layer.eval()
    x = torch.randn(6, 4, 10)
    before = _buffers(layer)

    with task_bn.batch_statistics(layer):
        out = layer(x)

    expected = nn.functional.batch_norm(
        x, None, None, layer.weight, layer.bias, True, 0.0, layer.eps
    )
    torch.testing.assert_close(out, expected)
    for key, value in _buffers(layer).items():
        torch.testing.assert_close(value, before[key])


def test_batch_statistics_restores_forward_on_error() -> None:
    layer = nn.BatchNorm1d(4).eval()
    with pytest.raises(RuntimeError):
        with task_bn.batch_statistics(layer):
            raise RuntimeError("boom")
    assert "forward" not in layer.__dict__


@pytest.mark.parametrize(
    "overrides, expected",
    [
        ({}, True),
        ({"eval_bn_stats": "running"}, False),
        ({"bn_mode": "task_specific"}, False),
        # Both loaders follow the same policy since 2026-09-16; the CIL
        # exclusion assumed single-task eval batches, but CIL eval sets are
        # cumulative and shuffled, so their batches span tasks.
        ({"loader": "class_incremental_loader"}, True),
        ({"loader": "class_incremental_loader", "eval_bn_stats": "running"}, False),
        ({"loader": "class_incremental_loader", "bn_mode": "task_specific"}, False),
    ],
)
def test_eval_uses_batch_statistics_gate(overrides, expected) -> None:
    assert task_bn.eval_uses_batch_statistics(_args(**overrides)) is expected


def test_eval_uses_batch_statistics_rejects_unknown_mode() -> None:
    with pytest.raises(ValueError):
        task_bn.eval_uses_batch_statistics(_args(eval_bn_stats="bogus"))


class _TinyNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv1d(2, 4, 3, padding=1)
        self.bn = nn.BatchNorm1d(4)
        self.head = nn.Linear(4, 3)

    def forward(self, x: torch.Tensor, t: int) -> torch.Tensor:
        return self.head(self.bn(self.conv(x)).mean(-1))


def test_metric_loop_uses_batch_statistics_for_shared_til() -> None:
    torch.manual_seed(0)
    model = _TinyNet()
    _skewed_layer(model.bn)
    model.eval()
    x = torch.randn(8, 2, 16)

    with torch.no_grad():
        batch_logits = model_forward_for_metric_loop(model, x, 0, _args())
        running_logits = model_forward_for_metric_loop(
            model, x, 0, _args(eval_bn_stats="running")
        )
        model.bn.train()
        reference = model(x, 0)

    torch.testing.assert_close(batch_logits, reference)
    assert not torch.allclose(batch_logits, running_logits)


def test_metric_loop_does_not_write_running_stats() -> None:
    torch.manual_seed(0)
    model = _TinyNet()
    _skewed_layer(model.bn)
    model.eval()
    before = _buffers(model)
    with torch.no_grad():
        for _ in range(3):
            model_forward_for_metric_loop(model, torch.randn(8, 2, 16), 0, _args())
    for key, value in _buffers(model).items():
        torch.testing.assert_close(value, before[key])
