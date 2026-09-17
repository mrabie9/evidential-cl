"""Regression tests for ``main.eval_tasks(..., specific_task=...)`` task ids."""

from __future__ import annotations

import argparse
from typing import List

import torch

from main import eval_tasks


class _TaskIdRecordingModel(torch.nn.Module):
    """Model that records every task id it is asked to forward with.

    Its logits are deterministic per task id so the caller can assert that the
    metrics were produced from the requested task's head.
    """

    def __init__(self, n_classes: int = 4) -> None:
        super().__init__()
        self.n_classes = n_classes
        self.seen_task_ids: List[int] = []

    def forward(self, x: torch.Tensor, task_id: int) -> torch.Tensor:
        self.seen_task_ids.append(int(task_id))
        logits = torch.zeros(x.size(0), self.n_classes)
        logits[:, int(task_id) % self.n_classes] = 1.0
        return logits


def _make_task_loader(label: int, n_classes: int = 4) -> List[tuple]:
    inputs = torch.randn(4, 8)
    targets = torch.full((4,), label, dtype=torch.long)
    assert label < n_classes
    return [(inputs, targets)]


def _args() -> argparse.Namespace:
    # Every sample counts toward the metric: there is no noise class to mask out
    # of the classification metrics.
    return argparse.Namespace(
        cuda=False,
        arch="linear",
        classes_per_task=4,
    )


def test_specific_task_forwards_true_task_id() -> None:
    """Selecting task ``t`` must query the model with ``t``, not position 0."""
    model = _TaskIdRecordingModel()
    tasks = [_make_task_loader(label) for label in (0, 1, 2)]

    eval_tasks(model, tasks, _args(), specific_task=2)

    assert model.seen_task_ids == [2]


def test_specific_task_metrics_match_full_sweep_entry() -> None:
    """``specific_task=t`` must reproduce entry ``t`` of the full evaluation."""
    args = _args()
    tasks = [_make_task_loader(label) for label in (0, 1, 2)]

    full_recalls = eval_tasks(_TaskIdRecordingModel(), tasks, args)[0]
    single_recalls = eval_tasks(_TaskIdRecordingModel(), tasks, args, specific_task=1)[
        0
    ]

    assert single_recalls == [full_recalls[1]]


def test_full_sweep_still_uses_sequential_task_ids() -> None:
    """Without ``specific_task`` every task id is visited in order."""
    model = _TaskIdRecordingModel()
    tasks = [_make_task_loader(label) for label in (0, 1, 2)]

    eval_tasks(model, tasks, args=_args())

    assert model.seen_task_ids == [0, 1, 2]
