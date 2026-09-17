"""CIL evaluation scope: per-task columns, current-task logit space, union headline.

Before this, the class-incremental loader's ``new_task`` handed back the pooled
tasks ``0..t`` test set and ``life_experience`` stored it as column ``t``. Every
column was therefore a nested prefix rather than a task, so:

* column ``j`` was masked to classes ``0..j``, hiding the classes learned since
  task ``j`` and quietly making it a task-incremental measurement;
* the headline averaged those prefixes, counting task 0 ``n_tasks`` times and the
  final task once;
* BWT / forgetting / FWT, which all assume column ``j`` is task ``j``, measured
  prefix pooling instead of retention.
"""

from __future__ import annotations

import argparse
from typing import List, Tuple

import torch

from main import eval_cil_pooled, eval_class_tasks, eval_tasks


class _MaskRecordingModel(torch.nn.Module):
    """Records ``(task_index, cil_all_seen_upto_task)`` for every forward."""

    def __init__(self, n_classes: int = 6) -> None:
        super().__init__()
        self.n_classes = n_classes
        self.calls: List[Tuple[int, object]] = []

    def forward(self, x: torch.Tensor, task_id: int, **kwargs) -> torch.Tensor:
        self.calls.append((int(task_id), kwargs.get("cil_all_seen_upto_task")))
        return torch.zeros(x.size(0), self.n_classes)


def _loader(label: int) -> List[tuple]:
    return [(torch.randn(4, 8), torch.full((4,), label, dtype=torch.long))]


def _args(loader: str) -> argparse.Namespace:
    return argparse.Namespace(
        cuda=False,
        arch="linear",
        classes_per_task=[2, 2, 2],
        loader=loader,
        model="iid2",
    )


def test_cil_columns_share_the_current_task_logit_space() -> None:
    """Scoring task j after training task i must keep classes 0..i active.

    Masking column j to 0..j would hide every class learned since task j, which
    turns the CIL number into a task-incremental one.
    """
    model = _MaskRecordingModel()
    tasks = [_loader(label) for label in (0, 1, 2)]

    eval_class_tasks(
        model, tasks, _args("class_incremental_loader"), cil_mask_upto_task=2
    )

    assert [task_id for task_id, _ in model.calls] == [0, 1, 2]
    assert {bound for _, bound in model.calls} == {2}


def test_cil_mask_defaults_to_column_index() -> None:
    """Without an explicit bound each column keeps its own index (old behaviour)."""
    model = _MaskRecordingModel()
    tasks = [_loader(label) for label in (0, 1, 2)]

    eval_class_tasks(model, tasks, _args("class_incremental_loader"))

    assert model.calls == [(0, 0), (1, 1), (2, 2)]


def test_pooled_pass_yields_headline_and_per_task_columns() -> None:
    """One pass over the pooled set gives both numbers, masked to every seen class."""
    model = _MaskRecordingModel()
    pooled = [
        (torch.randn(6, 8), torch.tensor([0, 1, 2, 3, 4, 5], dtype=torch.long)),
    ]

    columns, headline = eval_cil_pooled(
        model, pooled, 2, _args("class_incremental_loader")
    )

    assert model.calls == [(2, 2)]
    for value in headline:
        assert 0.0 <= value <= 1.0
    # classes_per_task [2, 2, 2] -> one column per seen task.
    for per_task in columns:
        assert len(per_task) == 3


def test_columns_are_sliced_from_the_pooled_predictions() -> None:
    """Columns must come from the pooled forward, not from task-pure batches.

    With ``--eval_bn_stats batch`` a task-pure batch and a mixed batch give
    different predictions for the same sample, so re-running the model per task
    would measure a BatchNorm mismatch rather than forgetting. One forward pass
    over the pooled loader is the guarantee against that.
    """

    class _PerfectOnTaskZero(torch.nn.Module):
        """Predicts class 0 for everything, so only task 0 scores."""

        def __init__(self) -> None:
            super().__init__()
            self.forward_calls = 0

        def forward(self, x: torch.Tensor, task_id: int, **kwargs) -> torch.Tensor:
            self.forward_calls += 1
            logits = torch.full((x.size(0), 6), -10.0)
            logits[:, 0] = 10.0
            return logits

    model = _PerfectOnTaskZero()
    pooled = [
        (torch.randn(6, 8), torch.tensor([0, 0, 2, 3, 4, 5], dtype=torch.long)),
    ]

    (col_rec, _, _), _ = eval_cil_pooled(
        model, pooled, 2, _args("class_incremental_loader")
    )

    assert model.forward_calls == 1, "columns must reuse the single pooled pass"
    assert col_rec[0] == 1.0  # task 0's samples are all class 0, all correct
    assert col_rec[1] == 0.0  # tasks 1 and 2 are swallowed by the class-0 bias
    assert col_rec[2] == 0.0


def test_til_never_receives_a_cil_bound() -> None:
    """The task-incremental path must not gain a CIL keyword from the plumbing."""
    model = _MaskRecordingModel()
    tasks = [_loader(label) for label in (0, 1)]

    eval_tasks(model, tasks, _args("task_incremental_loader"), cil_mask_upto_task=1)

    assert model.calls == [(0, None), (1, None)]


def test_evaluation_builds_no_autograd_graph() -> None:
    """The metric loop never backpropagates, so it must not retain a graph."""

    class _GraphProbe(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.linear = torch.nn.Linear(8, 4)
            self.grad_was_enabled: List[bool] = []

        def forward(self, x: torch.Tensor, task_id: int, **kwargs) -> torch.Tensor:
            self.grad_was_enabled.append(torch.is_grad_enabled())
            return self.linear(x)

    model = _GraphProbe()
    eval_tasks(model, [_loader(0)], _args("task_incremental_loader"))

    assert model.grad_was_enabled == [False]
