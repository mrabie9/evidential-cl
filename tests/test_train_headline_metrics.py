"""Tests for the training headline (``tr_macro_*``) returned by ``life_experience``."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, Iterable
from unittest.mock import patch

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from main import life_experience

# Training macro score reported for each task's batches, keyed by task id.
TASK_SCORES = {
    0: {"rec": 0.2, "prec": 0.3, "f1": 0.1},
    1: {"rec": 0.6, "prec": 0.5, "f1": 0.7},
}


class _TqdmPassthrough:
    """Iterable tqdm stand-in with the ``set_description`` hook main.py calls."""

    def __init__(self, iterable: Iterable[Any], **kwargs: Any) -> None:
        del kwargs
        self._iterable = iterable

    def __iter__(self) -> Any:
        return iter(self._iterable)

    def set_description(self, *args: Any, **kwargs: Any) -> None:
        del args, kwargs


class _TinyModel(torch.nn.Module):
    """Linear model whose ``observe`` returns logits, so no extra metric forward runs."""

    split = False

    def __init__(self) -> None:
        super().__init__()
        self.fc = torch.nn.Linear(4, 6)
        self.real_epoch = 0

    def forward(self, x: torch.Tensor, task_id: int) -> torch.Tensor:
        del task_id
        return self.fc(x)

    def observe(self, v_x: torch.Tensor, v_y: torch.Tensor, task_id: int):
        logits = self.forward(v_x, task_id)
        return 0.0, 0.0, logits.detach()


class _TwoTaskLoader:
    """Task 0 uses labels 0-2 and task 1 labels 3-5, each with one batch."""

    n_tasks = 2

    def __init__(self) -> None:
        self._next = 0

    def new_task(self):
        task = self._next
        self._next += 1
        labels = torch.arange(3, dtype=torch.long).repeat(2) + 3 * task
        loader = DataLoader(TensorDataset(torch.randn(6, 4), labels), batch_size=6)
        info = {"task": task, "task_name": "task_{}".format(task)}
        return info, loader, None, loader


def _score(metric: str):
    """Return a metric stub scoring a batch by the task its labels belong to."""

    def _stub(predictions: torch.Tensor, labels: torch.Tensor) -> float:
        del predictions
        return TASK_SCORES[int(labels.max()) // 3][metric]

    return _stub


def test_til_training_headline_averages_every_task(tmp_path) -> None:
    """TIL reports the mean of each task's final-epoch training metrics."""
    args = SimpleNamespace(
        state_logging=False,
        n_epochs=2,
        val_rate=1,
        loader="task_incremental_loader",
        cuda=False,
        arch="",
        model="stub_model_name",
        calc_test_accuracy=False,
        log_dir=str(tmp_path),
    )
    with patch("main.tqdm", _TqdmPassthrough), patch(
        "main.macro_recall", _score("rec")
    ), patch("main.macro_precision", _score("prec")), patch(
        "main.macro_f1", _score("f1")
    ):
        *_, headline = life_experience(_TinyModel(), _TwoTaskLoader(), args)

    assert headline["tr_macro_rec"] == pytest.approx(0.4)
    assert headline["tr_macro_prec"] == pytest.approx(0.4)
    assert headline["tr_macro_f1"] == pytest.approx(0.4)
