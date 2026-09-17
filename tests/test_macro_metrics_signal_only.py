"""Guard the macro metric API after the detection/noise removal.

All three metrics are plain macro averages over the classes present in the
batch. None of them takes a noise label any more.
"""

# ruff: noqa: E402

from __future__ import annotations

import inspect
import os
import sys

import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from utils import training_metrics
from utils.training_metrics import macro_f1, macro_precision, macro_recall


def test_metrics_take_only_predictions_and_targets() -> None:
    """No metric accepts a ``noise_label`` argument any more."""
    for metric in (macro_recall, macro_precision, macro_f1):
        parameters = list(inspect.signature(metric).parameters)
        assert parameters == ["preds", "targets"], (metric.__name__, parameters)


def test_removed_metric_names_are_gone() -> None:
    """The noise-aware spellings must not linger as aliases."""
    for removed in ("macro_precision_signal_only", "macro_f1_including_noise"):
        assert not hasattr(training_metrics, removed), removed


def test_perfect_predictions_score_one() -> None:
    """A perfect prediction scores 1.0 on all three metrics."""
    targets = torch.tensor([0, 1, 2, 1, 0])
    assert macro_recall(targets, targets) == 1.0
    assert macro_precision(targets, targets) == 1.0
    assert macro_f1(targets, targets) == 1.0


def test_metrics_average_over_every_class_present() -> None:
    """Class 2 is missed entirely, so macro recall drops to 2/3."""
    targets = torch.tensor([0, 1, 2])
    preds = torch.tensor([0, 1, 1])
    assert macro_recall(preds, targets) == 2.0 / 3.0


def test_empty_batch_returns_zero() -> None:
    """Empty batches score 0.0 rather than raising."""
    empty = torch.tensor([], dtype=torch.long)
    assert macro_recall(empty, empty) == 0.0
    assert macro_precision(empty, empty) == 0.0
    assert macro_f1(empty, empty) == 0.0
