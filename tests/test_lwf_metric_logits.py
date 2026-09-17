"""LwF must report training metrics in the global class space.

Regression: ``observe`` returned the task-local logit slice, so every task after
the first showed exactly 0.00 train recall/precision/F1 in ``life_experience``,
which scores ``argmax`` against global labels.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from model.lwf import Net as LwfNet  # noqa: E402
from utils.training_metrics import macro_recall  # noqa: E402

CLASSES_PER_TASK = [5, 6]
N_OUTPUTS = sum(CLASSES_PER_TASK)
LENGTH = 32


def _args() -> object:
    args = type("Args", (), {})()
    args.classes_per_task = CLASSES_PER_TASK
    args.nc_per_task_list = ""
    args.nc_per_task = None
    args.arch = "resnet1d"
    args.dataset = "iq"
    args.data_scaling = "none"
    args.loader = "task_incremental_loader"
    args.lr = 1e-2
    args.optimizer = "sgd"
    args.inner_steps = 1
    args.class_weighted_ce = False
    args.temperature = 2.0
    args.distill_lambda = 1.0
    args.cls_lambda = 1.0
    args.clipgrad = 0.0
    args.cuda = False
    return args


def _batch(task: int, n: int = 12):
    """Inputs plus GLOBAL labels drawn from ``task``'s class block."""
    offset = sum(CLASSES_PER_TASK[:task])
    y = torch.arange(n) % CLASSES_PER_TASK[task] + offset
    return torch.randn(n, 2, LENGTH), y


def test_metric_logits_are_global_width_on_later_tasks() -> None:
    torch.manual_seed(0)
    model = LwfNet(2 * LENGTH, N_OUTPUTS, 2, _args())
    model.observe(*_batch(0), 0)

    x, y = _batch(1)
    _, _, metric_logits = model.observe(x, y, 1)

    assert metric_logits.shape == (len(y), N_OUTPUTS)
    predictions = metric_logits.argmax(dim=1)
    # Task 1 owns global classes 5..10; nothing may be predicted outside it.
    assert bool(((predictions >= 5) & (predictions < N_OUTPUTS)).all())


def test_global_label_metrics_are_not_identically_zero() -> None:
    """The symptom: life_experience scores argmax against global labels."""
    torch.manual_seed(0)
    model = LwfNet(2 * LENGTH, N_OUTPUTS, 2, _args())
    model.observe(*_batch(0), 0)

    x, y = _batch(1)
    for _ in range(25):  # enough steps to beat chance on a 12-row batch
        _, _, metric_logits = model.observe(x, y, 1)

    recall = macro_recall(metric_logits.argmax(dim=1), y)
    assert recall > 0.0


def test_columns_map_to_the_right_global_classes() -> None:
    torch.manual_seed(0)
    model = LwfNet(2 * LENGTH, N_OUTPUTS, 2, _args())
    model.observe(*_batch(0), 0)

    x, y = _batch(1)
    _, _, metric_logits = model.observe(x, y, 1)

    class_ids = model.task_class_ids[1]
    local = model._select_task_logits(metric_logits, class_ids)
    # Every task-1 column survived the scatter, and the rest are masked.
    assert torch.isfinite(local).all()
    assert bool((local > -1e8).all())
    others = [c for c in range(N_OUTPUTS) if c not in class_ids]
    assert bool((metric_logits[:, others] <= -1e8).all())


def test_task_zero_still_reports_sensible_metrics() -> None:
    torch.manual_seed(0)
    model = LwfNet(2 * LENGTH, N_OUTPUTS, 2, _args())

    x, y = _batch(0)
    _, _, metric_logits = model.observe(x, y, 0)

    assert metric_logits.shape == (len(y), N_OUTPUTS)
    predictions = metric_logits.argmax(dim=1)
    assert bool((predictions < 5).all())
