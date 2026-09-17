"""iid2 maximal replay: task t trains on tasks 0..t with per-task logit masking."""

from __future__ import annotations

import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import parser as file_parser  # noqa: E402
from main import _cumulative_train_loader  # noqa: E402
from model import iid2  # noqa: E402


def _iid2_args(loader: str) -> object:
    args = file_parser.parse_args_from_yaml([str(ROOT / "configs" / "base.yaml")])
    args.cuda = False
    args.model = "iid2"
    args.arch = "resnet1d"
    args.dataset = "iq"
    args.data_scaling = "none"
    args.classes_per_task = [2, 3]
    args.nc_per_task_list = ""
    args.nc_per_task = None
    args.inner_steps = 1
    args.lr = 0.01
    args.class_weighted_ce = False
    args.loader = loader
    args.norm_type = "batchnorm"
    return args


def _task_loader(task_id: int, n: int, batch_size: int = 4) -> DataLoader:
    x = torch.full((n, 1), float(task_id))
    y = torch.full((n,), task_id, dtype=torch.long)
    return DataLoader(TensorDataset(x, y), batch_size=batch_size, shuffle=False)


def test_cumulative_loader_covers_every_seen_task() -> None:
    seen: list = []
    first = _cumulative_train_loader(seen, _task_loader(0, 6))
    assert len(first.dataset) == 6

    second = _cumulative_train_loader(seen, _task_loader(1, 10, batch_size=4))
    assert len(second.dataset) == 16
    assert second.batch_size == 4
    labels = torch.cat([y for _, y in second])
    assert torch.bincount(labels).tolist() == [6, 10]


def test_cumulative_loader_mixes_tasks_within_batches() -> None:
    """The concatenation is task-ordered, so without shuffling batches stay pure."""
    torch.manual_seed(0)
    seen: list = []
    _cumulative_train_loader(seen, _task_loader(0, 64))
    loader = _cumulative_train_loader(seen, _task_loader(1, 64, batch_size=16))
    assert any(y.unique().numel() == 2 for _, y in loader)


def test_model_opts_into_cumulative_replay() -> None:
    assert iid2.Net.cumulative_replay is True


def test_til_training_masks_each_row_to_its_own_task() -> None:
    torch.manual_seed(0)
    model = iid2.Net(2 * 32, 5, 2, _iid2_args("task_incremental_loader"))
    logits = torch.zeros(3, 5)
    targets = torch.tensor([0, 3, 4])

    masked = model._mask_training_logits(logits, targets, t=1)

    assert (masked[0, :2] == 0).all() and (masked[0, 2:] < -1e8).all()
    for row in (1, 2):
        assert (masked[row, :2] < -1e8).all() and (masked[row, 2:] == 0).all()


def test_cil_training_masks_only_future_classes() -> None:
    torch.manual_seed(0)
    model = iid2.Net(2 * 32, 5, 2, _iid2_args("class_incremental_loader"))
    targets = torch.tensor([0, 1])

    task0 = model._mask_training_logits(torch.zeros(2, 5), targets, t=0)
    assert (task0[:, :2] == 0).all() and (task0[:, 2:] < -1e8).all()

    task1 = model._mask_training_logits(torch.zeros(2, 5), targets, t=1)
    assert (task1 == 0).all()


def test_til_forward_masks_to_the_queried_task() -> None:
    torch.manual_seed(0)
    model = iid2.Net(2 * 32, 5, 2, _iid2_args("task_incremental_loader")).eval()
    x = torch.randn(4, 2, 32)

    with torch.no_grad():
        predictions = model(x, 1).argmax(dim=1)

    assert ((predictions >= 2) & (predictions < 5)).all()


def test_observe_trains_on_a_mixed_task_batch() -> None:
    torch.manual_seed(0)
    model = iid2.Net(2 * 32, 5, 2, _iid2_args("task_incremental_loader"))
    x = torch.randn(8, 2, 32)
    y = torch.tensor([0, 1, 2, 3, 4, 0, 2, 4])

    loss, _, metric_logits = model.observe(x, y, 1)

    assert torch.isfinite(torch.tensor(loss))
    predictions = metric_logits.argmax(dim=1)
    assert (model._label_to_task[predictions] == model._label_to_task[y]).all()
