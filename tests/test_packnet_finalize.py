"""PackNet end-of-task finalize: single pack, no repack on task switch."""

from __future__ import annotations

import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader, TensorDataset

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import parser as file_parser  # noqa: E402
from model.packnet import Net  # noqa: E402


def _tiny_args() -> object:
    """Minimal IQ-style args for a small PackNet ResNet1D."""
    chain = [
        str(ROOT / "configs" / "base.yaml"),
        str(ROOT / "configs" / "models" / "til" / "packnet.yaml"),
    ]
    args = file_parser.parse_args_from_yaml(chain)
    args.cuda = False
    args.model = "packnet"
    args.arch = "resnet1d"
    args.dataset = "iq"
    args.data_scaling = "none"
    args.classes_per_task = [2, 2]
    args.nc_per_task_list = ""
    args.nc_per_task = None
    args.batch_size = 8
    args.inner_steps = 1
    args.lr = 0.01
    args.optimizer = "sgd"
    args.post_prune_epochs = 0
    args.prune_perc = 0.5
    args.class_weighted_ce = False
    args.loader = "task_incremental_loader"
    return args


def _first_prunable_owner_snapshot(model: Net) -> tuple[str, torch.Tensor]:
    for name, _ in model._prunable_named_parameters():
        owner, _ = model._get_owner_and_frozen(name)
        return name, owner.detach().clone()
    raise AssertionError("expected at least one prunable parameter")


@pytest.mark.parametrize("with_post_prune", [False, True])
def test_finalize_once_and_no_owner_shift_on_task_switch(
    with_post_prune: bool,
) -> None:
    """Packing runs in finalize; switching to the next task does not repack."""
    torch.manual_seed(0)
    args = _tiny_args()
    if with_post_prune:
        args.post_prune_epochs = 1
    n_inputs = 2 * 32
    n_outputs = 4
    n_tasks = 2
    model = Net(n_inputs, n_outputs, n_tasks, args)

    x0 = torch.randn(4, 2, 32)
    y0 = torch.randint(0, 2, (4,))
    for _ in range(3):
        model.observe(x0, y0, 0)

    loader = DataLoader(TensorDataset(x0, y0), batch_size=4)
    model.finalize_task_after_training(train_loader=loader if with_post_prune else None)

    name, owner_after_finalize = _first_prunable_owner_snapshot(model)
    assert 0 in owner_after_finalize.unique().tolist()

    model.finalize_task_after_training(None)
    _, owner_second_finalize = _first_prunable_owner_snapshot(model)
    assert torch.equal(owner_second_finalize, owner_after_finalize)

    x1 = torch.randn(4, 2, 32)
    y1 = torch.randint(2, 4, (4,))
    model.observe(x1, y1, 1)

    _, owner_after_task1 = _first_prunable_owner_snapshot(model)
    committed = owner_after_finalize == 0
    if committed.any():
        assert torch.equal(
            owner_after_task1[committed],
            owner_after_finalize[committed],
        )
