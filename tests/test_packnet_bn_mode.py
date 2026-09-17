"""PackNet bn_mode option: shared vs task_specific BatchNorm handling.

Per-task BatchNorm statistics moved out of ``model/packnet.py`` into the shared
:mod:`model.task_bn` module, which ``main`` installs once per run. These tests
cover PackNet's remaining responsibility (validating ``bn_mode``) and the
conversion behaviour it now delegates.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import parser as file_parser  # noqa: E402
from model import task_bn  # noqa: E402
from model.packnet import Net  # noqa: E402


def _tiny_args(bn_mode: str) -> object:
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
    args.bn_mode = bn_mode
    return args


def _build_model(bn_mode: str) -> Net:
    torch.manual_seed(0)
    args = _tiny_args(bn_mode)
    return Net(2 * 32, 4, 2, args)


def test_invalid_bn_mode_rejected() -> None:
    args = _tiny_args("bogus")
    with pytest.raises(ValueError):
        Net(2 * 32, 4, 2, args)


def test_task_specific_keeps_per_task_stats() -> None:
    args = _tiny_args("task_specific")
    torch.manual_seed(0)
    model = Net(2 * 32, 4, 2, args)
    layers = task_bn.install(model, args, num_tasks=2)
    assert layers, "packnet's resnet1d backbone should expose BatchNorm1d layers"

    x0 = torch.randn(4, 2, 32)
    y0 = torch.randint(0, 2, (4,))
    x1 = torch.randn(4, 2, 32) * 5.0 - 3.0
    y1 = torch.randint(2, 4, (4,))

    task_bn.set_active_task(model, 0)
    for _ in range(3):
        model.observe(x0, y0, 0)
    model.finalize_task_after_training(train_loader=None)
    task0_mean = layers[0].task_running_mean[0].detach().clone()

    task_bn.set_active_task(model, 1)
    for _ in range(3):
        model.observe(x1, y1, 1)

    # Task 0's statistics are untouched by task 1's training, and the two rows
    # have genuinely diverged.
    assert torch.equal(layers[0].task_running_mean[0], task0_mean)
    assert not torch.allclose(
        layers[0].task_running_mean[0], layers[0].task_running_mean[1]
    )

    # Evaluating task 1 does not disturb task 0's row (the old snapshot/restore
    # implementation left the last-evaluated task's stats live).
    model.eval()
    task_bn.set_active_task(model, 1)
    model.forward(x1, 1)
    assert torch.equal(layers[0].task_running_mean[0], task0_mean)


def test_shared_bn_leaves_backbone_unconverted() -> None:
    args = _tiny_args("shared")
    torch.manual_seed(0)
    model = Net(2 * 32, 4, 2, args)
    assert model.bn_mode == "shared"

    layers = task_bn.install(model, args, num_tasks=2)
    assert layers == []
    assert task_bn.task_bn_layers(model) == []

    # Training still runs, on a single shared set of running statistics.
    x0 = torch.randn(4, 2, 32)
    y0 = torch.randint(0, 2, (4,))
    for _ in range(3):
        model.observe(x0, y0, 0)
    model.finalize_task_after_training(train_loader=None)
    model.observe(x0, y0, 1)


def test_build_model_helper_still_constructs() -> None:
    """Both modes construct a usable PackNet without task_bn installed."""
    for bn_mode in ("task_specific", "shared"):
        model = _build_model(bn_mode)
        assert model.bn_mode == bn_mode
