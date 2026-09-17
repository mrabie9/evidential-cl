"""Task-specific BatchNorm running statistics (:mod:`model.task_bn`)."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from model import task_bn  # noqa: E402
from model.adab1n import AdaB1N  # noqa: E402
from model.resnet1d import ResNet1D  # noqa: E402
from model.task_bn import TaskSpecificBatchNorm1d  # noqa: E402


def _args(**overrides: object) -> object:
    args = type("Args", (), {})()
    args.bn_mode = "task_specific"
    args.loader = "task_incremental_loader"
    args.model = "ewc"
    args.norm_type = "batchnorm"
    args.use_groupnorm = False
    for key, value in overrides.items():
        setattr(args, key, value)
    return args


def _tiny_backbone() -> nn.Module:
    torch.manual_seed(0)
    return nn.Sequential(nn.Conv1d(2, 4, 3, padding=1), nn.BatchNorm1d(4))


def _train(norm: TaskSpecificBatchNorm1d, x: torch.Tensor, steps: int = 5) -> None:
    norm.train()
    for _ in range(steps):
        norm(x)


# ----------------------------------------------------------------------
# The layer itself
# ----------------------------------------------------------------------
def test_buffer_rows_are_contiguous_views() -> None:
    """F.batch_norm updates in place, so each task row must write through."""
    norm = TaskSpecificBatchNorm1d(4, num_tasks=3)
    assert norm.task_running_mean[1].is_contiguous()

    norm.set_task(1)
    _train(norm, torch.randn(8, 4, 16) * 3.0 + 5.0)
    assert not torch.allclose(norm.task_running_mean[1], torch.zeros(4))


def test_per_task_statistics_are_isolated() -> None:
    norm = TaskSpecificBatchNorm1d(4, num_tasks=3)
    x0 = torch.randn(8, 4, 16) * 3.0 + 5.0
    x1 = torch.randn(8, 4, 16) * 0.5 - 2.0

    norm.set_task(0)
    _train(norm, x0)
    task0_mean = norm.task_running_mean[0].detach().clone()

    norm.set_task(1)
    _train(norm, x1)

    assert torch.equal(norm.task_running_mean[0], task0_mean)
    assert not torch.allclose(norm.task_running_mean[0], norm.task_running_mean[1])


def test_eval_reads_the_active_task_row() -> None:
    norm = TaskSpecificBatchNorm1d(4, num_tasks=3)
    x0 = torch.randn(8, 4, 16) * 3.0 + 5.0
    x1 = torch.randn(8, 4, 16) * 0.5 - 2.0

    norm.set_task(0)
    _train(norm, x0)
    norm.set_task(1)
    _train(norm, x1)

    probe = torch.randn(4, 4, 16)
    norm.eval()
    norm.set_task(0)
    out0 = norm(probe)
    norm.set_task(1)
    out1 = norm(probe)

    assert not torch.allclose(out0, out1)


def test_untrained_task_falls_back_to_last_trained() -> None:
    norm = TaskSpecificBatchNorm1d(4, num_tasks=3)
    norm.set_task(0)
    _train(norm, torch.randn(8, 4, 16) * 3.0 + 5.0)

    probe = torch.randn(4, 4, 16)
    norm.eval()
    norm.set_task(0)
    trained = norm(probe)
    norm.set_task(2)  # never trained
    untrained = norm(probe)

    assert torch.allclose(trained, untrained)


def test_new_task_warm_starts_from_previous_task() -> None:
    norm = TaskSpecificBatchNorm1d(4, num_tasks=3)
    norm.set_task(0)
    _train(norm, torch.randn(8, 4, 16) * 3.0 + 5.0)
    task0_mean = norm.task_running_mean[0].detach().clone()

    # First training batch on task 1 seeds from task 0 rather than (0, 1).
    norm.set_task(1)
    norm.train()
    assert torch.allclose(norm.task_running_mean[1], torch.zeros(4))
    norm(torch.randn(8, 4, 16) * 3.0 + 5.0)
    assert not torch.allclose(norm.task_running_mean[1], torch.zeros(4))
    # The seed moved it near task 0 rather than from the default origin.
    assert torch.linalg.vector_norm(
        norm.task_running_mean[1] - task0_mean
    ) < torch.linalg.vector_norm(task0_mean)


def test_set_task_rejects_out_of_range() -> None:
    norm = TaskSpecificBatchNorm1d(4, num_tasks=2)
    with pytest.raises(ValueError):
        norm.set_task(2)
    with pytest.raises(ValueError):
        norm.set_task(-1)


# ----------------------------------------------------------------------
# Freezing on replay passes
# ----------------------------------------------------------------------
def test_frozen_running_stats_writes_nothing() -> None:
    backbone = nn.Sequential(TaskSpecificBatchNorm1d(4, num_tasks=2))
    norm = backbone[0]
    norm.set_task(0)
    _train(norm, torch.randn(8, 4, 16))

    before_mean = norm.task_running_mean.clone()
    before_var = norm.task_running_var.clone()
    before_count = norm.task_num_batches_tracked.clone()

    norm.train()
    with task_bn.frozen_running_stats(backbone):
        norm(torch.randn(8, 4, 16) * 20.0 + 50.0)

    assert torch.equal(norm.task_running_mean, before_mean)
    assert torch.equal(norm.task_running_var, before_var)
    assert torch.equal(norm.task_num_batches_tracked, before_count)


def test_frozen_pass_matches_unfrozen_output_and_gradients() -> None:
    """Freezing must change only the buffer writes, never the computation.

    This is the claim the replay-pass design rests on: in training mode
    BatchNorm normalizes with batch statistics regardless of the buffers.
    """
    backbone = nn.Sequential(TaskSpecificBatchNorm1d(4, num_tasks=2))
    norm = backbone[0]
    norm.set_task(0)
    _train(norm, torch.randn(8, 4, 16))
    norm.train()

    x = torch.randn(8, 4, 16) * 2.0 + 1.0

    frozen_input = x.clone().requires_grad_(True)
    with task_bn.frozen_running_stats(backbone):
        frozen_out = norm(frozen_input)
    frozen_out.sum().backward()

    unfrozen_input = x.clone().requires_grad_(True)
    unfrozen_out = norm(unfrozen_input)
    unfrozen_out.sum().backward()

    assert torch.allclose(frozen_out, unfrozen_out, atol=1e-6)
    assert torch.allclose(frozen_input.grad, unfrozen_input.grad, atol=1e-6)


def test_frozen_running_stats_restores_previous_flag() -> None:
    backbone = nn.Sequential(TaskSpecificBatchNorm1d(4, num_tasks=2))
    norm = backbone[0]
    with task_bn.frozen_running_stats(backbone):
        assert norm.freeze_running_stats is True
        with task_bn.frozen_running_stats(backbone):
            assert norm.freeze_running_stats is True
        assert norm.freeze_running_stats is True
    assert norm.freeze_running_stats is False


# ----------------------------------------------------------------------
# Conversion
# ----------------------------------------------------------------------
def test_conversion_reuses_affine_parameters() -> None:
    backbone = _tiny_backbone()
    original_weight = backbone[1].weight
    original_bias = backbone[1].bias
    optimizer = torch.optim.SGD(backbone.parameters(), lr=0.1)

    layers = task_bn.convert_batchnorm_to_task_specific(backbone, num_tasks=3)

    assert len(layers) == 1
    assert backbone[1].weight is original_weight
    assert backbone[1].bias is original_bias
    optimizer_params = [p for g in optimizer.param_groups for p in g["params"]]
    assert any(p is original_weight for p in optimizer_params)


def test_conversion_skips_adab1n_and_groupnorm() -> None:
    backbone = nn.Sequential(AdaB1N(4, num_tasks=2), nn.GroupNorm(2, 4))
    assert task_bn.convert_batchnorm_to_task_specific(backbone, num_tasks=3) == []
    assert isinstance(backbone[0], AdaB1N)


def test_conversion_is_not_reentrant() -> None:
    backbone = _tiny_backbone()
    task_bn.convert_batchnorm_to_task_specific(backbone, num_tasks=3)
    assert task_bn.convert_batchnorm_to_task_specific(backbone, num_tasks=3) == []


def test_resnet1d_backbone_converts() -> None:
    args = _args()
    args.dataset = "iq"
    args.data_scaling = "none"
    backbone = ResNet1D(4, args)
    layers = task_bn.convert_batchnorm_to_task_specific(backbone, num_tasks=3)
    assert len(layers) > 1
    assert task_bn.task_bn_layers(backbone) == layers


# ----------------------------------------------------------------------
# Enablement gate
# ----------------------------------------------------------------------
@pytest.mark.parametrize(
    "overrides, expected",
    [
        ({}, True),
        ({"bn_mode": "shared"}, False),
        ({"loader": "class_incremental_loader"}, False),
        ({"model": "iid2"}, False),
        ({"model": "anml"}, False),
        ({"norm_type": "groupnorm"}, False),
        ({"use_groupnorm": True}, False),
        ({"norm_type": "adab1n"}, False),
    ],
)
def test_task_bn_enabled_gate(overrides: dict, expected: bool) -> None:
    assert task_bn.task_bn_enabled(_args(**overrides)) is expected


def test_install_rejects_unknown_bn_mode() -> None:
    backbone = _tiny_backbone()
    with pytest.raises(ValueError):
        task_bn.install(backbone, _args(bn_mode="bogus"), num_tasks=2)


def test_install_on_shared_mode_leaves_model_alone() -> None:
    backbone = _tiny_backbone()
    assert task_bn.install(backbone, _args(bn_mode="shared"), num_tasks=2) == []
    assert isinstance(backbone[1], nn.BatchNorm1d)
    assert not isinstance(backbone[1], TaskSpecificBatchNorm1d)


def test_set_active_task_is_a_noop_without_layers() -> None:
    backbone = _tiny_backbone()
    task_bn.install(backbone, _args(bn_mode="shared"), num_tasks=2)
    task_bn.set_active_task(backbone, 1)  # must not raise
    assert task_bn.get_active_task(backbone) is None


def test_helpers_tolerate_non_module_callers() -> None:
    """Test doubles and functional wrappers must not trip the helpers."""
    sentinel = object()
    task_bn.set_active_task(sentinel, 0)
    assert task_bn.get_active_task(sentinel) is None
    with task_bn.frozen_running_stats(sentinel):
        pass


# ----------------------------------------------------------------------
# Checkpointing
# ----------------------------------------------------------------------
def test_state_dict_round_trip_preserves_per_task_stats() -> None:
    norm = TaskSpecificBatchNorm1d(4, num_tasks=3)
    norm.set_task(0)
    _train(norm, torch.randn(8, 4, 16) * 3.0 + 5.0)
    norm.set_task(1)
    _train(norm, torch.randn(8, 4, 16) * 0.5 - 2.0)

    probe = torch.randn(4, 4, 16)
    norm.eval()
    norm.set_task(1)
    expected = norm(probe)

    reloaded = TaskSpecificBatchNorm1d(4, num_tasks=3)
    reloaded.load_state_dict(norm.state_dict())
    reloaded.eval()
    reloaded.set_task(1)

    assert torch.allclose(reloaded(probe), expected)
    assert reloaded._task_trained == [True, True, False]
    assert reloaded._last_trained_task == 1


def test_state_dict_carries_task_rows() -> None:
    norm = TaskSpecificBatchNorm1d(4, num_tasks=3)
    keys = norm.state_dict().keys()
    for key in (
        "task_running_mean",
        "task_running_var",
        "task_num_batches_tracked",
        "task_trained",
        "last_trained_task",
    ):
        assert key in keys
