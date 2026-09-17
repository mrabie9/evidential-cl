"""AdaB1N wiring: unbiased reweighting, task-counter advance, host integration."""

from __future__ import annotations

import random
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import parser as file_parser  # noqa: E402
from model.adab1n import (  # noqa: E402
    AdaB1N,
    adab1n_layers,
    clear_batch_task_counts,
    end_task_all,
    set_batch_task_counts,
)


def _sample_weight_sum(layer: AdaB1N) -> float:
    """Reproduce training_forward's weighting and return the total row weight."""
    concentration = (
        layer.task_weight[: layer.cur_tasks + 1].exp() + layer.task_counts_extended
    )
    concentration = concentration * (layer.task_counts_extended > 0)
    task_weights = concentration / concentration.sum()
    weights = task_weights[layer.sample_task_indices] / layer.sample_task_counts
    return float(weights.sum().detach())


@pytest.mark.parametrize("cur_tasks", [1, 3, 9])
@pytest.mark.parametrize("present", ["all", "current_only", "replay_only"])
def test_reweighting_is_unbiased_under_partial_task_coverage(
    cur_tasks: int, present: str
) -> None:
    """Row weights must sum to 1 however few tasks the batch covers.

    A weighted mean is only unbiased when the weights sum to 1. Before the
    partial-coverage fix, absent tasks kept their share and the batch mean shrank
    toward zero, progressively worse with each task.
    """
    layer = AdaB1N(num_features=4, num_tasks=20, init_weight=0.5)
    layer.cur_tasks.fill_(cur_tasks)

    if present == "all":
        indices = torch.arange(cur_tasks + 1).repeat_interleave(2)
    elif present == "current_only":
        indices = torch.full((6,), cur_tasks, dtype=torch.long)
    else:
        indices = torch.arange(cur_tasks).repeat_interleave(2)

    assert set_batch_task_counts([layer], indices) is True
    assert _sample_weight_sum(layer) == pytest.approx(1.0)


def test_set_batch_task_counts_derives_metadata() -> None:
    layer = AdaB1N(num_features=4, num_tasks=20)
    layer.cur_tasks.fill_(2)
    indices = torch.tensor([0, 0, 0, 2])

    set_batch_task_counts([layer], indices)

    assert torch.equal(layer.task_counts_extended, torch.tensor([3.0, 0.0, 1.0]))
    assert torch.equal(layer.sample_task_counts, torch.tensor([3.0, 3.0, 3.0, 1.0]))


def test_set_batch_task_counts_rejects_task_id_beyond_counter() -> None:
    """A row from a task the layer has not been told about is a wiring error."""
    layer = AdaB1N(num_features=4, num_tasks=20)
    with pytest.raises(ValueError, match="end_task_all"):
        set_batch_task_counts([layer], torch.tensor([0, 1]))


def test_empty_layer_list_is_a_noop() -> None:
    assert set_batch_task_counts([], torch.tensor([0])) is False
    clear_batch_task_counts([])
    end_task_all([])


def test_clear_restores_unweighted_statistics() -> None:
    layer = AdaB1N(num_features=4, num_tasks=20)
    layer.cur_tasks.fill_(1)
    set_batch_task_counts([layer], torch.tensor([0, 1]))
    assert layer.sample_task_indices is not None

    clear_batch_task_counts([layer])

    assert layer.sample_task_indices is None
    layer.train()
    out = layer(torch.randn(4, 4, 8))
    assert out.shape == (4, 4, 8)


def _host_args(model: str, norm_type: str) -> object:
    chain = [str(ROOT / "configs" / "base.yaml")]
    args = file_parser.parse_args_from_yaml(chain)
    args.cuda = False
    args.model = model
    args.arch = "resnet1d"
    args.dataset = "iq"
    args.data_scaling = "none"
    args.classes_per_task = [2, 2]
    args.nc_per_task_list = ""
    args.nc_per_task = None
    args.batch_size = 8
    args.inner_steps = 1
    args.lr = 0.01
    args.class_weighted_ce = False
    args.loader = "task_incremental_loader"
    args.norm_type = norm_type
    args.n_tasks = 2
    args.memories = 32
    args.replay_batch_size = 8
    args.learn_lr = False
    return args


@pytest.mark.parametrize("model_name", ["eralg4", "iid2"])
def test_host_has_no_adab1n_layers_under_batchnorm(model_name: str) -> None:
    """The default norm path must stay untouched: no layers, so every call no-ops."""
    module = __import__(f"model.{model_name}", fromlist=["Net"])
    torch.manual_seed(0)
    model = module.Net(2 * 32, 4, 2, _host_args(model_name, "batchnorm"))
    assert model._adab1n == []
    assert adab1n_layers(model.net) == []
    model.finalize_task_after_training(None)


@pytest.mark.parametrize("model_name", ["eralg4", "iid2"])
def test_host_advances_task_counter_at_boundary(model_name: str) -> None:
    module = __import__(f"model.{model_name}", fromlist=["Net"])
    torch.manual_seed(0)
    model = module.Net(2 * 32, 4, 2, _host_args(model_name, "adab1n"))
    assert len(model._adab1n) > 0
    assert all(int(layer.cur_tasks) == 0 for layer in model._adab1n)

    x = torch.randn(4, 2, 32)
    y = torch.randint(0, 2, (4,))
    model.observe(x, y, 0)
    model.finalize_task_after_training(None)

    assert all(int(layer.cur_tasks) == 1 for layer in model._adab1n)
    # Task 1 rows are now representable, which they were not before the boundary.
    model.observe(x, y, 1)


@pytest.mark.parametrize("model_name", ["eralg4", "iid2"])
def test_host_task_boundary_is_idempotent(model_name: str) -> None:
    """A repeated boundary must not double-advance the counter."""
    module = __import__(f"model.{model_name}", fromlist=["Net"])
    torch.manual_seed(0)
    model = module.Net(2 * 32, 4, 2, _host_args(model_name, "adab1n"))

    model.observe(torch.randn(4, 2, 32), torch.randint(0, 2, (4,)), 0)
    model.finalize_task_after_training(None)
    model.finalize_task_after_training(None)
    model.finalize_task_after_training(None)

    assert all(int(layer.cur_tasks) == 1 for layer in model._adab1n)


def test_eralg4_replay_batch_engages_reweighting() -> None:
    """task_weight trains only once a single forward's batch spans two tasks.

    With one task present the weights renormalise to a constant 1, so the
    gradient is identically zero however the logits move. On eralg4 that means
    the first step of task 1 (replay still drawn purely from task 0) cannot train
    task_weight, and later steps — once the reservoir holds both tasks — can.
    """
    from model.eralg4 import Net

    # eralg4's reservoir draws with the ``random`` module, so seeding torch alone
    # leaves which tasks a replay batch spans at the mercy of earlier tests.
    random.seed(0)
    torch.manual_seed(0)
    model = Net(2 * 32, 4, 2, _host_args("eralg4", "adab1n"))
    x0 = torch.randn(8, 2, 32)
    y0 = torch.randint(0, 2, (8,))
    for _ in range(3):
        model.observe(x0, y0, 0)
    model.finalize_task_after_training(None)

    layer = model._adab1n[0]
    assert all(int(each.cur_tasks) == 1 for each in model._adab1n)

    # First task-1 step: the reservoir holds only task-0 rows, so coverage is
    # single-task and no reweighting gradient exists.
    layer.task_weight.grad = None
    model.observe(torch.randn(8, 2, 32), torch.randint(2, 4, (8,)), 1)
    first = layer.task_weight.grad
    assert first is None or int((first.abs() > 0).sum()) == 0

    # Later steps draw a reservoir sample spanning both tasks.
    grads_seen = 0
    for _ in range(6):
        layer.task_weight.grad = None
        model.observe(torch.randn(8, 2, 32), torch.randint(2, 4, (8,)), 1)
        grad = layer.task_weight.grad
        if grad is not None and int((grad.abs() > 0).sum()) > 0:
            grads_seen += 1
    assert grads_seen > 0, "mixed-task replay must train task_weight"
