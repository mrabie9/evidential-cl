"""The evaluation pathway must read per-task BatchNorm stats without writing them.

``main.eval_tasks`` is the single funnel for every evaluation in a run (zero-shot,
mid-epoch validation, end-of-task validation, final test), and it reaches models
through :func:`utils.training_forward.model_forward_for_metric_loop`. These tests
pin the three properties that funnel has to provide:

1. an evaluation pass mutates **no** BatchNorm buffer, even mid-training with the
   model left in train mode by the caller;
2. task ``t``'s logits are produced with task ``t``'s statistics;
3. the task that was active before the evaluation is active again afterwards.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import main as main_module  # noqa: E402
import parser as file_parser  # noqa: E402
from model import task_bn  # noqa: E402
from utils.training_forward import model_forward_for_metric_loop  # noqa: E402


def _args(config_name: str) -> object:
    """Build args from a TIL config.

    ``args.model`` is left at whatever the YAML declares -- the config stem and the
    module name differ for some models (``ucl`` -> ``ucl_bresnet``, ``lamaml`` ->
    ``lamaml_cifar``), and ``args.model`` is what selects the module and what
    ``task_bn.task_bn_enabled`` checks against its exclusion list.
    """
    chain = [
        str(ROOT / "configs" / "base.yaml"),
        str(ROOT / "configs" / "models" / "til" / f"{config_name}.yaml"),
    ]
    args = file_parser.parse_args_from_yaml(chain)
    args.cuda = False
    args.arch = "resnet1d"
    args.dataset = "iq"
    args.data_scaling = "none"
    args.classes_per_task = [2, 2]
    args.nc_per_task_list = ""
    args.nc_per_task = None
    args.batch_size = 16
    args.inner_steps = 1
    # iCaRL asserts on this when no loader-backed resolver is present.
    args.samples_per_task = 16
    args.get_samples_per_task = None
    args.class_weighted_ce = False
    args.loader = "task_incremental_loader"
    args.bn_mode = "task_specific"
    args.norm_type = "batchnorm"
    args.use_groupnorm = False
    return args


def _build(config_name: str):
    """Build the model named by ``config_name``'s YAML, trained on 2 tasks."""
    torch.manual_seed(0)
    args = _args(config_name)
    module = __import__(f"model.{args.model}", fromlist=["Net"])
    model = module.Net(2 * 32, 4, 2, args)
    layers = task_bn.install(model, args, num_tasks=2)
    assert layers, f"{args.model} should have convertible BatchNorm1d layers"

    batches = {
        0: (torch.randn(16, 2, 32), torch.randint(0, 2, (16,))),
        1: (torch.randn(16, 2, 32) * 6.0 - 4.0, torch.randint(2, 4, (16,))),
    }
    for task in (0, 1):
        task_bn.set_active_task(model, task)
        x, y = batches[task]
        for _ in range(3):
            model.observe(x, y, task)
    return model, layers, args, batches


def _tasks_from(batches: dict) -> list:
    return [[(batches[0][0], batches[0][1])], [(batches[1][0], batches[1][1])]]


def _buffer_snapshot(layers) -> list:
    snapshot = []
    for layer in layers:
        snapshot.append(
            (
                layer.task_running_mean.detach().clone(),
                layer.task_running_var.detach().clone(),
                layer.task_num_batches_tracked.detach().clone(),
            )
        )
    return snapshot


def _assert_unchanged(layers, snapshot) -> None:
    for layer, (mean, var, count) in zip(layers, snapshot):
        assert torch.equal(layer.task_running_mean, mean)
        assert torch.equal(layer.task_running_var, var)
        assert torch.equal(layer.task_num_batches_tracked, count)


# ----------------------------------------------------------------------
@pytest.mark.parametrize(
    "model_name",
    [
        "ewc",
        "si",
        "lwf",
        "rwalk",
        "packnet",
        "eralg4",
        "er_ring",
        "hat",
        "ucl",
        "icarl",
        "lamaml",
    ],
)
def test_eval_tasks_never_mutates_bn_buffers(model_name: str) -> None:
    """A full eval_tasks sweep must be read-only w.r.t. every task's statistics.

    ``hat`` and ``ucl`` are the interesting cases: HAT's ``observe`` calls
    ``set_bn_eval(False)`` and UCL can force BatchNorm back into train mode for
    multi-sample evaluation, so both have a route to writing during eval.

    ``ctn`` is absent deliberately: its ``validation: 0.3`` memory split yields a
    1-row replay batch at this fixture's memory size, which BatchNorm rejects in
    train mode. That reproduces identically under ``bn_mode=shared`` (i.e. with no
    conversion at all), so it is a fixture limitation, not an eval-path property.
    """
    model, layers, args, batches = _build(model_name)
    snapshot = _buffer_snapshot(layers)

    main_module.eval_tasks(model, _tasks_from(batches), args)

    _assert_unchanged(layers, snapshot)


@pytest.mark.parametrize("model_name", ["ewc", "packnet"])
def test_eval_is_read_only_even_when_called_in_train_mode(model_name: str) -> None:
    """The mid-epoch probe evaluates while the loop still holds the model in train mode."""
    model, layers, args, batches = _build(model_name)
    model.train()
    task_bn.set_active_task(model, 1)
    snapshot = _buffer_snapshot(layers)

    main_module.eval_tasks(model, _tasks_from(batches), args)

    _assert_unchanged(layers, snapshot)


def test_eval_uses_the_queried_tasks_statistics() -> None:
    """Poisoning task 0's row must move task 0's logits and leave task 1's alone.

    Asserted on logits rather than macro recall: an under-trained model predicts a
    constant class, and macro recall over two classes is 0.5 whichever class that
    is, so the metric is blind to the substitution this test is pinning down.
    """
    model, layers, args, batches = _build("ewc")
    model.eval()
    probe = batches[0][0]

    with torch.no_grad():
        base_task0 = model_forward_for_metric_loop(model, probe, 0, args).clone()
        base_task1 = model_forward_for_metric_loop(model, probe, 1, args).clone()

    # Same input, different task id -> different statistics -> different logits.
    assert not torch.allclose(base_task0, base_task1)

    for layer in layers:
        layer.task_running_mean[0].fill_(50.0)
        layer.task_running_var[0].fill_(0.01)

    with torch.no_grad():
        poisoned_task0 = model_forward_for_metric_loop(model, probe, 0, args)
        poisoned_task1 = model_forward_for_metric_loop(model, probe, 1, args)

    assert not torch.allclose(
        base_task0, poisoned_task0
    ), "task 0's logits must depend on task 0's running statistics"
    assert torch.equal(
        base_task1, poisoned_task1
    ), "task 1's logits must not depend on task 0's running statistics"


@pytest.mark.parametrize("active_task", [0, 1])
def test_eval_restores_the_previously_active_task(active_task: int) -> None:
    """A validation sweep mid-task must leave the training task selected.

    Task 0 is covered explicitly because it is falsy: the funnel has to guard on
    ``is not None``, not on truthiness, or task 0 would never be restored.
    """
    model, layers, args, batches = _build("ewc")
    task_bn.set_active_task(model, active_task)
    assert task_bn.get_active_task(model) == active_task

    main_module.eval_tasks(model, _tasks_from(batches), args)

    assert task_bn.get_active_task(model) == active_task
    for layer in layers:
        assert layer.active_task == active_task


def test_specific_task_eval_uses_the_true_task_id() -> None:
    """``specific_task`` trims the list but must still query the real task id."""
    model, layers, args, batches = _build("ewc")
    tasks = _tasks_from(batches)

    for layer in layers:
        layer.task_running_mean[1].fill_(50.0)
        layer.task_running_var[1].fill_(0.01)

    full = main_module.eval_tasks(model, tasks, args)
    full_recall = full[0] if isinstance(full, tuple) else full

    only_task1 = main_module.eval_tasks(model, tasks, args, specific_task=1)
    only_recall = only_task1[0] if isinstance(only_task1, tuple) else only_task1

    # Evaluating task 1 alone must match its entry in the full sweep; if the
    # trimmed list caused task 0's row to be used, these would differ.
    assert only_recall[0] == full_recall[1]


def test_untrained_task_eval_does_not_write_a_row() -> None:
    """Zero-shot evaluation runs over tasks that were never trained."""
    torch.manual_seed(0)
    args = _args("ewc")
    module = __import__(f"model.{args.model}", fromlist=["Net"])
    model = module.Net(2 * 32, 4, 2, args)
    layers = task_bn.install(model, args, num_tasks=2)

    batches = {
        0: (torch.randn(4, 2, 32), torch.randint(0, 2, (4,))),
        1: (torch.randn(4, 2, 32), torch.randint(2, 4, (4,))),
    }
    snapshot = _buffer_snapshot(layers)

    # Nothing has been trained yet: this is main.py's pre-train zero-shot sweep.
    main_module.eval_tasks(model, _tasks_from(batches), args)

    _assert_unchanged(layers, snapshot)
    for layer in layers:
        assert layer._task_trained == [False, False]
        assert layer._last_trained_task == -1
