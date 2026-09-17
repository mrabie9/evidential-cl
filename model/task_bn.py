"""Task-specific BatchNorm running statistics for task-incremental learning.

In task-incremental learning (TIL) the task id is known at evaluation time, yet a
plain ``nn.BatchNorm1d`` normalizes every task with one shared set of running
statistics accumulated across all tasks. Task ``t``'s activations then get
normalized with a mean/variance dominated by whichever task was trained last,
which depresses backward transfer independently of what the continual-learning
algorithm does to the weights.

:class:`TaskSpecificBatchNorm1d` gives each task its own ``running_mean`` /
``running_var`` row while keeping a **single shared** affine ``weight``/``bias``
pair, so regularization-based learners (EWC, SI, RWalk, LwF, UCL) still penalise
drift of one parameter vector. The per-task rows are registered buffers, so they
survive ``state_dict()`` and the per-task checkpoints written by ``main.py``.

Three call sites drive it, all outside the model files:

* ``main.py`` installs the layers once after the model is built, and calls
  :func:`set_active_task` before every ``observe``.
* :func:`utils.training_forward.model_forward_for_metric_loop` selects the task
  being evaluated and restores the previous one afterwards.
* Replay-based learners wrap their replay forward in :func:`frozen_running_stats`
  so a mixed-task batch is not folded into the current task's statistics. In
  training mode BatchNorm normalizes with *batch* statistics regardless of the
  buffers, so freezing changes no output, loss or gradient -- only what gets
  written.

Usage:
    >>> import torch
    >>> norm = TaskSpecificBatchNorm1d(num_features=8, num_tasks=3)
    >>> norm.set_task(1)
    >>> norm.train()
    >>> norm(torch.randn(4, 8, 20)).shape
    torch.Size([4, 8, 20])
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator, List, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

# Models that must not get per-task normalization: ``iid2`` is the joint/IID
# control and deliberately discards the task id, and ``anml``'s forward takes no
# task id at all (special-cased in ``utils.training_forward``).
EXCLUDED_MODELS = frozenset({"iid2", "anml"})

_GROUPNORM_ALIASES = frozenset({"groupnorm", "group_norm", "gn"})
_ADAB1N_ALIASES = frozenset({"adab1n", "ada_b1n", "adab2n"})

BN_MODES = frozenset({"task_specific", "shared"})


class TaskSpecificBatchNorm1d(nn.BatchNorm1d):
    """``BatchNorm1d`` holding one set of running statistics per task.

    The inherited ``running_mean`` / ``running_var`` buffers are left registered
    but **unused**; read :attr:`task_running_mean` / :attr:`task_running_var`
    (indexed by task) instead.

    Args:
        num_features: Number of channels ``C`` in the expected ``(N, C, L)`` input.
        num_tasks: Number of tasks to allocate statistics for.
        eps: Passed through to ``nn.BatchNorm1d``.
        momentum: Passed through to ``nn.BatchNorm1d``. ``None`` selects a
            cumulative average, using this task's own batch count.
        affine: Passed through to ``nn.BatchNorm1d``. The affine parameters are
            shared across tasks.

    Usage:
        >>> norm = TaskSpecificBatchNorm1d(num_features=4, num_tasks=2)
        >>> norm.set_task(0)
        >>> norm.eval()
        >>> norm(torch.randn(2, 4, 6)).shape
        torch.Size([2, 4, 6])
    """

    def __init__(
        self,
        num_features: int,
        num_tasks: int,
        eps: float = 1e-5,
        momentum: Optional[float] = 0.1,
        affine: bool = True,
    ) -> None:
        super().__init__(num_features, eps, momentum, affine, track_running_stats=True)
        if num_tasks < 1:
            raise ValueError(f"num_tasks must be >= 1, got {num_tasks}.")

        self.num_tasks = int(num_tasks)
        self.register_buffer(
            "task_running_mean", torch.zeros(self.num_tasks, num_features)
        )
        self.register_buffer(
            "task_running_var", torch.ones(self.num_tasks, num_features)
        )
        self.register_buffer(
            "task_num_batches_tracked",
            torch.zeros(self.num_tasks, dtype=torch.long),
        )
        # Persisted mirrors of the Python-side bookkeeping below, so a reloaded
        # checkpoint knows which task rows were actually trained.
        self.register_buffer(
            "task_trained", torch.zeros(self.num_tasks, dtype=torch.uint8)
        )
        self.register_buffer("last_trained_task", torch.tensor(-1, dtype=torch.long))

        # Python-side mirrors, read on every forward. Keeping them off-device
        # avoids a host/device sync per BatchNorm layer per batch.
        self._task_trained: List[bool] = [False] * self.num_tasks
        self._last_trained_task: int = -1

        self.active_task: int = 0
        self.freeze_running_stats: bool = False

    # ------------------------------------------------------------------
    def set_task(self, task_index: int) -> None:
        """Select the task whose statistics the next forward should use.

        Args:
            task_index: Zero-based task id, in ``[0, num_tasks)``.

        Raises:
            ValueError: If ``task_index`` is out of range.

        Usage:
            >>> norm = TaskSpecificBatchNorm1d(num_features=4, num_tasks=3)
            >>> norm.set_task(2)
        """
        index = int(task_index)
        if not 0 <= index < self.num_tasks:
            raise ValueError(
                f"task_index {index} out of range for num_tasks={self.num_tasks}."
            )
        self.active_task = index

    # ------------------------------------------------------------------
    def _seed_untrained_task(self, task_index: int) -> None:
        """Warm-start an untrained task's statistics from the last trained task.

        Keeps a new task from starting at the ``(0, 1)`` defaults, which would
        also make the zero-shot evaluation of not-yet-trained tasks meaningless.

        Args:
            task_index: Task about to receive its first training batch.
        """
        source = self._last_trained_task
        if source >= 0 and source != task_index:
            self.task_running_mean[task_index].copy_(self.task_running_mean[source])
            self.task_running_var[task_index].copy_(self.task_running_var[source])
        self._task_trained[task_index] = True
        self._last_trained_task = task_index
        self.task_trained[task_index] = 1
        self.last_trained_task.fill_(task_index)

    def _read_task_index(self) -> int:
        """Return the task row to normalize with when not updating statistics."""
        index = self.active_task
        if self._task_trained[index] or self._last_trained_task < 0:
            return index
        return self._last_trained_task

    def _training_momentum(self, task_index: int) -> float:
        """Return the running-stat update rate for this task's next batch."""
        if self.momentum is not None:
            return float(self.momentum)
        return 1.0 / float(self.task_num_batches_tracked[task_index].item())

    # ------------------------------------------------------------------
    def forward(self, input: Tensor) -> Tensor:
        """Normalize ``input`` with the active task's statistics.

        Training forwards update the active task's running statistics, unless
        :attr:`freeze_running_stats` is set (replay passes), in which case the
        batch is normalized with its own statistics and nothing is written.
        Evaluation forwards read the active task's statistics without writing.

        Args:
            input: Batch of shape ``(N, C, L)`` or ``(N, C)``.

        Returns:
            The normalized (and, if affine, rescaled) batch.
        """
        self._check_input_dim(input)

        if self.training and self.freeze_running_stats:
            return F.batch_norm(
                input, None, None, self.weight, self.bias, True, 0.0, self.eps
            )

        if self.training:
            index = self.active_task
            if not self._task_trained[index]:
                self._seed_untrained_task(index)
            self.task_num_batches_tracked[index] += 1
            return F.batch_norm(
                input,
                self.task_running_mean[index],
                self.task_running_var[index],
                self.weight,
                self.bias,
                True,
                self._training_momentum(index),
                self.eps,
            )

        index = self._read_task_index()
        return F.batch_norm(
            input,
            self.task_running_mean[index],
            self.task_running_var[index],
            self.weight,
            self.bias,
            False,
            0.0,
            self.eps,
        )

    # ------------------------------------------------------------------
    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs) -> None:
        """Restore the Python-side mirrors after loading persisted buffers."""
        super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)
        self._task_trained = [bool(flag) for flag in self.task_trained.tolist()]
        self._last_trained_task = int(self.last_trained_task.item())

    def extra_repr(self) -> str:
        """Append the task count to the standard BatchNorm repr."""
        return f"{super().extra_repr()}, num_tasks={self.num_tasks}"


# ----------------------------------------------------------------------
def convert_batchnorm_to_task_specific(
    root: nn.Module, num_tasks: int
) -> List[TaskSpecificBatchNorm1d]:
    """Replace every plain ``nn.BatchNorm1d`` under ``root`` in place.

    The existing ``weight`` and ``bias`` ``Parameter`` objects are **reused** on
    the replacement module, so an optimizer already built over ``root`` keeps
    working and the affine parameters stay shared across tasks.

    ``AdaB1N`` (a ``BatchNorm1d`` subclass) and already-converted layers are left
    alone: only exact ``nn.BatchNorm1d`` instances are swapped.

    Args:
        root: Module tree to convert.
        num_tasks: Number of tasks to allocate statistics for.

    Returns:
        The replacement layers, in traversal order.

    Usage:
        >>> backbone = nn.Sequential(nn.Conv1d(2, 4, 3), nn.BatchNorm1d(4))
        >>> layers = convert_batchnorm_to_task_specific(backbone, num_tasks=3)
        >>> len(layers)
        1
    """
    converted: List[TaskSpecificBatchNorm1d] = []
    for parent in list(root.modules()):
        for name, child in list(parent.named_children()):
            if type(child) is not nn.BatchNorm1d:
                continue
            replacement = _build_replacement(child, num_tasks)
            setattr(parent, name, replacement)
            converted.append(replacement)
    return converted


def _build_replacement(
    source: nn.BatchNorm1d, num_tasks: int
) -> TaskSpecificBatchNorm1d:
    """Build a task-specific layer carrying ``source``'s parameters and stats."""
    replacement = TaskSpecificBatchNorm1d(
        source.num_features,
        num_tasks,
        eps=source.eps,
        momentum=source.momentum,
        affine=source.affine,
    )
    replacement.to(_module_device(source))
    if source.affine:
        replacement.weight = source.weight
        replacement.bias = source.bias
    if source.running_mean is not None:
        replacement.task_running_mean.copy_(source.running_mean.unsqueeze(0))
        replacement.task_running_var.copy_(source.running_var.unsqueeze(0))
    return replacement


def _module_device(module: nn.Module) -> torch.device:
    """Return the device a module's tensors live on, defaulting to CPU."""
    for tensor in list(module.parameters()) + list(module.buffers()):
        return tensor.device
    return torch.device("cpu")


# ----------------------------------------------------------------------
def task_bn_layers(root: nn.Module) -> List[TaskSpecificBatchNorm1d]:
    """Collect every :class:`TaskSpecificBatchNorm1d` layer inside ``root``."""
    return [m for m in root.modules() if isinstance(m, TaskSpecificBatchNorm1d)]


def _layers_of(model: object) -> Sequence[TaskSpecificBatchNorm1d]:
    """Return the cached layer list for ``model``, walking the tree if unset.

    Tolerates non-module callers (test doubles, functional wrappers) by
    reporting no layers rather than raising.
    """
    cached = getattr(model, "_task_bn_layers", None)
    if isinstance(cached, list):
        return cached
    if not isinstance(model, nn.Module):
        return []
    layers = task_bn_layers(model)
    model._task_bn_layers = layers
    return layers


def set_active_task(model: object, task_index: int) -> None:
    """Point every task-specific BatchNorm layer in ``model`` at ``task_index``.

    A no-op when the model has no task-specific layers (``--bn_mode shared``,
    CIL runs, GroupNorm/AdaB1N backbones).

    Args:
        model: Continual-learning module.
        task_index: Zero-based task id to activate.

    Usage:
        >>> set_active_task(model, task_info["task"])  # doctest: +SKIP
    """
    for layer in _layers_of(model):
        layer.set_task(task_index)


def get_active_task(model: object) -> Optional[int]:
    """Return the currently active task id, or ``None`` when not applicable."""
    layers = _layers_of(model)
    if not layers:
        return None
    return layers[0].active_task


@contextmanager
def frozen_running_stats(model: object) -> Iterator[None]:
    """Suspend running-statistic updates for the duration of the block.

    Wrap replay / mixed-task forwards in this so old-task activations are not
    folded into the current task's statistics. In training mode BatchNorm
    normalizes with batch statistics either way, so the outputs and gradients
    inside the block are unchanged -- only the buffer writes are suppressed.

    Args:
        model: Continual-learning module.

    Usage:
        >>> with frozen_running_stats(self):  # doctest: +SKIP
        ...     replay_logits = self.net.forward(replay_x)
    """
    layers = _layers_of(model)
    previous = [layer.freeze_running_stats for layer in layers]
    for layer in layers:
        layer.freeze_running_stats = True
    try:
        yield
    finally:
        for layer, was_frozen in zip(layers, previous):
            layer.freeze_running_stats = was_frozen


# ----------------------------------------------------------------------
EVAL_BN_STATS = frozenset({"batch", "running"})


def _batch_statistics_forward(
    self: nn.modules.batchnorm._BatchNorm, input: Tensor
) -> Tensor:
    """Normalize ``input`` with its own statistics, leaving every buffer untouched."""
    self._check_input_dim(input)
    return F.batch_norm(input, None, None, self.weight, self.bias, True, 0.0, self.eps)


@contextmanager
def batch_statistics(root: object) -> Iterator[None]:
    """Normalize every BatchNorm layer in ``root`` with batch statistics.

    Inside the block each layer (plain ``BatchNorm1d``, :class:`TaskSpecificBatchNorm1d`
    and ``AdaB1N`` alike) ignores its running statistics and its train/eval
    flag: it normalizes with the current batch's mean/variance and writes no
    buffer, so an evaluation pass stays side-effect free.

    Under ``--bn_mode shared`` the running statistics track whichever task was
    trained last, so reading them at evaluation normalizes every earlier task
    with the wrong mean/variance; LwF old-task recall fell from 0.66 to 0.44
    with the same weights. Evaluation loaders are per task, so batch statistics
    are task-conditional, which TIL permits because the task id is given.

    Args:
        root: Module whose BatchNorm layers are switched (non-modules: no-op).

    Usage:
        >>> with batch_statistics(model):  # doctest: +SKIP
        ...     logits = model(x, t)
    """
    if not isinstance(root, nn.Module):
        yield
        return
    layers = [
        m for m in root.modules() if isinstance(m, nn.modules.batchnorm._BatchNorm)
    ]
    previous = [layer.__dict__.get("forward") for layer in layers]
    for layer in layers:
        layer.forward = _batch_statistics_forward.__get__(layer)
    try:
        yield
    finally:
        for layer, forward in zip(layers, previous):
            if forward is None:
                del layer.forward
            else:
                layer.forward = forward


def eval_uses_batch_statistics(args: object) -> bool:
    """Report whether evaluation forwards should normalize with batch statistics.

    True with ``--eval_bn_stats batch`` (the default) and ``--bn_mode shared``,
    for both loaders. ``task_specific`` keeps reading its per-task rows.

    Class-incremental runs were previously excluded, on the grounds that batch
    statistics over a single-task eval batch leak the task id. That reasoning
    holds for the task-incremental sets, which are per task, but not for the
    class-incremental ones: set ``j`` holds every task up to ``j``, shuffled, so
    its batches span tasks and carry no task id to leak. Running statistics
    describe only the most recently trained task, which is what motivated batch
    statistics in the first place, so both loaders now follow the same policy
    (user decision, 2026-09-16). Note this makes evaluation transductive: a
    prediction depends on the other rows sharing its batch.

    Args:
        args: Experiment arguments (``eval_bn_stats``, ``bn_mode``, ``loader``).

    Returns:
        ``True`` when :func:`batch_statistics` should wrap evaluation forwards.

    Raises:
        ValueError: If ``args.eval_bn_stats`` is not one of :data:`EVAL_BN_STATS`.
    """
    mode = str(getattr(args, "eval_bn_stats", "batch")).lower()
    if mode not in EVAL_BN_STATS:
        raise ValueError(
            f"Unsupported eval_bn_stats {getattr(args, 'eval_bn_stats', None)!r}; "
            f"expected one of {sorted(EVAL_BN_STATS)}."
        )
    if mode != "batch":
        return False
    return str(getattr(args, "bn_mode", "shared")).lower() == "shared"


def task_bn_enabled(args: object) -> bool:
    """Report whether this run should use task-specific BatchNorm statistics.

    Enabled for ``--bn_mode task_specific`` on a ``task_incremental_loader`` run
    whose backbone actually uses ``nn.BatchNorm1d``.

    Args:
        args: Experiment arguments (``bn_mode``, ``loader``, ``model``,
            ``norm_type``, ``use_groupnorm``).

    Returns:
        ``True`` when the conversion should run.

    Usage:
        >>> task_bn_enabled(args)  # doctest: +SKIP
        True
    """
    if str(getattr(args, "bn_mode", "shared")).lower() != "task_specific":
        return False
    if str(getattr(args, "loader", "")) != "task_incremental_loader":
        return False
    if str(getattr(args, "model", "")).lower() in EXCLUDED_MODELS:
        return False

    norm_type = str(getattr(args, "norm_type", "batchnorm")).lower()
    if bool(getattr(args, "use_groupnorm", False)) or norm_type in _GROUPNORM_ALIASES:
        return False
    return norm_type not in _ADAB1N_ALIASES


def install(
    model: nn.Module, args: object, num_tasks: int
) -> List[TaskSpecificBatchNorm1d]:
    """Convert ``model``'s BatchNorm layers for TIL, when the run calls for it.

    Call once, immediately after the model is constructed and before it is moved
    to the GPU. Caches the converted layers on ``model._task_bn_layers`` for
    :func:`set_active_task` and :func:`frozen_running_stats`.

    Args:
        model: Freshly constructed continual-learning module.
        args: Experiment arguments; ``bn_mode`` is validated here.
        num_tasks: Number of tasks in the run.

    Returns:
        The converted layers (empty when task-specific BatchNorm is off).

    Raises:
        ValueError: If ``args.bn_mode`` is not one of :data:`BN_MODES`.

    Usage:
        >>> install(model, args, n_tasks)  # doctest: +SKIP
    """
    bn_mode = str(getattr(args, "bn_mode", "shared")).lower()
    if bn_mode not in BN_MODES:
        raise ValueError(
            f"Unsupported bn_mode {getattr(args, 'bn_mode', None)!r}; "
            f"expected one of {sorted(BN_MODES)}."
        )

    if not task_bn_enabled(args):
        model._task_bn_layers = []
        return []

    layers = convert_batchnorm_to_task_specific(model, max(1, int(num_tasks)))
    model._task_bn_layers = layers
    if layers:
        print(
            f"Task-specific BatchNorm enabled: {len(layers)} layers "
            f"x {max(1, int(num_tasks))} tasks (running statistics only; "
            "affine parameters shared)."
        )
    else:
        print(
            "bn_mode=task_specific has no effect: backbone has no BatchNorm1d "
            "layers (e.g. norm_type=groupnorm)."
        )
    return layers


__all__ = [
    "BN_MODES",
    "EVAL_BN_STATS",
    "EXCLUDED_MODELS",
    "TaskSpecificBatchNorm1d",
    "batch_statistics",
    "convert_batchnorm_to_task_specific",
    "eval_uses_batch_statistics",
    "frozen_running_stats",
    "get_active_task",
    "install",
    "set_active_task",
    "task_bn_enabled",
    "task_bn_layers",
]
