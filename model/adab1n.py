"""Task-aware adaptive BatchNorm1d, ported from AdaB2N for 1D signal backbones.

AdaB2N (https://github.com/wsmr11/AdaB2N, ``models/utils/adab2n.py``) is a
``BatchNorm2d`` drop-in for image continual-learning backbones: when a single
forward pass mixes current-task rows with a multi-task replay buffer, it
reweights each row's contribution to the batch mean/variance by a learned
per-task concentration, instead of letting whichever task dominates the batch
skew the pooled statistics. :class:`AdaB1N` is the same mechanism ported to
``BatchNorm1d`` (reducing over the length dimension instead of height/width)
for :mod:`model.resnet1d`, the RF/IQ backbone used by most models in this
repository.

Unlike upstream AdaB2N, which is constructed by wrapping an existing
``BatchNorm2d`` module (``AdaB2N(target=existing_bn, ...)``) via Mammoth's
``ContinualModel.replace_bn``, :class:`AdaB1N` is constructed directly from
``num_features`` so it can plug into :meth:`model.resnet1d.ResNet1D._build_norm_factory`
the same way ``nn.BatchNorm1d`` and ``nn.GroupNorm`` already do.

Per-task reweighting only activates once a caller opts in via
:meth:`AdaB1N.set_counts` before each forward pass and :meth:`AdaB1N.end_task`
at task boundaries (mirroring ``models/derpp_ada.py`` in the AdaB2N repo). No
model in this codebase currently forwards a single mixed current+replay batch
through the RF backbone (they intentionally split current/replay into
separate forward passes to keep BatchNorm statistics unmixed -- see
``model/lamaml_cifar.py``'s ``meta_loss`` docstring), so by default
(``sample_task_indices`` unset) :class:`AdaB1N` behaves like a plain
``BatchNorm1d`` with a kappa-scheduled running-stat momentum.

Usage:
    >>> import torch
    >>> norm = AdaB1N(num_features=16, num_tasks=3, kappa=1.0)
    >>> x = torch.randn(8, 16, 32)
    >>> out = norm(x)
    >>> out.shape
    torch.Size([8, 16, 32])
"""

from __future__ import annotations

from typing import Iterable, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn import BatchNorm1d, Parameter


class AdaB1N(BatchNorm1d):
    """Adaptive, task-reweighted BatchNorm1d for RF/IQ continual-learning backbones.

    Args:
        num_features: Number of channels ``C`` in the expected ``(N, C, L)`` input.
        num_tasks: Upper bound on the number of tasks this layer will see. The
            per-task concentration parameter is sized to this value, so
            :meth:`end_task` raises once more than ``num_tasks`` task
            boundaries have been crossed.
        num_classes_per_task: Optional bookkeeping value carried over from
            AdaB2N for caller convenience; unused internally.
        kappa: Interpolates the running-stat update rule between a cumulative
            average (``kappa=0``) and the fixed-momentum EMA used by ordinary
            BatchNorm (``kappa=1``). Must lie in ``[0, 1]``.
        init_weight: Initial value of the per-task concentration logits.
        eps: Passed through to ``nn.BatchNorm1d``.
        momentum: Passed through to ``nn.BatchNorm1d``; must not be ``None``.
        affine: Passed through to ``nn.BatchNorm1d``.
        track_running_stats: Passed through to ``nn.BatchNorm1d``; must be ``True``.

    Usage:
        >>> norm = AdaB1N(num_features=8, num_tasks=2)
        >>> norm.train()
        >>> x = torch.randn(4, 8, 20)
        >>> norm(x).shape
        torch.Size([4, 8, 20])
    """

    def __init__(
        self,
        num_features: int,
        num_tasks: int = 1,
        num_classes_per_task: Optional[int] = None,
        kappa: float = 1.0,
        init_weight: float = 0.0,
        eps: float = 1e-5,
        momentum: Optional[float] = 0.1,
        affine: bool = True,
        track_running_stats: bool = True,
    ) -> None:
        super().__init__(num_features, eps, momentum, affine, track_running_stats)
        assert self.momentum is not None
        assert self.track_running_stats
        assert 0.0 <= kappa <= 1.0
        assert num_tasks >= 1

        self.kappa = kappa
        self.last_eta: Optional[float] = None
        self.num_tasks = num_tasks
        self.num_classes_per_task = num_classes_per_task
        self.init_weight = init_weight
        self.register_buffer("cur_tasks", torch.tensor(0, dtype=torch.long))
        self.cur_tasks: Optional[Tensor]

        self.task_weight = Parameter(torch.full((num_tasks,), fill_value=init_weight))

        self.sample_task_indices: Optional[Tensor] = None
        self.sample_task_counts: Optional[Tensor] = None
        self.task_counts_extended: Optional[Tensor] = None
        self.loss: Tensor = torch.tensor(0.0)

    def set_counts(
        self,
        sample_task_indices: Tensor,
        sample_task_counts: Tensor,
        task_counts_extended: Tensor,
    ) -> None:
        """Register per-sample task metadata for the next forward pass.

        Args:
            sample_task_indices: Task id for each row in the upcoming batch.
            sample_task_counts: Count of rows sharing each row's task id.
            task_counts_extended: Per-task row counts, indexed by task id.

        Usage:
            >>> norm = AdaB1N(num_features=4, num_tasks=2)
            >>> norm.set_counts(
            ...     torch.tensor([0, 0, 1]),
            ...     torch.tensor([2, 2, 1]),
            ...     torch.tensor([2, 1]),
            ... )
        """
        self.sample_task_indices = sample_task_indices
        self.sample_task_counts = sample_task_counts
        self.task_counts_extended = task_counts_extended

    def end_task(self) -> None:
        """Advance the task counter at a task boundary.

        Usage:
            >>> norm = AdaB1N(num_features=4, num_tasks=2)
            >>> norm.end_task()
        """
        next_task_count = int(self.cur_tasks.item()) + 1
        if next_task_count >= self.num_tasks:
            raise ValueError(
                f"AdaB1N configured with num_tasks={self.num_tasks} cannot "
                f"advance past task index {next_task_count}; construct it "
                "with a larger num_tasks (e.g. via args.n_tasks)."
            )
        self.cur_tasks.add_(1)

    def get_eta(self) -> float:
        """Compute the running-stat update rate for the current batch.

        Usage:
            >>> norm = AdaB1N(num_features=4, kappa=1.0)
            >>> isinstance(norm.get_eta(), float)
            True
        """
        if self.kappa == 0.0:
            return 1.0 / float(self.num_batches_tracked)
        if self.kappa == 1.0:
            return self.momentum
        if self.last_eta is None:
            self.last_eta = self.momentum**self.kappa
            return self.last_eta
        eta = self.last_eta / (self.last_eta + (1 - self.momentum) ** self.kappa)
        self.last_eta = eta
        return eta

    def training_forward(self, input: Tensor) -> Tensor:
        """Normalize a training-mode batch, optionally task-reweighted.

        Args:
            input: Batch of shape ``(N, C, L)``.

        Returns:
            The normalized (and, if affine, rescaled) batch.
        """
        self.num_batches_tracked.add_(1)
        eta = self.get_eta()

        if self.sample_task_indices is not None and self.cur_tasks > 0:
            concentration = (
                self.task_weight[: self.cur_tasks + 1].exp() + self.task_counts_extended
            )
            # Only tasks with rows in this batch may hold weight: otherwise the
            # absent tasks' share is lost and the weighted mean shrinks toward
            # zero, worsening with every task (44% of true mean by task 9).
            concentration = concentration * (self.task_counts_extended > 0)
            task_weights = concentration / concentration.sum()
            sample_weights = (
                task_weights[self.sample_task_indices] / self.sample_task_counts
            )

            batch_mean = input.mean(2).t().matmul(sample_weights).view(1, -1, 1)
            batch_var = (
                (input - batch_mean)
                .square()
                .mean(2)
                .t()
                .matmul(sample_weights)
                .view(1, -1, 1)
            )
        else:
            batch_var, batch_mean = torch.var_mean(
                input, [0, 2], correction=0, keepdim=True
            )

        output = (input - batch_mean) / torch.sqrt(batch_var + self.eps)
        if self.affine:
            output = self.weight.view(1, -1, 1) * output + self.bias.view(1, -1, 1)

        self.running_mean.add_(
            batch_mean.detach().squeeze() - self.running_mean, alpha=eta
        )
        self.running_var.mul_(1 - eta).add_(batch_var.detach().squeeze(), alpha=eta)
        self.loss = F.mse_loss(batch_mean.squeeze(), self.running_mean) + F.mse_loss(
            batch_var.squeeze(), self.running_var
        )

        return output

    def eval_forward(self, input: Tensor) -> Tensor:
        """Normalize an eval-mode batch using the tracked running statistics."""
        return F.batch_norm(
            input,
            self.running_mean,
            self.running_var,
            self.weight,
            self.bias,
            False,
            0.0,
            self.eps,
        )

    def forward(self, input: Tensor) -> Tensor:
        """Dispatch to training- or eval-mode normalization.

        Usage:
            >>> norm = AdaB1N(num_features=4, num_tasks=1)
            >>> x = torch.randn(3, 4, 10)
            >>> norm(x).shape
            torch.Size([3, 4, 10])
        """
        self._check_input_dim(input)
        if self.training:
            return self.training_forward(input)
        return self.eval_forward(input)


def adab1n_layers(root: nn.Module) -> List[AdaB1N]:
    """Collect every :class:`AdaB1N` layer inside ``root``."""
    return [m for m in root.modules() if isinstance(m, AdaB1N)]


def set_batch_task_counts(
    layers: Iterable[AdaB1N], sample_task_indices: Tensor
) -> bool:
    """Broadcast per-row task metadata for the next forward to ``layers``.

    Derives ``sample_task_counts`` and ``task_counts_extended`` from
    ``sample_task_indices`` and hands them to each layer's
    :meth:`AdaB1N.set_counts`, sized to that layer's ``cur_tasks``. Callers
    holding a backbone should cache :func:`adab1n_layers` once rather than
    re-walking the module tree on every step.

    Args:
        layers: AdaB1N layers to configure; an empty iterable is a no-op.
        sample_task_indices: Task id for each row of the upcoming batch.

    Returns:
        ``True`` when at least one layer was configured, else ``False`` (so
        callers can skip the rest of their AdaB1N-specific bookkeeping).

    Raises:
        ValueError: If a row's task id exceeds a layer's ``cur_tasks``, which
            means :func:`end_task_all` was not called at a task boundary.

    Usage:
        >>> layer = AdaB1N(num_features=4, num_tasks=3)
        >>> set_batch_task_counts([layer], torch.tensor([0, 0, 0]))
        True
    """
    layers = list(layers)
    if not layers:
        return False

    indices = sample_task_indices.reshape(-1).long()
    highest = int(indices.max().item()) if indices.numel() else 0
    for layer in layers:
        span = int(layer.cur_tasks.item()) + 1
        if highest >= span:
            raise ValueError(
                f"batch contains task id {highest} but AdaB1N has seen "
                f"cur_tasks={span - 1}; call end_task_all() at task boundaries."
            )
        counts = torch.bincount(indices, minlength=span).to(
            device=indices.device, dtype=layer.task_weight.dtype
        )
        layer.set_counts(indices, counts[indices], counts)
    return True


def clear_batch_task_counts(layers: Iterable[AdaB1N]) -> None:
    """Drop per-row task metadata so subsequent forwards use unweighted stats."""
    for layer in layers:
        layer.set_counts(None, None, None)


def end_task_all(layers: Iterable[AdaB1N]) -> None:
    """Advance each layer's task counter at a task boundary."""
    for layer in layers:
        layer.end_task()


__all__ = [
    "AdaB1N",
    "adab1n_layers",
    "set_batch_task_counts",
    "clear_batch_task_counts",
    "end_task_all",
]
