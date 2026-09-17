"""Training-time forward helpers shared by ``main`` and continual-learning models."""

from __future__ import annotations

from contextlib import nullcontext
from typing import Any, Tuple

import torch

from model import task_bn


def model_forward_for_metric_loop(
    model: object,
    x: torch.Tensor,
    task_index: int,
    args: object,
    cil_mask_upto_task: int | None = None,
) -> torch.Tensor:
    """Run ``model`` forward for metric computation (validation, test, or train probe).

    For ``class_incremental_loader``, passes ``cil_all_seen_upto_task`` so
    :func:`utils.misc_utils.apply_task_incremental_logit_mask` is applied in
    model code with the **cumulative** class boundary (true CIL inference). For
    task-incremental loaders, no extra keyword is passed (per-task masking
    only).

    ``cil_mask_upto_task`` separates *which* task's data is being scored
    (``task_index``, used for head/BN selection) from *how wide* the CIL logit
    space is. Scoring task ``j`` in isolation after training through task ``i``
    must keep all classes ``0..i`` active, otherwise the classes learned since
    task ``j`` are hidden and the score is a task-incremental one. Defaults to
    ``task_index`` when not given, which is the cumulative-loader case.

    **iCaRL:** ``forward`` is the nearest-mean-of-exemplars classifier; see
    :meth:`model.icarl.Net.forward`.

    **Task-specific BatchNorm:** when the run uses per-task running statistics
    (see :mod:`model.task_bn`), ``task_index`` selects them for the duration of
    the forward and the previously active task is restored afterwards, so a
    mid-epoch validation pass does not leave the training task deselected.

    **Shared BatchNorm (TIL):** with ``--bn_mode shared`` and
    ``--eval_bn_stats batch`` the forward normalizes with batch statistics
    (see :func:`model.task_bn.batch_statistics`), because the shared running
    statistics describe only the most recently trained task.

    Args:
        model: Continual-learning module with ``forward(x, task_index, ...)``.
        x: Input batch on the correct device.
        task_index: Zero-based continual task id (same as ``task_info['task']``).
        args: Experiment arguments (``loader``, ``model`` id).
        cil_mask_upto_task: CIL logit-space bound; defaults to ``task_index``.

    Returns:
        Classifier logits tensor.

    Usage:
        logits = model_forward_for_metric_loop(model, batch_x, task_id, args)
    """
    previous_task = task_bn.get_active_task(model)
    if previous_task is not None:
        task_bn.set_active_task(model, task_index)
    normalization = (
        task_bn.batch_statistics(model)
        if task_bn.eval_uses_batch_statistics(args)
        else nullcontext()
    )
    try:
        with normalization:
            return _dispatch_metric_forward(
                model, x, task_index, args, cil_mask_upto_task=cil_mask_upto_task
            )
    finally:
        if previous_task is not None:
            task_bn.set_active_task(model, previous_task)


def _dispatch_metric_forward(
    model: object,
    x: torch.Tensor,
    task_index: int,
    args: object,
    cil_mask_upto_task: int | None = None,
) -> torch.Tensor:
    """Route a metric-loop forward to the right entry point for this model.

    Args:
        model: Continual-learning module.
        x: Input batch on the correct device.
        task_index: Zero-based continual task id.
        args: Experiment arguments (``loader``, ``model`` id).
        cil_mask_upto_task: CIL logit-space bound; defaults to ``task_index``.

    Returns:
        Classifier logits tensor.
    """
    forward_kw: dict[str, Any] = {}
    if getattr(args, "loader", "") == "class_incremental_loader":
        forward_kw["cil_all_seen_upto_task"] = (
            task_index if cil_mask_upto_task is None else int(cil_mask_upto_task)
        )
    if getattr(args, "model", "") == "anml":
        return model(x, fast_weights=None)  # type: ignore[operator]
    try:
        return model(x, task_index, **forward_kw)  # type: ignore[operator]
    except TypeError:
        return model(x, task_index)  # type: ignore[operator]


def unpack_observe_result(
    result: Tuple[float, float] | Tuple[float, float, torch.Tensor | None],
) -> Tuple[float, float, torch.Tensor | None]:
    """Unpack ``observe()`` return value (2- or 3-tuple).

    Args:
        result: ``(loss, cls_tr_rec)`` or ``(loss, cls_tr_rec, metric_logits)``.

    Returns:
        Loss scalar, training recall scalar, and optional detached logits for
        progress-bar metrics (``None`` when the model did not provide them).

    Usage:
        loss, rec, logits = unpack_observe_result(model.observe(x, y, t))
    """
    if len(result) == 3:
        loss, cls_tr_rec, metric_logits = result
        return float(loss), float(cls_tr_rec), metric_logits
    loss, cls_tr_rec = result
    return float(loss), float(cls_tr_rec), None
