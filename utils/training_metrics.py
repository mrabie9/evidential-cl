from typing import Union

import os
import numpy as np
import torch
from sklearn.metrics import f1_score, precision_score, recall_score

TensorLike = Union[torch.Tensor, np.ndarray]

_METRICS_DEBUG_ENABLED = os.getenv("LA_MAML_METRICS_DEBUG", "").strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}
_METRICS_DEBUG_MAX_PRINTS = int(os.getenv("LA_MAML_METRICS_DEBUG_MAX_PRINTS", "5"))
_METRICS_DEBUG_PRINT_COUNT = 0
_METRICS_UNIQUE_DENSITY_DEBUG_PRINT_COUNT = 0


def _unique_labels(values: np.ndarray, debug_context: str) -> np.ndarray:
    """Return unique integer class ids for sklearn's `labels=` argument.

    Args:
        values: Array of class ids (any numeric dtype).
        debug_context: Short label describing the callsite/metric.

    Returns:
        1D array of unique class ids cast to `np.int64`.

    Usage:
        labels = _unique_labels(targets_np, "macro_f1")
    """
    if values.size == 0:
        return np.asarray([], dtype=np.int64)

    global _METRICS_UNIQUE_DENSITY_DEBUG_PRINT_COUNT
    coerced = _coerce_to_class_ids(values)
    # sklearn expects `labels` to be list-like of class ids.
    unique_labels = np.unique(coerced).astype(np.int64, copy=False)
    unique_ratio = (
        float(unique_labels.size) / float(coerced.size) if coerced.size else 0.0
    )

    # sklearn emits a warning when it suspects regression-like targets:
    # "The number of unique classes is greater than 50%..."
    # We surface the actual batch stats here so we can verify the cause.
    # sklearn warns when it suspects that `n_unique > 0.5 * n_samples`.
    # We also check for very small sample sizes since that can make the
    # heuristic fire even for "normal" label distributions.
    should_print = (unique_ratio > 0.5) or (coerced.size <= 50)
    if (
        _METRICS_DEBUG_ENABLED
        and coerced.size > 0
        and should_print
        and _METRICS_UNIQUE_DENSITY_DEBUG_PRINT_COUNT < _METRICS_DEBUG_MAX_PRINTS
    ):
        sample_unique = unique_labels[: min(12, unique_labels.size)]
        ratio = unique_ratio
        print(
            "[metrics-debug:unique-density] %s n_samples=%d n_unique=%d "
            "unique_ratio=%.3f values_dtype=%s coerced_dtype=%s sample_unique=%s"
            % (
                debug_context,
                coerced.size,
                unique_labels.size,
                ratio,
                values.dtype,
                coerced.dtype,
                sample_unique,
            )
        )
        _METRICS_UNIQUE_DENSITY_DEBUG_PRINT_COUNT += 1

    return unique_labels


def _coerce_to_class_ids(values: np.ndarray) -> np.ndarray:
    """Coerce prediction/target arrays into integer class ids.

    Some dataloaders/tests provide labels as float arrays even when they
    represent discrete class IDs. sklearn can emit warnings when it sees
    float targets; converting to integer ids avoids that.
    """
    if values.size == 0:
        return values.astype(np.int64)

    global _METRICS_DEBUG_PRINT_COUNT
    if (
        _METRICS_DEBUG_ENABLED
        and _METRICS_DEBUG_PRINT_COUNT < _METRICS_DEBUG_MAX_PRINTS
    ):
        # Print only a tiny sample to avoid flooding logs.
        flat = values.reshape(-1)
        sample = flat[: min(8, flat.size)]
        print(
            "[metrics-debug:before] dtype=%s shape=%s sample=%s"
            % (values.dtype, values.shape, sample)
        )
        _METRICS_DEBUG_PRINT_COUNT += 1

    if np.issubdtype(values.dtype, np.floating):
        # Labels should be integral; round to the nearest integer.
        coerced = np.rint(values).astype(np.int64)
        if (
            _METRICS_DEBUG_ENABLED
            and _METRICS_DEBUG_PRINT_COUNT <= _METRICS_DEBUG_MAX_PRINTS
        ):
            print(
                "[metrics-debug:after] dtype=%s shape=%s sample=%s"
                % (
                    coerced.dtype,
                    coerced.shape,
                    coerced.reshape(-1)[: min(8, coerced.size)],
                )
            )
        return coerced

    if not np.issubdtype(values.dtype, np.integer):
        coerced = values.astype(np.int64)
        if (
            _METRICS_DEBUG_ENABLED
            and _METRICS_DEBUG_PRINT_COUNT <= _METRICS_DEBUG_MAX_PRINTS
        ):
            print(
                "[metrics-debug:after] dtype=%s shape=%s sample=%s"
                % (
                    coerced.dtype,
                    coerced.shape,
                    coerced.reshape(-1)[: min(8, coerced.size)],
                )
            )
        return coerced

    return values


def _to_numpy(preds: TensorLike, targets: TensorLike) -> tuple[np.ndarray, np.ndarray]:
    """Convert preds and targets to numpy arrays."""
    preds_np = (
        preds.detach().cpu().numpy()
        if isinstance(preds, torch.Tensor)
        else np.asarray(preds)
    )
    targets_np = (
        targets.detach().cpu().numpy()
        if isinstance(targets, torch.Tensor)
        else np.asarray(targets)
    )
    return preds_np, targets_np


def macro_precision(preds: TensorLike, targets: TensorLike) -> float:
    """Macro-averaged precision over the signal classes present in the batch.

    Args:
        preds: Predicted class ids.
        targets: Ground-truth class ids.

    Returns:
        Macro precision in ``[0, 1]``; ``0.0`` for an empty batch.

    Usage:
        precision = macro_precision(predictions, labels)
    """
    preds_np, targets_np = _to_numpy(preds, targets)
    if preds_np.size == 0 or targets_np.size == 0:
        return 0.0
    preds_np = _coerce_to_class_ids(preds_np)
    targets_np = _coerce_to_class_ids(targets_np)

    unique_labels = _unique_labels(targets_np, "macro_precision")
    return precision_score(
        targets_np,
        preds_np,
        labels=unique_labels,
        average="macro",
        zero_division=0,
    )


def macro_f1(preds: TensorLike, targets: TensorLike) -> float:
    """Macro-averaged F1 over the signal classes present in the batch.

    Args:
        preds: Predicted class ids.
        targets: Ground-truth class ids.

    Returns:
        Macro F1 in ``[0, 1]``; ``0.0`` for an empty batch.

    Usage:
        f1 = macro_f1(predictions, labels)
    """
    preds_np, targets_np = _to_numpy(preds, targets)
    if preds_np.size == 0 or targets_np.size == 0:
        return 0.0
    preds_np = _coerce_to_class_ids(preds_np)
    targets_np = _coerce_to_class_ids(targets_np)
    unique_labels = _unique_labels(targets_np, "macro_f1")
    return f1_score(
        targets_np, preds_np, labels=unique_labels, average="macro", zero_division=0
    )


def macro_recall(preds: TensorLike, targets: TensorLike) -> float:
    """Macro-averaged recall over the signal classes present in the batch.

    Args:
        preds: Predicted class ids.
        targets: Ground-truth class ids.

    Returns:
        Macro recall in ``[0, 1]``; ``0.0`` for an empty batch.

    Usage:
        recall = macro_recall(predictions, labels)
    """
    if isinstance(preds, torch.Tensor):
        preds_np = preds.detach().cpu().numpy()
    else:
        preds_np = np.asarray(preds)

    if isinstance(targets, torch.Tensor):
        targets_np = targets.detach().cpu().numpy()
    else:
        targets_np = np.asarray(targets)

    if preds_np.size == 0 or targets_np.size == 0:
        return 0.0
    preds_np = _coerce_to_class_ids(preds_np)
    targets_np = _coerce_to_class_ids(targets_np)

    unique_labels = _unique_labels(targets_np, "macro_recall")
    return recall_score(
        targets_np, preds_np, labels=unique_labels, average="macro", zero_division=0
    )
