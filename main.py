# TODO: hyperparameter tuner

import importlib
import datetime
import argparse
import atexit
import json
import math
import time
import os
import sys
from contextlib import nullcontext
from pathlib import Path
from typing import List, Tuple

from tqdm import tqdm

import numpy as np
import torch
from torch.autograd import Variable
from torch.utils.data import ConcatDataset, DataLoader, Dataset

import parser as file_parser
from metrics.metrics import append_metric_block, confusion_matrix
from model import task_bn
from utils import misc_utils
from utils.training_metrics import (
    macro_f1,
    macro_precision,
    macro_recall,
)
from utils.training_forward import (
    model_forward_for_metric_loop,
    unpack_observe_result,
)

# Backward-compatible alias for imports from ``main``.
_model_forward_for_metric_loop = model_forward_for_metric_loop


def log_state(enabled, message):
    """Print a timestamped state message when state logging is enabled."""
    if not enabled:
        return
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print("[STATE {}] {}".format(timestamp, message))


TASK_SPECIFIC_EPOCH_BASELINE = 10
TASK_SPECIFIC_EPOCH_SCHEDULE: dict[int, int] = {
    0: 10,
    1: 10,
    2: 3,
    3: 20,
    4: 20,
    5: 5,
    6: 20,
    7: 5,
    8: 20,
    9: 10,
}
LEGACY_USE_GLOBAL_N_EPOCHS = True


def _task_epoch_schedule_for_base_epochs(base_n_epochs: int) -> dict[int, int]:
    """Return the task-specific epoch schedule for matching baseline runs.

    Args:
        base_n_epochs: The globally configured number of epochs per task.

    Returns:
        The per-task schedule when the run uses the 10-epoch baseline, otherwise
        an empty mapping so other experiment configurations remain unchanged.
    """
    if int(base_n_epochs) != TASK_SPECIFIC_EPOCH_BASELINE:
        return {}
    return dict(TASK_SPECIFIC_EPOCH_SCHEDULE)


_OUTPUT_TEE_INITIALIZED = False
_OUTPUT_TEE_LOG_FILE = None
_OUTPUT_TEE_ORIGINAL_STDOUT = None
_OUTPUT_TEE_ORIGINAL_STDERR = None


class _OutputTee:
    """Mirror writes to terminal and a log file."""

    def __init__(self, terminal: object, log_file: object):
        self._terminal = terminal
        self._log_file = log_file
        self._pending_carriage_line = None

    def write(self, data: str) -> int:
        self._terminal.write(data)

        # Keep terminal behavior unchanged, but compact carriage-return based
        # progress updates in the log file so only the final line is persisted.
        for character in data:
            if character == "\r":
                self._pending_carriage_line = ""
                continue
            if character == "\n":
                if self._pending_carriage_line is not None:
                    self._log_file.write(self._pending_carriage_line)
                    self._pending_carriage_line = None
                self._log_file.write("\n")
                continue

            if self._pending_carriage_line is not None:
                self._pending_carriage_line += character
            else:
                self._log_file.write(character)

        return len(data)

    @property
    def encoding(self):
        return getattr(self._terminal, "encoding", None)

    def flush(self) -> None:
        # Keep terminal responsive and ensure log is up to date.
        self._terminal.flush()
        self._log_file.flush()

    def isatty(self) -> bool:
        if hasattr(self._terminal, "isatty"):
            return bool(self._terminal.isatty())
        return False

    def fileno(self):
        if hasattr(self._terminal, "fileno"):
            return self._terminal.fileno()
        return None


def enable_output_tee(log_file_path: str, append: bool = False) -> None:
    """Enable stdout/stderr mirroring to a log file (without hiding terminal output).

    Args:
        log_file_path: Full path to the log file to append/overwrite.
        append: When ``True``, append to an existing log (used when resuming an
            interrupted experiment) instead of truncating it.
    """
    global _OUTPUT_TEE_INITIALIZED, _OUTPUT_TEE_LOG_FILE
    global _OUTPUT_TEE_ORIGINAL_STDOUT, _OUTPUT_TEE_ORIGINAL_STDERR

    if _OUTPUT_TEE_INITIALIZED:
        return

    _OUTPUT_TEE_ORIGINAL_STDOUT = sys.stdout
    _OUTPUT_TEE_ORIGINAL_STDERR = sys.stderr

    # Use line buffering so the log closely matches what users see in the terminal.
    file_mode = "a" if append else "w"
    _OUTPUT_TEE_LOG_FILE = open(log_file_path, file_mode, encoding="utf-8", buffering=1)

    sys.stdout = _OutputTee(_OUTPUT_TEE_ORIGINAL_STDOUT, _OUTPUT_TEE_LOG_FILE)  # type: ignore[assignment]
    sys.stderr = _OutputTee(_OUTPUT_TEE_ORIGINAL_STDERR, _OUTPUT_TEE_LOG_FILE)  # type: ignore[assignment]

    def _shutdown_output_tee() -> None:
        """Restore stdout/stderr and close the log file on interpreter shutdown."""
        global _OUTPUT_TEE_LOG_FILE, _OUTPUT_TEE_INITIALIZED

        # Restore first so any final writes go to the real streams.
        if _OUTPUT_TEE_ORIGINAL_STDOUT is not None:
            sys.stdout = _OUTPUT_TEE_ORIGINAL_STDOUT
        if _OUTPUT_TEE_ORIGINAL_STDERR is not None:
            sys.stderr = _OUTPUT_TEE_ORIGINAL_STDERR

        if _OUTPUT_TEE_LOG_FILE is not None:
            try:
                _OUTPUT_TEE_LOG_FILE.flush()
            finally:
                _OUTPUT_TEE_LOG_FILE.close()
                _OUTPUT_TEE_LOG_FILE = None

        _OUTPUT_TEE_INITIALIZED = False

    atexit.register(_shutdown_output_tee)
    _OUTPUT_TEE_INITIALIZED = True


def _split_eval_output(output):
    """Return ``(macro_rec, macro_prec, macro_f1)``; missing values are ``None``.

    Args:
        output: Return value of :func:`eval_tasks` / :func:`eval_class_tasks`,
            or a bare per-task recall sequence.

    Returns:
        Three-tuple of per-task metric sequences.

    Usage:
        rec, prec, f1 = _split_eval_output(eval_tasks(model, tasks, args))
    """
    if isinstance(output, (tuple, list)) and len(output) == 3:
        return output[0], output[1], output[2]
    return output, None, None


def _scalar_metric_at_task_index(metrics: object, task_index: int) -> float:
    """Return the validation metric for one task index from evaluator output.

    Args:
        metrics: Per-task list/tuple from ``eval_tasks`` / ``eval_class_tasks``, or scalar.
        task_index: Zero-based continual task id.

    Returns:
        Metric as float, or NaN if missing or out of range.
    """
    if metrics is None:
        return float("nan")
    if isinstance(metrics, (list, tuple)):
        if task_index < 0 or task_index >= len(metrics):
            return float("nan")
        value = metrics[task_index]
        if torch.is_tensor(value):
            return float(value.detach().cpu().item())
        return float(value)
    if torch.is_tensor(metrics):
        return float(metrics.detach().cpu().item())
    return float(metrics)


def _mean_metric_across_tasks(metrics: object) -> float:
    """Mean of a per-task metric sequence (e.g. macro F1 averaged over tasks)."""
    if metrics is None:
        return float("nan")
    if isinstance(metrics, (list, tuple)):
        if not metrics:
            return float("nan")
        floats: List[float] = []
        for value in metrics:
            if torch.is_tensor(value):
                floats.append(float(value.detach().cpu().item()))
            else:
                floats.append(float(value))
        return float(sum(floats) / len(floats))
    if torch.is_tensor(metrics):
        return float(metrics.detach().cpu().item())
    return float(metrics)


def _per_task_metric_array(metrics: object, num_tasks: int) -> np.ndarray:
    """Build a fixed-length per-task metric vector for zero-shot NPZ storage.

    Args:
        metrics: Per-task list from an evaluator, or ``None``.
        num_tasks: Number of tasks seen so far (length of ``test_task_loaders``).

    Returns:
        ``float`` array of shape ``(num_tasks,)`` with NaNs for missing tasks/metrics.
    """
    row = np.full((num_tasks,), np.nan, dtype=float)
    if metrics is None or num_tasks <= 0:
        return row
    if isinstance(metrics, (list, tuple)):
        for task_index, value in enumerate(metrics):
            if task_index >= num_tasks:
                break
            if torch.is_tensor(value):
                row[task_index] = float(value.detach().cpu().item())
            else:
                row[task_index] = float(value)
    return row


def _labels_to_numpy(labels: object) -> np.ndarray:
    """Return labels as a NumPy array regardless of source container type."""
    if torch.is_tensor(labels):
        return labels.detach().cpu().numpy()
    return np.asarray(labels)


def _extract_task_labels(task: object) -> np.ndarray | None:
    """Extract raw labels for a task from tuple tasks or loader-backed datasets."""
    if isinstance(task, (list, tuple)) and len(task) == 3:
        return _labels_to_numpy(task[2])

    dataset = getattr(task, "dataset", None)
    if dataset is None:
        return None

    for attribute_name in ("targets", "labels", "y", "ys"):
        if hasattr(dataset, attribute_name):
            return _labels_to_numpy(getattr(dataset, attribute_name))

    dataset_tensors = getattr(dataset, "tensors", None)
    if isinstance(dataset_tensors, (list, tuple)) and len(dataset_tensors) >= 2:
        return _labels_to_numpy(dataset_tensors[1])

    return None


def _infer_class_counts_from_tasks(tasks: List[object]) -> List[int] | None:
    """Infer per-task class counts directly from task labels."""
    inferred_counts: List[int] = []
    for task in tasks:
        task_labels = _extract_task_labels(task)
        if task_labels is None:
            return None
        y_cls_array = np.asarray(task_labels).reshape(-1)
        inferred_counts.append(int(np.unique(y_cls_array).size))
    return inferred_counts


def _maybe_print_eval_prediction_debug(
    task_index: int,
    all_predictions: List[torch.Tensor],
    all_targets: List[torch.Tensor],
) -> None:
    """Print a compact eval prediction summary when debug mode is enabled.

    Args:
        task_index: Zero-based task id used in logging.
        all_predictions: Predicted class-id tensors accumulated across batches.
        all_targets: Ground-truth class-id tensors accumulated across batches.

    Usage:
        _maybe_print_eval_prediction_debug(task_index, preds, targets)
    """
    debug_enabled = os.getenv("LA_MAML_EVAL_DEBUG", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    if not debug_enabled or not all_predictions or not all_targets:
        return

    predictions = torch.cat(all_predictions).detach().cpu().long()
    targets = torch.cat(all_targets).detach().cpu().long()
    if predictions.numel() == 0 or targets.numel() == 0:
        return

    class_ids = torch.unique(targets).tolist()
    max_class_id = int(max(class_ids)) if class_ids else 0
    prediction_histogram = torch.bincount(predictions, minlength=max_class_id + 1)
    target_histogram = torch.bincount(targets, minlength=max_class_id + 1)

    per_class_recall_parts: List[str] = []
    for class_id in class_ids:
        class_mask = targets == int(class_id)
        class_total = int(class_mask.sum().item())
        class_correct = int((predictions[class_mask] == int(class_id)).sum().item())
        class_recall = (
            float(class_correct / class_total) if class_total > 0 else float("nan")
        )
        per_class_recall_parts.append(f"{int(class_id)}:{class_recall:.3f}")

    print(
        "[eval-debug] task={} pred_hist={} target_hist={} per_class_recall={}".format(
            task_index,
            prediction_histogram.tolist(),
            target_histogram.tolist(),
            ",".join(per_class_recall_parts),
        )
    )


def _maybe_print_train_metric_debug(
    task_index: int,
    epoch_index: int,
    batch_index: int,
    observe_cls_recall: float,
    metric_cls_recall: float,
    metric_precision: float,
    metric_f1: float,
    predictions: torch.Tensor,
    labels_for_metrics: torch.Tensor,
) -> None:
    """Print per-batch train metric tensors when debug mode is enabled.

    Args:
        task_index: Zero-based task id.
        epoch_index: Zero-based epoch index.
        batch_index: Zero-based batch index.
        observe_cls_recall: Recall returned by ``model.observe``.
        metric_cls_recall: Recall recomputed in training loop.
        metric_precision: Macro precision recomputed in training loop.
        metric_f1: Macro F1 recomputed in training loop.
        predictions: Argmax class predictions used for train metrics.
        labels_for_metrics: Class labels used for train metrics (already task-local).

    Usage:
        _maybe_print_train_metric_debug(..., pb, y_cls_for_metric)
    """
    debug_enabled = os.getenv("LA_MAML_TRAIN_DEBUG", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    if not debug_enabled:
        return

    debug_every = max(int(os.getenv("LA_MAML_TRAIN_DEBUG_EVERY", "50")), 1)
    if (batch_index + 1) % debug_every != 0:
        return

    predictions_cpu = predictions.detach().cpu().long()
    labels_cpu = labels_for_metrics.detach().cpu().long()
    if predictions_cpu.numel() == 0 or labels_cpu.numel() == 0:
        return

    max_class_id = int(
        max(
            int(predictions_cpu.max().item()),
            int(labels_cpu.max().item()),
        )
    )
    prediction_histogram = torch.bincount(predictions_cpu, minlength=max_class_id + 1)
    label_histogram = torch.bincount(labels_cpu, minlength=max_class_id + 1)

    print(
        "[train-debug] task={} ep={} batch={} observe_rec={:.4f} metric_rec={:.4f} prec={:.4f} f1={:.4f} n={} uniq_pred={} uniq_y={} pred_hist={} y_hist={}".format(
            task_index,
            epoch_index + 1,
            batch_index + 1,
            float(observe_cls_recall),
            float(metric_cls_recall),
            float(metric_precision),
            float(metric_f1),
            int(labels_cpu.numel()),
            predictions_cpu.unique(sorted=True).tolist(),
            labels_cpu.unique(sorted=True).tolist(),
            prediction_histogram.tolist(),
            label_histogram.tolist(),
        )
    )


def _evaluate_one_loader(
    model, loader, task_index, args, class_counts, cil_mask_upto_task=None
):
    """Score one dataloader, returning ``(macro_rec, macro_prec, macro_f1)``.

    Metrics are computed per batch and averaged over batches, which is the
    convention the whole repo's numbers are on; changing it would invalidate
    every stored result.

    Runs under ``torch.no_grad()``: the metric loop never backpropagates, but it
    used to build an autograd graph for every evaluation batch anyway. No model
    needs gradients in ``forward`` (every ``backward``/``autograd.grad`` call
    sits in a training path), so this is numerically identical and only changes
    peak memory and speed.

    Args:
        model: Continual-learning module (already in ``eval()`` mode).
        loader: Dataloader yielding ``(x, y)`` or ``(x, y, t)`` batches.
        task_index: Task id used for head/BN selection and label offsets.
        args: Experiment arguments.
        class_counts: Per-task class counts, or ``None`` to fall back to args.
        cil_mask_upto_task: CIL logit-space bound; defaults to ``task_index``.

    Returns:
        Tuple of three floats.
    """
    device = torch.device(
        "cuda" if getattr(args, "cuda", False) and torch.cuda.is_available() else "cpu"
    )
    recalls = []
    precisions = []
    f1s = []
    eval_debug_predictions: List[torch.Tensor] = []
    eval_debug_targets: List[torch.Tensor] = []
    with torch.no_grad():
        for batch in loader:
            if isinstance(batch, (list, tuple)) and len(batch) == 3:
                xb, yb, _ = batch
            else:
                xb, yb = batch
            xb = xb.to(device)
            if getattr(args, "arch", "").lower() == "linear":
                xb = xb.view(xb.size(0), -1)
            if not torch.is_tensor(yb):
                yb = torch.as_tensor(yb)

            logits = model_forward_for_metric_loop(
                model, xb, task_index, args, cil_mask_upto_task=cil_mask_upto_task
            )
            pb = torch.argmax(logits, dim=1).cpu()
            yb_cls_for_metrics = yb.detach().cpu()
            # Task-incremental learners (UCL and any model exposing ``split``)
            # emit task-local logits; shift global labels the same way as the
            # training metric loop in ``life_experience``
            # (``model.compute_offsets``).
            if getattr(model, "split", False):
                compute_offsets_fn = getattr(model, "compute_offsets", None)
                if callable(compute_offsets_fn):
                    offset1, _ = compute_offsets_fn(task_index)
                else:
                    offset1, _ = misc_utils.compute_offsets(
                        task_index,
                        class_counts if class_counts is not None else args.nc_per_task,
                    )
                yb_cls_for_metrics = yb_cls_for_metrics - offset1

            eval_debug_predictions.append(pb)
            eval_debug_targets.append(yb_cls_for_metrics)

            recalls.append(macro_recall(pb, yb_cls_for_metrics))
            precisions.append(macro_precision(pb, yb_cls_for_metrics))
            f1s.append(macro_f1(pb, yb_cls_for_metrics))

    _maybe_print_eval_prediction_debug(
        task_index=task_index,
        all_predictions=eval_debug_predictions,
        all_targets=eval_debug_targets,
    )

    return (
        sum(recalls) / len(recalls) if recalls else 0.0,
        sum(precisions) / len(precisions) if precisions else 0.0,
        sum(f1s) / len(f1s) if f1s else 0.0,
    )


def eval_tasks(
    model,
    tasks,
    args,
    specific_task=None,
    eval_epistemic=False,
    cil_mask_upto_task=None,
):
    """Evaluate per-task macro recall, precision and F1 over signal classes.

    Args:
        model: Continual-learning module to evaluate.
        tasks: Sequence of per-task dataloaders or ``(x, y, t)`` tuples.
        args: Experiment arguments (``cuda``, ``arch``, ``loader``, ...).
        specific_task: Evaluate only this task id when not ``None``.
        eval_epistemic: Accepted for call-site compatibility; unused.
        cil_mask_upto_task: CIL logit-space bound shared by every entry; see
            :func:`utils.training_forward.model_forward_for_metric_loop`.
            ``None`` keeps each entry masked to its own index.

    Returns:
        Tuple of three per-task lists: macro recall, macro precision, macro F1.

    Usage:
        rec, prec, f1 = eval_tasks(model, test_task_loaders, args)
    """
    model.eval()
    results = []
    prec_results = []
    f1_results = []
    class_counts = _infer_class_counts_from_tasks(tasks)
    if class_counts is None:
        class_counts = getattr(args, "classes_per_task", None)

    # ``specific_task`` selects a single task, but the model must still be
    # queried with that task's *true* id (head selection / label offsets),
    # not the position 0 it now occupies in the trimmed list.
    if specific_task is not None:
        task_ids = [int(specific_task)]
        tasks = [tasks[specific_task]]
    else:
        task_ids = list(range(len(tasks)))

    for task_position, task in enumerate(tasks):
        rec, prec, f1 = _evaluate_one_loader(
            model,
            task,
            task_ids[task_position],
            args,
            class_counts,
            cil_mask_upto_task=cil_mask_upto_task,
        )
        results.append(rec)
        prec_results.append(prec)
        f1_results.append(f1)

    return results, prec_results, f1_results


def eval_class_tasks(model, tasks, args, **kwargs):
    """Evaluate class-incremental runs with the same metrics as :func:`eval_tasks`.

    Args:
        model: Continual-learning module to evaluate.
        tasks: Sequence of per-task dataloaders.
        args: Experiment arguments.
        **kwargs: ``specific_task`` / ``eval_epistemic`` / ``cil_mask_upto_task``
            passthrough.

    Returns:
        Tuple of three per-task lists: macro recall, macro precision, macro F1.

    Usage:
        rec, prec, f1 = eval_class_tasks(model, test_task_loaders, args)
    """

    return eval_tasks(
        model,
        tasks,
        args,
        specific_task=kwargs.get("specific_task"),
        eval_epistemic=kwargs.get("eval_epistemic", False),
        cil_mask_upto_task=kwargs.get("cil_mask_upto_task"),
    )


def _global_label_to_task(class_counts, n_labels):
    """Map each global class label to the task that owns it."""
    counts = [int(c) for c in class_counts] if class_counts is not None else []
    mapping = torch.zeros(
        max(int(n_labels), sum(counts) if counts else 0), dtype=torch.long
    )
    start = 0
    for task_index, count in enumerate(counts):
        stop = min(start + count, mapping.numel())
        if start < stop:
            mapping[start:stop] = task_index
        start += count
    return mapping


def eval_cil_pooled(model, union_loader, current_task, args):
    """One pass over the pooled tasks ``0..current_task`` test set.

    Returns both CIL numbers from the *same* forward passes:

    * the **headline** -- macro rec/prec/F1 over every class seen so far, which
      must be measured on mixed batches because
      :func:`utils.training_metrics.macro_f1` macro-averages over the labels
      present in each batch;
    * the **per-task columns** -- the same predictions sliced by the task that
      owns each sample's true label.

    Scoring the columns from task-pure loaders instead looks equivalent but is
    not, whenever ``--eval_bn_stats batch`` is in play: BatchNorm then
    normalises with the current batch's statistics, so a task-pure batch and a
    mixed batch yield *different predictions for the same sample*. Models whose
    training batches mix tasks (iid2's cumulative replay, and every replay
    method) are then scored under a distribution they never trained on, and the
    columns measure that mismatch on top of forgetting -- for iid2 it dragged
    mean per-task recall from 0.68 to 0.28. Slicing one pooled pass keeps the
    columns and the headline on identical forward passes, and costs one pass
    instead of ``current_task + 2``.

    Args:
        model: Continual-learning module to evaluate.
        union_loader: Shuffled loader over the pooled tasks ``0..current_task``.
        current_task: Task just trained; also the CIL logit-space bound.
        args: Experiment arguments.

    Returns:
        ``((col_rec, col_prec, col_f1), (rec, prec, f1))`` -- per-task lists of
        length ``current_task + 1``, then the three headline floats.
    """
    model.eval()
    device = torch.device(
        "cuda" if getattr(args, "cuda", False) and torch.cuda.is_available() else "cpu"
    )
    task_index = int(current_task)
    class_counts = getattr(args, "classes_per_task", None)

    recalls = []
    precisions = []
    f1s = []
    pooled_preds = []
    pooled_metric_targets = []
    pooled_raw_targets = []
    with torch.no_grad():
        for batch in union_loader:
            if isinstance(batch, (list, tuple)) and len(batch) == 3:
                xb, yb, _ = batch
            else:
                xb, yb = batch
            xb = xb.to(device)
            if getattr(args, "arch", "").lower() == "linear":
                xb = xb.view(xb.size(0), -1)
            if not torch.is_tensor(yb):
                yb = torch.as_tensor(yb)

            logits = model_forward_for_metric_loop(
                model, xb, task_index, args, cil_mask_upto_task=task_index
            )
            pb = torch.argmax(logits, dim=1).cpu()
            raw_targets = yb.detach().cpu()
            metric_targets = raw_targets
            # UCL (``split``) emits task-local logits. A pooled batch spans many
            # tasks, so a single offset is already ill-defined here; keep the
            # pre-existing behaviour rather than change it silently.
            if getattr(model, "split", False):
                compute_offsets_fn = getattr(model, "compute_offsets", None)
                if callable(compute_offsets_fn):
                    offset1, _ = compute_offsets_fn(task_index)
                else:
                    offset1, _ = misc_utils.compute_offsets(
                        task_index,
                        class_counts if class_counts is not None else args.nc_per_task,
                    )
                metric_targets = metric_targets - offset1

            recalls.append(macro_recall(pb, metric_targets))
            precisions.append(macro_precision(pb, metric_targets))
            f1s.append(macro_f1(pb, metric_targets))

            pooled_preds.append(pb)
            pooled_metric_targets.append(metric_targets)
            pooled_raw_targets.append(raw_targets)

    headline = (
        sum(recalls) / len(recalls) if recalls else 0.0,
        sum(precisions) / len(precisions) if precisions else 0.0,
        sum(f1s) / len(f1s) if f1s else 0.0,
    )

    col_rec = []
    col_prec = []
    col_f1 = []
    if pooled_preds:
        preds = torch.cat(pooled_preds)
        metric_targets = torch.cat(pooled_metric_targets)
        raw_targets = torch.cat(pooled_raw_targets)
        label_task = _global_label_to_task(
            class_counts, int(raw_targets.max().item()) + 1
        )
        owning_task = label_task[raw_targets.clamp(min=0)]
        for t in range(task_index + 1):
            selected = owning_task == t
            if bool(selected.any()):
                col_rec.append(macro_recall(preds[selected], metric_targets[selected]))
                col_prec.append(
                    macro_precision(preds[selected], metric_targets[selected])
                )
                col_f1.append(macro_f1(preds[selected], metric_targets[selected]))
            else:
                col_rec.append(0.0)
                col_prec.append(0.0)
                col_f1.append(0.0)

    return (col_rec, col_prec, col_f1), headline


def _evaluate_validation(
    model, test_task_loaders, cil_union_loader, current_task, args, cil_mode
):
    """Per-task validation columns, plus the CIL headline when in CIL mode.

    CIL slices one pooled pass (see :func:`eval_cil_pooled`); TIL scores each
    task's own loader, where task-pure batches match how TIL models train.

    Returns:
        ``((rec, prec, f1), headline)`` with ``headline`` ``None`` under TIL.
    """
    if cil_mode and cil_union_loader is not None:
        return eval_cil_pooled(model, cil_union_loader, current_task, args)
    return eval_tasks(model, test_task_loaders, args), None


def _save_task_checkpoint(
    model: torch.nn.Module, experiment_log_dir: str, task_index: int
) -> str:
    """Persist ``model`` weights under ``experiment_log_dir/checkpoints``.

    Checkpoints are written after each continual-learning task completes (same
    experiment root as ``metrics/`` and ``results.pt``).

    Args:
        model: Trained module whose ``state_dict`` will be stored.
        experiment_log_dir: Run directory (typically ``args.log_dir``).
        task_index: Completed task id (``task_info['task']``).

    Returns:
        Absolute path to the saved ``.pt`` file.

    Usage:
        path = _save_task_checkpoint(model, args.log_dir, current_task)
    """
    checkpoints_dir = os.path.join(experiment_log_dir, "checkpoints")
    os.makedirs(checkpoints_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoints_dir, "task_{}.pt".format(task_index))
    torch.save(
        {"task": int(task_index), "state_dict": model.state_dict()},
        checkpoint_path,
        pickle_protocol=4,
    )
    return checkpoint_path


def _checkpoint_task_index(checkpoint_path: object) -> int:
    """Return the completed-task id encoded in a ``task_<i>.pt`` filename.

    Args:
        checkpoint_path: Path to a checkpoint written by ``_save_task_checkpoint``.

    Returns:
        The integer task id parsed from the filename (e.g. ``task_4.pt`` -> ``4``).

    Raises:
        ValueError: If the filename does not end with an integer task id.
    """
    stem = Path(checkpoint_path).stem
    return int(stem.split("_")[-1])


def _discover_task_checkpoints(checkpoints_dir: Path) -> dict[int, Path]:
    """Map completed task ids to their checkpoint files inside ``checkpoints_dir``.

    Args:
        checkpoints_dir: Directory containing ``task_<i>.pt`` files.

    Returns:
        Dict of ``task_id -> checkpoint Path`` for every parseable checkpoint.
    """
    discovered: dict[int, Path] = {}
    for candidate in checkpoints_dir.glob("task_*.pt"):
        try:
            discovered[_checkpoint_task_index(candidate)] = candidate
        except ValueError:
            continue
    return discovered


def _resolve_resume_plan(args: object, resume_request: str) -> dict:
    """Resolve where to resume an interrupted experiment from.

    The ``resume_request`` may point at an experiment log directory (whose
    ``checkpoints/`` folder is scanned) or directly at a ``task_<i>.pt`` file.
    The optional ``args.resume_task`` overrides which task training resumes at
    (loading ``task_<resume_task-1>.pt``); otherwise we continue one task past
    the latest available checkpoint.

    Args:
        args: Parsed experiment arguments (read ``resume_task``).
        resume_request: Value of ``--resume`` (directory or checkpoint file).

    Returns:
        Dict with ``experiment_dir`` (str), ``tf_dir`` (str),
        ``resume_from_task`` (int), and ``checkpoint_path`` (str or ``None``).

    Raises:
        SystemExit: If the path or requested checkpoint cannot be found.
    """
    request_path = Path(resume_request).expanduser()
    if request_path.is_file():
        checkpoints_dir = request_path.parent
        experiment_dir = checkpoints_dir.parent
        explicit_checkpoint: Path | None = request_path
    else:
        experiment_dir = request_path
        checkpoints_dir = request_path / "checkpoints"
        explicit_checkpoint = None

    if not checkpoints_dir.is_dir():
        raise SystemExit(
            "Cannot resume: no checkpoints directory at {}".format(checkpoints_dir)
        )

    available_checkpoints = _discover_task_checkpoints(checkpoints_dir)
    resume_task_override = getattr(args, "resume_task", None)

    if explicit_checkpoint is not None and resume_task_override is None:
        checkpoint_path: Path | None = explicit_checkpoint
        resume_from_task = _checkpoint_task_index(explicit_checkpoint) + 1
    elif resume_task_override is not None:
        resume_from_task = int(resume_task_override)
        needed_task = resume_from_task - 1
        if needed_task < 0:
            checkpoint_path = None
        elif needed_task in available_checkpoints:
            checkpoint_path = available_checkpoints[needed_task]
        else:
            raise SystemExit(
                "Cannot resume at task {}: required checkpoint task_{}.pt not found in {}".format(
                    resume_from_task, needed_task, checkpoints_dir
                )
            )
    else:
        if not available_checkpoints:
            raise SystemExit(
                "Cannot resume: no task_<i>.pt checkpoints found in {}".format(
                    checkpoints_dir
                )
            )
        latest_task = max(available_checkpoints)
        checkpoint_path = available_checkpoints[latest_task]
        resume_from_task = latest_task + 1

    tf_dir = experiment_dir / "tfdir"
    os.makedirs(tf_dir, exist_ok=True)

    return {
        "experiment_dir": str(experiment_dir),
        "tf_dir": str(tf_dir),
        "resume_from_task": resume_from_task,
        "checkpoint_path": str(checkpoint_path) if checkpoint_path else None,
    }


def _load_checkpoint_into_model(
    model: torch.nn.Module, checkpoint_path: str, args: object
) -> None:
    """Load a task checkpoint's weights into ``model`` for resuming.

    Args:
        model: Model instance to receive the checkpoint weights.
        checkpoint_path: Path to a ``task_<i>.pt`` file written by
            ``_save_task_checkpoint``.
        args: Parsed experiment arguments (read ``cuda`` for the map location).

    Raises:
        SystemExit: If the checkpoint cannot be read or has no usable state dict.
    """
    path = Path(checkpoint_path).expanduser()
    if not path.exists():
        raise SystemExit("Resume checkpoint does not exist: {}".format(path))

    map_location = (
        "cuda" if getattr(args, "cuda", False) and torch.cuda.is_available() else "cpu"
    )
    try:
        checkpoint = torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        checkpoint = torch.load(path, map_location=map_location)

    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        checkpoint_state_dict = checkpoint["state_dict"]
    elif isinstance(checkpoint, dict):
        checkpoint_state_dict = checkpoint
    else:
        raise SystemExit(
            "Unsupported checkpoint format at {}; expected a dict with 'state_dict'.".format(
                path
            )
        )

    model_state_dict = model.state_dict()
    filtered_state_dict = {
        key: value
        for key, value in checkpoint_state_dict.items()
        if key in model_state_dict
    }
    incompatible = model.load_state_dict(filtered_state_dict, strict=False)

    print(
        "Loaded resume checkpoint: {} (matched keys: {} / {})".format(
            path, len(filtered_state_dict), len(model_state_dict)
        )
    )
    if incompatible.missing_keys:
        print(
            "Missing {} model key(s) not present in checkpoint.".format(
                len(incompatible.missing_keys)
            )
        )


def _cumulative_train_loader(
    seen_train_datasets: List[Dataset], task_train_loader: DataLoader
) -> DataLoader:
    """Return a shuffled loader over this task's training set and all earlier ones.

    Args:
        seen_train_datasets: Training datasets of earlier tasks; the current
            task's dataset is appended in place.
        task_train_loader: The loader ``new_task()`` built for the current task;
            its batch size and worker count are reused.

    Returns:
        A loader over the union of every task seen so far. It is shuffled
        because the concatenation is otherwise ordered task by task.

    Usage:
        train_loader = _cumulative_train_loader(seen, train_loader)
    """
    seen_train_datasets.append(task_train_loader.dataset)
    print(
        "Maximal replay: training on {} task(s), {} samples.".format(
            len(seen_train_datasets), sum(len(d) for d in seen_train_datasets)
        )
    )
    return DataLoader(
        ConcatDataset(seen_train_datasets),
        batch_size=task_train_loader.batch_size,
        shuffle=True,
        num_workers=task_train_loader.num_workers,
    )


def life_experience(model, inc_loader, args):
    result_val_a = []
    result_test_a = []
    result_val_prec = []
    result_val_f1 = []

    result_val_t = []
    result_test_t = []

    last_tr_cls_rec = last_tr_cls_prec = last_tr_cls_f1 = None
    # (rec, prec, f1) from each task's final training epoch, in training order.
    final_epoch_tr_metrics: List[Tuple[float, float, float]] = []
    base_n_epochs = int(args.n_epochs)
    force_global_n_epochs_legacy = bool(LEGACY_USE_GLOBAL_N_EPOCHS)
    task_epoch_schedule = (
        {}
        if force_global_n_epochs_legacy
        else _task_epoch_schedule_for_base_epochs(base_n_epochs)
    )
    args.task_epoch_schedule = task_epoch_schedule

    time_start = time.time()
    # Only the current task's train loader is ever used; previously every task's
    # train loader was retained for the whole run (each holding a permuted copy of
    # that task's training set), which grew host memory as new tasks began
    # training without ever being read. Keep only the per-task local below.
    test_task_loaders = []
    # Maximal-replay models (``cumulative_replay``, e.g. iid2) are the exception:
    # they train task t on tasks 0..t, so every task's training set is retained.
    cumulative_replay = bool(getattr(model, "cumulative_replay", False))
    seen_train_datasets = []
    # Validation routing lives in ``_evaluate_validation``: CIL slices one pooled
    # pass, TIL scores each task's own loader.
    cil_mode = args.loader == "class_incremental_loader"
    # CIL headline per task: macro rec/prec/f1 on the pooled tasks 0..t test set.
    cil_union_rec: list[float] = []
    cil_union_prec: list[float] = []
    cil_union_f1: list[float] = []

    interactive_terminal = sys.stdout.isatty()
    amp_dtype = (
        torch.bfloat16
        if getattr(args, "amp_dtype", "bfloat16") == "bfloat16"
        else torch.float16
    )
    use_amp = bool(
        getattr(args, "amp", False) and args.cuda and not getattr(args, "no-amp", False)
    )
    if getattr(args, "model", "") == "eucr":
        use_amp = False
    print("use amp:", use_amp)
    resume_from_task = int(getattr(args, "resume_from_task", 0) or 0)
    log_state(
        args.state_logging,
        "Life experience start: {} tasks queued".format(inc_loader.n_tasks),
    )
    if resume_from_task > 0:
        log_state(
            args.state_logging,
            "Resuming from checkpoint: tasks 0-{} will be replayed to rebuild "
            "their loaders without retraining; training continues at task {}.".format(
                resume_from_task - 1, resume_from_task
            ),
        )
        print(
            "Resuming experiment at task {} (skipping {} completed task(s)).".format(
                resume_from_task, resume_from_task
            )
        )
    if task_epoch_schedule:
        log_state(
            args.state_logging,
            "Using task-specific epoch schedule: {}".format(task_epoch_schedule),
        )
    elif force_global_n_epochs_legacy:
        log_state(
            args.state_logging,
            "Legacy epoch behavior enabled: using n_epochs={} for all tasks".format(
                base_n_epochs
            ),
        )

    for task_i in range(inc_loader.n_tasks):
        result_epoch_loss = []
        result_acc_val = []
        result_acc_tr = []
        task_info, train_loader, _, test_loader = inc_loader.new_task()
        current_task = task_info["task"]
        # Under the CIL loader ``new_task`` hands back the *pooled* tasks 0..t
        # test set. Keeping that as column t made every column a nested prefix,
        # so the per-task retention the BWT/forgetting/FWT machinery expects was
        # never measured (and the mean over columns counted early tasks up to
        # n_tasks times). Columns are the per-task splits; the pooled loader is
        # scored separately as the CIL headline.
        cil_union_loader = test_loader if cil_mode else None
        if cil_mode:
            test_task_loaders.append(inc_loader.get_tasks("test")[current_task])
        else:
            test_task_loaders.append(test_loader)
        if cumulative_replay:
            train_loader = _cumulative_train_loader(seen_train_datasets, train_loader)

        # When resuming an interrupted experiment, advance the loader for tasks
        # already trained (so their loaders exist for evaluating retention) but
        # skip retraining, zero-shot eval, metric dumps, and checkpoint writes.
        # The model weights were restored from the resume checkpoint in main().
        if current_task < resume_from_task:
            log_state(
                args.state_logging,
                "Skipping completed task {} ({}/{}) while resuming".format(
                    current_task, task_i + 1, inc_loader.n_tasks
                ),
            )
            print(
                "Skipping completed task {} (loaded from checkpoint).".format(
                    current_task
                )
            )
            continue

        task_n_epochs = task_epoch_schedule.get(current_task, base_n_epochs)
        args.n_epochs = task_n_epochs

        log_state(
            args.state_logging,
            "Task {}: zero-shot validation (pre-train)".format(current_task),
        )
        zero_shot_raw, _ = _evaluate_validation(
            model, test_task_loaders, cil_union_loader, current_task, args, cil_mode
        )
        zs_rec, zs_prec, zs_f1 = _split_eval_output(zero_shot_raw)
        num_tasks_now = len(test_task_loaders)
        current_task_idx = task_info["task"]
        zero_shot_rec_cls = _scalar_metric_at_task_index(zs_rec, current_task_idx)
        zero_shot_prec_cls = _scalar_metric_at_task_index(zs_prec, current_task_idx)
        zero_shot_f1_cls = _scalar_metric_at_task_index(zs_f1, current_task_idx)
        zero_shot_total_f1 = _mean_metric_across_tasks(zs_f1)
        zero_shot_per_task_rec_cls = _per_task_metric_array(zs_rec, num_tasks_now)
        zero_shot_per_task_prec_cls = _per_task_metric_array(zs_prec, num_tasks_now)
        zero_shot_per_task_f1_cls = _per_task_metric_array(zs_f1, num_tasks_now)
        print(
            "---- Zero-shot (pre-train) task {}: macro_rec {:.4f} | macro_prec {:.4f} | macro_f1 {:.4f} | total_f1 {:.4f} ----".format(
                current_task,
                zero_shot_rec_cls,
                zero_shot_prec_cls,
                zero_shot_f1_cls,
                zero_shot_total_f1,
            )
        )

        # Per-epoch training metrics for this task.
        per_epoch_train_cls_rec = []
        per_epoch_train_cls_prec = []
        per_epoch_train_f1 = []

        # Per-evaluation validation metrics for this task.
        per_epoch_val_cls_rec = []
        per_epoch_val_cls_prec = []
        per_epoch_val_f1 = []

        log_state(
            args.state_logging,
            "Starting task {} ({}/{})".format(
                current_task, task_i + 1, inc_loader.n_tasks
            ),
        )
        for ep in range(task_n_epochs):
            model.real_epoch = ep
            epoch_losses = []
            epoch_train_accs = []
            epoch_precisions = []
            epoch_f1s = []
            epoch_eval_mode_recalls = []
            epoch_start_time = time.time()
            epoch_eval_time = 0.0
            log_state(
                args.state_logging,
                "Task {} Epoch {}/{}: entering train loop".format(
                    current_task, ep + 1, task_n_epochs
                ),
            )

            prog_bar = tqdm(train_loader, disable=not interactive_terminal)
            for i, (x, y) in enumerate(prog_bar):

                v_x = x
                y_cls = y if torch.is_tensor(y) else torch.as_tensor(y)
                v_y = y_cls
                if args.cuda:
                    v_x = v_x.cuda()
                    v_y = v_y.cuda()
                model.train()
                task_bn.set_active_task(model, task_info["task"])
                amp_context = (
                    torch.autocast(device_type="cuda", dtype=amp_dtype)
                    if use_amp
                    else nullcontext()
                )
                with amp_context:
                    observe_result = model.observe(
                        Variable(v_x), v_y, task_info["task"]
                    )
                loss, cls_tr_rec, metric_logits = unpack_observe_result(observe_result)
                observe_cls_tr_rec = float(cls_tr_rec)
                result_acc_tr.append(cls_tr_rec)
                result_epoch_loss.append(loss)
                epoch_losses.append(loss)
                epoch_train_accs.append(cls_tr_rec)

                # Batch-level macro metrics for the progress bar. Prefer observe()
                # predictions when available to avoid a second stochastic forward.
                y_cls_for_metric = y_cls.cpu()

                # For split (task-incremental) models, forward returns task-local logits so pb is in [0, C_t-1].
                # Convert labels to task-local so Train Rec / Prec / F1 match.
                if getattr(model, "split", False):
                    offset1, _ = model.compute_offsets(task_info["task"])
                    y_cls_for_metric = y_cls_for_metric - offset1

                if metric_logits is not None:
                    pb = torch.argmax(metric_logits, dim=1).cpu()
                else:
                    model.eval()
                    with torch.no_grad():
                        logits = model_forward_for_metric_loop(
                            model, v_x, task_info["task"], args
                        )
                        pb = torch.argmax(logits, dim=1).cpu()
                    model.train()

                prec = macro_precision(pb, y_cls_for_metric)
                f1 = macro_f1(pb, y_cls_for_metric)
                cls_tr_rec = macro_recall(pb, y_cls_for_metric)

                result_acc_tr[-1] = cls_tr_rec
                epoch_train_accs[-1] = cls_tr_rec

                epoch_precisions.append(prec)
                epoch_f1s.append(f1)
                _maybe_print_train_metric_debug(
                    task_index=task_info["task"],
                    epoch_index=ep,
                    batch_index=i,
                    observe_cls_recall=observe_cls_tr_rec,
                    metric_cls_recall=cls_tr_rec,
                    metric_precision=prec,
                    metric_f1=f1,
                    predictions=pb,
                    labels_for_metrics=y_cls_for_metric,
                )

                prog_bar.set_description(
                    "T{}| Ep: {}/{}| Loss: {}| Rec: {}| Prec: {}| F1: {}".format(
                        task_info["task"],
                        ep + 1,
                        task_n_epochs,
                        round(loss, 3),
                        round(cls_tr_rec, 2),
                        round(prec, 2),
                        round(f1, 2),
                    )
                )

                # prog_bar.set_description(
                #     "Task: {} | Epoch: {}/{} | Iter: {} | Loss: {} | Acc: Total: {} Current Task: {} ".format(
                #         task_info["task"], ep+1, args.n_epochs, i%(1000*args.n_epochs), round(loss, 3),
                #         round(sum(result_val_a[-1]).item()/len(result_val_a[-1]), 5), round(result_val_a[-1][task_info["task"]].item(), 5)
                #     )
                # )

            # Run validation at end of epoch (after last batch) so val scores reflect current task
            if (ep % args.val_rate) == 0:
                eval_start = time.time()
                log_state(
                    args.state_logging,
                    "Task {} Epoch {}/{}: running validation (end of epoch)".format(
                        current_task, ep + 1, task_n_epochs
                    ),
                )
                val_acc, _ = _evaluate_validation(
                    model,
                    test_task_loaders,
                    cil_union_loader,
                    current_task,
                    args,
                    cil_mode,
                )
                val_acc, val_prec, val_f1 = _split_eval_output(val_acc)
                epoch_eval_time += time.time() - eval_start
                result_acc_val.append(val_acc)
                result_val_a.append(val_acc)
                if val_prec is not None:
                    result_val_prec.append(val_prec)
                if val_f1 is not None:
                    result_val_f1.append(val_f1)
                result_val_t.append(task_info["task"])
                print("---- Eval at Epoch {}: {} ----".format(ep, val_acc))

                # Store per-evaluation validation metrics for this task (current epoch).
                # Index into the evaluator outputs with the current task id where possible.
                current_task_idx = task_info["task"]
                if isinstance(val_acc, (list, tuple)) and current_task_idx < len(
                    val_acc
                ):
                    per_epoch_val_cls_rec.append(float(val_acc[current_task_idx]))
                elif not isinstance(val_acc, (list, tuple)):
                    per_epoch_val_cls_rec.append(float(val_acc))
                else:
                    per_epoch_val_cls_rec.append(float("nan"))

                if val_prec is not None:
                    if isinstance(val_prec, (list, tuple)) and current_task_idx < len(
                        val_prec
                    ):
                        per_epoch_val_cls_prec.append(float(val_prec[current_task_idx]))
                    elif not isinstance(val_prec, (list, tuple)):
                        per_epoch_val_cls_prec.append(float(val_prec))
                    else:
                        per_epoch_val_cls_prec.append(float("nan"))
                else:
                    per_epoch_val_cls_prec.append(float("nan"))

                if val_f1 is not None:
                    if isinstance(val_f1, (list, tuple)) and current_task_idx < len(
                        val_f1
                    ):
                        per_epoch_val_f1.append(float(val_f1[current_task_idx]))
                    elif not isinstance(val_f1, (list, tuple)):
                        per_epoch_val_f1.append(float(val_f1))
                    else:
                        per_epoch_val_f1.append(float("nan"))
                else:
                    per_epoch_val_f1.append(float("nan"))

            epoch_duration = time.time() - epoch_start_time
            epoch_train_time = max(epoch_duration - epoch_eval_time, 0.0)
            avg_loss = (
                float(sum(epoch_losses) / len(epoch_losses))
                if epoch_losses
                else float("nan")
            )
            avg_cls_tr_rec = (
                float(sum(epoch_train_accs) / len(epoch_train_accs))
                if epoch_train_accs
                else float("nan")
            )
            avg_prec = (
                float(sum(epoch_precisions) / len(epoch_precisions))
                if epoch_precisions
                else float("nan")
            )
            avg_f1 = (
                float(sum(epoch_f1s) / len(epoch_f1s)) if epoch_f1s else float("nan")
            )

            # Track the last training metrics we saw (for summary logging).
            last_tr_cls_rec = avg_cls_tr_rec
            last_tr_cls_prec = avg_prec
            last_tr_cls_f1 = avg_f1

            # Persist per-epoch training metrics for this task.
            per_epoch_train_cls_rec.append(avg_cls_tr_rec)
            per_epoch_train_cls_prec.append(avg_prec)
            per_epoch_train_f1.append(avg_f1)

            if not interactive_terminal:
                print(
                    "T{} Ep {}/{} | L {:.4f} | Rec {:.2f} | Prec {:.2f} | F1 {:.2f} | Epoch Time {:.2f}s (Eval {:.2f}s, Train {:.2f}s)".format(
                        task_info["task"],
                        ep + 1,
                        task_n_epochs,
                        avg_loss,
                        avg_cls_tr_rec,
                        avg_prec,
                        avg_f1,
                        epoch_duration,
                        epoch_eval_time,
                        epoch_train_time,
                    )
                )
                log_state(
                    args.state_logging,
                    "T{} Ep {}/{} complete: Prec {:.4f} F1 {:.4f} | {:.2f}s total ({:.2f}s eval/{:.2f}s train)".format(
                        current_task,
                        ep + 1,
                        task_n_epochs,
                        avg_prec,
                        avg_f1,
                        epoch_duration,
                        epoch_eval_time,
                        epoch_train_time,
                    ),
                )
            if epoch_train_accs and epoch_eval_mode_recalls:
                avg_tr_recall = float(sum(epoch_train_accs) / len(epoch_train_accs))
                avg_eval_recall = float(
                    sum(epoch_eval_mode_recalls) / len(epoch_eval_mode_recalls)
                )
                print(
                    "Task {} Epoch {}/{} | Avg Train Recall {:.5f} | Avg Eval-Mode Recall {:.5f}".format(
                        task_info["task"],
                        ep + 1,
                        task_n_epochs,
                        avg_tr_recall,
                        avg_eval_recall,
                    )
                )
        if last_tr_cls_rec is not None:
            final_epoch_tr_metrics.append(
                (last_tr_cls_rec, last_tr_cls_prec, last_tr_cls_f1)
            )
            last_tr_cls_rec = last_tr_cls_prec = last_tr_cls_f1 = None

        finalize_fn = getattr(model, "finalize_task_after_training", None)
        if callable(finalize_fn):
            finalize_fn(train_loader)
        log_state(
            args.state_logging,
            "Task {}: running final validation.".format(current_task),
        )
        val_acc, val_headline = _evaluate_validation(
            model, test_task_loaders, cil_union_loader, current_task, args, cil_mode
        )
        val_acc, val_prec, val_f1 = _split_eval_output(val_acc)
        result_val_a.append(val_acc)
        if val_prec is not None:
            result_val_prec.append(val_prec)
        if val_f1 is not None:
            result_val_f1.append(val_f1)
        result_val_t.append(task_info["task"])

        if val_headline is not None:
            union_rec, union_prec, union_f1 = val_headline
            cil_union_rec.append(union_rec)
            cil_union_prec.append(union_prec)
            cil_union_f1.append(union_f1)
            print(
                "---- CIL all-seen-classes (tasks 0..{}): macro_rec {:.4f} | "
                "macro_prec {:.4f} | macro_f1 {:.4f} ----".format(
                    current_task, union_rec, union_prec, union_f1
                )
            )

        losses = np.array(result_epoch_loss)
        result_acc_tr = np.array(
            [x.cpu().item() if torch.is_tensor(x) else x for x in result_acc_tr]
        )
        result_acc_val = np.array(
            [
                x.detach().cpu().item() if torch.is_tensor(x) else x
                for sublist in result_acc_val
                for x in sublist
            ]
        )
        # Flatten validation F1 scores in the same order as result_acc_val, if available.
        if result_val_f1:
            result_val_f1_flat = np.array(
                [
                    x.detach().cpu().item() if torch.is_tensor(x) else x
                    for sublist in result_val_f1
                    for x in sublist
                ]
            )
        else:
            result_val_f1_flat = None

        logs_dir = os.path.join(args.log_dir, "metrics")
        os.makedirs(logs_dir, exist_ok=True)
        save_payload = {
            "losses": losses,
            "tr_macro_rec": result_acc_tr,
            "val_macro_rec": result_acc_val,
            "n_epochs": np.int64(task_n_epochs),
            "zero_shot_macro_rec": np.float64(zero_shot_rec_cls),
            "zero_shot_macro_prec": np.float64(zero_shot_prec_cls),
            "zero_shot_macro_f1": np.float64(zero_shot_f1_cls),
            "zero_shot_total_macro_f1": np.float64(zero_shot_total_f1),
            "zero_shot_per_task_macro_rec": zero_shot_per_task_rec_cls,
            "zero_shot_per_task_macro_prec": zero_shot_per_task_prec_cls,
            "zero_shot_per_task_macro_f1": zero_shot_per_task_f1_cls,
        }
        if result_val_f1_flat is not None:
            save_payload["val_macro_f1"] = result_val_f1_flat
        if cil_mode and cil_union_f1:
            # One entry per task completed so far: the CIL score over every class
            # seen up to that task. Element -1 is this task's headline; the whole
            # vector is the decay curve to plot.
            save_payload["cil_union_macro_rec"] = np.asarray(cil_union_rec, dtype=float)
            save_payload["cil_union_macro_prec"] = np.asarray(
                cil_union_prec, dtype=float
            )
            save_payload["cil_union_macro_f1"] = np.asarray(cil_union_f1, dtype=float)
        # Optional: per-epoch training metrics for this task.
        if per_epoch_train_cls_rec:
            save_payload["train_macro_rec"] = np.asarray(
                per_epoch_train_cls_rec, dtype=float
            )
        if per_epoch_train_cls_prec:
            save_payload["train_macro_prec"] = np.asarray(
                per_epoch_train_cls_prec, dtype=float
            )
        if per_epoch_train_f1:
            save_payload["train_macro_f1"] = np.asarray(per_epoch_train_f1, dtype=float)

        # Optional: per-evaluation validation metrics for this task (one entry per eval/epoch).
        if per_epoch_val_cls_rec:
            save_payload["val_macro_rec_per_epoch"] = np.asarray(
                per_epoch_val_cls_rec, dtype=float
            )
        if per_epoch_val_cls_prec:
            save_payload["val_macro_prec_per_epoch"] = np.asarray(
                per_epoch_val_cls_prec, dtype=float
            )
        if per_epoch_val_f1:
            save_payload["val_macro_f1_per_epoch"] = np.asarray(
                per_epoch_val_f1, dtype=float
            )

        # Persist per-task metrics and a human-readable task order file.
        np.savez(os.path.join(logs_dir, "task" + str(task_i) + ".npz"), **save_payload)

        task_order_path = os.path.join(logs_dir, "task_order.txt")
        try:
            task_name = task_info.get("task_name", f"task{task_i}")
        except AttributeError:
            task_name = f"task{task_i}"
        with open(task_order_path, "a", encoding="utf-8") as f_task_order:
            f_task_order.write(str(task_name) + "\n")

        if args.calc_test_accuracy:
            test_acc, _ = _evaluate_validation(
                model, test_task_loaders, cil_union_loader, current_task, args, cil_mode
            )
            test_acc, test_prec, test_f1 = _split_eval_output(test_acc)
            result_test_a.append(test_acc)
            result_test_t.append(task_info["task"])

        if getattr(args, "save_checkpoints", True):
            checkpoint_path = _save_task_checkpoint(model, args.log_dir, current_task)
            print("Saved task checkpoint: {}".format(checkpoint_path))
            log_state(
                args.state_logging,
                "Saved task checkpoint to {}".format(checkpoint_path),
            )

        log_state(
            args.state_logging,
            "Completed task {} ({}/{})".format(
                current_task, task_i + 1, inc_loader.n_tasks
            ),
        )

    print("####Final Validation Accuracy####")
    print(
        "Final Results:- \n Total Recall: {} \n Individual Recall: {}".format(
            sum(result_val_a[-1]) / len(result_val_a[-1]), result_val_a[-1]
        )
    )

    def _nan_mean(values):
        finite = [float(v) for v in values if v is not None and not math.isnan(v)]
        return sum(finite) / len(finite) if finite else None

    def _mean(x):
        if x is None or (isinstance(x, (list, tuple)) and len(x) == 0):
            return None
        if isinstance(x, (list, tuple)):
            return sum(float(v) for v in x) / len(x)
        return float(x)

    # Headline macro metrics after the last task, returned so main() can
    # record them for the cross-seed sweep summary. Missing values stay None.
    headline = {
        "val_macro_rec": None,
        "val_macro_prec": None,
        "val_macro_f1": None,
        "tr_macro_rec": None,
        "tr_macro_prec": None,
        "tr_macro_f1": None,
    }

    if final_epoch_tr_metrics:
        # CIL scores training in the global class space, so the last task's
        # final epoch already speaks for the run. TIL scores each task in its
        # own label space, so average every task's final epoch instead.
        tr_rows = final_epoch_tr_metrics[-1:] if cil_mode else final_epoch_tr_metrics
        tr_rec, tr_prec, tr_f1 = (
            _nan_mean([row[col] for row in tr_rows]) for col in range(3)
        )
        headline["tr_macro_rec"] = tr_rec
        headline["tr_macro_prec"] = tr_prec
        headline["tr_macro_f1"] = tr_f1
        parts = []
        if tr_rec is not None:
            parts.append("macro_rec={:.4f}".format(tr_rec))
        if tr_prec is not None:
            parts.append("macro_prec={:.4f}".format(tr_prec))
        if tr_f1 is not None:
            parts.append("macro_f1={:.4f}".format(tr_f1))
        if parts:
            print("SUMMARY_TR " + " ".join(parts))

    if result_val_a:
        if cil_mode and cil_union_f1:
            # CIL headline is the score over every class seen after the last
            # task, not the mean of the per-task columns.
            te_rec = cil_union_rec[-1]
            te_prec = cil_union_prec[-1]
            te_f1 = cil_union_f1[-1]
        else:
            te_rec = _mean(result_val_a[-1])
            te_prec = _mean(result_val_prec[-1]) if result_val_prec else None
            te_f1 = _mean(result_val_f1[-1]) if result_val_f1 else None
        headline["val_macro_rec"] = float(te_rec)
        headline["val_macro_prec"] = float(te_prec) if te_prec is not None else None
        headline["val_macro_f1"] = float(te_f1) if te_f1 is not None else None
        parts = ["macro_rec={:.4f}".format(te_rec)]
        if te_prec is not None:
            parts.append("macro_prec={:.4f}".format(te_prec))
        if te_f1 is not None:
            parts.append("macro_f1={:.4f}".format(te_f1))
        print("SUMMARY_TE " + " ".join(parts))

    if args.calc_test_accuracy:
        print("####Final Test Accuracy####")
        print(
            "Final Results:- \n Total Accuracy: {} \n Individual Accuracy: {}".format(
                sum(result_test_a[-1]) / len(result_test_a[-1]), result_test_a[-1]
            )
        )

    time_end = time.time()
    time_spent = time_end - time_start
    args.n_epochs = base_n_epochs

    def _pad_results(result_list: list[object], pad_value: float = 0.0) -> torch.Tensor:
        """Pad ragged per-task results into a dense 2D tensor.

        Args:
            result_list: Sequence of per-eval results, each being a list/array/tensor
                of task metrics or a scalar.
            pad_value: Value used to pad missing task entries.

        Returns:
            A 2D tensor of shape (num_evals, max_tasks).

        Usage:
            results = _pad_results([[0.1, 0.2], [0.3]])
        """
        if not result_list:
            return torch.empty((0, 0), dtype=torch.float)

        def _flatten_to_floats(value: object) -> list[float]:
            if isinstance(value, torch.Tensor):
                return [float(x) for x in value.detach().cpu().flatten().tolist()]
            if isinstance(value, np.ndarray):
                return [float(x) for x in value.flatten().tolist()]
            if isinstance(value, (list, tuple)):
                flattened: list[float] = []
                for item in value:
                    flattened.extend(_flatten_to_floats(item))
                return flattened
            return [float(value)]

        normalized_rows: list[torch.Tensor] = []
        for row in result_list:
            if row is None:
                row_tensor = torch.empty((0,), dtype=torch.float)
            else:
                row_tensor = torch.as_tensor(_flatten_to_floats(row), dtype=torch.float)
            normalized_rows.append(row_tensor)

        max_len = max(row.numel() for row in normalized_rows)
        padded = torch.full(
            (len(normalized_rows), max_len), float(pad_value), dtype=torch.float
        )
        for row_idx, row_tensor in enumerate(normalized_rows):
            if row_tensor.numel() == 0:
                continue
            padded[row_idx, : row_tensor.numel()] = row_tensor
        return padded

    return (
        torch.Tensor(result_val_t),
        _pad_results(result_val_a),
        _pad_results(result_val_prec),
        _pad_results(result_val_f1),
        torch.Tensor(result_test_t),
        _pad_results(result_test_a),
        time_spent,
        headline,
    )


def estimate_memory_buffer_size_bytes(model: torch.nn.Module) -> int:
    """Estimate total bytes used by replay/memory buffers in a model.

    Uses the same categorisation and walk as
    :func:`summarise_persistent_state_bytes` (``replay`` category), including
    exemplars stored as NumPy arrays inside reservoir lists such as ER's
    ``M`` / ``M_new``.

    Args:
        model: Torch module whose memory/replay buffers will be inspected.

    Returns:
        Total number of bytes occupied by replay buffers.

    Usage:
        buffer_bytes = estimate_memory_buffer_size_bytes(model)
    """
    return summarise_persistent_state_bytes(model)["replay"]


# Attribute names whose value is a full snapshot of the live network (an
# ``nn.Module``). Their parameters/buffers are persistent regularization state,
# not part of the deployable model, even though PyTorch registers them as
# submodules so they would otherwise inflate the trainable-parameter count
# (e.g. LWF's distillation ``teacher``, UCL's previous-task ``model_old``).
SNAPSHOT_MODULE_ATTRIBUTE_NAMES = frozenset(
    {"teacher", "model_old", "old_model", "old_net", "prev_model", "prev_net"}
)

# Ordered persistent-state categories reported by
# :func:`summarise_persistent_state_bytes`.
PERSISTENT_STATE_CATEGORIES = (
    "model_params",
    "replay",
    "regularization",
    "arch_mask",
    "other",
)


def _persistent_storage_byte_size(value: torch.Tensor | np.ndarray) -> int:
    """Return the size in bytes of a tensor or NumPy array backing store.

    Args:
        value: A :class:`torch.Tensor` or :class:`numpy.ndarray`.

    Returns:
        Number of bytes occupied by the array storage.
    """
    if torch.is_tensor(value):
        return value.numel() * value.element_size()
    return int(value.nbytes)


def _persistent_storage_pointer(value: torch.Tensor | np.ndarray) -> int:
    """Return a stable data pointer for deduplicating persistent storage.

    Args:
        value: A :class:`torch.Tensor` or :class:`numpy.ndarray`.

    Returns:
        Integer address of the underlying data buffer.
    """
    if torch.is_tensor(value):
        return value.data_ptr()
    return int(value.__array_interface__["data"][0])


def _iter_state_tensors(value: object):
    """Yield every tensor or NumPy array reachable inside a nested container.

    Args:
        value: A tensor, ndarray, or a list/tuple/set/dict that may contain
            them at any depth (for example ER reservoir entries in ``M``).

    Yields:
        Each :class:`torch.Tensor` or :class:`numpy.ndarray` found by walking
        the container.
    """
    if torch.is_tensor(value):
        yield value
    elif isinstance(value, np.ndarray):
        yield value
    elif isinstance(value, np.generic):
        yield np.asarray(value)
    elif isinstance(value, dict):
        for inner_value in value.values():
            yield from _iter_state_tensors(inner_value)
    elif isinstance(value, (list, tuple, set)):
        for inner_value in value:
            yield from _iter_state_tensors(inner_value)


def _categorize_state_attribute(attribute_name: str) -> str:
    """Classify a non-parameter persistent tensor by its attribute name.

    Args:
        attribute_name: Name of the module attribute or registered buffer that
            holds the tensor (e.g. ``"memory_data"``, ``"fisher"``,
            ``"weight_owner"``).

    Returns:
        One of ``"replay"``, ``"regularization"``, ``"arch_mask"`` or
        ``"other"``.

    Usage:
        >>> _categorize_state_attribute("memory_labs")
        'replay'
        >>> _categorize_state_attribute("conv_weight_si_omega")
        'regularization'
    """
    if attribute_name in ("M", "M_new"):
        return "replay"
    lowered_name = attribute_name.lower()
    replay_keywords = ("mem", "exemplar", "replay")
    if any(keyword in lowered_name for keyword in replay_keywords):
        return "replay"
    regularization_keywords = (
        "fisher",
        "omega",
        "importance",
        "star",
        "si_prev",
        "si_p_old",
        "si_w",
    )
    if any(keyword in lowered_name for keyword in regularization_keywords):
        return "regularization"
    architecture_keywords = ("owner", "frozen", "mask")
    if any(keyword in lowered_name for keyword in architecture_keywords):
        return "arch_mask"
    return "other"


def summarise_persistent_state_bytes(model: torch.nn.Module) -> dict[str, int]:
    """Break a model's persistent storage footprint down by category, in bytes.

    This walks every module and accounts for all persistent tensor and NumPy
    state, including replay/exemplar buffers stored in Python lists or dicts
    (for example ``eralg4``'s ``M``), regularization state (Fisher information,
    Synaptic Intelligence buffers, weight snapshots), and architecture masks
    (e.g. PackNet ``*_owner`` / ``*_frozen`` buffers). Snapshot submodules such
    as a distillation teacher are attributed to ``regularization`` rather than
    ``model_params``.

    Each underlying storage is counted once (deduplicated by data pointer) and
    optimizer state is not inspected.

    Args:
        model: Model to inspect, ideally at the end of training when replay and
            regularization buffers are populated.

    Returns:
        Mapping from each category in :data:`PERSISTENT_STATE_CATEGORIES` to the
        number of bytes it occupies (always includes every key, possibly zero).

    Usage:
        breakdown = summarise_persistent_state_bytes(model)
        total_gb = sum(breakdown.values()) / (1024**3)
    """
    parameter_storage_pointers = {
        parameter.data_ptr() for parameter in model.parameters()
    }

    snapshot_module_ids: set[int] = set()
    for attribute_name in SNAPSHOT_MODULE_ATTRIBUTE_NAMES:
        snapshot_candidate = getattr(model, attribute_name, None)
        if isinstance(snapshot_candidate, torch.nn.Module):
            for snapshot_submodule in snapshot_candidate.modules():
                snapshot_module_ids.add(id(snapshot_submodule))

    category_bytes: dict[str, int] = {
        category: 0 for category in PERSISTENT_STATE_CATEGORIES
    }
    seen_storage_pointers: set[int] = set()

    for module in model.modules():
        module_is_snapshot = id(module) in snapshot_module_ids

        for parameter in module.parameters(recurse=False):
            storage_pointer = parameter.data_ptr()
            if storage_pointer in seen_storage_pointers:
                continue
            seen_storage_pointers.add(storage_pointer)
            parameter_bytes = parameter.numel() * parameter.element_size()
            target_category = "regularization" if module_is_snapshot else "model_params"
            category_bytes[target_category] += parameter_bytes

        named_state_arrays: list[tuple[str, torch.Tensor | np.ndarray]] = []
        for attribute_name, attribute_value in module.__dict__.items():
            if attribute_name in ("_parameters", "_buffers", "_modules"):
                continue
            for state_array in _iter_state_tensors(attribute_value):
                named_state_arrays.append((attribute_name, state_array))
        for buffer_name, buffer_tensor in module._buffers.items():
            if buffer_tensor is not None:
                named_state_arrays.append((buffer_name, buffer_tensor))

        for attribute_name, state_array in named_state_arrays:
            storage_pointer = _persistent_storage_pointer(state_array)
            if storage_pointer in parameter_storage_pointers:
                continue
            if storage_pointer in seen_storage_pointers:
                continue
            seen_storage_pointers.add(storage_pointer)
            array_bytes = _persistent_storage_byte_size(state_array)
            if module_is_snapshot:
                category_bytes["regularization"] += array_bytes
            else:
                category_bytes[
                    _categorize_state_attribute(attribute_name)
                ] += array_bytes

    return category_bytes


def save_results(
    args,
    result_val_t,
    result_val_a,
    result_val_prec,
    result_val_f1,
    result_test_t,
    result_test_a,
    model,
    spent_time,
    headline=None,
):
    """Write results.txt / results.pt for one seed.

    results.txt holds the recall task matrix (written by ``confusion_matrix``,
    kept first so older parsers still find the zero-shot row and ``Backward:``),
    then the precision and F1 matrices, then a per-metric summary table.

    ``final`` is the mean of the last matrix row, so ``bwt = final - diagonal``
    exactly. When ``headline`` is given, the headline macro metrics are recorded
    beside it (summary table ``headline`` column and the one-liner). Under CIL
    the headline is scored over every seen class at once and differs from the
    row mean, which weights classes unequally when tasks differ in size; under
    TIL it equals the row mean.

    Args:
        headline: Dict from :func:`life_experience` with ``val_macro_rec``,
            ``val_macro_prec`` and ``val_macro_f1``.

    Returns:
        ``(val_stats, test_stats, val_bwt)`` where ``val_bwt`` maps ``rec``,
        ``prec`` and ``f1`` to the mean validation BWT (None if unavailable).
    """
    fname = os.path.join(args.log_dir, "results")
    log_state(args.state_logging, "Saving results to {}".format(fname))

    size_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    size_gb = size_bytes / (1024**3)
    buffer_bytes = estimate_memory_buffer_size_bytes(model)
    buffer_gb = buffer_bytes / (1024**3)
    print("Model size: {:.4f} GB".format(size_gb))
    print("Memory buffer size: {:.4f} GB".format(buffer_gb))

    state_breakdown_bytes = summarise_persistent_state_bytes(model)
    state_breakdown_gb = {
        category: byte_count / (1024**3)
        for category, byte_count in state_breakdown_bytes.items()
    }
    total_state_gb = sum(state_breakdown_gb.values())
    state_breakdown_text = " ".join(
        "{}={:.4f}".format(category, state_breakdown_gb[category])
        for category in PERSISTENT_STATE_CATEGORIES
    )
    print(
        "Persistent state sizes (GB): {} total={:.4f}".format(
            state_breakdown_text, total_state_gb
        )
    )

    headline_scores = {
        key: (headline or {}).get("val_macro_" + key) for key in ("rec", "prec", "f1")
    }

    # save confusion matrix and print one line of stats
    val_stats = confusion_matrix(
        result_val_t, result_val_a, args.log_dir, "results.txt"
    )

    one_liner = str(vars(args)) + " # val: "
    one_liner += " ".join(["%.3f" % stat for stat in val_stats])

    test_stats = 0
    if args.calc_test_accuracy:
        test_stats = confusion_matrix(
            result_test_t, result_test_a, args.log_dir, "results.txt"
        )
        one_liner += " # test: " + " ".join(["%.3f" % stat for stat in test_stats])

    # Append precision and F1 task matrices plus a per-metric summary table.
    results_path = os.path.join(args.log_dir, "results.txt")
    metric_stats = {
        "rec": {
            "diag": float(val_stats[0]),
            "final": float(val_stats[1]),
            "bwt": float(val_stats[2]),
            "fwt": float(val_stats[3]),
        },
        "prec": append_metric_block(
            results_path, "Precision", result_val_t, result_val_prec
        ),
        "f1": append_metric_block(results_path, "F1", result_val_t, result_val_f1),
    }
    try:
        with open(results_path, "a", encoding="utf-8") as results_file:
            print("", file=results_file)
            print("Summary (validation):", file=results_file)
            print(
                "{:<10} {:>8} {:>8} {:>8} {:>8} {:>8}".format(
                    "metric", "diagonal", "final", "bwt", "fwt", "headline"
                ),
                file=results_file,
            )
            for key, label in (("rec", "recall"), ("prec", "precision"), ("f1", "f1")):
                stats = metric_stats[key]
                if stats is None:
                    continue
                headline_value = headline_scores[key]
                print(
                    "{:<10} {:>8.4f} {:>8.4f} {:>8.4f} {:>8.4f} {:>8}".format(
                        label,
                        stats["diag"],
                        stats["final"],
                        stats["bwt"],
                        stats["fwt"],
                        (
                            "n/a"
                            if headline_value is None
                            else "{:.4f}".format(headline_value)
                        ),
                    ),
                    file=results_file,
                )
            print(
                "final = mean of the last row (bwt = final - diagonal); "
                "headline = macro score over every seen class.",
                file=results_file,
            )
    except OSError:
        pass

    val_bwt = {
        key: (stats["bwt"] if stats is not None else None)
        for key, stats in metric_stats.items()
    }
    one_liner += " # bwt: " + " ".join(
        "{}={}".format(key, "n/a" if value is None else "{:.4f}".format(value))
        for key, value in val_bwt.items()
    )

    if any(value is not None for value in headline_scores.values()):
        one_liner += " # headline: " + " ".join(
            "{}={}".format(key, "n/a" if value is None else "{:.4f}".format(value))
            for key, value in headline_scores.items()
        )

    one_liner += " # sizes: model_gb={:.4f} mem_gb={:.4f}".format(size_gb, buffer_gb)
    one_liner += " # state_gb: {} total={:.4f}".format(
        state_breakdown_text, total_state_gb
    )

    print(fname + ": " + one_liner + " # " + str(spent_time))

    # save all results in binary file
    state_dict = model.state_dict()
    if getattr(args, "state_logging", False):

        def _tensor_storage_size(t):
            return t.element_size() * t.numel() if torch.is_tensor(t) else 0

        state_dict_bytes = sum(_tensor_storage_size(v) for v in state_dict.values())
        val_t_bytes = _tensor_storage_size(result_val_t)
        val_a_bytes = _tensor_storage_size(result_val_a)
        log_state(
            args.state_logging,
            "results.pt components (approx): state_dict {:.1f} MB, result_val_t {:.1f} KB, result_val_a {:.1f} KB".format(
                state_dict_bytes / (1024 * 1024), val_t_bytes / 1024, val_a_bytes / 1024
            ),
        )
    # Bound loader methods must not reach the pickle: they drag the whole
    # IncrementalLoader (every task's arrays, ~GB) into results.pt, or fail.
    for _loader_attr in ("get_samples_per_task", "get_task_train_loader"):
        if hasattr(args, _loader_attr):
            try:
                delattr(args, _loader_attr)
            except AttributeError:
                setattr(args, _loader_attr, None)

    torch.save(
        (result_val_t, result_val_a, state_dict, val_stats, one_liner, args),
        fname + ".pt",
        pickle_protocol=4,
    )
    return val_stats, test_stats, val_bwt


def _default_main_config_chain() -> List[str]:
    chain: List[str] = []
    base_cfg = Path("configs/base.yaml")
    if base_cfg.exists():
        chain.append(str(base_cfg))
    legacy = Path("config_all.yaml")
    if legacy.exists():
        chain.append(str(legacy))
    return chain


def _parse_seed_list(raw: str) -> List[int]:
    """Parse a comma-separated seed string like "0,39,55" into a list of ints."""
    seeds: List[int] = []
    for token in raw.split(","):
        token = token.strip()
        if not token:
            continue
        seeds.append(int(token))
    return seeds


# Final macro metrics recorded per seed, paired with display labels.
SWEEP_FINAL_FIELDS = [
    ("val_macro_rec", "Validation macro_rec"),
    ("val_macro_prec", "Validation macro_prec"),
    ("val_macro_f1", "Validation macro_f1"),
    ("tr_macro_rec", "Training macro_rec"),
    ("tr_macro_prec", "Training macro_prec"),
    ("tr_macro_f1", "Training macro_f1"),
]

# Per-metric validation backward transfer recorded per seed.
SWEEP_BWT_FIELDS = [
    ("val_bwt_rec", "Validation BWT rec"),
    ("val_bwt_prec", "Validation BWT prec"),
    ("val_bwt_f1", "Validation BWT f1"),
]


def _write_seed_metrics(args, spent_time, headline, val_bwt):
    """Write a small machine-readable metrics file into the seed's log dir.

    Records the headline macro recall/precision/F1 returned by
    ``life_experience`` and the per-metric validation BWT returned by
    ``save_results``. The multi-seed launcher reads these back to build the
    cross-seed summary.
    """
    payload = {
        "seed": args.seed,
        "task_order_seed": getattr(args, "task_order_seed", None),
        "task_order_seed_source": getattr(args, "task_order_seed_source", None),
    }
    for field, _label in SWEEP_FINAL_FIELDS:
        payload[field] = headline.get(field)
    for key in ("rec", "prec", "f1"):
        payload["val_bwt_" + key] = val_bwt.get(key)
    payload["runtime_seconds"] = float(spent_time)
    path = os.path.join(args.log_dir, "seed_metrics.json")
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
    except OSError:
        pass


def _write_sweep_summary(experiment_root, seeds):
    """Aggregate per-seed metrics into a cross-seed results.txt summary.

    Reads each seed's seed_metrics.json and writes mean +/- std of the final
    macro recall/precision/F1, their validation BWT, and the runtime to
    ``<experiment_root>/results.txt``.
    """
    per_seed = []
    for seed in seeds:
        path = os.path.join(experiment_root, str(seed), "seed_metrics.json")
        try:
            with open(path, "r", encoding="utf-8") as f:
                per_seed.append(json.load(f))
        except (OSError, ValueError):
            per_seed.append({"seed": seed})

    def _summary_line(field, label):
        """Build a 'mean +/- std [per-seed]' line for one metric, or None."""
        vals = [
            m.get(field) for m in per_seed if isinstance(m.get(field), (int, float))
        ]
        if not vals:
            return None
        mean = sum(vals) / len(vals)
        if len(vals) > 1:
            var = sum((x - mean) ** 2 for x in vals) / (len(vals) - 1)
            std = var**0.5
        else:
            std = 0.0
        per_seed_str = ", ".join("{:.4f}".format(x) for x in vals)
        return "  {:<22} (n={}): {:.4f} +/- {:.4f}   [{}]".format(
            label, len(vals), mean, std, per_seed_str
        )

    lines = [
        "Seed-sweep summary (classification metrics)",
        "Seeds: {}".format(", ".join(str(s) for s in seeds)),
        "Runs:  {}".format(len(seeds)),
    ]

    for title, fields in (
        ("Final macro metrics mean +/- std:", SWEEP_FINAL_FIELDS),
        ("Backward transfer (BWT) mean +/- std:", SWEEP_BWT_FIELDS),
    ):
        lines.append("")
        lines.append(title)
        for field, label in fields:
            line = _summary_line(field, label)
            if line is not None:
                lines.append(line)
    lines.append("")

    runtimes = [
        m.get("runtime_seconds")
        for m in per_seed
        if isinstance(m.get("runtime_seconds"), (int, float))
    ]
    if runtimes:
        lines.append(
            "Total runtime (all seeds): {:.2f} hours".format(sum(runtimes) / 3600.0)
        )
        lines.append(
            "Per-seed runtime hours: {}".format(
                ", ".join("{:.2f}".format(r / 3600.0) for r in runtimes)
            )
        )

    out = os.path.join(experiment_root, "results.txt")
    try:
        with open(out, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")
        print("[seed-sweep] wrote cross-seed summary to {}".format(out))
    except OSError:
        pass


def _parse_seed_gpu_ids(raw: str) -> List[str]:
    """Parse a comma-separated GPU id string like "0,1,2" into a list of strings.

    Empty or whitespace-only input yields an empty list, signalling that
    ``CUDA_VISIBLE_DEVICES`` should be left untouched for child processes.
    """
    ids: List[str] = []
    for token in (raw or "").split(","):
        token = token.strip()
        if token:
            ids.append(token)
    return ids


def _run_seeds_sequential(executable, script, base_argv, seeds, shared_timestamp):
    """Run each seed one after another, aborting the sweep on the first failure.

    Args:
        executable: Path to the Python interpreter (``sys.executable``).
        script: Path to this script (``sys.argv[0]``).
        base_argv: Argv (without seed/timestamp flags) shared by every child.
        seeds: Ordered list of integer seeds to run.
        shared_timestamp: Timestamp shared by all children so they group under
            one experiment directory.
    """
    import subprocess

    for index, seed in enumerate(seeds):
        child_argv = _build_seed_child_argv(base_argv, seed, shared_timestamp)
        print(
            "[seed-sweep] launching seed {} ({} of {})".format(
                seed, index + 1, len(seeds)
            )
        )
        code = subprocess.call([executable, script, *child_argv])
        if code != 0:
            raise SystemExit(
                "[seed-sweep] seed {} failed with exit code {}; "
                "aborting remaining seeds.".format(seed, code)
            )


def _run_seeds_parallel(
    executable, script, base_argv, seeds, shared_timestamp, max_parallel, gpu_ids
):
    """Run seeds concurrently, up to ``max_parallel`` child processes at a time.

    Each worker is optionally pinned to a GPU by round-robin assignment of
    ``gpu_ids`` via ``CUDA_VISIBLE_DEVICES``. All seeds are attempted; if any
    child exits non-zero the sweep raises ``SystemExit`` after the rest finish.

    Args:
        executable: Path to the Python interpreter (``sys.executable``).
        script: Path to this script (``sys.argv[0]``).
        base_argv: Argv (without seed/timestamp flags) shared by every child.
        seeds: Ordered list of integer seeds to run.
        shared_timestamp: Timestamp shared by all children so they group under
            one experiment directory.
        max_parallel: Maximum number of concurrent child processes.
        gpu_ids: GPU ids to distribute workers across, or empty to leave
            ``CUDA_VISIBLE_DEVICES`` untouched.
    """
    import subprocess

    pending = list(enumerate(seeds))
    running = {}  # subprocess.Popen -> seed
    failures = []
    worker_slot = 0

    while pending or running:
        while pending and len(running) < max_parallel:
            index, seed = pending.pop(0)
            child_argv = _build_seed_child_argv(base_argv, seed, shared_timestamp)
            env = os.environ.copy()
            if gpu_ids:
                env["CUDA_VISIBLE_DEVICES"] = gpu_ids[worker_slot % len(gpu_ids)]
                worker_slot += 1
            print(
                "[seed-sweep] launching seed {} ({} of {}){}".format(
                    seed,
                    index + 1,
                    len(seeds),
                    " on GPU {}".format(env["CUDA_VISIBLE_DEVICES"]) if gpu_ids else "",
                )
            )
            process = subprocess.Popen([executable, script, *child_argv], env=env)
            running[process] = seed

        finished = None
        while finished is None:
            for process in list(running):
                code = process.poll()
                if code is not None:
                    finished = process
                    break
            if finished is None:
                time.sleep(1.0)

        seed = running.pop(finished)
        if finished.returncode != 0:
            failures.append((seed, finished.returncode))
            print(
                "[seed-sweep] seed {} failed with exit code {}.".format(
                    seed, finished.returncode
                )
            )

    if failures:
        detail = ", ".join(
            "seed {} (exit {})".format(seed, code) for seed, code in failures
        )
        raise SystemExit(
            "[seed-sweep] {} seed(s) failed: {}".format(len(failures), detail)
        )


def _build_seed_child_argv(base_argv, seed, shared_timestamp):
    """Append the per-seed single-run flags to the shared base argv."""
    return base_argv + [
        "--single-seed",
        "--seed",
        str(seed),
        "--timestamp",
        shared_timestamp,
    ]


def _strip_argv_flags(argv: List[str], flags: set) -> List[str]:
    """Drop the given flags (and their values) from an argv list.

    Handles both ``--flag value`` and ``--flag=value`` forms. ``--single-seed``
    is a boolean flag with no value; the others consume the following token.
    """
    boolean_flags = {"--single-seed"}
    result: List[str] = []
    i = 0
    while i < len(argv):
        tok = argv[i]
        name = tok.split("=", 1)[0]
        if name in flags:
            # Skip the following value token for non-boolean flags using the
            # space-separated form (i.e. no "=" embedded in this token).
            if name not in boolean_flags and "=" not in tok:
                i += 2
            else:
                i += 1
            continue
        result.append(tok)
        i += 1
    return result


def main():
    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument(
        "--config",
        action="append",
        default=[],
        metavar="FILE",
        help="YAML config fragment to apply (may be provided multiple times).",
    )
    config_parser.add_argument(
        "--config-dir",
        action="append",
        default=[],
        metavar="DIR",
        help="Directory of YAML fragments to apply in alphabetical order.",
    )
    config_parser.add_argument(
        "--no-config",
        action="store_true",
        help="Skip loading YAML configs and rely solely on CLI arguments.",
    )
    config_cli, remaining = config_parser.parse_known_args()

    config_chain: List[str] = []
    if not config_cli.no_config:
        # Apply defaults first so explicit model configs override them.
        config_chain.extend(config_cli.config_dir)
        config_chain.extend(config_cli.config)
        if not config_chain:
            config_chain = _default_main_config_chain()

    base_args = file_parser.parse_args_from_yaml(config_chain or None)
    parser = file_parser.get_parser()
    args = parser.parse_args(remaining, namespace=base_args)

    # Resolve the seed list. --single-seed forces the legacy single-run path
    # (one seed = args.seed); otherwise --seeds (default "0,39,55") drives a sweep.
    if getattr(args, "single_seed", False):
        seeds = [args.seed]
    else:
        seeds = _parse_seed_list(getattr(args, "seeds", "") or "")
        if not seeds:
            seeds = [args.seed]

    # When more than one seed is requested, act as a launcher: re-invoke this
    # script once per seed in a fresh process so each run starts with clean RNG,
    # CUDA, and global state. Each child is forced single-seed and shares one
    # timestamp so all seeds group under the same experiment directory.
    if len(seeds) > 1:
        if (getattr(args, "resume", "") or "").strip():
            raise SystemExit(
                "--resume targets a single experiment directory; pass "
                "--single-seed or a single --seeds value when resuming."
            )
        shared_timestamp = misc_utils.get_date_time()
        # Reconstruct the shared experiment directory (parent of the per-seed
        # dirs) using the same layout as misc_utils.log_dir().
        sweep_config_name = Path(config_chain[-1]).stem if config_chain else None
        dir_name = sweep_config_name if sweep_config_name else args.model
        experiment_root = os.path.join(
            args.log_dir, dir_name, "{}_{}".format(shared_timestamp, args.expt_name)
        )
        base_argv = _strip_argv_flags(
            sys.argv[1:],
            {"--seeds", "--seed", "--single-seed", "--timestamp", "--parallel-seeds"},
        )
        max_parallel = max(1, int(getattr(args, "parallel_seeds", 1) or 1))
        if max_parallel > 1:
            gpu_ids = _parse_seed_gpu_ids(getattr(args, "seed_gpu_ids", "") or "")
            print(
                "[seed-sweep] running {} seeds with up to {} in parallel{}".format(
                    len(seeds),
                    min(max_parallel, len(seeds)),
                    " across GPUs {}".format(",".join(gpu_ids)) if gpu_ids else "",
                )
            )
            _run_seeds_parallel(
                sys.executable,
                sys.argv[0],
                base_argv,
                seeds,
                shared_timestamp,
                max_parallel,
                gpu_ids,
            )
        else:
            _run_seeds_sequential(
                sys.executable, sys.argv[0], base_argv, seeds, shared_timestamp
            )
        _write_sweep_summary(experiment_root, seeds)
        raise SystemExit(0)

    # Single-seed run: ensure args.seed reflects the resolved seed.
    args.seed = seeds[0]

    # Task presentation order follows the training seed unless --task-order-seed
    # pins it. Resolved here so log_dir() records both the value and its origin.
    misc_utils.resolve_task_order_seed(args)

    # Scale learning rate based on batch size (reference batch size = 128).
    # This applies uniformly across all models that rely on args.lr.
    args.lr = misc_utils.scale_learning_rate_for_batch_size(args.lr, args.batch_size)

    # Setup logging early so we can mirror all prints to a log file.
    # Honor a timestamp passed down from the multi-seed launcher so all seeds
    # in a sweep share one experiment directory.
    timestamp = (
        getattr(args, "timestamp", "") or ""
    ).strip() or misc_utils.get_date_time()
    config_name = Path(config_chain[-1]).stem if config_chain else None

    # When resuming an interrupted run, reuse its existing log directory and
    # continue after the latest task checkpoint instead of starting fresh.
    resume_request = (getattr(args, "resume", "") or "").strip()
    if resume_request:
        resume_plan = _resolve_resume_plan(args, resume_request)
        args.log_dir = resume_plan["experiment_dir"]
        args.tf_dir = resume_plan["tf_dir"]
        args.resume_from_task = resume_plan["resume_from_task"]
        args.resume_checkpoint = resume_plan["checkpoint_path"]
    else:
        args.log_dir, args.tf_dir = misc_utils.log_dir(args, timestamp, config_name)
        args.resume_from_task = 0
        args.resume_checkpoint = None

    if getattr(args, "state_logging", False):
        enable_output_tee(
            os.path.join(args.log_dir, "terminal.log"), append=bool(resume_request)
        )
        log_state(
            args.state_logging,
            "Enabling terminal logging to {}".format(
                os.path.join(args.log_dir, "terminal.log")
            ),
        )

    print("New Experiment Starting...")
    print("Running model: ", args.model)
    log_state(
        args.state_logging,
        "Experiment '{}' starting with model '{}' (seed {})".format(
            args.expt_name, args.model, args.seed
        ),
    )

    # initialize seeds
    misc_utils.init_seed(args.seed)
    if args.cuda:
        log_state(
            args.state_logging,
            "Runtime accel: cudnn.benchmark={} amp={} amp_dtype={}".format(
                torch.backends.cudnn.benchmark,
                bool(getattr(args, "amp", False)),
                getattr(args, "amp_dtype", "bfloat16"),
            ),
        )

    # set up loader
    # 2 options: class_incremental and task_incremental
    # experiments in the paper only use task_incremental
    Loader = importlib.import_module("dataloaders." + args.loader)
    loader = Loader.IncrementalLoader(args, seed=args.seed)
    n_inputs, n_outputs, n_tasks = loader.get_dataset_info()
    args.n_tasks = n_tasks
    args.get_samples_per_task = getattr(loader, "get_samples_per_task", None)
    # Bound `IncrementalLoader.get_tasks`, for learners that need a task's *full*
    # data outside the training stream (WoE-SI's frozen-mu pre-pass, PR-3).
    # `get_tasks` rebuilds loaders from the retained per-task arrays and does not
    # advance the one-way `new_task()` cursor, so it is safe to call mid-run.
    args.get_task_train_loader = getattr(loader, "get_tasks", None)
    args.classes_per_task = getattr(loader, "classes_per_task", None)
    print("Classes per task:", args.classes_per_task)
    if args.classes_per_task is None or len(args.classes_per_task) == 0:
        args.classes_per_task = misc_utils.build_task_class_list(
            n_tasks,
            n_outputs,
            nc_per_task=(
                args.nc_per_task_list
                if getattr(args, "nc_per_task_list", "")
                else args.nc_per_task
            ),
            classes_per_task=getattr(args, "classes_per_task", None),
        )
        print("Built classes_per_task:", args.classes_per_task)
    log_state(
        args.state_logging,
        "Loader '{}' ready: {} inputs, {} outputs, {} tasks".format(
            args.loader, n_inputs, n_outputs, n_tasks
        ),
    )

    print("n_outputs:", n_outputs, "\tn_tasks:", n_tasks)

    log_state(args.state_logging, "Logging to {}".format(args.log_dir))

    # load model
    Model = importlib.import_module("model." + args.model)
    model = Model.Net(n_inputs, n_outputs, n_tasks, args)
    # Per-task BatchNorm running statistics for task-incremental runs. Must run
    # before ``.cuda()`` and after the model built its optimizer (the converted
    # layers reuse the existing affine Parameters, so param groups stay valid).
    task_bn.install(model, args, n_tasks)
    # print(model)
    if args.cuda:
        try:
            model.cuda()
        except RuntimeError:
            pass
    print(args.cuda)
    print("Model device:", next(model.parameters()).device)
    log_state(
        args.state_logging,
        "Model initialized on device {}".format(next(model.parameters()).device),
    )

    # Restore weights from the resume checkpoint before training continues.
    if getattr(args, "resume_checkpoint", None):
        _load_checkpoint_into_model(model, args.resume_checkpoint, args)
        log_state(
            args.state_logging,
            "Resumed model weights from {}; training continues at task {}".format(
                args.resume_checkpoint, args.resume_from_task
            ),
        )

    # run model on loader (iid2 included: it is the maximal-replay upper bound)
    log_state(args.state_logging, "Invoking continual life experience flow")
    (
        result_val_t,
        result_val_a,
        result_val_prec,
        result_val_f1,
        result_test_t,
        result_test_a,
        spent_time,
        headline,
    ) = life_experience(model, loader, args)

    spent_time_hours = spent_time / 3600.0

    # save results in files or print on terminal
    _, _, val_bwt = save_results(
        args,
        result_val_t,
        result_val_a,
        result_val_prec,
        result_val_f1,
        result_test_t,
        result_test_a,
        model,
        spent_time,
        headline=headline,
    )
    log_state(
        args.state_logging,
        "Results saved; total runtime {:.2f}h".format(spent_time_hours),
    )

    # Print and append total runtime for this experiment.
    print("Total runtime: {:.2f} hours".format(spent_time / 3600.0))
    results_txt_path = os.path.join(args.log_dir, "results.txt")
    try:
        with open(results_txt_path, "a", encoding="utf-8") as results_file:
            results_file.write("total_runtime_seconds: {:.3f}\n".format(spent_time))
    except OSError:
        # If results.txt cannot be written, fail silently to avoid breaking experiments.
        pass

    # Emit a machine-readable per-seed metrics file for the sweep summary.
    _write_seed_metrics(args, spent_time, headline, val_bwt)


if __name__ == "__main__":
    main()
