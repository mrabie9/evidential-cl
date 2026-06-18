# TODO: hyperparameter tuner

import importlib
import datetime
import argparse
import atexit
import time
import os
import sys
from contextlib import nullcontext
from pathlib import Path
from typing import List

from tqdm import tqdm

import numpy as np
import torch
from torch.autograd import Variable

import parser as file_parser
from metrics.metrics import confusion_matrix
from utils import misc_utils
from utils.training_metrics import (
    macro_f1_including_noise,
    macro_precision_signal_only,
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


def _split_labels(y):
    """Extract and return class labels from a batch label object.

    Supports multiple label formats:
    - `(y_cls, det_targets)` tuples/lists: returns `y_cls`
    - dict-like labels with `y_cls` or `y`: returns `y_cls`
    - 2D arrays/tensors shaped `[N, 2]`: returns the first column (class label)
    - otherwise: returns `y` as-is
    """
    if isinstance(y, dict):
        return y.get("y_cls", y.get("y"))
    if isinstance(y, (tuple, list)) and len(y) == 2:
        return y[0]
    if isinstance(y, np.ndarray) and y.ndim == 2 and y.shape[1] == 2:
        return y[:, 0]
    if torch.is_tensor(y) and y.dim() == 2 and y.size(1) == 2:
        return y[:, 0]
    return y


def _split_eval_output(output):
    """Return (cls_rec, cls_prec, cls_f1, det, fa). Missing values are None."""
    if isinstance(output, (tuple, list)):
        if len(output) == 5:
            return output[0], output[1], output[2], output[3], output[4]
        if len(output) == 3:
            return output[0], None, None, output[1], output[2]
        if len(output) == 2:
            return output[0], None, None, output[1], None
    return output, None, None, None, None


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


def _get_det_logits(model, xb, t):
    if hasattr(model, "forward_heads"):
        det_logits, _ = model.forward_heads(xb)
        return det_logits
    if hasattr(model, "net") and hasattr(model.net, "forward_heads"):
        det_logits, _ = model.net.forward_heads(xb)
        return det_logits
    if (
        hasattr(model, "net")
        and hasattr(model.net, "forward_features")
        and hasattr(model.net, "forward_detection")
    ):
        feats = model.net.forward_features(xb)
        return model.net.forward_detection(feats)
    return None


def _false_alarm_rate(preds: torch.Tensor, targets: torch.Tensor) -> float:
    neg_mask = targets == 0
    if not neg_mask.any():
        print(
            "Warning: No negative samples in _false_alarm_rate calculation, returning 0.0"
        )
        return 0.0
    neg_targets = targets[neg_mask]  # true noise label
    neg_preds = preds[neg_mask]  # predicted noise label
    fp = (neg_preds == 1).sum().item()  # predicted noise but actually signal
    tn = (neg_targets == 0).sum().item() - fp  # predicted noise and actually noise
    denom = fp + tn
    return float(fp / denom) if denom > 0 else -1


def _noise_label_for_task(
    args, task_idx: int, class_counts: List[int] | None = None
) -> int | None:
    if class_counts is None:
        class_counts = getattr(args, "classes_per_task", None)
    if class_counts is None:
        return None
    _, offset2 = misc_utils.compute_offsets(task_idx, class_counts)
    return offset2 - 1  # Assume noise label is highest in task


def _noise_label_max_for_task(task: object) -> int | None:
    """Compute noise label as the maximum class label value in a task.

    This is the "largest label in each task" rule used for detection metrics.
    """
    task_labels = _extract_task_labels(task)
    if task_labels is not None:
        y_cls = _split_labels(task_labels)
        y_cls_np = _labels_to_numpy(y_cls).reshape(-1)
        if y_cls_np.size == 0:
            return None
        return int(np.max(y_cls_np))

    max_label: int | None = None
    for batch in task:  # type: ignore[assignment]
        if isinstance(batch, (list, tuple)) and len(batch) >= 2:
            _, y_batch = batch[:2]
        else:
            continue
        y_cls = _split_labels(y_batch)
        y_cls_tensor = y_cls if torch.is_tensor(y_cls) else torch.as_tensor(y_cls)
        if y_cls_tensor.numel() == 0:
            continue
        batch_max = int(y_cls_tensor.reshape(-1).max().item())
        max_label = batch_max if max_label is None else max(max_label, batch_max)
    return max_label


def _noise_label_for_metrics(args: object, task: object) -> int | None:
    """Resolve the noise class id for masking classification / detection metrics.

    IQ class-incremental loaders set ``args.noise_label`` once to a **global**
    index shared by every task. Eval and train-side metric masking must use that
    value. Falling back to the per-split maximum label is incorrect for CIL when
    the split omits noise or when the last signal class equals that maximum.

    When ``args.noise_label`` is unset, we keep the legacy behavior of using the
    largest label observed in the task's dataloader (older TIL setups).

    Args:
        args: Parsed experiment arguments (may carry ``noise_label``).
        task: Per-task dataloader or a ``(x, y, t)`` task tuple for evaluation.

    Returns:
        Noise class index in **global** label space, or ``None`` if unknown.
    """
    raw = getattr(args, "noise_label", None)
    if raw is not None:
        return int(raw)
    return _noise_label_max_for_task(task)


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
        y_cls = _split_labels(task_labels)
        y_cls_array = np.asarray(y_cls).reshape(-1)
        inferred_counts.append(int(np.unique(y_cls_array).size))
    return inferred_counts


def _maybe_print_eval_prediction_debug(
    task_index: int,
    all_predictions: List[torch.Tensor],
    all_targets: List[torch.Tensor],
    noise_label: int | None,
) -> None:
    """Print a compact eval prediction summary when debug mode is enabled.

    Args:
        task_index: Zero-based task id used in logging.
        all_predictions: Predicted class-id tensors accumulated across batches.
        all_targets: Ground-truth class-id tensors accumulated across batches.
        noise_label: Optional class id reserved for noise in the current task.

    Usage:
        _maybe_print_eval_prediction_debug(task_index, preds, targets, noise_label)
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

    noise_prediction_rate = float("nan")
    if noise_label is not None:
        noise_prediction_rate = float(
            (predictions == int(noise_label)).float().mean().item()
        )

    print(
        "[eval-debug] task={} noise_label={} noise_pred_rate={:.4f} pred_hist={} target_hist={} per_class_recall={}".format(
            task_index,
            noise_label,
            noise_prediction_rate,
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
    metric_det_recall: float,
    metric_det_false_alarm: float,
    predictions: torch.Tensor,
    labels_for_metrics: torch.Tensor,
    noise_label_for_metrics: int | None,
) -> None:
    """Print per-batch train metric tensors when debug mode is enabled.

    Args:
        task_index: Zero-based task id.
        epoch_index: Zero-based epoch index.
        batch_index: Zero-based batch index.
        observe_cls_recall: Recall returned by ``model.observe``.
        metric_cls_recall: Recall recomputed in training loop.
        metric_precision: Signal-only precision in training loop.
        metric_f1: Macro F1 including noise class.
        metric_det_recall: Detection recall computed in training loop.
        metric_det_false_alarm: Detection false-alarm rate computed in training loop.
        predictions: Argmax class predictions used for train metrics.
        labels_for_metrics: Class labels used for train metrics (already task-local).
        noise_label_for_metrics: Optional task-local noise label.

    Usage:
        _maybe_print_train_metric_debug(..., pb, y_cls_for_metric, noise_label)
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

    if noise_label_for_metrics is None:
        signal_mask = torch.ones_like(labels_cpu, dtype=torch.bool)
    else:
        signal_mask = labels_cpu != int(noise_label_for_metrics)
    signal_count = int(signal_mask.sum().item())
    signal_prediction_unique = predictions_cpu[signal_mask].unique(sorted=True).tolist()
    signal_label_unique = labels_cpu[signal_mask].unique(sorted=True).tolist()
    noise_prediction_rate = (
        float((predictions_cpu == int(noise_label_for_metrics)).float().mean().item())
        if noise_label_for_metrics is not None
        else float("nan")
    )

    print(
        "[train-debug] task={} ep={} batch={} observe_rec={:.4f} metric_rec={:.4f} prec={:.4f} f1={:.4f} det_rec={:.4f} det_fa={:.4f} noise_label={} noise_pred_rate={:.4f} signal_n={} uniq_pred_signal={} uniq_y_signal={} pred_hist={} y_hist={}".format(
            task_index,
            epoch_index + 1,
            batch_index + 1,
            float(observe_cls_recall),
            float(metric_cls_recall),
            float(metric_precision),
            float(metric_f1),
            float(metric_det_recall),
            float(metric_det_false_alarm),
            noise_label_for_metrics,
            noise_prediction_rate,
            signal_count,
            signal_prediction_unique,
            signal_label_unique,
            prediction_histogram.tolist(),
            label_histogram.tolist(),
        )
    )


def _maybe_print_eval_detection_alignment_debug(
    task_index: int,
    batch_index: int,
    yb_cls_for_metrics: torch.Tensor,
    predictions: torch.Tensor,
    noise_label_for_metrics: int | None,
) -> None:
    """Print one-line eval detection alignment diagnostics when enabled.

    Args:
        task_index: Zero-based task id for the current eval loop.
        batch_index: Zero-based batch index inside the eval task loader.
        yb_cls_for_metrics: Task-local labels used by eval metrics.
        predictions: Class predictions used to derive detection decisions.
        noise_label_for_metrics: Task-local noise label or ``None``.

    Usage:
        _maybe_print_eval_detection_alignment_debug(t, i, y, pb, noise_label)
    """
    debug_enabled = os.getenv("LA_MAML_EVAL_DET_DEBUG", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    if not debug_enabled:
        return
    if batch_index != 0:
        return

    labels_cpu = yb_cls_for_metrics.detach().cpu().long()
    predictions_cpu = predictions.detach().cpu().long()
    if labels_cpu.numel() == 0 or predictions_cpu.numel() == 0:
        return

    if noise_label_for_metrics is None:
        det_targets = torch.ones_like(labels_cpu, dtype=torch.long)
        det_predictions = torch.ones_like(predictions_cpu, dtype=torch.long)
        noise_prediction_rate = float("nan")
    else:
        det_targets = (labels_cpu != int(noise_label_for_metrics)).long()
        det_predictions = (predictions_cpu != int(noise_label_for_metrics)).long()
        noise_prediction_rate = float(
            (predictions_cpu == int(noise_label_for_metrics)).float().mean().item()
        )

    print(
        "[eval-det-debug] task={} batch={} noise_label={} uniq_y={} uniq_pred={} noise_pred_rate={:.4f} det_target_signal_rate={:.4f} det_pred_signal_rate={:.4f}".format(
            task_index,
            batch_index + 1,
            noise_label_for_metrics,
            labels_cpu.unique(sorted=True).tolist(),
            predictions_cpu.unique(sorted=True).tolist(),
            noise_prediction_rate,
            float(det_targets.float().mean().item()),
            float(det_predictions.float().mean().item()),
        )
    )


def eval_tasks(model, tasks, args, specific_task=None, eval_epistemic=False):
    model.eval()
    device = torch.device(
        "cuda" if getattr(args, "cuda", False) and torch.cuda.is_available() else "cpu"
    )
    results = []
    prec_results = []
    f1_results = []
    class_counts = _infer_class_counts_from_tasks(tasks)
    if class_counts is None:
        class_counts = getattr(args, "classes_per_task", None)

    if specific_task is not None:
        tasks = [tasks[specific_task]]

    det_results = []
    det_fa_results = []
    det_metrics_active = False
    for task_position, task in enumerate(tasks):
        t = task_position
        recalls = []
        precisions = []
        f1s = []
        det_recalls = []
        det_false_alarms = []
        eval_debug_predictions: List[torch.Tensor] = []
        eval_debug_targets: List[torch.Tensor] = []
        noise_label = _noise_label_for_metrics(args, task)
        task_noise_label_for_metrics = noise_label
        for batch_index, batch in enumerate(task):
            if isinstance(batch, (list, tuple)) and len(batch) == 3:
                xb, yb, _ = batch
            else:
                xb, yb = batch
            xb = xb.to(device)
            if getattr(args, "arch", "").lower() == "linear":
                xb = xb.view(xb.size(0), -1)
            yb_cls = _split_labels(yb)
            if not torch.is_tensor(yb_cls):
                yb_cls = torch.as_tensor(yb_cls)

            logits = model_forward_for_metric_loop(model, xb, t, args)
            pb = torch.argmax(logits, dim=1).cpu()
            yb_cls_cpu = yb_cls.detach().cpu()
            yb_cls_for_metrics = yb_cls_cpu
            noise_label_for_metrics = noise_label
            # Task-incremental learners (UCL and any model exposing ``split``) emit
            # task-local logits; shift global labels the same way as the training
            # metric loop in ``life_experience`` (``model.compute_offsets``).
            if getattr(model, "split", False):
                compute_offsets_fn = getattr(model, "compute_offsets", None)
                if callable(compute_offsets_fn):
                    offset1, _ = compute_offsets_fn(t)
                else:
                    offset1, _ = misc_utils.compute_offsets(
                        t,
                        class_counts if class_counts is not None else args.nc_per_task,
                    )
                if getattr(args, "use_detector_arch", False):
                    yb_cls_for_metrics = yb_cls_cpu.clone()
                    if noise_label_for_metrics is not None:
                        signal_mask = yb_cls_for_metrics != noise_label_for_metrics
                        if signal_mask.any():
                            yb_cls_for_metrics[signal_mask] = (
                                yb_cls_for_metrics[signal_mask] - offset1
                            )
                else:
                    yb_cls_for_metrics = yb_cls_cpu - offset1
                    if noise_label_for_metrics is not None:
                        noise_label_for_metrics = noise_label_for_metrics - offset1
            task_noise_label_for_metrics = noise_label_for_metrics

            eval_debug_predictions.append(pb)
            eval_debug_targets.append(yb_cls_for_metrics)
            _maybe_print_eval_detection_alignment_debug(
                task_index=t,
                batch_index=batch_index,
                yb_cls_for_metrics=yb_cls_for_metrics,
                predictions=pb,
                noise_label_for_metrics=noise_label_for_metrics,
            )
            # Record total F1 score for all classes including noise
            if not getattr(args, "use_detector_arch", False):
                f1s.append(macro_f1_including_noise(pb, yb_cls_for_metrics))
            else:
                print("[WARNING] F1 not supported for detection architecture.")
                f1s.append(0.0)

            if noise_label_for_metrics is not None:
                cls_mask = yb_cls_for_metrics != noise_label_for_metrics
                if cls_mask.any():
                    recalls.append(
                        macro_recall(pb[cls_mask], yb_cls_for_metrics[cls_mask])
                    )
                    precisions.append(
                        macro_precision_signal_only(
                            pb[cls_mask],
                            yb_cls_for_metrics[cls_mask],
                            noise_label_for_metrics,
                        )
                    )
            else:
                recalls.append(macro_recall(pb, yb_cls_for_metrics))
                precisions.append(
                    macro_precision_signal_only(
                        pb, yb_cls_for_metrics, noise_label_for_metrics
                    )
                )

            if noise_label_for_metrics is not None:
                det_targets = (yb_cls_for_metrics != noise_label_for_metrics).long()
                det_logits = None
                if det_logits is not None:
                    det_pred = (det_logits >= 0).long().cpu()
                else:
                    det_pred = (pb != noise_label_for_metrics).long()
                det_recalls.append(macro_recall(det_pred, det_targets))
                det_false_alarms.append(_false_alarm_rate(det_pred, det_targets))

        results.append(sum(recalls) / len(recalls) if recalls else 0.0)
        prec_results.append(sum(precisions) / len(precisions) if precisions else 0.0)
        f1_results.append(sum(f1s) / len(f1s) if f1s else 0.0)
        if det_recalls:
            det_results.append(sum(det_recalls) / len(det_recalls))
            det_fa_results.append(sum(det_false_alarms) / len(det_false_alarms))
            det_metrics_active = True
        else:
            det_results.append(0.0)
            det_fa_results.append(0.0)

        _maybe_print_eval_prediction_debug(
            task_index=t,
            all_predictions=eval_debug_predictions,
            all_targets=eval_debug_targets,
            noise_label=task_noise_label_for_metrics,
        )

    if det_metrics_active:
        return results, prec_results, f1_results, det_results, det_fa_results
    return results, prec_results, f1_results, None, None


def eval_class_tasks(model, tasks, args, **kwargs):
    """Evaluate class-incremental runs with the same metrics as :func:`eval_tasks`.

    The previous implementation returned only coarse per-task accuracy and
    ``None`` for precision, F1, and detection, which made zero-shot / val
    log lines show ``nan`` for those fields.
    """

    return eval_tasks(
        model,
        tasks,
        args,
        specific_task=kwargs.get("specific_task"),
        eval_epistemic=kwargs.get("eval_epistemic", False),
    )


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


def life_experience(model, inc_loader, args):
    result_val_a = []
    result_test_a = []
    result_val_prec = []
    result_val_f1 = []
    result_val_det_a = []
    result_test_det_a = []
    result_val_det_fa = []
    result_test_det_fa = []

    result_val_t = []
    result_test_t = []

    last_tr_cls_rec = last_tr_cls_prec = last_tr_cls_f1 = None
    last_tr_det = last_tr_fa = None
    base_n_epochs = int(args.n_epochs)
    force_global_n_epochs_legacy = bool(LEGACY_USE_GLOBAL_N_EPOCHS)
    task_epoch_schedule = (
        {}
        if force_global_n_epochs_legacy
        else _task_epoch_schedule_for_base_epochs(base_n_epochs)
    )
    args.task_epoch_schedule = task_epoch_schedule

    time_start = time.time()
    train_task_loaders = []
    test_task_loaders = []
    evaluator = eval_tasks
    if args.loader == "class_incremental_loader":
        evaluator = eval_class_tasks

    interactive_terminal = sys.stdout.isatty()
    amp_dtype = (
        torch.bfloat16
        if getattr(args, "amp_dtype", "bfloat16") == "bfloat16"
        else torch.float16
    )
    use_amp = bool(getattr(args, "amp", False) and args.cuda)
    if getattr(args, "model", "") == "eucr":
        use_amp = False
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
        train_task_loaders.append(train_loader)
        test_task_loaders.append(test_loader)
        current_task = task_info["task"]

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
        noise_label_for_task = _noise_label_for_metrics(args, train_loader)

        log_state(
            args.state_logging,
            "Task {}: zero-shot validation (pre-train)".format(current_task),
        )
        zero_shot_raw = evaluator(model, test_task_loaders, args)
        zs_rec, zs_prec, zs_f1, zs_det, zs_pfa = _split_eval_output(zero_shot_raw)
        num_tasks_now = len(test_task_loaders)
        current_task_idx = task_info["task"]
        zero_shot_rec_cls = _scalar_metric_at_task_index(zs_rec, current_task_idx)
        zero_shot_prec_cls = _scalar_metric_at_task_index(zs_prec, current_task_idx)
        zero_shot_f1_cls = _scalar_metric_at_task_index(zs_f1, current_task_idx)
        zero_shot_det = _scalar_metric_at_task_index(zs_det, current_task_idx)
        zero_shot_pfa = _scalar_metric_at_task_index(zs_pfa, current_task_idx)
        zero_shot_total_f1 = _mean_metric_across_tasks(zs_f1)
        zero_shot_per_task_rec_cls = _per_task_metric_array(zs_rec, num_tasks_now)
        zero_shot_per_task_prec_cls = _per_task_metric_array(zs_prec, num_tasks_now)
        zero_shot_per_task_f1_cls = _per_task_metric_array(zs_f1, num_tasks_now)
        zero_shot_per_task_det = _per_task_metric_array(zs_det, num_tasks_now)
        zero_shot_per_task_pfa = _per_task_metric_array(zs_pfa, num_tasks_now)
        print(
            "---- Zero-shot (pre-train) task {}: rec_cls {:.4f} | prec_cls {:.4f} | f1_cls {:.4f} | det {:.4f} | pfa {:.4f} | total_f1 {:.4f} ----".format(
                current_task,
                zero_shot_rec_cls,
                zero_shot_prec_cls,
                zero_shot_f1_cls,
                zero_shot_det,
                zero_shot_pfa,
                zero_shot_total_f1,
            )
        )

        # Per-epoch training metrics for this task (classification + detection).
        per_epoch_train_cls_rec = []
        per_epoch_train_cls_prec = []
        per_epoch_train_det_rec = []
        per_epoch_train_det_pfa = []
        per_epoch_train_f1 = []

        # Per-evaluation validation metrics for this task (classification + detection).
        per_epoch_val_cls_rec = []
        per_epoch_val_cls_prec = []
        per_epoch_val_det_rec = []
        per_epoch_val_det_pfa = []
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
            epoch_det_recalls = []
            epoch_det_fas = []
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
                y_cls = _split_labels(y)
                if not torch.is_tensor(y_cls):
                    y_cls = torch.as_tensor(y_cls)

                # Hybrid mode: keep passing detector targets into `model.observe`
                # when the dataloader provides them, but only use class labels
                # for metric computation.
                if getattr(args, "use_detector_arch", False):
                    if isinstance(y, (tuple, list)) and len(y) == 2:
                        cls_part, det_part = y[0], y[1]
                        if not torch.is_tensor(cls_part):
                            cls_part = torch.as_tensor(cls_part)
                        if not torch.is_tensor(det_part):
                            det_part = torch.as_tensor(det_part)
                        v_y = (cls_part, det_part)
                    elif torch.is_tensor(y) and y.dim() == 2 and y.size(1) == 2:
                        v_y = (y[:, 0], y[:, 1])
                    elif isinstance(y, np.ndarray) and y.ndim == 2 and y.shape[1] == 2:
                        v_y = (y[:, 0], y[:, 1])
                    else:
                        v_y = y_cls
                else:
                    v_y = y_cls
                if args.cuda:
                    v_x = v_x.cuda()
                    if isinstance(v_y, (tuple, list)) and len(v_y) == 2:
                        v_y = (v_y[0].cuda(), v_y[1].cuda())
                    else:
                        v_y = v_y.cuda()
                model.train()
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
                # debug_noise_label = _noise_label_max_for_task(train_loader)
                # model.eval()
                # with torch.no_grad():
                #     debug_logits = (
                #         model.forward_training(v_x, task_info["task"])
                #         if args.model != "anml"
                #         else model(v_x, fast_weights=None)
                #     )
                #     debug_preds = torch.argmax(debug_logits, dim=1).cpu()
                #     debug_y_cls = _split_labels(v_y)
                #     debug_y_cls_cpu = (
                #         debug_y_cls.detach().cpu()
                #         if torch.is_tensor(debug_y_cls)
                #         else torch.as_tensor(debug_y_cls)
                #     )
                #     debug_mask = (
                #         debug_y_cls_cpu != debug_noise_label
                #         if debug_noise_label is not None
                #         else torch.ones_like(debug_y_cls_cpu, dtype=torch.bool)
                #     )
                #     debug_eval_recall = (
                #         macro_recall(debug_preds[debug_mask], debug_y_cls_cpu[debug_mask])
                #         if debug_mask.any()
                #         else 0.0
                #     )
                # model.train()
                # logits = model(x, task_i) if args.model != 'anml' else model(x, task_i, fast_weights=None)
                # pb = torch.argmax(logits, dim=1)
                # correct += (pb == y).sum().item()
                # cls_tr_rec = correct / x.size(0)
                result_acc_tr.append(cls_tr_rec)
                result_epoch_loss.append(loss)
                epoch_losses.append(loss)
                epoch_train_accs.append(cls_tr_rec)

                # Batch-level precision (signal only) and F1 (all classes incl. noise) for progress bar.
                # Prefer observe() predictions when available to avoid a second stochastic forward.
                noise_label = noise_label_for_task
                y_cls_for_metric = (
                    y_cls.cpu() if torch.is_tensor(y_cls) else torch.as_tensor(y_cls)
                )
                noise_label_for_metric = noise_label

                # For split (task-incremental) models, forward returns task-local logits so pb is in [0, C_t-1].
                # Convert labels to task-local so Train Acc / Prec / F1 match.
                if getattr(model, "split", False):
                    offset1, _ = model.compute_offsets(task_info["task"])
                    y_cls_for_metric = y_cls_for_metric - offset1
                    if noise_label_for_metric is not None:
                        noise_label_for_metric = noise_label_for_metric - offset1

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
                det_logits = None

                prec = macro_precision_signal_only(
                    pb, y_cls_for_metric, noise_label_for_metric
                )
                f1 = macro_f1_including_noise(pb, y_cls_for_metric)

                if noise_label_for_metric is not None:
                    cls_mask = y_cls_for_metric != noise_label_for_metric
                    if cls_mask.any():
                        cls_tr_rec = macro_recall(
                            pb[cls_mask], y_cls_for_metric[cls_mask]
                        )
                    else:
                        cls_tr_rec = 0.0
                else:
                    cls_tr_rec = macro_recall(pb, y_cls_for_metric)

                det_rec = 0.0
                det_fa = 0.0
                if noise_label_for_metric is not None:
                    det_targets = (y_cls_for_metric != noise_label_for_metric).long()
                    if det_logits is not None:
                        det_pred = (det_logits >= 0).long().cpu()
                    else:
                        det_pred = (pb != noise_label_for_metric).long()
                    det_rec = macro_recall(det_pred, det_targets)
                    det_fa = _false_alarm_rate(det_pred, det_targets)

                result_acc_tr[-1] = cls_tr_rec
                epoch_train_accs[-1] = cls_tr_rec

                epoch_precisions.append(prec)
                epoch_f1s.append(f1)
                epoch_det_recalls.append(det_rec)
                epoch_det_fas.append(det_fa)
                _maybe_print_train_metric_debug(
                    task_index=task_info["task"],
                    epoch_index=ep,
                    batch_index=i,
                    observe_cls_recall=observe_cls_tr_rec,
                    metric_cls_recall=cls_tr_rec,
                    metric_precision=prec,
                    metric_f1=f1,
                    metric_det_recall=det_rec,
                    metric_det_false_alarm=det_fa,
                    predictions=pb,
                    labels_for_metrics=y_cls_for_metric,
                    noise_label_for_metrics=noise_label_for_metric,
                )

                prog_bar.set_description(
                    "T{}| Ep: {}/{}| Loss: {}| Rec: {}| Prec: {}| F1: {}| DetRec: {}| DetFA: {}".format(
                        task_info["task"],
                        ep + 1,
                        task_n_epochs,
                        round(loss, 3),
                        round(cls_tr_rec, 2),
                        round(prec, 2),
                        round(f1, 2),
                        round(det_rec, 2),
                        round(det_fa, 2),
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
                val_acc = evaluator(model, test_task_loaders, args)
                val_acc, val_prec, val_f1, val_det_acc, val_det_fa = _split_eval_output(
                    val_acc
                )
                epoch_eval_time += time.time() - eval_start
                result_acc_val.append(val_acc)
                result_val_a.append(val_acc)
                if val_prec is not None:
                    result_val_prec.append(val_prec)
                if val_f1 is not None:
                    result_val_f1.append(val_f1)
                if val_det_acc is not None:
                    result_val_det_a.append(val_det_acc)
                    if isinstance(val_det_acc, (list, tuple)):
                        last_tr_det = (
                            sum(val_det_acc) / len(val_det_acc) if val_det_acc else None
                        )
                    else:
                        last_tr_det = float(val_det_acc)
                if val_det_fa is not None:
                    result_val_det_fa.append(val_det_fa)
                    if isinstance(val_det_fa, (list, tuple)):
                        last_tr_fa = (
                            sum(val_det_fa) / len(val_det_fa) if val_det_fa else None
                        )
                    else:
                        last_tr_fa = float(val_det_fa)
                result_val_t.append(task_info["task"])
                if val_det_acc is not None:
                    print(
                        "---- Eval at Epoch {}: cls {} | det_recall {} | det_fa {} ----".format(
                            ep, val_acc, val_det_acc, val_det_fa
                        )
                    )
                else:
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

                if val_det_acc is not None:
                    if isinstance(
                        val_det_acc, (list, tuple)
                    ) and current_task_idx < len(val_det_acc):
                        per_epoch_val_det_rec.append(
                            float(val_det_acc[current_task_idx])
                        )
                    elif not isinstance(val_det_acc, (list, tuple)):
                        per_epoch_val_det_rec.append(float(val_det_acc))
                    else:
                        per_epoch_val_det_rec.append(float("nan"))
                else:
                    per_epoch_val_det_rec.append(float("nan"))

                if val_det_fa is not None:
                    if isinstance(val_det_fa, (list, tuple)) and current_task_idx < len(
                        val_det_fa
                    ):
                        per_epoch_val_det_pfa.append(
                            float(val_det_fa[current_task_idx])
                        )
                    elif not isinstance(val_det_fa, (list, tuple)):
                        per_epoch_val_det_pfa.append(float(val_det_fa))
                    else:
                        per_epoch_val_det_pfa.append(float("nan"))
                else:
                    per_epoch_val_det_pfa.append(float("nan"))

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
            avg_det_rec = (
                float(sum(epoch_det_recalls) / len(epoch_det_recalls))
                if epoch_det_recalls
                else float("nan")
            )
            avg_det_fa = (
                float(sum(epoch_det_fas) / len(epoch_det_fas))
                if epoch_det_fas
                else float("nan")
            )

            # Track the last training metrics we saw (for summary logging).
            last_tr_cls_rec = avg_cls_tr_rec
            last_tr_cls_prec = avg_prec
            last_tr_cls_f1 = avg_f1

            # Persist per-epoch training metrics for this task.
            per_epoch_train_cls_rec.append(avg_cls_tr_rec)
            per_epoch_train_cls_prec.append(avg_prec)
            per_epoch_train_det_rec.append(avg_det_rec)
            per_epoch_train_det_pfa.append(avg_det_fa)
            per_epoch_train_f1.append(avg_f1)

            if not interactive_terminal:
                print(
                    "T{} Ep {}/{} | L {:.4f} | Train Acc {:.2f} | Prec {:.2f} | F1 {:.2f} | Det Rec {:.2f} | Det FA {:.2f} | Epoch Time {:.2f}s (Eval {:.2f}s, Train {:.2f}s)".format(
                        task_info["task"],
                        ep + 1,
                        task_n_epochs,
                        avg_loss,
                        avg_cls_tr_rec,
                        avg_prec,
                        avg_f1,
                        avg_det_rec,
                        avg_det_fa,
                        epoch_duration,
                        epoch_eval_time,
                        epoch_train_time,
                    )
                )
                log_state(
                    args.state_logging,
                    "T{} Ep {}/{} complete: Prec {:.4f} F1 {:.4f} DetRec {:.4f} DetFA {:.4f} | {:.2f}s total ({:.2f}s eval/{:.2f}s train)".format(
                        current_task,
                        ep + 1,
                        task_n_epochs,
                        avg_prec,
                        avg_f1,
                        avg_det_rec,
                        avg_det_fa,
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
        finalize_fn = getattr(model, "finalize_task_after_training", None)
        if callable(finalize_fn):
            finalize_fn(train_loader)
        log_state(
            args.state_logging,
            "Task {}: running final validation.".format(current_task),
        )
        val_acc = evaluator(model, test_task_loaders, args)
        val_acc, val_prec, val_f1, val_det_acc, val_det_fa = _split_eval_output(val_acc)
        result_val_a.append(val_acc)
        if val_prec is not None:
            result_val_prec.append(val_prec)
        if val_f1 is not None:
            result_val_f1.append(val_f1)
        if val_det_acc is not None:
            result_val_det_a.append(val_det_acc)
        if val_det_fa is not None:
            result_val_det_fa.append(val_det_fa)
        result_val_t.append(task_info["task"])

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
            "cls_tr_rec": result_acc_tr,
            "val_acc": result_acc_val,
            "n_epochs": np.int64(task_n_epochs),
            "zero_shot_rec_cls": np.float64(zero_shot_rec_cls),
            "zero_shot_prec_cls": np.float64(zero_shot_prec_cls),
            "zero_shot_f1_cls": np.float64(zero_shot_f1_cls),
            "zero_shot_det": np.float64(zero_shot_det),
            "zero_shot_pfa": np.float64(zero_shot_pfa),
            "zero_shot_total_f1": np.float64(zero_shot_total_f1),
            "zero_shot_per_task_rec_cls": zero_shot_per_task_rec_cls,
            "zero_shot_per_task_prec_cls": zero_shot_per_task_prec_cls,
            "zero_shot_per_task_f1_cls": zero_shot_per_task_f1_cls,
            "zero_shot_per_task_det": zero_shot_per_task_det,
            "zero_shot_per_task_pfa": zero_shot_per_task_pfa,
        }
        if result_val_f1_flat is not None:
            save_payload["val_f1"] = result_val_f1_flat
        # Optional: per-epoch training metrics for this task.
        if per_epoch_train_cls_rec:
            save_payload["train_cls_rec"] = np.asarray(
                per_epoch_train_cls_rec, dtype=float
            )
        if per_epoch_train_cls_prec:
            save_payload["train_cls_prec"] = np.asarray(
                per_epoch_train_cls_prec, dtype=float
            )
        if per_epoch_train_det_rec:
            save_payload["train_det_rec"] = np.asarray(
                per_epoch_train_det_rec, dtype=float
            )
        if per_epoch_train_det_pfa:
            save_payload["train_det_pfa"] = np.asarray(
                per_epoch_train_det_pfa, dtype=float
            )
        if per_epoch_train_f1:
            save_payload["train_f1"] = np.asarray(per_epoch_train_f1, dtype=float)

        # Optional: per-evaluation validation metrics for this task (one entry per eval/epoch).
        if per_epoch_val_cls_rec:
            save_payload["val_cls_rec"] = np.asarray(per_epoch_val_cls_rec, dtype=float)
        if per_epoch_val_cls_prec:
            save_payload["val_cls_prec"] = np.asarray(
                per_epoch_val_cls_prec, dtype=float
            )
        if per_epoch_val_det_rec:
            save_payload["val_det_rec"] = np.asarray(per_epoch_val_det_rec, dtype=float)
        if per_epoch_val_det_pfa:
            save_payload["val_det_pfa"] = np.asarray(per_epoch_val_det_pfa, dtype=float)
        if per_epoch_val_f1:
            save_payload["val_f1_per_epoch"] = np.asarray(per_epoch_val_f1, dtype=float)

        if result_val_det_a:
            save_payload["val_det_acc"] = np.array(result_val_det_a[-1])
        if result_val_det_fa:
            save_payload["val_det_fa"] = np.array(result_val_det_fa[-1])

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
            test_acc = evaluator(model, test_task_loaders, args)
            test_acc, test_prec, test_f1, test_det_acc, test_det_fa = (
                _split_eval_output(test_acc)
            )
            result_test_a.append(test_acc)
            if test_det_acc is not None:
                result_test_det_a.append(test_det_acc)
            if test_det_fa is not None:
                result_test_det_fa.append(test_det_fa)
            result_test_t.append(task_info["task"])

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
    if result_val_det_a:
        print(
            "Final Detection Results:- \n Total Detection: {} \n Individual Detection: {}".format(
                sum(result_val_det_a[-1]) / len(result_val_det_a[-1]),
                result_val_det_a[-1],
            )
        )
    if result_val_det_fa:
        print(
            "Final Detection False Alarm:- \n Total False Alarm: {} \n Individual False Alarm: {}".format(
                sum(result_val_det_fa[-1]) / len(result_val_det_fa[-1]),
                result_val_det_fa[-1],
            )
        )

    def _mean(x):
        if x is None or (isinstance(x, (list, tuple)) and len(x) == 0):
            return None
        if isinstance(x, (list, tuple)):
            return sum(float(v) for v in x) / len(x)
        return float(x)

    if (
        last_tr_cls_rec is not None
        or last_tr_cls_prec is not None
        or last_tr_cls_f1 is not None
    ):
        tr_rec = float(last_tr_cls_rec) if last_tr_cls_rec is not None else None
        tr_prec = float(last_tr_cls_prec) if last_tr_cls_prec is not None else None
        tr_f1 = float(last_tr_cls_f1) if last_tr_cls_f1 is not None else None
        tr_det = last_tr_det
        tr_fa = last_tr_fa
        parts = []
        if tr_rec is not None:
            parts.append("cls_rec={:.4f}".format(tr_rec))
        if tr_prec is not None:
            parts.append("cls_prec={:.4f}".format(tr_prec))
        if tr_f1 is not None:
            parts.append("cls_f1={:.4f}".format(tr_f1))
        if tr_det is not None:
            parts.append("det={:.4f}".format(tr_det))
        if tr_fa is not None:
            parts.append("fa={:.4f}".format(tr_fa))
        if parts:
            print("SUMMARY_TR " + " ".join(parts))

    if result_val_a:
        te_rec = _mean(result_val_a[-1])
        te_prec = _mean(result_val_prec[-1]) if result_val_prec else None
        te_f1 = _mean(result_val_f1[-1]) if result_val_f1 else None
        te_det = _mean(result_val_det_a[-1]) if result_val_det_a else None
        te_fa = _mean(result_val_det_fa[-1]) if result_val_det_fa else None
        parts = ["cls_rec={:.4f}".format(te_rec)]
        if te_prec is not None:
            parts.append("cls_prec={:.4f}".format(te_prec))
        if te_f1 is not None:
            parts.append("cls_f1={:.4f}".format(te_f1))
        if te_det is not None:
            parts.append("det={:.4f}".format(te_det))
        if te_fa is not None:
            parts.append("fa={:.4f}".format(te_fa))
        print("SUMMARY_TE " + " ".join(parts))

    if args.calc_test_accuracy:
        print("####Final Test Accuracy####")
        print(
            "Final Results:- \n Total Accuracy: {} \n Individual Accuracy: {}".format(
                sum(result_test_a[-1]) / len(result_test_a[-1]), result_test_a[-1]
            )
        )
        if result_test_det_a:
            print(
                "Final Detection Results:- \n Total Detection: {} \n Individual Detection: {}".format(
                    sum(result_test_det_a[-1]) / len(result_test_det_a[-1]),
                    result_test_det_a[-1],
                )
            )
        if result_test_det_fa:
            print(
                "Final Detection False Alarm:- \n Total False Alarm: {} \n Individual False Alarm: {}".format(
                    sum(result_test_det_fa[-1]) / len(result_test_det_fa[-1]),
                    result_test_det_fa[-1],
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
        torch.Tensor(result_test_t),
        _pad_results(result_test_a),
        _pad_results(result_val_det_a),
        _pad_results(result_val_det_fa),
        _pad_results(result_test_det_a),
        _pad_results(result_test_det_fa),
        time_spent,
    )


def estimate_memory_buffer_size_bytes(model: torch.nn.Module) -> int:
    """Estimate total bytes used by replay/memory buffers in a model.

    This scans all modules for tensor attributes whose names suggest they are
    part of a replay or memory buffer (for example, attributes containing
    ``\"mem\"``) while excluding tensors already counted as parameters and
    avoiding double-counting shared storages.

    Args:
        model: Torch module whose memory/replay buffers will be inspected.

    Returns:
        Total number of bytes occupied by the matching tensors.

    Usage:
        buffer_bytes = estimate_memory_buffer_size_bytes(model)
    """
    parameter_data_ids = {id(parameter.data) for parameter in model.parameters()}
    seen_tensor_ids: set[int] = set()
    total_bytes = 0

    for module in model.modules():
        for attribute_name, value in vars(module).items():
            if not torch.is_tensor(value):
                continue
            if "mem" not in attribute_name.lower():
                continue
            tensor_data = value
            tensor_id = id(tensor_data)
            if tensor_id in seen_tensor_ids or tensor_id in parameter_data_ids:
                continue
            seen_tensor_ids.add(tensor_id)
            total_bytes += tensor_data.numel() * tensor_data.element_size()

    return total_bytes


def save_results(
    args, result_val_t, result_val_a, result_test_t, result_test_a, model, spent_time
):
    fname = os.path.join(args.log_dir, "results")
    log_state(args.state_logging, "Saving results to {}".format(fname))

    size_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    size_gb = size_bytes / (1024**3)
    buffer_bytes = estimate_memory_buffer_size_bytes(model)
    buffer_gb = buffer_bytes / (1024**3)
    print("Model size: {:.4f} GB".format(size_gb))
    print("Memory buffer size: {:.4f} GB".format(buffer_gb))

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
    one_liner += " # sizes: model_gb={:.4f} mem_gb={:.4f}".format(size_gb, buffer_gb)

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
    if hasattr(args, "get_samples_per_task"):
        try:
            delattr(args, "get_samples_per_task")
        except AttributeError:
            args.get_samples_per_task = None

    torch.save(
        (result_val_t, result_val_a, state_dict, val_stats, one_liner, args),
        fname + ".pt",
        pickle_protocol=4,
    )
    return val_stats, test_stats


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

        import subprocess

        shared_timestamp = misc_utils.get_date_time()
        base_argv = _strip_argv_flags(
            sys.argv[1:], {"--seeds", "--seed", "--single-seed", "--timestamp"}
        )
        for idx, seed in enumerate(seeds):
            child_argv = base_argv + [
                "--single-seed",
                "--seed",
                str(seed),
                "--timestamp",
                shared_timestamp,
            ]
            print(
                "[seed-sweep] launching seed {} ({} of {})".format(
                    seed, idx + 1, len(seeds)
                )
            )
            code = subprocess.call([sys.executable, sys.argv[0], *child_argv])
            if code != 0:
                raise SystemExit(
                    "[seed-sweep] seed {} failed with exit code {}; "
                    "aborting remaining seeds.".format(seed, code)
                )
        raise SystemExit(0)

    # Single-seed run: ensure args.seed reflects the resolved seed.
    args.seed = seeds[0]

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
    args.get_samples_per_task = getattr(loader, "get_samples_per_task", None)
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

    # run model on loader
    if args.model == "iid2":
        # `iid2` is handled by the single-round entrypoint; delegate so we
        # never depend on `main_multi_task.py`.
        #
        # We preserve CLI compatibility by forwarding the original argv.
        import subprocess

        log_state(
            args.state_logging,
            "Delegating iid2 to main_single_round.py (no main_multi_task).",
        )
        exit_code = subprocess.call(
            [sys.executable, "main_single_round.py"] + sys.argv[1:]
        )
        raise SystemExit(exit_code)
    else:
        # for all the CL baselines
        log_state(args.state_logging, "Invoking continual life experience flow")
        (
            result_val_t,
            result_val_a,
            result_test_t,
            result_test_a,
            _,
            _,
            _,
            _,
            spent_time,
        ) = life_experience(model, loader, args)

        spent_time_hours = spent_time / 3600.0

        # save results in files or print on terminal
        save_results(
            args,
            result_val_t,
            result_val_a,
            result_test_t,
            result_test_a,
            model,
            spent_time,
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


if __name__ == "__main__":
    main()
