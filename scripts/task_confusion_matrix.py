#!/usr/bin/env python3
"""Task-wise confusion matrices for a run directory, averaged over seeds.

Two different matrices answer two different questions, and this script builds
either or both:

``perf`` (default)
    The task-by-task **performance** matrix already produced per seed by
    :func:`metrics.metrics.confusion_matrix` -- row ``i`` is "after finishing
    task ``i``", column ``j`` is the validation metric on task ``j``. Read
    straight out of each seed's ``results.pt`` / ``results.txt``, so it is cheap
    and needs no GPU or dataset. Diagonal accuracy, final accuracy, BWT and FWT
    come along with it.

``pred``
    The true-task x predicted-task **confusion** matrix: for validation samples
    belonging to task ``i``, which task owns the class the model actually
    predicts? This exposes the recency / head bias that the performance matrix
    hides. Nothing per-sample is logged during training, so this mode rebuilds
    the loader and the model from the saved args, loads a checkpoint, and runs
    inference (dataset + optionally GPU required).

    Task-incremental runs mask logits to the evaluated task, which would make
    the matrix trivially diagonal, so by default this mode forces cumulative
    class-incremental masking over every class seen up to the checkpoint's task
    (``--masking cil``). Use ``--masking native`` to keep the run's own masking.

    ``--class-breakdown`` adds a true-class x predicted-class matrix that lays
    out every seen task's classes on one grid (``t<task>c<class>``). For a
    task-incremental run it is drawn twice: with the TID mask (each task's
    block outlined, since predictions cannot leave it) and without it (the
    ``--masking`` view, or ``cil`` when that is ``native``). Multi-head
    models cannot be scored without a TID, so only the masked view exists.

When the run directory holds one sub-directory per seed, every matrix is
averaged over seeds (sample std alongside).

Usage:
    python scripts/task_confusion_matrix.py logs/eralg4/<run-dir>
    python scripts/task_confusion_matrix.py logs/eralg4/<run-dir> --metric macro_f1
    python scripts/task_confusion_matrix.py logs/ctn/<run-dir> --mode pred --device cuda
    python scripts/task_confusion_matrix.py logs/ctn/<run-dir> --mode both --json
    python scripts/task_confusion_matrix.py logs/ctn/<run-dir> --mode pred --class-breakdown
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import importlib
import json
import os
import re
import sys
import warnings
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from metrics.metrics import task_changes  # noqa: E402
from utils import misc_utils  # noqa: E402

# Metrics available in ``perf`` mode, in display order.
PERF_METRICS = ("cls_rec", "cls_prec", "macro_f1")

# Sequential blue ramp (100 -> 700) used for every heatmap: one hue, light to
# dark, lightest step meaning "near zero".
SEQUENTIAL_BLUE = [
    "#cde2fb",
    "#b7d3f6",
    "#9ec5f4",
    "#86b6ef",
    "#6da7ec",
    "#5598e7",
    "#3987e5",
    "#2a78d6",
    "#256abf",
    "#1c5cab",
    "#184f95",
    "#104281",
    "#0d366b",
]
SURFACE = "#fcfcfb"
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#8a8984"
GRID_INK = "#c3c2b7"

# "0.5766 0.0000 0.0000" -- a row of the reduced matrix in results.txt.
_MATRIX_ROW_RE = re.compile(r"^\s*-?\d+\.\d+(\s+-?\d+\.\d+)*\s*$")


# --------------------------------------------------------------------------
# run-directory discovery
# --------------------------------------------------------------------------


def _seed_sort_key(name: str):
    """Sort seed dir names numerically when possible, else lexically."""
    return (0, int(name)) if name.isdigit() else (1, name)


def discover_seed_dirs(run_dir: Path) -> List[Tuple[str, Path]]:
    """Return ``(seed_name, path)`` for every seed under ``run_dir``.

    A run directory normally holds one sub-directory per seed (``0/``, ``39/``,
    ...), each with its own ``results.pt``. A seed directory passed directly is
    also accepted and treated as a single-seed run.

    Args:
        run_dir: Experiment directory, or one seed directory inside it.

    Returns:
        Sorted list of ``(seed_name, seed_dir)`` pairs.

    Usage:
        seeds = discover_seed_dirs(Path("logs/eralg4/my-run"))
    """
    if (run_dir / "results.pt").is_file():
        return [(run_dir.name, run_dir)]

    seeds: List[Tuple[str, Path]] = []
    for name in sorted(os.listdir(run_dir), key=_seed_sort_key):
        sub = run_dir / name
        if sub.is_dir() and (sub / "results.pt").is_file():
            seeds.append((name, sub))
    return seeds


def load_results_bundle(seed_dir: Path) -> Tuple[Any, ...]:
    """Load one seed's ``results.pt`` 6-tuple with argparse namespaces allowed."""
    add_safe_globals = getattr(torch.serialization, "add_safe_globals", None)
    if add_safe_globals is not None:
        add_safe_globals([argparse.Namespace])
    bundle = torch.load(seed_dir / "results.pt", map_location="cpu", weights_only=False)
    if not isinstance(bundle, (tuple, list)) or len(bundle) != 6:
        raise RuntimeError(
            "{}: unexpected results.pt structure (expected a 6-tuple)".format(seed_dir)
        )
    return tuple(bundle)


# --------------------------------------------------------------------------
# mode "perf": task-by-task performance matrix from the logs
# --------------------------------------------------------------------------


def reduce_round_matrix(result_t: torch.Tensor, result_a: torch.Tensor) -> np.ndarray:
    """Reduce an (eval rounds x tasks) matrix to one row per finished task.

    Mirrors the reduction in :func:`metrics.metrics.confusion_matrix`: keep only
    the last evaluation round of each task, so row ``i`` is the state after
    training task ``i``.

    Args:
        result_t: 1D tensor of task ids, one entry per evaluation round.
        result_a: 2D tensor (rounds x tasks) of per-task metric values.

    Returns:
        A ``(n_tasks, n_tasks)`` float array.

    Usage:
        matrix = reduce_round_matrix(result_val_t, result_val_a)
    """
    n_tasks, changes = task_changes(result_t)
    change_indices = torch.LongTensor(changes + [result_a.size(0)]) - 1
    reduced = result_a[change_indices]
    if reduced.size(0) != n_tasks:
        raise RuntimeError(
            "reduced matrix has {} rows for {} tasks".format(reduced.size(0), n_tasks)
        )
    return reduced.detach().cpu().numpy().astype(float)


def parse_metric_matrix(results_txt: Path, header_prefix: str) -> np.ndarray | None:
    """Return the task matrix of one metric block in ``results.txt``, if present.

    ``main.save_results`` writes a ``"Precision (...)"`` and an ``"F1 (...)"``
    block after the recall matrix: a header line, the zero-shot row, a ``|``
    separator, then the T x T matrix. Older runs lack these blocks, in which
    case ``None`` is returned.

    Args:
        results_txt: Path to a seed's ``results.txt``.
        header_prefix: Start of the block header, e.g. ``"F1 ("``.

    Returns:
        The T x T matrix, or ``None`` when the block is missing or ragged.

    Usage:
        f1_matrix = parse_metric_matrix(seed_dir / "results.txt", "F1 (")
    """
    if not results_txt.is_file():
        return None
    rows: List[List[float]] = []
    in_block = False
    past_separator = False
    with open(results_txt, "r", errors="replace") as handle:
        for line in handle:
            if not in_block:
                in_block = line.startswith(header_prefix)
                continue
            if not past_separator:
                past_separator = line.strip() == "|"
                continue
            if _MATRIX_ROW_RE.match(line):
                rows.append([float(value) for value in line.split()])
            else:
                break
    if not rows:
        return None
    widths = {len(row) for row in rows}
    if len(widths) != 1:
        return None
    return np.asarray(rows, dtype=float)


def parse_baseline_row(results_txt: Path) -> np.ndarray | None:
    """Return the zero-shot (pre-training) per-task row printed first in results.txt."""
    if not results_txt.is_file():
        return None
    with open(results_txt, "r", errors="replace") as handle:
        first = handle.readline()
    if not _MATRIX_ROW_RE.match(first):
        return None
    return np.asarray([float(value) for value in first.split()], dtype=float)


def perf_matrices_for_seed(seed_dir: Path) -> Dict[str, np.ndarray]:
    """Build every available task-by-task performance matrix for one seed.

    Returns:
        Dict with ``cls_rec`` always present, ``cls_prec`` and ``macro_f1``
        present only when ``results.txt`` carries those metric blocks, and
        ``baseline`` (the zero-shot row) when it can be parsed.
    """
    result_t, result_a = load_results_bundle(seed_dir)[:2]
    matrices: Dict[str, np.ndarray] = {
        "cls_rec": reduce_round_matrix(result_t, result_a)
    }

    results_txt = seed_dir / "results.txt"
    for name, header_prefix in (("cls_prec", "Precision ("), ("macro_f1", "F1 (")):
        matrix = parse_metric_matrix(results_txt, header_prefix)
        if matrix is not None and matrix.shape == matrices["cls_rec"].shape:
            matrices[name] = matrix

    baseline = parse_baseline_row(results_txt)
    if baseline is not None and baseline.shape[0] == matrices["cls_rec"].shape[1]:
        matrices["baseline"] = baseline
    return matrices


def matrix_scalars(matrix: np.ndarray, baseline: np.ndarray | None) -> Dict[str, float]:
    """Diagonal / final / BWT (and FWT when a baseline row is available)."""
    n_tasks = matrix.shape[0]
    diagonal = np.diag(matrix)
    final = matrix[n_tasks - 1]
    scalars = {
        "diagonal": float(np.nanmean(diagonal)),
        "final": float(np.nanmean(final)),
        "bwt": float(np.nanmean(final - diagonal)),
    }
    if baseline is not None and n_tasks > 1:
        forward = np.zeros(n_tasks, dtype=float)
        for task in range(1, n_tasks):
            forward[task] = matrix[task - 1, task] - baseline[task]
        scalars["fwt"] = float(np.nanmean(forward))
    return scalars


# --------------------------------------------------------------------------
# mode "pred": true-task x predicted-task confusion from inference
# --------------------------------------------------------------------------


@contextlib.contextmanager
def forced_cil_masking(upto_task: int | None):
    """Patch logit masking so every task is scored against one joint head.

    Models call :func:`utils.misc_utils.apply_task_incremental_logit_mask` via
    the module, passing the loader name baked in at construction time. For a
    task-incremental run that restricts each forward pass to the evaluated
    task's own classes, which would force the confusion matrix onto its
    diagonal. This wrapper overrides the loader argument so the cumulative CIL
    branch runs with a fixed bound instead.

    Args:
        upto_task: Cumulative task bound (inclusive) to keep unmasked, or
            ``None`` to disable masking entirely.

    Usage:
        with forced_cil_masking(3):
            logits = model_forward_for_metric_loop(model, x, t, args)
    """
    original = misc_utils.apply_task_incremental_logit_mask

    def patched(logits, task_index, nc_per_task, n_outputs, **kwargs):
        if upto_task is None:
            return logits.clone()
        kwargs["loader"] = "class_incremental_loader"
        kwargs["cil_all_seen_upto_task"] = upto_task
        return original(logits, task_index, nc_per_task, n_outputs, **kwargs)

    misc_utils.apply_task_incremental_logit_mask = patched
    try:
        yield
    finally:
        misc_utils.apply_task_incremental_logit_mask = original


def rebuild_run(seed_dir: Path, device: torch.device) -> Tuple[Any, Any, Any, int]:
    """Rebuild the loader and untrained model of a finished run from its saved args.

    Args:
        seed_dir: Seed directory containing ``results.pt``.
        device: Device the model should end up on.

    Returns:
        ``(model, loader, args, n_tasks)``.

    Raises:
        RuntimeError: If the dataset on disk no longer produces the per-task
            class counts the run was trained with.

    Usage:
        model, loader, args, n_tasks = rebuild_run(seed_dir, torch.device("cpu"))
    """
    _, _, state_dict, _, _, args = load_results_bundle(seed_dir)
    saved_classes_per_task = list(getattr(args, "classes_per_task", None) or [])
    args.cuda = device.type == "cuda"

    misc_utils.init_seed(args.seed)
    loader_module = importlib.import_module("dataloaders." + args.loader)
    loader = loader_module.IncrementalLoader(args, seed=args.seed)
    n_inputs, n_outputs, n_tasks = loader.get_dataset_info()

    args.get_samples_per_task = getattr(loader, "get_samples_per_task", None)
    rebuilt_classes_per_task = list(getattr(loader, "classes_per_task", None) or [])
    if rebuilt_classes_per_task:
        args.classes_per_task = rebuilt_classes_per_task

    if saved_classes_per_task and rebuilt_classes_per_task != saved_classes_per_task:
        raise RuntimeError(
            "{}: the dataset under '{}' no longer matches this run. It was trained "
            "with classes_per_task={} but rebuilding the loader now gives {}. "
            "Inference would score a different label space, so 'pred' mode cannot "
            "run for this directory.".format(
                seed_dir,
                getattr(args, "data_path", "?"),
                saved_classes_per_task,
                rebuilt_classes_per_task,
            )
        )

    model_module = importlib.import_module("model." + args.model)
    model = model_module.Net(n_inputs, n_outputs, n_tasks, args)
    model.to(device)
    return model, loader, args, n_tasks


def load_checkpoint_state(
    seed_dir: Path, after_task: int, n_tasks: int
) -> Dict[str, torch.Tensor]:
    """Return the state dict saved after ``after_task``, preferring checkpoints/."""
    checkpoint_path = seed_dir / "checkpoints" / "task_{}.pt".format(after_task)
    if checkpoint_path.is_file():
        payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        return payload["state_dict"] if isinstance(payload, dict) else payload
    if after_task == n_tasks - 1:
        return load_results_bundle(seed_dir)[2]
    raise FileNotFoundError(
        "{}: no checkpoint for task {} (only results.pt holds the final model)".format(
            seed_dir, after_task
        )
    )


def class_to_task_map(
    classes_per_task: Sequence[int], n_outputs: int, noise_label: int | None
) -> np.ndarray:
    """Map each global class index to the task that owns it.

    Args:
        classes_per_task: Per-task signal-class counts.
        n_outputs: Width of the classifier head.
        noise_label: Global shared noise class, or ``None`` when noise is folded
            into each task's own class block.

    Returns:
        Array of length ``n_outputs``; entry ``c`` is the owning task index, or
        ``-1`` for the shared noise class and for any class beyond the last task.

    Usage:
        owner = class_to_task_map([7, 7, 6, 6], 26, noise_label=None)
    """
    owner = np.full(n_outputs, -1, dtype=int)
    boundary = 0
    for task_index, count in enumerate(classes_per_task):
        upper = min(boundary + int(count), n_outputs)
        if boundary < upper:
            owner[boundary:upper] = task_index
        boundary = upper
    if noise_label is not None and 0 <= int(noise_label) < n_outputs:
        owner[int(noise_label)] = -1
    return owner


def masking_variants(
    loader_name: str, masking: str, class_breakdown: bool
) -> List[str]:
    """Return the masking modes to run inference under, primary first.

    The primary mode (``masking``) feeds the task-level matrix. A class
    breakdown of a task-incremental run also needs the complementary view, so
    both "with TID" (the run's own per-task mask) and "without TID" appear.

    Args:
        loader_name: The run's ``args.loader``.
        masking: The ``--masking`` choice.
        class_breakdown: Whether the class-level matrix was requested.

    Returns:
        Ordered, duplicate-free list of masking modes.

    Usage:
        masking_variants("task_incremental_loader", "cil", True)  # ["cil", "native"]
    """
    variants = [masking]
    if class_breakdown and loader_name != "class_incremental_loader":
        variants.append("cil" if masking == "native" else "native")
    return variants


def variant_uses_tid(loader_name: str, masking: str) -> bool:
    """True when predictions under ``masking`` are restricted to the true task's block."""
    return masking == "native" and loader_name != "class_incremental_loader"


def variant_title(loader_name: str, masking: str) -> str:
    """Human-readable name of a masking mode for plot titles and logs."""
    if variant_uses_tid(loader_name, masking):
        return "with TID mask"
    if masking == "native":
        return "native CIL mask"
    if masking == "cil":
        return "without TID (all seen classes)"
    return "without TID (raw head)"


def collect_predictions(
    model: Any,
    tasks: Sequence[Any],
    args: Any,
    after_task: int,
    local_head_offsets: Sequence[int] | None,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray]:
    """Run inference over every task and return global true / predicted classes.

    Args:
        model: Trained model in eval mode.
        tasks: Per-task evaluation dataloaders (index = true task).
        args: Run args (used for the architecture-specific forward).
        after_task: Checkpoint task, used as the cumulative CIL logit bound.
        local_head_offsets: For multi-head models whose logits are task-local,
            the global offset of each task's first class; ``None`` when the
            logits already span the global class space.
        device: Device to run inference on.

    Returns:
        ``(true_classes, predicted_classes)``, two aligned 1D int arrays.

    Usage:
        y_true, y_pred = collect_predictions(model, tasks, args, 3, None, dev)
    """
    from model.replay_utils import unpack_y_to_class_labels
    from utils.training_forward import model_forward_for_metric_loop

    is_linear = str(getattr(args, "arch", "")).lower() == "linear"
    true_parts: List[np.ndarray] = []
    pred_parts: List[np.ndarray] = []

    model.eval()
    with torch.no_grad():
        for true_task, task_loader in enumerate(tasks):
            offset = 0 if local_head_offsets is None else local_head_offsets[true_task]
            for batch in task_loader:
                if isinstance(batch, (list, tuple)) and len(batch) == 3:
                    xb, yb, _ = batch
                else:
                    xb, yb = batch
                xb = xb.to(device)
                if is_linear:
                    xb = xb.view(xb.size(0), -1)

                labels = unpack_y_to_class_labels(yb).detach().cpu().reshape(-1)
                logits = model_forward_for_metric_loop(
                    model, xb, true_task, args, cil_mask_upto_task=after_task
                )
                predictions = torch.argmax(logits, dim=1).cpu().numpy() + offset
                true_parts.append(labels.numpy().astype(np.int64))
                pred_parts.append(predictions.astype(np.int64))

    if not true_parts:
        return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64)
    return np.concatenate(true_parts), np.concatenate(pred_parts)


def task_counts_from_predictions(
    true_classes: np.ndarray,
    predicted_classes: np.ndarray,
    owner: np.ndarray,
    n_tasks: int,
) -> np.ndarray:
    """Count, per true task, how often each task's classes are predicted.

    Args:
        true_classes: Global true class per sample.
        predicted_classes: Global predicted class per sample.
        owner: Class-index -> owning-task map from :func:`class_to_task_map`.
        n_tasks: Number of evaluated tasks.

    Returns:
        Integer array ``(n_tasks, n_tasks + 1)``; the last column collects
        predictions of the shared noise class or any unassigned head slot.

    Usage:
        counts = task_counts_from_predictions(y_true, y_pred, owner, 4)
    """
    counts = np.zeros((n_tasks, n_tasks + 1), dtype=np.int64)
    true_owner = owner[true_classes]
    predicted_owner = owner[predicted_classes]
    rows = true_owner >= 0
    # -1 (shared noise, or an unassigned head slot) lands in the last column.
    columns = np.where(predicted_owner < 0, n_tasks, predicted_owner)
    np.add.at(counts, (true_owner[rows], columns[rows]), 1)
    return counts


def class_counts_from_predictions(
    true_classes: np.ndarray, predicted_classes: np.ndarray, n_seen: int
) -> np.ndarray:
    """Count true-class x predicted-class pairs over the seen class space.

    Args:
        true_classes: Global true class per sample (all ``< n_seen``).
        predicted_classes: Global predicted class per sample.
        n_seen: Number of classes in tasks ``0..after_task``.

    Returns:
        Integer array ``(n_seen, n_seen + 1)``; the last column collects
        predictions of classes outside the seen space (future-task head slots).

    Usage:
        counts = class_counts_from_predictions(y_true, y_pred, 26)
    """
    counts = np.zeros((n_seen, n_seen + 1), dtype=np.int64)
    rows = (true_classes >= 0) & (true_classes < n_seen)
    columns = np.where(
        (predicted_classes >= 0) & (predicted_classes < n_seen),
        predicted_classes,
        n_seen,
    )
    np.add.at(counts, (true_classes[rows], columns[rows]), 1)
    return counts


def evaluation_loaders(
    loader: Any, loader_name: str, split: str, n_tasks: int
) -> List[Any]:
    """Return the per-task test loaders exactly as ``main.py`` scores them.

    Task-incremental training scores the loader ``new_task()`` returns, whose
    test samples are drawn through a per-task permutation. ``get_tasks()``
    instead hands back the stored arrays in file order, which for some
    datasets (e.g. DeepRad) is sorted by class. Under ``--eval_bn_stats
    batch`` a class-sorted batch is normalised with single-class statistics
    and collapses to chance, so the two streams give different numbers.
    Class-incremental training already scores ``get_tasks("test")``, whose
    per-task sets are permuted.

    Args:
        loader: Freshly built ``IncrementalLoader`` (no task consumed yet).
        loader_name: The run's ``args.loader``.
        split: ``val`` or ``test``; both map to the test arrays.
        n_tasks: Number of leading tasks to return.

    Returns:
        List of ``n_tasks`` dataloaders.
    """
    if loader_name == "class_incremental_loader" or not hasattr(loader, "new_task"):
        return list(loader.get_tasks(split))[:n_tasks]
    return [loader.new_task()[3] for _ in range(n_tasks)]


def load_trained_model(seed_dir: Path, device: torch.device, after_task_arg: int):
    """Rebuild a run and load the checkpoint taken after ``after_task_arg``.

    Returns:
        ``(model, loader, args, n_tasks, after_task)``.
    """
    model, loader, args, n_tasks = rebuild_run(seed_dir, device)
    after_task = n_tasks - 1 if after_task_arg < 0 else after_task_arg
    if not 0 <= after_task < n_tasks:
        raise ValueError(
            "--after-task {} out of range for a {}-task run".format(after_task, n_tasks)
        )

    state_dict = load_checkpoint_state(seed_dir, after_task, n_tasks)
    try:
        model.load_state_dict(state_dict, strict=True)
    except RuntimeError as error:
        incompatible = model.load_state_dict(state_dict, strict=False)
        if incompatible.missing_keys or incompatible.unexpected_keys:
            raise RuntimeError(
                "{}: checkpoint does not fit the rebuilt model ({}). The run's "
                "config or dataset has most likely changed since it was "
                "trained.".format(seed_dir, error)
            ) from error
    return model, loader, args, n_tasks, after_task


def confusion_for_seed(
    seed_dir: Path,
    split: str,
    after_task_arg: int,
    masking: str,
    signal_only: bool,
    class_breakdown: bool,
    device: torch.device,
) -> Dict[str, Any]:
    """Run inference for one seed under every requested masking mode.

    Returns:
        Dict with ``loader``, ``classes_per_task`` (seen tasks only),
        ``after_task`` and ``variants``: masking mode -> ``{"task_counts",
        "class_counts"}``. A mode is absent when the model cannot be scored
        under it (multi-head models have no head to use without a TID).
    """
    model, loader, args, _, after_task = load_trained_model(
        seed_dir, device, after_task_arg
    )
    loader_name = str(getattr(args, "loader", ""))
    tasks = evaluation_loaders(loader, loader_name, split, after_task + 1)
    classes_per_task = [int(c) for c in (getattr(args, "classes_per_task", None) or [])]
    n_outputs = loader.get_dataset_info()[1]
    noise_label = getattr(args, "noise_label", None)
    noise_label = None if noise_label is None else int(noise_label)
    owner = class_to_task_map(classes_per_task, n_outputs, noise_label)
    seen_classes_per_task = classes_per_task[: len(tasks)]
    n_seen = int(sum(seen_classes_per_task))

    # Multi-head models (e.g. UCL with split heads) emit task-local logits.
    multi_head = bool(getattr(model, "split", False))
    local_head_offsets = None
    if multi_head:
        local_head_offsets = [
            misc_utils.compute_offsets(task, classes_per_task)[0]
            for task in range(len(tasks))
        ]

    variants: Dict[str, Dict[str, np.ndarray]] = {}
    for mode in masking_variants(loader_name, masking, class_breakdown):
        if multi_head and mode != "native":
            print(
                "[pred] {}: multi-head model needs the task id to pick a head; "
                "skipping masking={}".format(seed_dir, mode)
            )
            continue
        bound = None if mode == "none" else after_task
        context = (
            contextlib.nullcontext() if mode == "native" else forced_cil_masking(bound)
        )
        with context:
            true_classes, predicted_classes = collect_predictions(
                model, tasks, args, after_task, local_head_offsets, device
            )
        if signal_only and noise_label is not None:
            keep = true_classes != noise_label
            true_classes = true_classes[keep]
            predicted_classes = predicted_classes[keep]
        # Out-of-range predictions (e.g. a raw head wider than the class map)
        # count as unassigned rather than indexing past ``owner``.
        predicted_classes = np.where(
            (predicted_classes >= 0) & (predicted_classes < n_outputs),
            predicted_classes,
            n_outputs,
        )
        owner_padded = np.append(owner, -1)
        variants[mode] = {
            "task_counts": task_counts_from_predictions(
                true_classes, predicted_classes, owner_padded, len(tasks)
            ),
            "class_counts": class_counts_from_predictions(
                true_classes, predicted_classes, n_seen
            ),
        }

    return {
        "loader": loader_name,
        "classes_per_task": seen_classes_per_task,
        "task_order": repr(
            (
                getattr(args, "task_order_seed", None),
                getattr(args, "task_order_files", None),
            )
        ),
        "noise_label": noise_label,
        "after_task": after_task,
        "variants": variants,
    }


def row_normalise(counts: np.ndarray, empty_value: float = 0.0) -> np.ndarray:
    """Row-normalise a count matrix; all-zero rows are filled with ``empty_value``."""
    totals = counts.sum(axis=1, keepdims=True).astype(float)
    with np.errstate(divide="ignore", invalid="ignore"):
        fractions = np.where(totals > 0, counts / totals, empty_value)
    return fractions


# --------------------------------------------------------------------------
# aggregation and output
# --------------------------------------------------------------------------


def mean_and_std(stack: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Mean and sample std (ddof=1) over the leading axis, ignoring NaNs."""
    mean = np.nanmean(stack, axis=0)
    if stack.shape[0] < 2:
        return mean, np.full(mean.shape, np.nan)
    return mean, np.nanstd(stack, axis=0, ddof=1)


def stack_matrices(per_seed: List[np.ndarray], label: str) -> np.ndarray:
    """Stack per-seed matrices, refusing to average runs of different shapes."""
    shapes = {matrix.shape for matrix in per_seed}
    if len(shapes) != 1:
        raise RuntimeError(
            "{}: seeds disagree on matrix shape ({}); they are not the same "
            "experiment.".format(label, sorted(shapes))
        )
    return np.stack(per_seed, axis=0)


def format_matrix(
    mean: np.ndarray,
    std: np.ndarray | None,
    row_labels: Sequence[str],
    col_labels: Sequence[str],
) -> str:
    """Render a matrix as a fixed-width text table, with +/- std when available."""
    show_std = std is not None and not np.all(np.isnan(std))
    cell_width = 15 if show_std else 8
    row_width = max(len(label) for label in row_labels) + 2

    header = " " * row_width + "".join(label.rjust(cell_width) for label in col_labels)
    lines = [header]
    for row_index, row_label in enumerate(row_labels):
        cells = []
        for col_index in range(len(col_labels)):
            value = mean[row_index, col_index]
            if show_std:
                deviation = std[row_index, col_index]  # type: ignore[index]
                text = (
                    "{:.4f}+-{:.4f}".format(value, deviation)
                    if not np.isnan(deviation)
                    else "{:.4f}".format(value)
                )
            else:
                text = "{:.4f}".format(value)
            cells.append(text.rjust(cell_width))
        lines.append(row_label.ljust(row_width) + "".join(cells))
    return "\n".join(lines)


def write_matrix_csv(
    path: Path,
    mean: np.ndarray,
    std: np.ndarray | None,
    row_labels: Sequence[str],
    col_labels: Sequence[str],
) -> None:
    """Write a matrix (and its std, when defined) to CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    show_std = std is not None and not np.all(np.isnan(std))
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        header = [""] + list(col_labels)
        if show_std:
            header += ["{} (std)".format(label) for label in col_labels]
        writer.writerow(header)
        for row_index, row_label in enumerate(row_labels):
            row = [row_label] + ["{:.6f}".format(v) for v in mean[row_index]]
            if show_std:
                row += ["{:.6f}".format(v) for v in std[row_index]]  # type: ignore[index]
            writer.writerow(row)


def plot_heatmap(
    path: Path,
    matrix: np.ndarray,
    row_labels: Sequence[str],
    col_labels: Sequence[str],
    title: str,
    subtitle: str,
    value_label: str,
    mask_upper_triangle: bool = False,
) -> None:
    """Save a single-hue sequential heatmap of ``matrix``.

    Magnitude is the encoded quantity, so the ramp is one hue light->dark with
    every cell directly labelled; axes and frame stay recessive.

    Args:
        path: Destination PNG path.
        matrix: Values to draw.
        row_labels: Row tick labels.
        col_labels: Column tick labels.
        title: Chart title.
        subtitle: One-line note under the title (may be empty).
        value_label: Colorbar label.
        mask_upper_triangle: Blank cells above the diagonal, which in a
            performance matrix are tasks that had not been seen yet rather than
            a measured zero.

    Usage:
        plot_heatmap(path, mean, rows, cols, "Title", "mean over 3 seeds", "f1")
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap

    colormap = LinearSegmentedColormap.from_list("sequential_blue", SEQUENTIAL_BLUE)
    colormap.set_bad(SURFACE)
    matrix = np.asarray(matrix, dtype=float)
    if mask_upper_triangle:
        matrix = np.where(
            np.triu(np.ones_like(matrix, dtype=bool), k=1), np.nan, matrix
        )
    finite = matrix[np.isfinite(matrix)]
    vmax = float(finite.max()) if finite.size and finite.max() > 0 else 1.0

    n_rows, n_cols = matrix.shape
    figure, axes = plt.subplots(
        figsize=(max(4.5, 0.95 * n_cols + 2.2), max(3.8, 0.8 * n_rows + 2.0))
    )
    figure.patch.set_facecolor(SURFACE)
    axes.set_facecolor(SURFACE)

    image = axes.imshow(
        np.ma.masked_invalid(matrix),
        cmap=colormap,
        vmin=0.0,
        vmax=vmax,
        aspect="auto",
    )

    axes.set_xticks(range(n_cols), labels=col_labels, color=INK_SECONDARY, fontsize=9)
    axes.set_yticks(range(n_rows), labels=row_labels, color=INK_SECONDARY, fontsize=9)
    axes.tick_params(length=0)
    for spine in axes.spines.values():
        spine.set_visible(False)

    # 2px surface gap between cells, drawn as minor gridlines on the surface color.
    axes.set_xticks(np.arange(-0.5, n_cols, 1), minor=True)
    axes.set_yticks(np.arange(-0.5, n_rows, 1), minor=True)
    axes.grid(which="minor", color=SURFACE, linewidth=2)
    axes.tick_params(which="minor", length=0)

    for row in range(n_rows):
        for col in range(n_cols):
            value = matrix[row, col]
            if not np.isfinite(value):
                continue
            dark_cell = value >= 0.55 * vmax
            axes.text(
                col,
                row,
                "{:.2f}".format(value),
                ha="center",
                va="center",
                fontsize=8,
                color=SURFACE if dark_cell else INK_PRIMARY,
            )

    axes.set_title(title, color=INK_PRIMARY, fontsize=12, loc="left", pad=26)
    if subtitle:
        # Offset in points so the gap does not scale with the axes height.
        axes.annotate(
            subtitle,
            xy=(0.0, 1.0),
            xycoords="axes fraction",
            xytext=(0, 6),
            textcoords="offset points",
            color=INK_SECONDARY,
            fontsize=9,
            ha="left",
            va="bottom",
        )

    colorbar = figure.colorbar(image, ax=axes, fraction=0.035, pad=0.02)
    colorbar.set_label(value_label, color=INK_SECONDARY, fontsize=9)
    colorbar.ax.tick_params(colors=INK_SECONDARY, labelsize=8, length=0)
    colorbar.outline.set_visible(False)

    path.parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout()
    figure.savefig(path, dpi=200, facecolor=SURFACE)
    plt.close(figure)


def class_labels_for_tasks(classes_per_task: Sequence[int]) -> List[str]:
    """Tick labels ``t<task>c<local class>`` for every class, in global order."""
    return [
        "t{}c{}".format(task, local)
        for task, count in enumerate(classes_per_task)
        for local in range(int(count))
    ]


def plot_class_confusion(
    path: Path,
    matrix: np.ndarray,
    classes_per_task: Sequence[int],
    col_labels: Sequence[str],
    title: str,
    subtitle: str,
    draw_task_boxes: bool,
) -> None:
    """Save a true-class x predicted-class heatmap spanning every seen task.

    Classes are laid out in global order, so each task is a contiguous block,
    and the axes are labelled per task block rather than per class. Recessive
    separators mark every task boundary. With a TID mask a sample can only be
    assigned to its own task's classes, so the diagonal blocks are also
    outlined to show the region predictions are confined to. The colour scale
    is fixed to ``[0, 1]`` so the with- and without-TID figures can be
    compared side by side.

    Args:
        path: Destination PNG path.
        matrix: Row-normalised ``(n_seen, n_cols)`` fractions; NaN rows are
            classes with no evaluation samples.
        classes_per_task: Class count of each seen task.
        col_labels: Column labels (the seen classes, plus an optional trailing
            "other" column); only the trailing "other" is drawn as a tick.
        title: Chart title.
        subtitle: Second title line (may be empty).
        draw_task_boxes: Outline each task's diagonal block.

    Usage:
        plot_class_confusion(path, mean, [7, 7, 6], cols, "Title", "", True)
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    from matplotlib.patches import Rectangle

    colormap = LinearSegmentedColormap.from_list("sequential_blue", SEQUENTIAL_BLUE)
    colormap.set_bad(SURFACE)
    matrix = np.asarray(matrix, dtype=float)
    n_rows, n_cols = matrix.shape
    counts = [int(count) for count in classes_per_task]
    owner = np.repeat(np.arange(len(counts)), counts)

    figure, axes = plt.subplots(figsize=(11, 9.5), facecolor=SURFACE)
    axes.set_facecolor(SURFACE)
    image = axes.imshow(
        np.ma.masked_invalid(matrix),
        cmap=colormap,
        vmin=0.0,
        vmax=1.0,
        interpolation="nearest",
    )

    # Task-block separators: recessive, so the data reads first. The trailing
    # "other" column gets one too, since it belongs to no task.
    boundaries = list(np.flatnonzero(np.diff(owner)) + 0.5)
    for edge in boundaries:
        axes.axhline(edge, color=GRID_INK, linewidth=0.8)
        axes.axvline(edge, color=GRID_INK, linewidth=0.8)
    if n_cols > n_rows:
        axes.axvline(n_rows - 0.5, color=GRID_INK, linewidth=0.8)

    if draw_task_boxes:
        start = 0
        for count in counts:
            if count > 0:
                axes.add_patch(
                    Rectangle(
                        (start - 0.5, start - 0.5),
                        count,
                        count,
                        fill=False,
                        edgecolor=INK_PRIMARY,
                        linewidth=1.2,
                    )
                )
            start += count

    tasks = np.unique(owner)
    centres = [float(np.mean(np.flatnonzero(owner == task))) for task in tasks]
    labels = ["T{}".format(task) for task in tasks]
    x_centres = list(centres)
    x_labels = list(labels)
    if n_cols > n_rows:
        x_centres += list(range(n_rows, n_cols))
        x_labels += list(col_labels[n_rows:])
    axes.set_xticks(x_centres)
    axes.set_yticks(centres)
    axes.set_xticklabels(x_labels, color=INK_SECONDARY, fontsize=9)
    axes.set_yticklabels(labels, color=INK_SECONDARY, fontsize=9)
    axes.set_xlabel(
        "predicted class (grouped by task)", color=INK_SECONDARY, fontsize=10
    )
    axes.set_ylabel("true class (grouped by task)", color=INK_SECONDARY, fontsize=10)
    full_title = title if not subtitle else "{}\n{}".format(title, subtitle)
    axes.set_title(full_title, color=INK_PRIMARY, fontsize=12, pad=12)
    for spine in axes.spines.values():
        spine.set_color(GRID_INK)
    axes.tick_params(colors=GRID_INK, length=3)

    bar = figure.colorbar(image, ax=axes, fraction=0.045, pad=0.02)
    bar.set_label("row-normalised rate", color=INK_SECONDARY, fontsize=9)
    bar.ax.tick_params(colors=INK_SECONDARY, labelsize=8)
    bar.outline.set_edgecolor(GRID_INK)

    path.parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout()
    figure.savefig(path, dpi=160, facecolor=SURFACE)
    plt.close(figure)


# --------------------------------------------------------------------------
# drivers
# --------------------------------------------------------------------------


def run_perf_mode(
    seeds: List[Tuple[str, Path]],
    metrics_wanted: Sequence[str],
    out_dir: Path,
    make_plots: bool,
) -> Dict[str, Any]:
    """Average the task-by-task performance matrices over seeds and report them."""
    per_seed: Dict[str, List[np.ndarray]] = {}
    # One entry per seed, aligned with ``used_seeds``; None when results.txt has
    # no parseable zero-shot row (so FWT is simply omitted for that seed).
    baselines: List[np.ndarray | None] = []
    used_seeds: List[str] = []

    for seed_name, seed_dir in seeds:
        matrices = perf_matrices_for_seed(seed_dir)
        used_seeds.append(seed_name)
        baselines.append(matrices.pop("baseline", None))
        for name, matrix in matrices.items():
            per_seed.setdefault(name, []).append(matrix)

    known_baselines = [b for b in baselines if b is not None]
    mean_baseline = (
        np.nanmean(np.stack(known_baselines, axis=0), axis=0)
        if len(known_baselines) == len(used_seeds) and known_baselines
        else None
    )

    report: Dict[str, Any] = {"seeds": used_seeds, "metrics": {}}
    for name in metrics_wanted:
        if name not in per_seed:
            if name != "cls_rec":
                print(
                    "[skip] {}: not available (results.txt has no matching metric "
                    "block for these seeds)".format(name)
                )
            continue
        if len(per_seed[name]) != len(used_seeds):
            print(
                "[skip] {}: only {}/{} seeds provide it".format(
                    name, len(per_seed[name]), len(used_seeds)
                )
            )
            continue

        stack = stack_matrices(per_seed[name], name)
        mean, std = mean_and_std(stack)
        n_tasks = mean.shape[0]
        row_labels = ["after task {}".format(index) for index in range(n_tasks)]
        col_labels = ["t{}".format(index) for index in range(n_tasks)]

        scalars_per_seed = [
            matrix_scalars(matrix, base)
            for matrix, base in zip(per_seed[name], baselines)
        ]
        scalar_summary = {}
        for key in ("diagonal", "final", "bwt", "fwt"):
            values = [s[key] for s in scalars_per_seed if key in s]
            if not values:
                continue
            array = np.asarray(values, dtype=float)
            scalar_summary[key] = {
                "mean": float(np.nanmean(array)),
                "std": (
                    float(np.nanstd(array, ddof=1)) if array.size > 1 else float("nan")
                ),
                "per_seed": [float(v) for v in array],
            }

        print()
        print("=" * 78)
        print(
            "Performance matrix [{}] -- mean over {} seed(s): {}".format(
                name, len(used_seeds), ", ".join(used_seeds)
            )
        )
        print("rows: state after finishing task i | cols: validation on task j")
        print("=" * 78)
        print(format_matrix(mean, std, row_labels, col_labels))
        print()
        for key, summary in scalar_summary.items():
            deviation = summary["std"]
            suffix = "" if np.isnan(deviation) else " +/- {:.4f}".format(deviation)
            print("  {:<10} {:.4f}{}".format(key + ":", summary["mean"], suffix))

        write_matrix_csv(
            out_dir / "perf_{}.csv".format(name), mean, std, row_labels, col_labels
        )
        if make_plots:
            plot_heatmap(
                out_dir / "perf_{}.png".format(name),
                mean,
                row_labels,
                col_labels,
                "Task-wise performance matrix ({})".format(name),
                "mean over {} seed(s)".format(len(used_seeds)),
                name,
                mask_upper_triangle=True,
            )

        report["metrics"][name] = {
            "mean": mean.tolist(),
            "std": std.tolist(),
            "scalars": scalar_summary,
        }

    if mean_baseline is not None:
        report["baseline"] = mean_baseline.tolist()
    return report


def variant_tag(loader_name: str, masking: str) -> str:
    """File-name tag for a masking mode (``with_tid``, ``no_tid_cil``, ...)."""
    if variant_uses_tid(loader_name, masking):
        return "with_tid"
    if masking == "native":
        return "native_cil"
    return "no_tid_cil" if masking == "cil" else "no_tid_raw"


def report_task_confusion(
    per_seed_counts: List[np.ndarray],
    used_seeds: List[str],
    heading: str,
    out_dir: Path,
    make_plots: bool,
) -> Dict[str, Any]:
    """Average, print and save the true-task x predicted-task matrix."""
    n_tasks = per_seed_counts[0].shape[0]
    labels = ["task {}".format(index) for index in range(n_tasks)] + ["noise"]
    stack = stack_matrices(
        [row_normalise(counts) for counts in per_seed_counts],
        "predicted-task confusion",
    )
    mean, std = mean_and_std(stack)
    total_counts = np.sum(np.stack(per_seed_counts, axis=0), axis=0)

    # Drop the shared-noise column when no run has one.
    if total_counts[:, -1].sum() == 0:
        mean = mean[:, :-1]
        std = std[:, :-1]
        total_counts = total_counts[:, :-1]
        labels = labels[:-1]

    row_labels = ["true task {}".format(index) for index in range(mean.shape[0])]
    col_labels = list(labels)

    print()
    print("=" * 78)
    print(
        "Predicted-task confusion -- mean over {} seed(s): {}".format(
            len(used_seeds), ", ".join(used_seeds)
        )
    )
    print("rows: true task | cols: task owning the predicted class | " + heading)
    print("=" * 78)
    print(format_matrix(mean, std, row_labels, col_labels))
    print()
    diagonal = float(np.nanmean(np.diag(mean)))
    print("  task-identification rate (mean diagonal): {:.4f}".format(diagonal))

    write_matrix_csv(out_dir / "pred_confusion.csv", mean, std, row_labels, col_labels)
    write_matrix_csv(
        out_dir / "pred_confusion_counts.csv",
        total_counts.astype(float),
        None,
        row_labels,
        col_labels,
    )
    if make_plots:
        plot_heatmap(
            out_dir / "pred_confusion.png",
            mean,
            row_labels,
            col_labels,
            "True task vs predicted task",
            "row-normalised, mean over {} seed(s)".format(len(used_seeds)),
            "fraction of samples",
        )

    return {
        "labels": col_labels,
        "mean": mean.tolist(),
        "std": std.tolist(),
        "counts": total_counts.tolist(),
        "diagonal": diagonal,
    }


def report_class_confusion(
    per_seed_counts: List[np.ndarray],
    classes_per_task: Sequence[int],
    used_seeds: List[str],
    tag: str,
    title: str,
    heading: str,
    draw_task_boxes: bool,
    out_dir: Path,
    make_plots: bool,
) -> Dict[str, Any]:
    """Average, print a summary of, and save one true-class x predicted-class matrix.

    Args:
        per_seed_counts: ``(n_seen, n_seen + 1)`` count matrices, one per seed.
        classes_per_task: Class count of each seen task.
        used_seeds: Seed names, aligned with ``per_seed_counts``.
        tag: File-name tag of the masking mode.
        title: Masking-mode description for the plot title.
        heading: Run settings printed with the summary.
        draw_task_boxes: Outline each task's block (TID-masked view).
        out_dir: Output directory.
        make_plots: Whether to write the PNG.

    Returns:
        JSON-serialisable summary of the matrix.
    """
    # Classes without evaluation samples stay NaN rather than reading as 0.
    stack = stack_matrices(
        [row_normalise(counts, empty_value=np.nan) for counts in per_seed_counts],
        "class confusion ({})".format(tag),
    )
    with np.errstate(invalid="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        mean, std = mean_and_std(stack)
    total_counts = np.sum(np.stack(per_seed_counts, axis=0), axis=0)

    col_labels = class_labels_for_tasks(classes_per_task) + ["other"]
    if total_counts[:, -1].sum() == 0:
        mean = mean[:, :-1]
        std = std[:, :-1]
        total_counts = total_counts[:, :-1]
        col_labels = col_labels[:-1]
    row_labels = class_labels_for_tasks(classes_per_task)

    n_seen = len(row_labels)
    class_accuracy = np.diag(mean[:, :n_seen])
    owner = np.repeat(np.arange(len(classes_per_task)), classes_per_task)
    within_task = np.array(
        [np.nansum(mean[row, :n_seen][owner == owner[row]]) for row in range(n_seen)]
    )
    has_samples = np.isfinite(class_accuracy)

    print()
    print("-" * 78)
    print(
        "Class confusion [{}] -- mean over {} seed(s) | {}".format(
            title, len(used_seeds), heading
        )
    )
    print("-" * 78)
    print(
        "  mean per-class recall:            {:.4f}".format(
            float(np.mean(class_accuracy[has_samples])) if has_samples.any() else np.nan
        )
    )
    print(
        "  mass kept inside the true task:   {:.4f}".format(
            float(np.mean(within_task[has_samples])) if has_samples.any() else np.nan
        )
    )
    if "other" in col_labels:
        print(
            "  mass on classes of unseen tasks:  {:.4f}".format(
                float(np.nanmean(mean[:, -1]))
            )
        )

    stem = "pred_class_confusion_{}".format(tag)
    write_matrix_csv(out_dir / "{}.csv".format(stem), mean, std, row_labels, col_labels)
    write_matrix_csv(
        out_dir / "{}_counts.csv".format(stem),
        total_counts.astype(float),
        None,
        row_labels,
        col_labels,
    )
    if make_plots:
        plot_class_confusion(
            out_dir / "{}.png".format(stem),
            mean,
            classes_per_task,
            col_labels,
            "True class vs predicted class, {}".format(title),
            "row-normalised %, mean over {} seed(s) | {}".format(
                len(used_seeds), heading
            ),
            draw_task_boxes,
        )

    return {
        "masking": title,
        "row_labels": row_labels,
        "col_labels": col_labels,
        "mean": np.where(np.isfinite(mean), mean, None).tolist(),
        "counts": total_counts.tolist(),
        "per_class_recall": np.where(has_samples, class_accuracy, None).tolist(),
        "within_task": np.where(has_samples, within_task, None).tolist(),
    }


def run_pred_mode(
    seeds: List[Tuple[str, Path]],
    split: str,
    after_task: int,
    masking: str,
    signal_only: bool,
    class_breakdown: bool,
    device: torch.device,
    out_dir: Path,
    make_plots: bool,
) -> Dict[str, Any]:
    """Average the predicted-task (and optionally per-class) confusion over seeds.

    With ``class_breakdown`` a true-class x predicted-class matrix covering
    every seen task is also produced. Task-incremental runs get it twice: with
    the TID mask (task blocks outlined) and without it.
    """
    seed_results: List[Tuple[str, Dict[str, Any]]] = []
    for seed_name, seed_dir in seeds:
        print("[pred] seed {}: running inference...".format(seed_name))
        seed_results.append(
            (
                seed_name,
                confusion_for_seed(
                    seed_dir,
                    split,
                    after_task,
                    masking,
                    signal_only,
                    class_breakdown,
                    device,
                ),
            )
        )
    if not seed_results:
        return {}

    loader_names = {result["loader"] for _, result in seed_results}
    if len(loader_names) != 1:
        raise RuntimeError(
            "seeds mix loaders ({}); they are not the same experiment".format(
                sorted(loader_names)
            )
        )
    loader_name = loader_names.pop()
    used_seeds = [name for name, _ in seed_results]
    resolved_after = seed_results[0][1]["after_task"]
    heading = "split={} after task {}{}".format(
        split, resolved_after, " signal-only" if signal_only else ""
    )

    def counts_for(mode: str, key: str) -> List[np.ndarray] | None:
        found = [
            result["variants"][mode][key]
            for _, result in seed_results
            if mode in result["variants"]
        ]
        return found if len(found) == len(seed_results) else None

    report: Dict[str, Any] = {"seeds": used_seeds, "loader": loader_name}
    primary = counts_for(masking, "task_counts")
    if primary is None:
        print(
            "[skip] task confusion: masking={} unavailable for this model".format(
                masking
            )
        )
    else:
        report.update(
            report_task_confusion(
                primary,
                used_seeds,
                "{} masking={}".format(heading, masking),
                out_dir,
                make_plots,
            )
        )

    if class_breakdown:
        layouts = {tuple(result["classes_per_task"]) for _, result in seed_results}
        if len(layouts) != 1:
            raise RuntimeError(
                "seeds disagree on classes_per_task ({}); their class indices do "
                "not line up, so a per-class average is meaningless. Run one seed "
                "directory at a time.".format(sorted(layouts))
            )
        orders = {result["task_order"] for _, result in seed_results}
        if len(orders) != 1:
            raise RuntimeError(
                "seeds were trained on different task orders ({}); class indices "
                "do not refer to the same classes, so a per-class average is "
                "meaningless. Run one seed directory at a time.".format(sorted(orders))
            )
        classes_per_task = list(layouts.pop())
        report["class_breakdown"] = {}
        for mode in masking_variants(loader_name, masking, True):
            per_seed = counts_for(mode, "class_counts")
            if per_seed is None:
                print("[skip] class confusion: masking={} unavailable".format(mode))
                continue
            tag = variant_tag(loader_name, mode)
            report["class_breakdown"][tag] = report_class_confusion(
                per_seed,
                classes_per_task,
                used_seeds,
                tag,
                variant_title(loader_name, mode),
                heading,
                variant_uses_tid(loader_name, mode),
                out_dir,
                make_plots,
            )

    return report


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("run_dir", type=Path, help="Run directory (or one seed dir).")
    parser.add_argument(
        "--mode",
        choices=("perf", "pred", "both"),
        default="perf",
        help="perf: task-by-task metric matrix from logs (default). "
        "pred: true-task x predicted-task confusion, via inference.",
    )
    parser.add_argument(
        "--metric",
        choices=PERF_METRICS + ("all",),
        default="all",
        help="Which metric fills the perf matrix (default: all available).",
    )
    parser.add_argument(
        "--split",
        choices=("val", "test"),
        default="val",
        help="Evaluation split for pred mode (default: val).",
    )
    parser.add_argument(
        "--after-task",
        type=int,
        default=-1,
        help="Checkpoint to run pred mode on; -1 (default) = after the last task.",
    )
    parser.add_argument(
        "--masking",
        choices=("cil", "none", "native"),
        default="cil",
        help="Logit masking for pred mode. cil (default): one joint head over "
        "every class seen so far. none: raw head. native: the run's own masking "
        "(diagonal by construction for task-incremental runs).",
    )
    parser.add_argument(
        "--signal-only",
        action="store_true",
        help="Drop samples whose true label is the noise class.",
    )
    parser.add_argument(
        "--class-breakdown",
        action="store_true",
        help="pred mode: also plot a true-class x predicted-class matrix over "
        "every seen task. TIL runs get it with and without the TID mask.",
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device for pred-mode inference.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Where to write CSV/PNG output (default: <run_dir>/task_confusion).",
    )
    parser.add_argument("--no-plot", action="store_true", help="Skip the heatmap PNGs.")
    parser.add_argument(
        "--json",
        type=Path,
        default=None,
        help="Also write the full report as JSON to this path.",
    )
    return parser.parse_args()


def main() -> int:
    """Entry point."""
    args = parse_args()
    # Saved args carry dataset paths relative to the repo root.
    os.chdir(REPO_ROOT)

    run_dir = args.run_dir.resolve()
    if not run_dir.is_dir():
        print("No such run directory: {}".format(run_dir), file=sys.stderr)
        return 2

    seeds = discover_seed_dirs(run_dir)
    if not seeds:
        print(
            "No seed directories with results.pt under {}".format(run_dir),
            file=sys.stderr,
        )
        return 2

    out_dir = args.out_dir.resolve() if args.out_dir else run_dir / "task_confusion"
    metrics_wanted = PERF_METRICS if args.metric == "all" else (args.metric,)
    report: Dict[str, Any] = {"run_dir": str(run_dir), "seeds": [s for s, _ in seeds]}

    try:
        if args.mode in ("perf", "both"):
            report["perf"] = run_perf_mode(
                seeds, metrics_wanted, out_dir, not args.no_plot
            )

        if args.mode in ("pred", "both"):
            report["pred"] = run_pred_mode(
                seeds,
                args.split,
                args.after_task,
                args.masking,
                args.signal_only,
                args.class_breakdown,
                torch.device(args.device),
                out_dir,
                not args.no_plot,
            )
    except (RuntimeError, FileNotFoundError, ValueError) as error:
        print("\nError: {}".format(error), file=sys.stderr)
        return 1

    print()
    print("Wrote output to {}".format(out_dir))
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        with open(args.json, "w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2)
        print("Wrote JSON report to {}".format(args.json))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
