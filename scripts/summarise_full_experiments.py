#!/usr/bin/env python
"""Summarise full_experiments logs by algorithm performance.

Parses a `full_experiments_*.log` file and prints, for each algorithm:
- status (completed or failed)
- total classification accuracy (validation)
- total detection recall and false-alarm rate
- training precision and F1 (from last epoch line when present; left blank if
  the log only has "Train Acc | Val Acc" to avoid misleading values)
- time taken (from results line when completed, else sum of Epoch Time)

Time is computed per algorithm as follows:
- If the run completed: the value is taken from the final results line
  (regex RESULTS_TIME_RE matches "# val: ... # <seconds>" at end of line).
- If the run did not finish or that line is missing: time is the sum of all
  "Epoch Time <seconds>s" lines (EPOCH_TIME_RE) seen for that algorithm.

Usage:
    python scripts/summarise_full_experiments.py --log \
        logs/full_experiments/full_experiments_20260305_175812.log
    python scripts/summarise_full_experiments.py --log \
        logs/full_experiments/run_20260325_154832_lnx-elkk-1

If --log is omitted, the newest `run_*` directory under
`logs/full_experiments/` is used when available; otherwise the newest
`full_experiments_*.log` file is used.

Memory sweep comparison (TR and TE: rec, prec, f1):
    python scripts/summarise_full_experiments.py --mem-compare-runs \\
        logs/full_experiments/run_20260511_221802_lnx-elkk-1_mem_512 \\
        logs/full_experiments/run_20260513_173647_lnx-elkk-1_mem_1024
    python scripts/summarise_full_experiments.py --log run_A --mem-compare-runs run_B run_C
    python scripts/summarise_full_experiments.py --mem-compare-runs \\
        logs/full_experiments/run_*_mem_* \\
        --base-mem logs/full_experiments/full-til_10epochs_w-zs

Task order seed sweep (same table layout; label from ``_task_order_seed_<n>``):
    python scripts/summarise_full_experiments.py --task-order-seed-compare-runs \\
        logs/full_experiments/run_20260101_120000_lnx-elkk-1_task_order_seed_57 \\
        logs/full_experiments/run_20260101_180000_lnx-elkk-1_task_order_seed_1040
    python scripts/summarise_full_experiments.py --task-order-seed-compare-runs \\
        logs/full_experiments/run_*_task_order_seed_* \\
        --base-seed logs/full_experiments/full-til_10epochs_w-zs

Multi-seed summary (mean± over seed-* folders; sharded runs merged per seed):
    python scripts/summarise_full_experiments.py --multi-seed \\
        logs/full_experiments/one-shot_til
"""

from __future__ import annotations

import argparse
import glob
import os
import re
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from collections import Counter
from typing import Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils.metric_keys import metric_aliases  # noqa: E402


@dataclass
class AlgoSummary:
    name: str
    status: str = "unknown"
    exit_code: Optional[int] = None
    cls_rec_tr: Optional[float] = None
    cls_prec_tr: Optional[float] = None
    cls_f1_tr: Optional[float] = None
    cls_rec_te: Optional[float] = None
    cls_prec_te: Optional[float] = None
    cls_f1_te: Optional[float] = None
    size_gb: Optional[float] = None
    time_sec: Optional[float] = None


RUN_RE = re.compile(r"--- Running: base \+ (?P<algo>[\w\-]+) ---")
DISPATCH_RE = re.compile(
    r"--- Dispatching:\s+base \+\s+(?P<algo>[\w\-]+)\s+\(job log:\s+(?P<path>[^)]+)\)\s+---"
)
COMPLETED_RE = re.compile(r"Completed:\s+(?P<algo>[\w\-]+)\s+\(exit\s+(?P<code>\d+)\)")
ERROR_RE = re.compile(
    r"ERROR:\s+(?P<algo>[\w\-]+)\s+failed with exit code\s+(?P<code>\d+)"
)
TOTAL_ACC_RE = re.compile(r"Total Accuracy:\s+(?P<val>[0-9.]+)")
# Results line: ... # val: ... # 2276.83
RESULTS_TIME_RE = re.compile(r"#\s+val:\s+[^#]+#\s+(?P<sec>[0-9.]+)\s*$")
# Epoch line with Prec and F1: Task 0 Epoch 1/1 | Loss 1.38 | Train Acc 0.47 | Prec 0.50 | F1 0.45 | Epoch Time ...
EPOCH_PREC_F1_RE = re.compile(
    r"Task\s+\d+\s+Epoch\s+\d+/\d+\s+\|\s+Loss\s+[0-9.]+\s+\|\s+Train Acc\s+[0-9.]+\s+\|\s+Prec\s+(?P<prec>[0-9.]+)\s+\|\s+F1\s+(?P<f1>[0-9.]+)\s+\|"
)
# Epoch Time for fallback when results line has no time (e.g. incomplete run)
EPOCH_TIME_RE = re.compile(r"Epoch Time\s+(?P<sec>[0-9.]+)s")
# ``macro_*`` is the current spelling; ``cls_*`` is accepted so that runs logged
# before the detection-metric removal still parse. Trailing ``det=``/``fa=``
# tokens from those older logs are ignored.
_SUMMARY_FIELDS = (
    r"(?:(?:macro|cls)_rec=(?P<cls_rec>[0-9.]+)\s*)?"
    r"(?:(?:macro|cls)_prec=(?P<cls_prec>[0-9.]+)\s*)?"
    r"(?:(?:macro|cls)_f1=(?P<cls_f1>[0-9.]+)\s*)?"
)
SUMMARY_TR_RE = re.compile(r"SUMMARY_TR\s+" + _SUMMARY_FIELDS)
SUMMARY_TE_RE = re.compile(r"SUMMARY_TE\s+" + _SUMMARY_FIELDS)
MODEL_SIZE_RE = re.compile(r"Model size:\s+(?P<gb>[0-9.]+)\s+GB")
LOG_TIMESTAMP_RE = re.compile(r"^\[(?P<stamp>\d{4}-\d{2}-\d{2}T[^]]+)\]")
LOGGING_TO_RE = re.compile(r"Logging to\s+(?P<path>\S+)")
RESULTS_DICT_RE = re.compile(r"'log_dir':\s*'(?P<path>[^']+)'")
RUN_MEM_SUFFIX_RE = re.compile(r"_mem_(?P<mem>\d+)$")
RUN_TASK_ORDER_SEED_SUFFIX_RE = re.compile(r"_task_order_seed_(?P<seed>\d+)$")
SEED_DIRECTORY_RE = re.compile(r"^seed[-_](?P<seed>\d+)$")
BASE_SWEEP_LABEL = "base"
SweepLabelValue = Union[int, str]
SweepMetricValues = Tuple[
    Optional[float],
    Optional[float],
    Optional[float],
    Optional[float],
    Optional[float],
    Optional[float],
]


def _find_default_log() -> str:
    run_dirs = sorted(glob.glob(os.path.join("logs", "full_experiments", "run_*")))
    if run_dirs:
        return run_dirs[-1]

    candidates = sorted(
        glob.glob(os.path.join("logs", "full_experiments", "full_experiments_*.log"))
    )
    if candidates:
        return candidates[-1]
    raise SystemExit("No full_experiments logs found.")


def parse_log(path: str) -> Dict[str, AlgoSummary]:
    summaries: Dict[str, AlgoSummary] = {}
    current_algo: Optional[str] = None

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            # Running block
            m = RUN_RE.search(line)
            if m:
                current_algo = m.group("algo")
                summaries.setdefault(current_algo, AlgoSummary(name=current_algo))
                continue

            # Completion / error lines
            m = COMPLETED_RE.search(line)
            if m:
                algo = m.group("algo")
                code = int(m.group("code"))
                s = summaries.setdefault(algo, AlgoSummary(name=algo))
                s.status = "completed" if code == 0 else "failed"
                s.exit_code = code
                continue

            m = ERROR_RE.search(line)
            if m:
                algo = m.group("algo")
                code = int(m.group("code"))
                s = summaries.setdefault(algo, AlgoSummary(name=algo))
                s.status = "failed"
                s.exit_code = code
                continue

            # Metrics: only make sense in context of current_algo
            if current_algo is None:
                continue

            s = summaries.setdefault(current_algo, AlgoSummary(name=current_algo))

            m = SUMMARY_TR_RE.search(line)
            if m:
                if m.group("cls_rec"):
                    s.cls_rec_tr = float(m.group("cls_rec"))
                if m.group("cls_prec"):
                    s.cls_prec_tr = float(m.group("cls_prec"))
                if m.group("cls_f1"):
                    s.cls_f1_tr = float(m.group("cls_f1"))
                continue

            m = SUMMARY_TE_RE.search(line)
            if m:
                if m.group("cls_rec"):
                    s.cls_rec_te = float(m.group("cls_rec"))
                if m.group("cls_prec"):
                    s.cls_prec_te = float(m.group("cls_prec"))
                if m.group("cls_f1"):
                    s.cls_f1_te = float(m.group("cls_f1"))
                continue

            m = MODEL_SIZE_RE.search(line)
            if m:
                s.size_gb = float(m.group("gb"))
                continue

            m = TOTAL_ACC_RE.search(line)
            if m:
                if s.cls_rec_te is None:
                    s.cls_rec_te = float(m.group("val"))
                continue

                continue

                continue

            m = EPOCH_PREC_F1_RE.search(line)
            if m:
                if s.cls_prec_tr is None:
                    s.cls_prec_tr = float(m.group("prec"))
                if s.cls_f1_tr is None:
                    s.cls_f1_tr = float(m.group("f1"))
                continue

            m = EPOCH_TIME_RE.search(line)
            if m:
                epoch_sec = float(m.group("sec"))
                s.time_sec = (s.time_sec or 0.0) + epoch_sec
                continue

            m = RESULTS_TIME_RE.search(line)
            if m:
                s.time_sec = float(m.group("sec"))
                continue

    return summaries


def parse_job_log(path: str, algo_name: Optional[str] = None) -> Dict[str, AlgoSummary]:
    summaries: Dict[str, AlgoSummary] = {}
    current_algo = algo_name

    with open(path, "r", encoding="utf-8") as file_handle:
        for line in file_handle:
            if current_algo is None:
                match = re.search(r"Running model:\s+(?P<algo>[\w\-]+)", line)
                if match:
                    current_algo = match.group("algo")
                    summaries.setdefault(current_algo, AlgoSummary(name=current_algo))
                    continue
                continue

            summary = summaries.setdefault(current_algo, AlgoSummary(name=current_algo))

            match = SUMMARY_TR_RE.search(line)
            if match:
                if match.group("cls_rec"):
                    summary.cls_rec_tr = float(match.group("cls_rec"))
                if match.group("cls_prec"):
                    summary.cls_prec_tr = float(match.group("cls_prec"))
                if match.group("cls_f1"):
                    summary.cls_f1_tr = float(match.group("cls_f1"))
                continue

            match = SUMMARY_TE_RE.search(line)
            if match:
                if match.group("cls_rec"):
                    summary.cls_rec_te = float(match.group("cls_rec"))
                if match.group("cls_prec"):
                    summary.cls_prec_te = float(match.group("cls_prec"))
                if match.group("cls_f1"):
                    summary.cls_f1_te = float(match.group("cls_f1"))
                continue

            match = MODEL_SIZE_RE.search(line)
            if match:
                summary.size_gb = float(match.group("gb"))
                continue

            match = TOTAL_ACC_RE.search(line)
            if match and summary.cls_rec_te is None:
                summary.cls_rec_te = float(match.group("val"))
                continue

                continue

                continue

            match = EPOCH_PREC_F1_RE.search(line)
            if match:
                if summary.cls_prec_tr is None:
                    summary.cls_prec_tr = float(match.group("prec"))
                if summary.cls_f1_tr is None:
                    summary.cls_f1_tr = float(match.group("f1"))
                continue

            match = EPOCH_TIME_RE.search(line)
            if match:
                epoch_sec = float(match.group("sec"))
                summary.time_sec = (summary.time_sec or 0.0) + epoch_sec
                continue

            match = RESULTS_TIME_RE.search(line)
            if match:
                summary.time_sec = float(match.group("sec"))
                continue

            match = re.search(r"Total runtime:\s+(?P<hours>[0-9.]+)\s+hours", line)
            if match:
                summary.time_sec = float(match.group("hours")) * 3600

    return summaries


def _to_float_or_none(raw_value: object) -> Optional[float]:
    """Convert an NP value/array to the latest finite float.

    Args:
        raw_value: Raw object loaded from an NPZ entry.

    Returns:
        Last finite numeric value if available, otherwise None.
    """
    array = np.asarray(raw_value)
    if array.size == 0:
        return None
    flattened_values = array.astype(float).reshape(-1)
    finite_values = flattened_values[np.isfinite(flattened_values)]
    if finite_values.size == 0:
        return None
    return float(finite_values[-1])


def _extract_latest_metric(
    metrics_data: np.lib.npyio.NpzFile, candidate_keys: List[str]
) -> Optional[float]:
    """Return the latest finite value from the first present metric key.

    Args:
        metrics_data: Opened task NPZ payload.
        candidate_keys: Ordered keys to try.

    Returns:
        Latest finite float from the first matching key, otherwise None.
    """
    for metric_key in candidate_keys:
        if metric_key in metrics_data:
            metric_value = _to_float_or_none(metrics_data[metric_key])
            if metric_value is not None:
                return metric_value
    return None


def _extract_log_dir_from_job_log(job_log_path: str) -> Optional[str]:
    """Extract the per-run logging directory from a job log.

    Args:
        job_log_path: Path to a single algorithm job log.

    Returns:
        Reported run directory path if found, else None.
    """
    extracted_log_dir: Optional[str] = None
    with open(job_log_path, "r", encoding="utf-8") as file_handle:
        for line in file_handle:
            logging_match = LOGGING_TO_RE.search(line)
            if logging_match:
                extracted_log_dir = logging_match.group("path").strip()
            results_match = RESULTS_DICT_RE.search(line)
            if results_match:
                extracted_log_dir = results_match.group("path").strip()
    return extracted_log_dir


def _resolve_existing_log_dir(job_log_path: str, raw_log_dir: str) -> Optional[Path]:
    """Resolve a raw run directory string to an existing path.

    Args:
        job_log_path: Path to the job log containing the directory hint.
        raw_log_dir: Directory string parsed from that job log.

    Returns:
        Existing directory path if found, otherwise None.
    """
    cleaned_relative_path = raw_log_dir.strip().replace("//", "/").lstrip("./")
    candidate_paths: List[Path] = []
    parsed_path = Path(cleaned_relative_path)
    if parsed_path.is_absolute():
        candidate_paths.append(parsed_path)
    else:
        candidate_paths.append(Path.cwd() / parsed_path)
        candidate_paths.append(Path(job_log_path).resolve().parent / parsed_path)
    for candidate_path in candidate_paths:
        if candidate_path.exists():
            return candidate_path
    return None


def _fill_missing_metrics_from_npz(
    summary: AlgoSummary, metrics_file_path: Path
) -> None:
    """Populate missing summary metrics from a task metrics NPZ.

    Args:
        summary: Algorithm summary object to augment in-place.
        metrics_file_path: Path to the selected task metrics file.

    Returns:
        None.
    """
    with np.load(metrics_file_path, allow_pickle=False) as metrics_data:
        train_recall = _extract_latest_metric(
            metrics_data, metric_aliases("train_macro_rec")
        )
        train_precision = _extract_latest_metric(
            metrics_data, metric_aliases("train_macro_prec")
        )
        train_f1 = _extract_latest_metric(
            metrics_data, metric_aliases("train_macro_f1")
        )

        validation_recall = _extract_latest_metric(
            metrics_data, metric_aliases("val_macro_rec_per_epoch")
        )
        validation_precision = _extract_latest_metric(
            metrics_data, metric_aliases("val_macro_prec_per_epoch")
        )
        validation_f1 = _extract_latest_metric(
            metrics_data, metric_aliases("val_macro_f1")
        )

    if summary.cls_rec_tr is None:
        summary.cls_rec_tr = train_recall
    if summary.cls_prec_tr is None:
        summary.cls_prec_tr = train_precision
    if summary.cls_f1_tr is None:
        summary.cls_f1_tr = train_f1

    if summary.cls_rec_te is None:
        summary.cls_rec_te = validation_recall
    if summary.cls_prec_te is None:
        summary.cls_prec_te = validation_precision
    if summary.cls_f1_te is None:
        summary.cls_f1_te = validation_f1


def _apply_metrics_fallback_from_job_log(
    summary: AlgoSummary, job_log_path: str
) -> None:
    """Use task metrics NPZ files when terminal log metrics are missing.

    Args:
        summary: Algorithm summary object to augment in-place.
        job_log_path: Path to the algorithm job log.

    Returns:
        None.
    """
    raw_log_directory = _extract_log_dir_from_job_log(job_log_path)
    if not raw_log_directory:
        return
    resolved_log_directory = _resolve_existing_log_dir(job_log_path, raw_log_directory)
    if resolved_log_directory is None:
        return
    task_metric_paths = sorted(
        resolved_log_directory.glob("metrics/task*.npz"),
        key=lambda path: int(path.stem.replace("task", "")),
    )
    if not task_metric_paths:
        return
    _fill_missing_metrics_from_npz(summary, task_metric_paths[-1])


def _merge_summary(into: AlgoSummary, source: AlgoSummary) -> None:
    for field_name in (
        "cls_rec_tr",
        "cls_prec_tr",
        "cls_f1_tr",
        "cls_rec_te",
        "cls_prec_te",
        "cls_f1_te",
        "size_gb",
        "time_sec",
    ):
        value = getattr(source, field_name)
        if value is not None:
            setattr(into, field_name, value)
    if source.exit_code is not None:
        into.exit_code = source.exit_code
        into.status = source.status


def _resolve_job_log_path(run_directory: str, job_log_path: str) -> Optional[str]:
    """Resolve a coordinator job log path, including after host or prefix moves.

    Args:
        run_directory: Coordinator run directory containing ``job_logs/``.
        job_log_path: Path recorded in the coordinator log (may be absolute elsewhere).

    Returns:
        Existing path to read, or None if no matching file is found.

    Usage:
        >>> _resolve_job_log_path(
        ...     "logs/full_experiments/run_20260101_120000_lnx-elkk-1",
        ...     "logs/full_experiments/run_20260101_120000_lnx-elkk-1/job_logs/job_gem.log",
        ... )
    """
    if os.path.isfile(job_log_path):
        return job_log_path
    job_log_file_name = os.path.basename(job_log_path)
    candidate_under_run = os.path.join(run_directory, "job_logs", job_log_file_name)
    if os.path.isfile(candidate_under_run):
        return candidate_under_run
    return None


def _discover_coordinator_run_directories(parent_directory: str) -> List[str]:
    """List coordinator run directories under a parent or return the parent itself.

    Args:
        parent_directory: Either a coordinator run (contains ``full_experiments_*.log``)
            or a parent folder whose immediate children are coordinator runs
            (e.g. ``logs/full_experiments/full-til_10epochs_w-zs``).

    Returns:
        Sorted list of coordinator run directory paths.

    Raises:
        SystemExit: If ``parent_directory`` is missing or contains no coordinator runs.

    Usage:
        >>> _discover_coordinator_run_directories(
        ...     "logs/full_experiments/full-til_10epochs_w-zs"
        ... )  # doctest: +SKIP
    """
    parent_path = Path(parent_directory)
    if not parent_path.is_dir():
        raise SystemExit(f"Not a directory: {parent_directory}")
    if sorted(parent_path.glob("full_experiments_*.log")):
        return [str(parent_path)]
    coordinator_runs = sorted(
        child_path
        for child_path in parent_path.iterdir()
        if child_path.is_dir() and sorted(child_path.glob("full_experiments_*.log"))
    )
    if not coordinator_runs:
        raise SystemExit(
            f"No coordinator run directories (full_experiments_*.log) found under: "
            f"{parent_directory}"
        )
    return [str(coordinator_run) for coordinator_run in coordinator_runs]


def _discover_seed_directories(parent_directory: str) -> List[Tuple[int, str]]:
    """List immediate ``seed-<n>`` or ``seed_<n>`` subdirectories under a parent folder.

    Args:
        parent_directory: Parent folder containing per-seed experiment trees
            (e.g. ``logs/full_experiments/one-shot_til``).

    Returns:
        Sorted list of ``(seed_value, seed_directory_path)`` pairs.

    Raises:
        SystemExit: If ``parent_directory`` is missing or has no seed subdirectories.

    Usage:
        >>> _discover_seed_directories("logs/full_experiments/one-shot_til")  # doctest: +SKIP
    """
    parent_path = Path(parent_directory)
    if not parent_path.is_dir():
        raise SystemExit(f"Not a directory: {parent_directory}")
    seed_directories: List[Tuple[int, str]] = []
    for child_path in sorted(parent_path.iterdir()):
        if not child_path.is_dir():
            continue
        match = SEED_DIRECTORY_RE.match(child_path.name)
        if match:
            seed_directories.append((int(match.group("seed")), str(child_path)))
    if not seed_directories:
        raise SystemExit(
            f"No seed-* or seed_* subdirectories found under: {parent_directory}"
        )
    return seed_directories


def _discover_coordinator_runs_under(root_directory: str) -> List[str]:
    """Recursively find coordinator run directories under a seed folder.

    Args:
        root_directory: Seed directory or any subtree to search.

    Returns:
        Sorted unique paths to directories containing ``full_experiments_*.log``.

    Raises:
        SystemExit: If no coordinator runs are found.

    Usage:
        >>> _discover_coordinator_runs_under(
        ...     "logs/full_experiments/one-shot_til/seed-39"
        ... )  # doctest: +SKIP
    """
    root_path = Path(root_directory)
    if not root_path.is_dir():
        raise SystemExit(f"Not a directory: {root_directory}")
    run_directories: set[str] = set()
    for log_file in root_path.rglob("full_experiments_*.log"):
        run_directories.add(str(log_file.parent))
    if not run_directories:
        raise SystemExit(
            f"No coordinator run directories (full_experiments_*.log) found under: "
            f"{root_directory}"
        )
    return sorted(run_directories)


def _summarize_seed_directory(seed_directory: str) -> Dict[str, AlgoSummary]:
    """Parse and merge all coordinator runs under one seed directory.

    Args:
        seed_directory: Path to a ``seed-<n>`` folder (may contain sharded host runs).

    Returns:
        Merged algorithm summaries for that seed.

    Usage:
        >>> _summarize_seed_directory(
        ...     "logs/full_experiments/one-shot_til/seed-39"
        ... )  # doctest: +SKIP
    """
    coordinator_run_paths = _discover_coordinator_runs_under(seed_directory)
    summary_collections = [
        parse_run_directory(run_path) for run_path in coordinator_run_paths
    ]
    return _merge_many_summaries(summary_collections)


def parse_run_directory(path: str) -> Dict[str, AlgoSummary]:
    main_log_candidates = sorted(
        glob.glob(os.path.join(path, "full_experiments_*.log"))
    )
    if not main_log_candidates:
        raise SystemExit(f"No full_experiments_*.log found under run directory: {path}")
    main_log_path = main_log_candidates[-1]

    summaries = parse_log(main_log_path)
    job_logs_by_algo: Dict[str, str] = {}

    with open(main_log_path, "r", encoding="utf-8") as file_handle:
        for line in file_handle:
            dispatch_match = DISPATCH_RE.search(line)
            if dispatch_match:
                algo_name = dispatch_match.group("algo")
                job_logs_by_algo[algo_name] = dispatch_match.group("path").strip()

    if not job_logs_by_algo:
        for job_log_path in sorted(
            glob.glob(os.path.join(path, "job_logs", "job_*.log"))
        ):
            file_name = os.path.basename(job_log_path)
            job_match = re.match(
                r"job_(?P<algo>[\w\-]+)_\d{8}_\d{6}_\d+\.log", file_name
            )
            if not job_match:
                continue
            job_logs_by_algo[job_match.group("algo")] = job_log_path

    for algo_name, job_log_path in job_logs_by_algo.items():
        resolved_job_log_path = _resolve_job_log_path(path, job_log_path)
        if resolved_job_log_path is None:
            continue
        job_summaries = parse_job_log(resolved_job_log_path, algo_name=algo_name)
        if algo_name in job_summaries:
            summary = summaries.setdefault(algo_name, AlgoSummary(name=algo_name))
            _merge_summary(summary, job_summaries[algo_name])
            _apply_metrics_fallback_from_job_log(summary, resolved_job_log_path)

    return summaries


def parse_path(path: str) -> Dict[str, AlgoSummary]:
    if os.path.isdir(path):
        return parse_run_directory(path)
    return parse_log(path)


def _format_time(sec: Optional[float]) -> str:
    if sec is None:
        return "    -"
    if sec >= 3600:
        return f"{sec / 3600:.1f}h"
    if sec >= 60:
        return f"{sec / 60:.1f}m"
    return f"{sec:.0f}s"


def _fmt(v: Optional[float]) -> str:
    return f"{v:.3f}" if v is not None else "  -"


def _classification_f1_from_recall_precision(
    recall_value: Optional[float], precision_value: Optional[float]
) -> Optional[float]:
    if recall_value is None or precision_value is None:
        return None
    denominator = recall_value + precision_value
    if denominator == 0:
        return 0.0
    return 2 * (recall_value * precision_value) / denominator


def _compute_concurrent_runtime_seconds(log_path: str) -> Optional[float]:
    """Estimate wall-clock concurrent runtime from file timestamps."""
    if os.path.isdir(log_path):
        candidates = sorted(glob.glob(os.path.join(log_path, "full_experiments_*.log")))
        if not candidates:
            return None
        coordinator_log_path = Path(candidates[-1])
    else:
        coordinator_log_path = Path(log_path)

    file_stat = coordinator_log_path.stat()
    creation_timestamp = getattr(file_stat, "st_birthtime", None)
    if creation_timestamp is None:
        with open(coordinator_log_path, "r", encoding="utf-8") as file_handle:
            first_line = file_handle.readline()
        timestamp_match = LOG_TIMESTAMP_RE.search(first_line)
        if timestamp_match:
            creation_timestamp = datetime.fromisoformat(
                timestamp_match.group("stamp")
            ).timestamp()
        else:
            creation_timestamp = file_stat.st_ctime
    modification_timestamp = file_stat.st_mtime
    concurrent_runtime_seconds = modification_timestamp - creation_timestamp
    return max(0.0, concurrent_runtime_seconds)


def print_summary(
    summaries: Dict[str, AlgoSummary],
    concurrent_runtime_seconds: Optional[float] = None,
    output_format: str = "readable",
) -> None:
    if not summaries:
        print("No algorithm runs found in log.")
        return

    sorted_summaries = sorted(
        summaries.values(),
        key=lambda summary: (
            summary.cls_f1_te is None,
            summary.cls_f1_te if summary.cls_f1_te is not None else float("inf"),
            summary.name,
        ),
    )
    if output_format == "markdown":
        markdown_header_columns = [
            "Algo",
            "Exit",
            "TR rec",
            "TR prec",
            "TR F1_c",
            "TR f1",
            "TE rec",
            "TE prec",
            "TE F1_c",
            "TE f1",
            "Size_GB",
            "Time",
        ]
        print("| " + " | ".join(markdown_header_columns) + " |")
        print("| " + " | ".join(["---"] * len(markdown_header_columns)) + " |")
    else:
        w_algo, w_exit = 8, 5
        w_num = 7
        w_time = 8
        header = (
            f"{'Algo':<{w_algo}} {'Exit':<{w_exit}} "
            f"{'rec':>{w_num}} {'prec':>{w_num}} {'F1_c':>{w_num}} {'f1':>{w_num}} "
            f"| "
            f"{'rec':>{w_num}} {'prec':>{w_num}} {'F1_c':>{w_num}} {'f1':>{w_num}} "
            f"{'Size_GB':>{w_num}} {'Time':>{w_time}}"
        )
        print(header)
        print("-" * len(header))

    for s in sorted_summaries:
        train_classification_f1 = _classification_f1_from_recall_precision(
            s.cls_rec_tr, s.cls_prec_tr
        )
        test_classification_f1 = _classification_f1_from_recall_precision(
            s.cls_rec_te, s.cls_prec_te
        )
        time_str = _format_time(s.time_sec)
        if output_format == "markdown":
            size_str = f"{s.size_gb:.3f}" if s.size_gb is not None else "-"
            exit_str = str(s.exit_code) if s.exit_code is not None else "-"
            row_values = [
                s.name,
                exit_str,
                _fmt(s.cls_rec_tr).strip(),
                _fmt(s.cls_prec_tr).strip(),
                _fmt(train_classification_f1).strip(),
                _fmt(s.cls_f1_tr).strip(),
                _fmt(s.cls_rec_te).strip(),
                _fmt(s.cls_prec_te).strip(),
                _fmt(test_classification_f1).strip(),
                _fmt(s.cls_f1_te).strip(),
                size_str,
                time_str,
            ]
            print("| " + " | ".join(row_values) + " |")
        else:
            size_str = f"{s.size_gb:.3f}" if s.size_gb is not None else "  -"
            exit_str = str(s.exit_code) if s.exit_code is not None else ""
            print(
                f"{s.name:<{w_algo}} {exit_str:<{w_exit}} "
                f"{_fmt(s.cls_rec_tr):>{w_num}} {_fmt(s.cls_prec_tr):>{w_num}} "
                f"{_fmt(train_classification_f1):>{w_num}} {_fmt(s.cls_f1_tr):>{w_num}} "
                f"| "
                f"{_fmt(s.cls_rec_te):>{w_num}} {_fmt(s.cls_prec_te):>{w_num}} "
                f"{_fmt(test_classification_f1):>{w_num}} {_fmt(s.cls_f1_te):>{w_num}} "
                f"{size_str:>{w_num}} {time_str:>{w_time}}"
            )

    total_serial_seconds = sum(
        summary.time_sec
        for summary in summaries.values()
        if summary.time_sec is not None
    )
    if total_serial_seconds > 0:
        if output_format == "markdown":
            print()
            print(
                f"- Total serial time (all models): `{_format_time(total_serial_seconds)}`"
            )
            if concurrent_runtime_seconds is not None:
                print(
                    "- Total concurrent time (coordinator log wall-clock): "
                    f"`{_format_time(concurrent_runtime_seconds)}`"
                )
        else:
            print(
                f"Total serial time (all models): {_format_time(total_serial_seconds)}"
            )
            if concurrent_runtime_seconds is not None:
                print(
                    "Total concurrent time (coordinator log wall-clock): "
                    f"{_format_time(concurrent_runtime_seconds)}"
                )


def _parse_memory_buffer_size_from_run_directory(run_directory: str) -> Optional[int]:
    """Extract replay buffer size from a run folder name ending in ``_mem_<n>``.

    Args:
        run_directory: Path to a coordinator run directory whose basename may
            end with ``_mem_<digits>`` (as produced by ``full_experiments_mem_sweep.sh``).

    Returns:
        Parsed non-negative buffer size if the basename matches, else None.

    Usage:
        >>> _parse_memory_buffer_size_from_run_directory(
        ...     "logs/full_experiments/run_20260511_221802_lnx-elkk-1_mem_512"
        ... )
        512
    """
    base_name = Path(run_directory).name
    match = RUN_MEM_SUFFIX_RE.search(base_name)
    if match:
        return int(match.group("mem"))
    return None


def _sweep_label_sort_key(
    sweep_label_value: Optional[SweepLabelValue],
) -> Tuple[int, float | str]:
    """Order sweep labels with baseline ``base`` first, then numeric sweep values.

    Args:
        sweep_label_value: Parsed mem/seed integer, ``base``, or None.

    Returns:
        Tuple suitable for sorting comparison table rows.

    Usage:
        >>> _sweep_label_sort_key("base")
        (0, 0)
        >>> _sweep_label_sort_key(57)
        (1, 57.0)
    """
    if sweep_label_value == BASE_SWEEP_LABEL:
        return (0, 0)
    if sweep_label_value is None:
        return (2, float("inf"))
    if isinstance(sweep_label_value, int):
        return (1, float(sweep_label_value))
    return (1, str(sweep_label_value))


def _parse_task_order_seed_from_run_directory(run_directory: str) -> Optional[int]:
    """Extract task-order seed from a run folder name ending in ``_task_order_seed_<n>``.

    Args:
        run_directory: Path to a coordinator run directory whose basename may
            end with ``_task_order_seed_<digits>`` (as produced by
            ``full_experiments_task_order_seed_sweep.sh``).

    Returns:
        Parsed seed if the basename matches, else None.

    Usage:
        >>> _parse_task_order_seed_from_run_directory(
        ...     "logs/full_experiments/run_20260101_120000_lnx-elkk-1_task_order_seed_57"
        ... )
        57
    """
    base_name = Path(run_directory).name
    match = RUN_TASK_ORDER_SEED_SUFFIX_RE.search(base_name)
    if match:
        return int(match.group("seed"))
    return None


def _print_sweep_algorithm_separator(
    output_format: str,
    separator_width: int,
) -> None:
    """Print a dashed line between algorithm groups in a sweep comparison table.

    Args:
        output_format: ``readable`` or ``markdown``.
        separator_width: Character width for readable-mode dashes.

    Returns:
        None.

    Usage:
        >>> _print_sweep_algorithm_separator("readable", 40)  # doctest: +SKIP
    """
    if output_format == "markdown":
        print()
        print("---")
        print()
        return
    print("-" * separator_width)


def _metric_values_from_summary(summary: AlgoSummary) -> SweepMetricValues:
    """Extract TR/TE sweep table metrics from an algorithm summary.

    Args:
        summary: Parsed algorithm summary.

    Returns:
        Tuple of train rec, prec, f1, then test rec, prec, f1.
    """
    return (
        summary.cls_rec_tr,
        summary.cls_prec_tr,
        summary.cls_f1_tr,
        summary.cls_rec_te,
        summary.cls_prec_te,
        summary.cls_f1_te,
    )


def _format_mean_plus_minus(values: Sequence[Optional[float]]) -> str:
    """Format mean and sample standard deviation for a list of metric values.

    Args:
        values: Metric values from sweep runs (``None`` entries are skipped).

    Returns:
        ``mean±std`` when at least two values exist, else mean only, else ``  -``.

    Usage:
        >>> _format_mean_plus_minus([0.5, 0.6, 0.7])
        '0.600±0.100'
    """
    numeric_values = [float(value) for value in values if value is not None]
    if not numeric_values:
        return "  -"
    mean_value = float(np.mean(numeric_values))
    if len(numeric_values) < 2:
        return f"{mean_value:.3f}"
    std_value = float(np.std(numeric_values, ddof=1))
    return f"{mean_value:.3f}±{std_value:.3f}"


def _aggregate_sweep_mean_pm_cells(
    sweep_metric_rows: Sequence[SweepMetricValues],
) -> List[str]:
    """Compute mean± cells for each TR/TE metric column from sweep-only rows.

    Args:
        sweep_metric_rows: Per-run metric tuples (non-baseline sweep runs only).

    Returns:
        Six formatted strings: TR rec, prec, f1, then TE rec, prec, f1.
    """
    if not sweep_metric_rows:
        return [_format_mean_plus_minus([]) for _ in range(6)]
    column_values = list(zip(*sweep_metric_rows))
    return [_format_mean_plus_minus(column) for column in column_values]


def _collect_sweep_metrics_by_algorithm(
    table_rows: Sequence[Tuple[str, Optional[SweepLabelValue], str, AlgoSummary]],
) -> Dict[str, List[SweepMetricValues]]:
    """Group non-baseline sweep metric tuples by algorithm name.

    Args:
        table_rows: Rows from a sweep comparison table build.

    Returns:
        Mapping from algorithm name to sweep-run metric tuples (baseline excluded).
    """
    sweep_metrics_by_algorithm: Dict[str, List[SweepMetricValues]] = {}
    for algorithm_name, sweep_label_value, _run_label, summary in table_rows:
        if sweep_label_value == BASE_SWEEP_LABEL:
            continue
        sweep_metrics_by_algorithm.setdefault(algorithm_name, []).append(
            _metric_values_from_summary(summary)
        )
    return sweep_metrics_by_algorithm


def _mean_te_val_f1_for_sort(sweep_metric_rows: Sequence[SweepMetricValues]) -> float:
    """Return mean TE validation F1 (``cls_f1_te``) for sorting; ``inf`` if missing.

    Args:
        sweep_metric_rows: Non-baseline sweep metric tuples for one algorithm.

    Returns:
        Mean TE F1, or positive infinity when no values exist.
    """
    te_f1_values = [metric_row[5] for metric_row in sweep_metric_rows]
    numeric_values = [float(value) for value in te_f1_values if value is not None]
    if not numeric_values:
        return float("inf")
    return float(np.mean(numeric_values))


def _print_sweep_mean_pm_summary_table(
    sweep_metrics_by_algorithm: Dict[str, List[SweepMetricValues]],
    output_format: str,
    *,
    readable_title_line: str,
    markdown_section_title: str,
) -> None:
    """Print a second table of per-algorithm mean± over non-baseline sweep runs.

    Rows are sorted by ascending mean TE validation F1 (``cls_f1_te``).

    Args:
        sweep_metrics_by_algorithm: Sweep metrics grouped by algorithm (no baseline).
        output_format: ``readable`` or ``markdown``.
        readable_title_line: Title line for readable output.
        markdown_section_title: Markdown subsection title.

    Returns:
        None.
    """
    sorted_algorithms = sorted(
        sweep_metrics_by_algorithm.keys(),
        key=lambda algorithm_name: _mean_te_val_f1_for_sort(
            sweep_metrics_by_algorithm[algorithm_name]
        ),
    )
    if not sorted_algorithms:
        return

    if output_format == "markdown":
        print()
        print(markdown_section_title)
        print()
        header_cells = [
            "Algo",
            "tr_rec",
            "tr_prec",
            "tr_f1",
            "te_rec",
            "te_prec",
            "te_f1",
        ]
        print("| " + " | ".join(header_cells) + " |")
        print("| " + " | ".join(["---"] * len(header_cells)) + " |")
        for algorithm_name in sorted_algorithms:
            mean_pm_cells = _aggregate_sweep_mean_pm_cells(
                sweep_metrics_by_algorithm[algorithm_name]
            )
            data_cells = [algorithm_name] + [cell.strip() for cell in mean_pm_cells]
            print("| " + " | ".join(data_cells) + " |")
        return

    print()
    print(readable_title_line)
    width_algorithm = max(len(algorithm_name) for algorithm_name in sorted_algorithms)
    width_algorithm = max(width_algorithm, len("Algo"))
    width_numeric = 11
    header_parts = [
        f"{'Algo':<{width_algorithm}}",
        f"{'rec':>{width_numeric}}",
        f"{'prec':>{width_numeric}}",
        f"{'f1':>{width_numeric}}",
        "|",
        f"{'rec':>{width_numeric}}",
        f"{'prec':>{width_numeric}}",
        f"{'f1':>{width_numeric}}",
    ]
    header_line = " ".join(header_parts)
    print(header_line)
    print("-" * len(header_line))
    for algorithm_name in sorted_algorithms:
        mean_pm_cells = _aggregate_sweep_mean_pm_cells(
            sweep_metrics_by_algorithm[algorithm_name]
        )
        row_parts = [f"{algorithm_name:<{width_algorithm}}"]
        for cell_index, formatted_cell in enumerate(mean_pm_cells):
            stripped_cell = formatted_cell.strip()
            if cell_index == 3:
                row_parts.append("|")
            row_parts.append(f"{stripped_cell:>{width_numeric}}")
        print(" ".join(row_parts))


def _print_compare_runs_path_intro(
    run_paths: List[str],
    markdown_heading: str,
    readable_label: str,
    output_format: str,
) -> None:
    """Print a short preamble listing coordinator run paths for a sweep comparison.

    Args:
        run_paths: Run directories passed on the CLI.
        markdown_heading: Markdown heading line (e.g. ``## Memory buffer comparison``).
        readable_label: One-line prefix before the path list in readable mode.
        output_format: ``readable`` or ``markdown``.

    Returns:
        None.

    Usage:
        >>> _print_compare_runs_path_intro(
        ...     ["logs/full_experiments/run_A_mem_512"],
        ...     "## Memory buffer comparison",
        ...     "Memory compare runs",
        ...     "markdown",
        ... )
    """
    if output_format == "markdown":
        print(markdown_heading)
        print()
        for run_path in run_paths:
            print(f"- `{run_path}`")
        print()
        return
    print(f"{readable_label} ({len(run_paths)}): {', '.join(run_paths)}")


def _print_tr_te_sweep_comparison(
    run_directory_paths: List[str],
    output_format: str,
    *,
    sweep_label_header: str,
    sweep_label_from_run: Callable[[str], Optional[int]],
    markdown_section_title: str,
    readable_title_line: str,
    empty_message: str,
    base_run_directory_paths: Optional[List[str]] = None,
    separate_algorithms: bool = False,
) -> None:
    """Print TR/TE rec, prec, f1 rows keyed by a per-run integer parsed from the path.

    Args:
        run_directory_paths: Coordinator run directories to load with :func:`parse_path`.
        output_format: ``readable`` or ``markdown``.
        sweep_label_header: Column name for the parsed sweep parameter (e.g. ``mem``).
        sweep_label_from_run: Returns the sweep label integer from a run directory path.
        markdown_section_title: Markdown subsection title (``### ...``).
        readable_title_line: First-line description in readable mode.
        empty_message: Printed when no algorithm rows are produced.
        base_run_directory_paths: Optional baseline runs (label ``base`` in the sweep
            column); only algorithms present in ``run_directory_paths`` are included.
            When set, also prints a second table of per-algorithm ``mean±`` over
            non-baseline sweep runs, sorted by ascending TE validation F1 mean.
        separate_algorithms: When True, print a dashed line between each algorithm's
            rows in the comparison table.

    Returns:
        None.
    """
    show_sweep_statistics = bool(base_run_directory_paths)
    table_rows: List[Tuple[str, Optional[SweepLabelValue], str, AlgoSummary]] = []
    comparison_algorithm_names: Optional[set[str]] = None
    if base_run_directory_paths:
        comparison_algorithm_names = set()
        for run_directory in run_directory_paths:
            comparison_algorithm_names.update(parse_path(run_directory).keys())

    for run_directory in base_run_directory_paths or []:
        run_label = Path(run_directory).name
        summaries = parse_path(run_directory)
        for algorithm_name, summary in summaries.items():
            if (
                comparison_algorithm_names is not None
                and algorithm_name not in comparison_algorithm_names
            ):
                continue
            table_rows.append((algorithm_name, BASE_SWEEP_LABEL, run_label, summary))
    for run_directory in run_directory_paths:
        sweep_label_value = sweep_label_from_run(run_directory)
        run_label = Path(run_directory).name
        summaries = parse_path(run_directory)
        for algorithm_name, summary in summaries.items():
            table_rows.append((algorithm_name, sweep_label_value, run_label, summary))

    if not table_rows:
        print(empty_message)
        return

    pair_counts = Counter(
        (algorithm_name, sweep_label_value)
        for algorithm_name, sweep_label_value, _, _ in table_rows
    )
    show_run_column = any(count > 1 for count in pair_counts.values())

    label_width = len(sweep_label_header)
    for _algorithm_name, sweep_label_value, _run_label, _summary in table_rows:
        if sweep_label_value is not None:
            label_width = max(label_width, len(str(sweep_label_value)))
    label_width = max(label_width, 6)

    sweep_metrics_by_algorithm = (
        _collect_sweep_metrics_by_algorithm(table_rows) if show_sweep_statistics else {}
    )

    def row_sort_key(
        item: Tuple[str, Optional[SweepLabelValue], str, AlgoSummary],
    ) -> Tuple[str, Tuple[int, float | str], str]:
        algorithm_name, sweep_label_value, run_label, _summary = item
        return (algorithm_name, _sweep_label_sort_key(sweep_label_value), run_label)

    table_rows.sort(key=row_sort_key)

    if output_format == "markdown":
        print()
        print(markdown_section_title)
        print()
        header_cells = ["Algo", sweep_label_header]
        if show_run_column:
            header_cells.append("run_dir")
        header_cells += [
            "tr_rec",
            "tr_prec",
            "tr_f1",
            "te_rec",
            "te_prec",
            "te_f1",
        ]
        print("| " + " | ".join(header_cells) + " |")
        print("| " + " | ".join(["---"] * len(header_cells)) + " |")
        previous_algorithm_name: Optional[str] = None
        for algorithm_name, sweep_label_value, run_label, summary in table_rows:
            if separate_algorithms and previous_algorithm_name is not None:
                if algorithm_name != previous_algorithm_name:
                    _print_sweep_algorithm_separator(output_format, 0)
            metric_values = _metric_values_from_summary(summary)
            label_cell = (
                str(sweep_label_value) if sweep_label_value is not None else "-"
            )
            data_cells = [
                algorithm_name,
                label_cell,
            ]
            if show_run_column:
                data_cells.append(run_label)
            data_cells += [_fmt(value).strip() for value in metric_values]
            print("| " + " | ".join(data_cells) + " |")
            previous_algorithm_name = algorithm_name
    else:
        print()
        print(readable_title_line)
        width_algorithm = max(len(item[0]) for item in table_rows)
        width_algorithm = max(width_algorithm, len("Algo"))
        width_run = max(len(item[2]) for item in table_rows) if show_run_column else 0
        if show_run_column:
            width_run = max(width_run, len("run_dir"))
        width_numeric = 7
        header_parts = [
            f"{'Algo':<{width_algorithm}}",
            f"{sweep_label_header:>{label_width}}",
        ]
        if show_run_column:
            header_parts.append(f"{'run_dir':<{width_run}}")
        header_parts += [
            f"{'rec':>{width_numeric}}",
            f"{'prec':>{width_numeric}}",
            f"{'f1':>{width_numeric}}",
            "|",
            f"{'rec':>{width_numeric}}",
            f"{'prec':>{width_numeric}}",
            f"{'f1':>{width_numeric}}",
        ]
        header_line = " ".join(header_parts)
        print(header_line)
        print("-" * len(header_line))
        previous_algorithm_name = None
        for algorithm_name, sweep_label_value, run_label, summary in table_rows:
            if separate_algorithms and previous_algorithm_name is not None:
                if algorithm_name != previous_algorithm_name:
                    _print_sweep_algorithm_separator(output_format, len(header_line))
            metric_values = _metric_values_from_summary(summary)
            label_cell = (
                f"{sweep_label_value:>{label_width}}"
                if sweep_label_value is not None
                else f"{'-':>{label_width}}"
            )
            row_parts = [
                f"{algorithm_name:<{width_algorithm}}",
                label_cell,
            ]
            if show_run_column:
                row_parts.append(f"{run_label:<{width_run}}")
            row_parts += [
                f"{_fmt(metric_values[0]):>{width_numeric}}",
                f"{_fmt(metric_values[1]):>{width_numeric}}",
                f"{_fmt(metric_values[2]):>{width_numeric}}",
                "|",
                f"{_fmt(metric_values[3]):>{width_numeric}}",
                f"{_fmt(metric_values[4]):>{width_numeric}}",
                f"{_fmt(metric_values[5]):>{width_numeric}}",
            ]
            print(" ".join(row_parts))
            previous_algorithm_name = algorithm_name

    if show_sweep_statistics and sweep_metrics_by_algorithm:
        _print_sweep_mean_pm_summary_table(
            sweep_metrics_by_algorithm,
            output_format,
            readable_title_line=(
                "Sweep mean± over non-baseline runs "
                "(sorted by TE val F1 mean, low to high)"
            ),
            markdown_section_title=(
                "### Sweep mean± (sorted by TE val F1 mean, low to high)"
            ),
        )


def print_mem_buffer_comparison(
    run_directory_paths: List[str],
    output_format: str = "readable",
    base_directory_path: Optional[str] = None,
) -> None:
    """Print compact TR and TE metrics across replay buffer sizes (see sweep script).

    Args:
        run_directory_paths: Coordinator run directories (typically
            ``logs/full_experiments/run_*_mem_*``).
        output_format: ``readable`` or ``markdown`` table style.
        base_directory_path: Optional baseline logs (default buffer size). May be a
            single coordinator run or a parent folder of runs (e.g.
            ``logs/full_experiments/full-til_10epochs_w-zs``); rows use sweep label
            ``base``. Only algorithms present in ``run_directory_paths`` are included.
            Also prints a second ``mean±`` table over non-baseline buffer sizes.

    Returns:
        None.

    Usage:
        >>> print_mem_buffer_comparison(
        ...     [
        ...         "logs/full_experiments/run_20260511_221802_lnx-elkk-1_mem_512",
        ...         "logs/full_experiments/run_20260513_173647_lnx-elkk-1_mem_1024",
        ...     ],
        ...     base_directory_path="logs/full_experiments/full-til_10epochs_w-zs",
        ... )
    """
    base_run_paths: Optional[List[str]] = None
    if base_directory_path is not None:
        base_run_paths = _discover_coordinator_run_directories(base_directory_path)
    _print_tr_te_sweep_comparison(
        run_directory_paths,
        output_format,
        sweep_label_header="mem",
        sweep_label_from_run=_parse_memory_buffer_size_from_run_directory,
        markdown_section_title=(
            "### Memory buffer comparison (TR and TE: rec, prec, f1)"
        ),
        readable_title_line=(
            "Memory buffer comparison (TR: rec prec f1 | TE: rec prec f1)"
        ),
        empty_message="No algorithm runs found for memory buffer comparison.",
        base_run_directory_paths=base_run_paths,
        separate_algorithms=True,
    )


def print_task_order_seed_comparison(
    run_directory_paths: List[str],
    output_format: str = "readable",
    base_directory_path: Optional[str] = None,
) -> None:
    """Print compact TR and TE metrics across task-order seeds (see sweep script).

    Args:
        run_directory_paths: Coordinator run directories (typically
            ``logs/full_experiments/run_*_task_order_seed_*``).
        output_format: ``readable`` or ``markdown`` table style.
        base_directory_path: Optional baseline logs (default task order). May be a
            single coordinator run or a parent directory of runs (e.g.
            ``logs/full_experiments/full-til_10epochs_w-zs``); rows use sweep label
            ``base``. Only algorithms present in ``run_directory_paths`` are included.
            Also prints a second ``mean±`` table over non-baseline task-order seeds.

    Returns:
        None.

    Usage:
        >>> print_task_order_seed_comparison(
        ...     [
        ...         "logs/full_experiments/run_20260101_120000_lnx-elkk-1_task_order_seed_57",
        ...         "logs/full_experiments/run_20260101_180000_lnx-elkk-1_task_order_seed_1040",
        ...     ],
        ...     base_directory_path="logs/full_experiments/full-til_10epochs_w-zs",
        ... )
    """
    base_run_paths: Optional[List[str]] = None
    if base_directory_path is not None:
        base_run_paths = _discover_coordinator_run_directories(base_directory_path)
    _print_tr_te_sweep_comparison(
        run_directory_paths,
        output_format,
        sweep_label_header="task_order_seed",
        sweep_label_from_run=_parse_task_order_seed_from_run_directory,
        markdown_section_title=(
            "### Task order seed comparison (TR and TE: rec, prec, f1)"
        ),
        readable_title_line=(
            "Task order seed comparison (TR: rec prec f1 | TE: rec prec f1)"
        ),
        empty_message="No algorithm runs found for task order seed comparison.",
        base_run_directory_paths=base_run_paths,
        separate_algorithms=True,
    )


def print_multi_seed_summary(
    parent_directory: str,
    output_format: str = "readable",
) -> None:
    """Print TR/TE mean± over random seeds from ``seed-*`` subdirectories.

    Recursively collects coordinator and job logs under each seed folder, merges
    sharded host runs per seed, then aggregates metrics across seeds.

    Args:
        parent_directory: Folder containing ``seed-<n>`` or ``seed_<n>`` children
            (e.g. ``logs/full_experiments/one-shot_til``).
        output_format: ``readable`` or ``markdown`` table style.

    Returns:
        None.

    Usage:
        >>> print_multi_seed_summary("logs/full_experiments/one-shot_til")  # doctest: +SKIP
    """
    seed_directories = _discover_seed_directories(parent_directory)
    seed_values = [seed_value for seed_value, _ in seed_directories]
    metrics_by_algorithm: Dict[str, List[SweepMetricValues]] = {}

    for _seed_value, seed_directory in seed_directories:
        seed_summaries = _summarize_seed_directory(seed_directory)
        for algorithm_name, summary in seed_summaries.items():
            metrics_by_algorithm.setdefault(algorithm_name, []).append(
                _metric_values_from_summary(summary)
            )

    if not metrics_by_algorithm:
        print("No algorithm runs found for multi-seed summary.")
        return

    seed_list_text = ", ".join(str(seed_value) for seed_value in seed_values)
    seed_count = len(seed_values)
    _print_sweep_mean_pm_summary_table(
        metrics_by_algorithm,
        output_format,
        readable_title_line=(
            f"Multi-seed results ({seed_count} seeds: {seed_list_text}; "
            "mean± over seeds, sorted by TE f1 mean)"
        ),
        markdown_section_title=(
            f"### Multi-seed results ({seed_count} seeds; mean± over seeds)"
        ),
    )


def _print_metadata(log_paths: List[str], output_format: str) -> None:
    if output_format == "markdown":
        print("## Full Experiments Summary")
        print()
        if len(log_paths) == 1:
            print(f"- Log: `{log_paths[0]}`")
        else:
            print(f"- Logs ({len(log_paths)}):")
            for log_path in log_paths:
                print(f"  - `{log_path}`")
        print()
        return

    if len(log_paths) == 1:
        print(f"Log: {log_paths[0]}")
    else:
        print(f"Logs ({len(log_paths)}): {', '.join(log_paths)}")


def _parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarise full_experiments log file by algorithm performance."
    )
    sweep_group = parser.add_mutually_exclusive_group()
    sweep_group.add_argument(
        "--multi-seed",
        type=str,
        default=None,
        metavar="PARENT_DIR",
        help=(
            "Parent folder with seed-* or seed_* subdirectories. Recursively collects "
            "coordinator/job logs per seed (merging sharded runs), then prints only "
            "the TR/TE mean± table aggregated across seeds "
            "(see full_experiments_seed_sweep.sh)."
        ),
    )
    sweep_group.add_argument(
        "--mem-compare-runs",
        type=str,
        nargs="+",
        default=None,
        metavar="RUN_DIR",
        help=(
            "Coordinator run directories (typically logs/full_experiments/run_*_mem_*) "
            "to compare TR and TE rec, prec, and f1 across buffer sizes. "
            "Printed after the main summary when --log/--logs are also used; "
            "buffer size is read from the directory name suffix _mem_<n> "
            "(see full_experiments_mem_sweep.sh)."
        ),
    )
    sweep_group.add_argument(
        "--task-order-seed-compare-runs",
        type=str,
        nargs="+",
        default=None,
        metavar="RUN_DIR",
        help=(
            "Coordinator run directories (typically run_*_task_order_seed_<n>) "
            "to compare TR and TE rec, prec, and f1 across task-order seeds. "
            "Same table as --mem-compare-runs; seed is read from suffix "
            "_task_order_seed_<n> (see full_experiments_task_order_seed_sweep.sh)."
        ),
    )
    parser.add_argument(
        "--log",
        type=str,
        default=None,
        help="Path to full_experiments_*.log (default: latest in logs/full_experiments/).",
    )
    parser.add_argument(
        "--logs",
        type=str,
        nargs="+",
        default=None,
        help=(
            "One or more log paths (files or run directories) to aggregate into a single "
            "summary. Example: --logs run_elkk1 run_elkk2"
        ),
    )
    parser.add_argument(
        "--output-format",
        type=str,
        choices=("readable", "markdown"),
        default="readable",
        help="Output format for the summary table.",
    )
    parser.add_argument(
        "--base-mem",
        type=str,
        default=None,
        metavar="BASE_DIR",
        help=(
            "Baseline experiment directory for --mem-compare-runs (default replay "
            "buffer, no _mem_<n> suffix). May be one coordinator run or a parent "
            "folder of runs (e.g. logs/full_experiments/full-til_10epochs_w-zs). "
            "Adds rows labelled 'base' in the comparison table."
        ),
    )
    parser.add_argument(
        "--base-seed",
        type=str,
        default=None,
        metavar="BASE_DIR",
        help=(
            "Baseline experiment directory for --task-order-seed-compare-runs "
            "(default task order, no _task_order_seed_ suffix). May be one coordinator "
            "run or a parent folder of runs (e.g. logs/full_experiments/"
            "full-til_10epochs_w-zs). Adds rows labelled 'base' in the comparison table."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_arguments()

    if args.multi_seed is not None:
        print_multi_seed_summary(
            args.multi_seed,
            output_format=args.output_format,
        )
        return

    any_sweep_compare = (
        args.mem_compare_runs is not None
        or args.task_order_seed_compare_runs is not None
    )
    if args.logs is not None:
        log_paths: List[str] = list(args.logs)
    elif args.log is not None:
        log_paths = [args.log]
    elif any_sweep_compare:
        log_paths = []
    else:
        log_paths = [_find_default_log()]

    summary_collections: List[Dict[str, AlgoSummary]] = []
    total_concurrent_runtime_seconds: Optional[float] = 0.0
    if log_paths:
        for log_path in log_paths:
            summary_collections.append(parse_path(log_path))
            concurrent_runtime_seconds = _compute_concurrent_runtime_seconds(log_path)
            if concurrent_runtime_seconds is None:
                total_concurrent_runtime_seconds = None
            elif total_concurrent_runtime_seconds is not None:
                total_concurrent_runtime_seconds += concurrent_runtime_seconds

        merged_summaries = _merge_many_summaries(summary_collections)
        _print_metadata(log_paths, output_format=args.output_format)
        print_summary(
            merged_summaries,
            concurrent_runtime_seconds=total_concurrent_runtime_seconds,
            output_format=args.output_format,
        )

    if args.mem_compare_runs:
        if not log_paths:
            intro_paths = list(args.mem_compare_runs)
            if args.base_mem is not None:
                intro_paths = [args.base_mem] + intro_paths
            _print_compare_runs_path_intro(
                intro_paths,
                "## Memory buffer comparison",
                "Memory compare runs",
                args.output_format,
            )
        print_mem_buffer_comparison(
            list(args.mem_compare_runs),
            output_format=args.output_format,
            base_directory_path=args.base_mem,
        )

    if args.task_order_seed_compare_runs:
        if not log_paths:
            intro_paths = list(args.task_order_seed_compare_runs)
            if args.base_seed is not None:
                intro_paths = [args.base_seed] + intro_paths
            _print_compare_runs_path_intro(
                intro_paths,
                "## Task order seed comparison",
                "Task order seed compare runs",
                args.output_format,
            )
        print_task_order_seed_comparison(
            list(args.task_order_seed_compare_runs),
            output_format=args.output_format,
            base_directory_path=args.base_seed,
        )


def _merge_many_summaries(
    summary_collections: List[Dict[str, AlgoSummary]],
) -> Dict[str, AlgoSummary]:
    merged_summaries: Dict[str, AlgoSummary] = {}
    for summary_collection in summary_collections:
        for algorithm_name, summary in summary_collection.items():
            merged_summary = merged_summaries.setdefault(
                algorithm_name, AlgoSummary(name=algorithm_name)
            )
            _merge_summary(merged_summary, summary)
    return merged_summaries


if __name__ == "__main__":
    main()
