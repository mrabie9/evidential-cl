#!/usr/bin/env python
"""Collect zero-shot validation metrics per task from continual-learning logs.

This script reads ``task*.npz`` files from one or more ``metrics`` directories and
extracts the zero-shot (pre-train) validation metrics written by ``main.py``:

- ``zero_shot_macro_rec``
- ``zero_shot_macro_prec``
- ``zero_shot_macro_f1`` (also emitted as ``total_macro_f1_zs``)

Runs written before the detection-metric removal spell these ``zero_shot_rec_cls``
/ ``zero_shot_prec_cls`` / ``zero_shot_f1_cls``; those names are still read.

By default, one row is emitted per task checkpoint (task 0, task 1, ...). This is
the task-level series typically used to compute forward transfer (FWT).

Optionally, ``--include-per-task`` also exports the full zero-shot metric matrix
for all seen tasks at each checkpoint using the ``zero_shot_per_task_*`` arrays.

Usage:
    python scripts/collect_zero_shot_validation_metrics.py --algo cmaml

    python scripts/collect_zero_shot_validation_metrics.py \
        --algo cmaml,hat --logs-root logs --run-index 0 --csv outputs/zs.csv

    python scripts/collect_zero_shot_validation_metrics.py \
        --algo cmaml --metrics-dir logs/cmaml/my-run/0/metrics --include-per-task
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

import plot_multi_algorithms as plot_multi  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils.metric_keys import extract_metric  # noqa: E402

TaskMetrics = Dict[str, Any]


def _extract_algo_name_from_job_filename(log_path: Path) -> str | None:
    """Extract algorithm/config stem from a job log filename.

    Expected patterns include ``job_<algo>.log`` and
    ``job_<algo>_<timestamp>_<id>.log``.

    Args:
        log_path: Path to a single ``job_*.log`` file.

    Returns:
        Algorithm/config stem, or ``None`` when the filename cannot be parsed.

    Usage:
        >>> isinstance(_extract_algo_name_from_job_filename, object)
        True
    """
    filename_stem = log_path.stem
    if not filename_stem.startswith("job_"):
        return None
    stem_without_prefix = filename_stem[len("job_") :]
    if not stem_without_prefix:
        return None
    timestamp_suffix_match = re.match(
        r"^(?P<algo>.+)_\d{8}_\d{6}_\d+$",
        stem_without_prefix,
    )
    if timestamp_suffix_match:
        return timestamp_suffix_match.group("algo")
    return stem_without_prefix


def _extract_model_name_from_log_header(log_path: Path) -> str | None:
    """Extract model name from a job log header.

    Args:
        log_path: Path to a single ``job_*.log`` file.

    Returns:
        Parsed model name, or ``None`` when unavailable.

    Usage:
        >>> isinstance(_extract_model_name_from_log_header, object)
        True
    """
    max_lines_to_scan = 40
    with log_path.open("r", encoding="utf-8", errors="replace") as log_file:
        for line_index, line_text in enumerate(log_file):
            if line_index >= max_lines_to_scan:
                break
            model_match = re.search(r"Running model:\s*([A-Za-z0-9_.-]+)", line_text)
            if model_match:
                return model_match.group(1).strip()
    return None


def _extract_logged_output_dir_from_log(log_path: Path) -> Path | None:
    """Extract experiment output directory from a job log.

    Args:
        log_path: Path to a single ``job_*.log`` file.

    Returns:
        Relative or absolute output directory path, or ``None`` when not found.

    Usage:
        >>> isinstance(_extract_logged_output_dir_from_log, object)
        True
    """
    max_lines_to_scan = 80
    with log_path.open("r", encoding="utf-8", errors="replace") as log_file:
        for line_index, line_text in enumerate(log_file):
            if line_index >= max_lines_to_scan:
                break
            logging_to_match = re.search(r"Logging to\s+(.+?)\s*$", line_text)
            if logging_to_match:
                return Path(logging_to_match.group(1).strip())
            terminal_log_match = re.search(
                r"Enabling terminal logging to\s+(.+?)/terminal\.log\s*$",
                line_text,
            )
            if terminal_log_match:
                return Path(terminal_log_match.group(1).strip())
    return None


def _resolve_output_dir_to_metrics_dir(output_dir: Path) -> Path:
    """Resolve an experiment output directory to its metrics subdirectory.

    Args:
        output_dir: Directory where a run writes logs and artifacts.

    Returns:
        Path to the metrics directory under output dir.

    Usage:
        >>> isinstance(_resolve_output_dir_to_metrics_dir, object)
        True
    """
    return output_dir / "metrics"


def _candidate_metrics_dirs_for_logged_output(
    logged_output_dir: Path,
    algorithm_name: str,
) -> List[Path]:
    """Build candidate metrics dirs from a logged output path.

    This supports both the original training location and synchronized/moved
    artifacts under ``logs/00_sync/*/saved_models``.

    Args:
        logged_output_dir: Path extracted from a job log's ``Logging to ...`` line.
        algorithm_name: Algorithm name parsed from the job log filename/header.

    Returns:
        Candidate metrics directories ordered by preference.

    Usage:
        >>> isinstance(_candidate_metrics_dirs_for_logged_output, object)
        True
    """
    normalized_logged_output_dir = Path(str(logged_output_dir).replace("//", "/"))
    if normalized_logged_output_dir.is_absolute():
        resolved_output_dir = normalized_logged_output_dir
    else:
        resolved_output_dir = (
            _SCRIPT_DIR.parent / normalized_logged_output_dir
        ).resolve()

    candidate_metrics_dirs: List[Path] = [
        _resolve_output_dir_to_metrics_dir(resolved_output_dir).resolve()
    ]

    relative_parts = normalized_logged_output_dir.parts
    if len(relative_parts) >= 3 and relative_parts[0] == "logs":
        run_subpath = Path(*relative_parts[2:])
        for sync_dir_name in (
            "one-shot_CIL",
            "one-shot_TIL",
            "full-til_10epochs_w-zs",
            "full-cil_10epochs",
        ):
            sync_base = (
                _SCRIPT_DIR.parent
                / "logs"
                / "00_sync"
                / sync_dir_name
                / "saved_models"
                / algorithm_name
            )
            candidate_metrics_dirs.append(
                _resolve_output_dir_to_metrics_dir((sync_base / run_subpath).resolve())
            )

    deduplicated_candidates: List[Path] = []
    seen_candidate_dirs: set[Path] = set()
    for candidate_metrics_dir in candidate_metrics_dirs:
        if candidate_metrics_dir in seen_candidate_dirs:
            continue
        deduplicated_candidates.append(candidate_metrics_dir)
        seen_candidate_dirs.add(candidate_metrics_dir)
    return deduplicated_candidates


def _discover_runs_from_run_dir(run_dir: Path) -> tuple[List[str], List[Path]]:
    """Discover model names and metrics dirs from a run folder.

    This scans ``run_dir/job_logs`` for ``job_*.log`` files, extracts each model
    name and its logged output directory, and resolves that to ``.../metrics``.

    Args:
        run_dir: Full run directory (for example under ``logs/full_experiments``).

    Returns:
        Tuple ``(algorithm_names, metrics_dirs)`` ordered by model name.

    Raises:
        SystemExit: If no usable job logs or metrics directories are found.

    Usage:
        >>> isinstance(_discover_runs_from_run_dir, object)
        True
    """
    if not run_dir.is_dir():
        raise SystemExit(f"--run-dir is not a directory: {run_dir}")

    job_logs_dir = run_dir / "job_logs"
    if not job_logs_dir.is_dir():
        raise SystemExit(f"--run-dir does not contain job_logs/: {run_dir}")

    discovered_by_model: Dict[str, Path] = {}
    for job_log in sorted(job_logs_dir.glob("job_*.log")):
        algorithm_name = _extract_algo_name_from_job_filename(job_log)
        if algorithm_name is None:
            algorithm_name = _extract_model_name_from_log_header(job_log)
        logged_output_dir = _extract_logged_output_dir_from_log(job_log)
        if algorithm_name is None or logged_output_dir is None:
            continue

        candidate_metrics_dirs = _candidate_metrics_dirs_for_logged_output(
            logged_output_dir=logged_output_dir,
            algorithm_name=algorithm_name,
        )
        selected_metrics_dir: Path | None = None
        for metrics_dir in candidate_metrics_dirs:
            if not metrics_dir.is_dir():
                continue
            if not any(metrics_dir.glob("task*.npz")):
                continue
            selected_metrics_dir = metrics_dir
            break
        if selected_metrics_dir is None:
            continue
        discovered_by_model[algorithm_name] = selected_metrics_dir

    if not discovered_by_model:
        raise SystemExit(
            "No metrics directories discovered from job logs under "
            f"{job_logs_dir}. Ensure the run has completed and wrote task*.npz files."
        )

    sorted_models = sorted(discovered_by_model.keys())
    algorithm_names = [model_name for model_name in sorted_models]
    metrics_dirs = [discovered_by_model[model_name] for model_name in sorted_models]
    return algorithm_names, metrics_dirs


def _comma_separated_list(argument_value: str) -> List[str]:
    """Split a single CLI token into stripped non-empty values.

    Args:
        argument_value: Raw argument string (for example ``--algo cmaml,hat``).

    Returns:
        List of trimmed, non-empty values.

    Usage:
        >>> _comma_separated_list("cmaml, hat , ewc")
        ['cmaml', 'hat', 'ewc']
    """
    return [part.strip() for part in argument_value.split(",") if part.strip()]


def _safe_float(value: Any) -> float:
    """Convert metric-like values to ``float`` while preserving NaN.

    Args:
        value: Input scalar or array-like value.

    Returns:
        Float representation; ``nan`` on unsupported/missing values.

    Usage:
        >>> math.isnan(_safe_float(None))
        True
        >>> _safe_float(np.float64(0.25))
        0.25
    """
    if value is None:
        return float("nan")
    array_value = np.asarray(value)
    if array_value.size == 0:
        return float("nan")
    try:
        return float(array_value.reshape(-1)[0])
    except (TypeError, ValueError):
        return float("nan")


def _harmonic_mean_f1(precision: float, recall: float) -> float:
    """Compute F1 from precision and recall using harmonic mean.

    Args:
        precision: Precision metric in [0, 1] or NaN.
        recall: Recall metric in [0, 1] or NaN.

    Returns:
        ``2 * P * R / (P + R)`` when defined, else NaN.

    Usage:
        >>> _harmonic_mean_f1(0.5, 0.5)
        0.5
    """
    if math.isnan(precision) or math.isnan(recall):
        return float("nan")
    denominator = precision + recall
    if denominator <= 0.0:
        return 0.0
    return float((2.0 * precision * recall) / denominator)


def _extract_task_scalar_rows(run: plot_multi.AlgoRun) -> List[Dict[str, Any]]:
    """Build one row per task checkpoint with scalar zero-shot metrics.

    Args:
        run: Loaded algorithm run with per-task metric dictionaries.

    Returns:
        List of row dictionaries for each task in run order.

    Usage:
        >>> isinstance(_extract_task_scalar_rows, object)
        True
    """
    task_names = run.task_names or []
    rows: List[Dict[str, Any]] = []
    for task_index, task_metrics in enumerate(run.tasks):
        task_name = task_names[task_index] if task_index < len(task_names) else ""
        macro_rec = _safe_float(extract_metric(task_metrics, "zero_shot_macro_rec"))
        macro_prec = _safe_float(extract_metric(task_metrics, "zero_shot_macro_prec"))
        total_macro_f1_zs = _safe_float(
            extract_metric(task_metrics, "zero_shot_macro_f1")
        )
        f1_from_rec_prec = _harmonic_mean_f1(macro_prec, macro_rec)
        rows.append(
            {
                "algo": run.name,
                "metrics_dir": str(run.metrics_dir),
                "task_index": task_index,
                "task_name": task_name,
                "zero_shot_macro_rec": macro_rec,
                "zero_shot_macro_prec": macro_prec,
                "zero_shot_macro_f1": f1_from_rec_prec,
                "zero_shot_total_macro_f1_zs": total_macro_f1_zs,
            }
        )
    return rows


def _extract_per_task_matrix_rows(
    run: plot_multi.AlgoRun,
) -> List[Dict[str, Any]]:
    """Build full pre-train metric matrix rows for each checkpoint/task pair.

    Args:
        run: Loaded algorithm run.

    Returns:
        Row dictionaries with both ``checkpoint_task_index`` and
        ``evaluated_task_index``.

    Usage:
        >>> isinstance(_extract_per_task_matrix_rows, object)
        True
    """
    task_names = run.task_names or []
    rows: List[Dict[str, Any]] = []
    for checkpoint_task_index, task_metrics in enumerate(run.tasks):
        per_task_f1 = np.asarray(
            extract_metric(task_metrics, "zero_shot_per_task_macro_f1", []), dtype=float
        ).reshape(-1)
        per_task_rec = np.asarray(
            extract_metric(task_metrics, "zero_shot_per_task_macro_rec", []),
            dtype=float,
        ).reshape(-1)
        per_task_prec = np.asarray(
            extract_metric(task_metrics, "zero_shot_per_task_macro_prec", []),
            dtype=float,
        ).reshape(-1)
        num_tasks_now = max(
            len(per_task_f1),
            len(per_task_rec),
            len(per_task_prec),
        )
        for evaluated_task_index in range(num_tasks_now):
            evaluated_task_name = (
                task_names[evaluated_task_index]
                if evaluated_task_index < len(task_names)
                else ""
            )
            rows.append(
                {
                    "algo": run.name,
                    "metrics_dir": str(run.metrics_dir),
                    "checkpoint_task_index": checkpoint_task_index,
                    "evaluated_task_index": evaluated_task_index,
                    "evaluated_task_name": evaluated_task_name,
                    "zero_shot_per_task_macro_rec": _safe_float(
                        per_task_rec[evaluated_task_index]
                        if evaluated_task_index < len(per_task_rec)
                        else float("nan")
                    ),
                    "zero_shot_per_task_macro_prec": _safe_float(
                        per_task_prec[evaluated_task_index]
                        if evaluated_task_index < len(per_task_prec)
                        else float("nan")
                    ),
                    "zero_shot_per_task_macro_f1": _safe_float(
                        per_task_f1[evaluated_task_index]
                        if evaluated_task_index < len(per_task_f1)
                        else float("nan")
                    ),
                }
            )
    return rows


def _prepare_runs(
    algorithm_names: Sequence[str],
    metrics_dirs: Sequence[Path] | None,
    logs_root: Path,
    run_index: int,
) -> List[plot_multi.AlgoRun]:
    """Resolve algorithms to loaded runs.

    Args:
        algorithm_names: Algorithm labels from CLI.
        metrics_dirs: Optional explicit metrics directories.
        logs_root: Root directory for auto-discovery.
        run_index: Recency index for auto-discovery (0 = latest).

    Returns:
        List of resolved runs with loaded task metrics.

    Usage:
        >>> isinstance(_prepare_runs, object)
        True
    """
    if metrics_dirs and len(metrics_dirs) not in (0, len(algorithm_names)):
        raise SystemExit(
            "Number of --metrics-dir entries must be zero or match --algo count."
        )
    runs: List[plot_multi.AlgoRun] = []
    for index, algorithm_name in enumerate(algorithm_names):
        if metrics_dirs and len(metrics_dirs) == len(algorithm_names):
            metrics_dir = metrics_dirs[index]
        else:
            metrics_dir = plot_multi.find_metrics_dir_for_algo(
                algorithm_name, logs_root, run_index=run_index
            )
        metrics_dir = metrics_dir.resolve()
        if not metrics_dir.is_dir():
            raise SystemExit(f"Not a directory: {metrics_dir}")
        tasks = plot_multi.load_metrics(metrics_dir)
        name = plot_multi._resolve_algo_name(metrics_dir, explicit_name=algorithm_name)
        runs.append(
            plot_multi.AlgoRun(
                name=name,
                metrics_dir=metrics_dir,
                tasks=tasks,
                task_names=plot_multi.load_task_names(metrics_dir),
            )
        )
    return runs


def _parse_args() -> argparse.Namespace:
    """Parse command line arguments.

    Returns:
        Parsed argument namespace.

    Usage:
        >>> isinstance(_parse_args, object)
        True
    """
    parser = argparse.ArgumentParser(
        description=(
            "Collect zero-shot validation metrics per task from task*.npz files."
        )
    )
    parser.add_argument(
        "--algo",
        type=str,
        required=False,
        help=(
            "Comma-separated algorithm names (for example 'cmaml,hat') or a single "
            "name. Optional when --run-dir is provided."
        ),
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=None,
        help=(
            "Run directory containing job_logs (for example a full_experiments "
            "run folder). When provided, models are auto-discovered from job logs "
            "and their metrics directories are resolved automatically."
        ),
    )
    parser.add_argument(
        "--metrics-dir",
        type=str,
        default=None,
        help=(
            "Optional comma-separated metrics directories in the same order as "
            "--algo. If omitted, latest run under logs/<algo>/ is used."
        ),
    )
    parser.add_argument(
        "--logs-root",
        type=Path,
        default=Path("logs"),
        help="Root path for auto-discovering metrics directories.",
    )
    parser.add_argument(
        "--run-index",
        type=int,
        default=0,
        help="Run recency index for auto-discovery (0 = latest).",
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=None,
        help="Optional CSV output path for per-task scalar zero-shot rows.",
    )
    parser.add_argument(
        "--json",
        type=Path,
        default=None,
        help="Optional JSON output path for per-task scalar zero-shot rows.",
    )
    parser.add_argument(
        "--baseline-json",
        type=Path,
        default=None,
        help=(
            "Optional baseline JSON (e.g., zs_baselines.json). When provided, "
            "prints baseline total_macro_f1_zs and forward transfer "
            "(zero_shot_total_macro_f1_zs - baseline_total_macro_f1_zs)."
        ),
    )
    parser.add_argument(
        "--include-per-task",
        action="store_true",
        help=(
            "Also export full zero-shot matrix rows (all seen tasks at each "
            "checkpoint)."
        ),
    )
    parser.add_argument(
        "--per-task-csv",
        type=Path,
        default=None,
        help=(
            "Optional CSV output path for --include-per-task rows. Ignored unless "
            "--include-per-task is set."
        ),
    )
    return parser.parse_args()


def _write_csv(rows: Sequence[Dict[str, Any]], output_path: Path) -> None:
    """Write a list of dictionaries to CSV.

    Args:
        rows: Table rows with homogeneous keys.
        output_path: Destination CSV path.

    Usage:
        >>> isinstance(_write_csv, object)
        True
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    field_names = list(rows[0].keys()) if rows else []
    with output_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=field_names)
        writer.writeheader()
        writer.writerows(rows)


def _nan_to_none(value: Any) -> Any:
    """Convert NaN floats recursively to ``None`` for JSON compatibility.

    Args:
        value: Potentially nested structure.

    Returns:
        JSON-safe structure with NaN replaced by ``None``.

    Usage:
        >>> _nan_to_none(float("nan")) is None
        True
    """
    if isinstance(value, dict):
        return {key: _nan_to_none(sub_value) for key, sub_value in value.items()}
    if isinstance(value, list):
        return [_nan_to_none(item) for item in value]
    if isinstance(value, float) and math.isnan(value):
        return None
    return value


def _build_baseline_lookup(
    baseline_rows: Sequence[Dict[str, Any]],
) -> Dict[tuple[str, int], float]:
    """Build ``(algo, task) -> baseline total_macro_f1_zs`` lookup map.

    Args:
        baseline_rows: Parsed baseline rows from JSON.

    Returns:
        Lookup keyed by normalized algorithm name and task index.

    Usage:
        >>> isinstance(_build_baseline_lookup, object)
        True
    """
    baseline_lookup: Dict[tuple[str, int], float] = {}
    for row in baseline_rows:
        algo_name = str(row.get("algo", "")).strip().lower()
        if not algo_name:
            continue
        try:
            task_index = int(row.get("task"))
        except (TypeError, ValueError):
            continue
        baseline_lookup[(algo_name, task_index)] = _safe_float(
            row.get("total_macro_f1_zs", row.get("total_f1_zs"))
        )
    return baseline_lookup


def main() -> None:
    """Collect and print zero-shot validation metrics across task checkpoints."""
    args = _parse_args()
    if args.run_dir is not None:
        if args.algo is not None:
            raise SystemExit("Use either --run-dir or --algo/--metrics-dir, not both.")
        if args.metrics_dir is not None:
            raise SystemExit("Use either --run-dir or --metrics-dir, not both.")
        algorithm_names, metrics_dirs_list = _discover_runs_from_run_dir(args.run_dir)
    else:
        if args.algo is None:
            raise SystemExit("Provide --algo (or use --run-dir for auto-discovery).")
        algorithm_names = _comma_separated_list(args.algo)
        if not algorithm_names:
            raise SystemExit("--algo must contain at least one algorithm name.")

        metrics_dirs_list: List[Path] | None
        if args.metrics_dir is None:
            metrics_dirs_list = None
        else:
            metrics_dirs_list = [
                Path(part) for part in _comma_separated_list(args.metrics_dir)
            ]

    runs = _prepare_runs(
        algorithm_names=algorithm_names,
        metrics_dirs=metrics_dirs_list,
        logs_root=args.logs_root,
        run_index=args.run_index,
    )
    baseline_lookup: Dict[tuple[str, int], float] = {}
    if args.baseline_json is not None:
        if not args.baseline_json.is_file():
            raise SystemExit(f"--baseline-json file not found: {args.baseline_json}")
        with args.baseline_json.open("r", encoding="utf-8") as baseline_file:
            baseline_payload = json.load(baseline_file)
        if not isinstance(baseline_payload, list):
            raise SystemExit("--baseline-json must contain a JSON list of rows.")
        baseline_lookup = _build_baseline_lookup(baseline_payload)

    scalar_rows: List[Dict[str, Any]] = []
    matrix_rows: List[Dict[str, Any]] = []
    for run in runs:
        scalar_rows.extend(_extract_task_scalar_rows(run))
        if args.include_per_task:
            matrix_rows.extend(_extract_per_task_matrix_rows(run))

    scalar_rows.sort(key=lambda row: (str(row["algo"]), int(row["task_index"])))
    matrix_rows.sort(
        key=lambda row: (
            str(row["algo"]),
            int(row["checkpoint_task_index"]),
            int(row["evaluated_task_index"]),
        )
    )

    include_baseline_columns = bool(baseline_lookup)
    if include_baseline_columns:
        for row in scalar_rows:
            baseline_value = baseline_lookup.get(
                (str(row["algo"]).strip().lower(), int(row["task_index"])),
                float("nan"),
            )
            row["baseline_total_macro_f1_zs"] = baseline_value
            row["forward_transfer_total_macro_f1_zs"] = (
                _safe_float(row["zero_shot_total_macro_f1_zs"]) - baseline_value
                if not math.isnan(baseline_value)
                else float("nan")
            )

    header = (
        f"{'algo':<12} {'task':>4} {'macro_f1':>10} {'macro_rec':>10} "
        f"{'macro_prec':>10} {'total_macro_f1_zs':>18}"
    )
    if include_baseline_columns:
        header += f" {'baseline':>10} {'fwt':>10}"
    print(header)
    print("-" * len(header))
    for row in scalar_rows:
        line = (
            f"{row['algo']:<12} {row['task_index']:4d} "
            f"{row['zero_shot_macro_f1']:10.6f} "
            f"{row['zero_shot_macro_rec']:10.6f} "
            f"{row['zero_shot_macro_prec']:10.6f} "
            f"{row['zero_shot_total_macro_f1_zs']:18.6f}"
        )
        if include_baseline_columns:
            line += (
                f" {row['baseline_total_macro_f1_zs']:10.6f} "
                f"{row['forward_transfer_total_macro_f1_zs']:10.6f}"
            )
        print(line)
    print(
        "\nNote: macro_f1 is recomputed from macro_rec/macro_prec as 2PR/(P+R). "
        "total_macro_f1_zs is the raw stored zero-shot macro F1 from logs."
    )
    if include_baseline_columns:
        print(
            "Forward transfer (fwt) is computed as "
            "total_macro_f1_zs - baseline_total_macro_f1_zs; "
            "positive means validation is above baseline."
        )
        average_fwt_by_algo: Dict[str, List[float]] = {}
        for row in scalar_rows:
            algorithm_name = str(row["algo"])
            task_index = int(row["task_index"])
            forward_transfer_value = _safe_float(
                row["forward_transfer_total_macro_f1_zs"]
            )
            if task_index < 1 or task_index > 9 or math.isnan(forward_transfer_value):
                continue
            average_fwt_by_algo.setdefault(algorithm_name, []).append(
                forward_transfer_value
            )

        print("\nAverage FWT for tasks 1-9 (per algo; task 0 excluded):")
        for algorithm_name in sorted({str(row["algo"]) for row in scalar_rows}):
            algo_task_indices = {
                int(row["task_index"])
                for row in scalar_rows
                if str(row["algo"]) == algorithm_name
            }
            if 9 not in algo_task_indices:
                continue
            algo_values = average_fwt_by_algo.get(algorithm_name, [])
            if not algo_values:
                print(f"- {algorithm_name}: NaN")
                continue
            print(f"- {algorithm_name}: {float(np.mean(algo_values)):.6f}")

    if args.csv is not None:
        _write_csv(scalar_rows, args.csv)
    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        with args.json.open("w", encoding="utf-8") as json_file:
            json.dump(_nan_to_none(scalar_rows), json_file, indent=2)
            json_file.write("\n")
    if args.include_per_task and args.per_task_csv is not None:
        _write_csv(matrix_rows, args.per_task_csv)

    if args.include_per_task:
        print(f"\nCollected {len(matrix_rows)} full per-task zero-shot matrix rows.")
    if args.csv is not None:
        print(f"Saved scalar rows CSV to {args.csv}")
    if args.json is not None:
        print(f"Saved scalar rows JSON to {args.json}")
    if args.include_per_task and args.per_task_csv is not None:
        print(f"Saved per-task matrix CSV to {args.per_task_csv}")


if __name__ == "__main__":
    main()
