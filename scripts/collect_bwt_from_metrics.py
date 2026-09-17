#!/usr/bin/env python
"""Collect backward transfer (BWT) from continual-learning metrics on disk.

``scripts/plot_multi_algorithms.py`` defines **average forgetting** at each
checkpoint (after training task *k*) using the same peak-vs-current rule. The
**final** average forgetting is the last element of that curve (after all tasks).

This script reports that final value and **BWT** as its negative, matching the
convention described alongside ``plot_multi_algorithms`` average-forgetting plots:

- ``avg_forgetting_final``: last checkpoint average forgetting
- ``bwt``: ``-avg_forgetting_final``

Metrics are read from ``task*.npz`` under a ``metrics`` directory, identical to
the plotting workflow.

Usage:
    python scripts/collect_bwt_from_metrics.py --algo cmaml,hat

    python scripts/collect_bwt_from_metrics.py --algo cmaml

    python scripts/collect_bwt_from_metrics.py \\
        --algo cmaml,hat \\
        --metrics-dir logs/cmaml/run-a/0/metrics,logs/hat/run-b/0/metrics

    python scripts/collect_bwt_from_metrics.py \\
        --logs-root logs --run-index 0 --val-metric macro_f1 \\
        --algo ewc,lamaml
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

# Sibling import: ``python scripts/collect_bwt_from_metrics.py`` puts this
# directory on ``sys.path`` first.
_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

import plot_multi_algorithms as plot_multi  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils.metric_keys import first_present_key  # noqa: E402

TaskMetrics = Dict[str, Any]


def _extract_algo_name_from_job_filename(log_path: Path) -> str | None:
    """Extract algorithm/config stem from a job log filename.

    Args:
        log_path: Path to a single ``job_*.log`` file.

    Returns:
        Algorithm/config stem, or ``None`` when parsing fails.

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
    """Extract model name from the beginning of a job log.

    Args:
        log_path: Path to a single ``job_*.log`` file.

    Returns:
        Parsed runtime model name, or ``None`` when unavailable.

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
    """Resolve an experiment output directory to its metrics subdirectory."""
    return output_dir / "metrics"


def _candidate_metrics_dirs_for_logged_output(
    logged_output_dir: Path,
    algorithm_name: str,
) -> List[Path]:
    """Build candidate metrics dirs from a logged output path.

    This supports both the original training location and synchronized/moved
    artifacts under ``logs/00_sync/*/saved_models`` (same layout as
    ``collect_zero_shot_validation_metrics.py``).

    Args:
        logged_output_dir: Path extracted from a job log's ``Logging to ...`` line.
        algorithm_name: Algorithm name parsed from the job log filename/header.

    Returns:
        Candidate metrics directories ordered by preference.
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
    """Discover algorithm names and metrics directories from a run folder.

    Args:
        run_dir: Full run directory containing ``job_logs``.

    Returns:
        Tuple ``(algorithm_names, metrics_dirs)`` sorted by algorithm name.

    Raises:
        SystemExit: If run dir is invalid or no usable metrics dirs are found.

    Usage:
        >>> isinstance(_discover_runs_from_run_dir, object)
        True
    """
    if not run_dir.is_dir():
        raise SystemExit(f"--run-dir is not a directory: {run_dir}")

    job_logs_dir = run_dir / "job_logs"
    if not job_logs_dir.is_dir():
        raise SystemExit(f"--run-dir does not contain job_logs/: {run_dir}")

    discovered_by_algo: Dict[str, Path] = {}
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
        discovered_by_algo[algorithm_name] = selected_metrics_dir

    if not discovered_by_algo:
        raise SystemExit(
            "No metrics directories discovered from job logs under "
            f"{job_logs_dir}. Ensure the run has completed and wrote task*.npz files."
        )

    sorted_algorithms = sorted(discovered_by_algo.keys())
    algorithm_names = [name for name in sorted_algorithms]
    metrics_dirs = [discovered_by_algo[name] for name in sorted_algorithms]
    return algorithm_names, metrics_dirs


def _comma_separated_list(argument_value: str) -> List[str]:
    """Split a single CLI token into stripped non-empty parts (comma-separated).

    Args:
        argument_value: Raw string from ``--algo`` or ``--metrics-dir``.

    Returns:
        List of trimmed substrings; empty entries are dropped.

    Usage:
        >>> _comma_separated_list("cmaml, hat , ewc")
        ['cmaml', 'hat', 'ewc']
        >>> _comma_separated_list("cmaml")
        ['cmaml']
    """
    return [part.strip() for part in argument_value.split(",") if part.strip()]


def resolve_val_metric_key(
    tasks: Sequence[TaskMetrics], val_metric_choice: str
) -> Tuple[str, str]:
    """Map CLI validation choice to ``task*.npz`` keys (same as plotter).

    Args:
        tasks: Loaded per-task metric dicts.
        val_metric_choice: ``macro_f1`` or ``macro_rec``.

    Returns:
        Tuple of (npz key, human label). Pre-removal key spellings are resolved
        via :func:`utils.metric_keys.first_present_key`.

    Usage:
        >>> resolve_val_metric_key([{"val_macro_f1": np.array([1.0])}], "macro_f1")
        ('val_macro_f1', 'Macro F1')
    """
    recall_key = next(
        (k for task in tasks if (k := first_present_key(task, ["val_macro_rec"]))),
        "val_macro_rec",
    )
    if val_metric_choice == "macro_rec":
        return recall_key, "Macro recall"
    f1_key = next(
        (k for task in tasks if (k := first_present_key(task, ["val_macro_f1"]))),
        None,
    )
    if f1_key is not None:
        return f1_key, "Macro F1"
    print(
        "[WARN] val-metric=macro_f1 but no macro-F1 key; using macro recall.",
        file=sys.stderr,
    )
    return recall_key, "Macro recall"


def final_average_forgetting_and_bwt(
    tasks: Sequence[TaskMetrics], val_metric_key: str
) -> Tuple[float, float]:
    """Return final average forgetting and BWT (-forgetting).

    Args:
        tasks: Per-task metrics from ``load_metrics``.
        val_metric_key: e.g. ``val_f1`` or ``val_acc``.

    Returns:
        ``(avg_forgetting_final, bwt)`` where ``bwt == -avg_forgetting_final``.

    Usage:
        >>> import numpy as np
        >>> t0 = {"val_f1": np.array([0.9, 0.5])}
        >>> t1 = {"val_f1": np.array([0.9, 0.85])}
        >>> af, bwt = final_average_forgetting_and_bwt([t0, t1], "val_f1")
        >>> af >= 0
        True
    """
    _checkpoint_indices, avg_forgetting = plot_multi.compute_average_forgetting(
        tasks, val_metric_key
    )
    if avg_forgetting.size == 0:
        raise ValueError("No tasks: cannot compute forgetting.")
    final_forgetting = float(avg_forgetting[-1])
    return final_forgetting, -final_forgetting


def _prepare_runs(
    algos: Sequence[str],
    metrics_dirs: Sequence[Path] | None,
    logs_root: Path,
    run_index: int,
) -> List[plot_multi.AlgoRun]:
    """Resolve algorithms to ``AlgoRun`` (same rules as plotting script)."""
    if metrics_dirs and len(metrics_dirs) not in (0, len(algos)):
        raise SystemExit(
            "Number of --metrics-dir entries must be zero or match --algo count."
        )
    runs: List[plot_multi.AlgoRun] = []
    for idx, algo in enumerate(algos):
        if metrics_dirs and len(metrics_dirs) == len(algos):
            metrics_dir = metrics_dirs[idx]
        else:
            metrics_dir = plot_multi.find_metrics_dir_for_algo(
                algo, logs_root, run_index=run_index
            )
        metrics_dir = metrics_dir.resolve()
        if not metrics_dir.is_dir():
            raise SystemExit(f"Not a directory: {metrics_dir}")
        tasks = plot_multi.load_metrics(metrics_dir)
        name = plot_multi._resolve_algo_name(metrics_dir, explicit_name=algo)
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
    parser = argparse.ArgumentParser(
        description=(
            "Print final average forgetting and BWT (-forgetting) from metrics "
            "directories (same sources as plot_multi_algorithms)."
        )
    )
    parser.add_argument(
        "--algo",
        type=str,
        required=False,
        help=(
            "Comma-separated algorithm names (e.g. 'cmaml,hat') or a single "
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
            "and metrics directories are resolved automatically."
        ),
    )
    parser.add_argument(
        "--metrics-dir",
        type=str,
        default=None,
        help=(
            "Optional comma-separated metrics directories (same order as --algo). "
            "If omitted, latest run under logs/<algo>/ is used per algorithm."
        ),
    )
    parser.add_argument(
        "--logs-root",
        type=Path,
        default=Path("logs"),
        help="Root for auto-discovering metrics (default: logs).",
    )
    parser.add_argument(
        "--run-index",
        type=int,
        default=0,
        help="Which run by recency when auto-discovering (0 = latest).",
    )
    parser.add_argument(
        "--val-metric",
        type=str,
        choices=("macro_f1", "macro_rec"),
        default="macro_f1",
        help="Validation signal for forgetting (default: macro_f1).",
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=None,
        help="Optional path to write results as CSV.",
    )
    parser.add_argument(
        "--sort",
        type=str,
        choices=("algo", "bwt", "forgetting"),
        default="algo",
        help="Sort rows by algorithm name, BWT (desc), or forgetting (asc).",
    )
    return parser.parse_args()


def main() -> None:
    """Load metrics per algorithm and print final forgetting and BWT."""
    args = _parse_args()
    context_dir = args.run_dir if args.run_dir is not None else args.logs_root
    print(f"Using context dir: {context_dir}")
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
        algos=algorithm_names,
        metrics_dirs=metrics_dirs_list,
        logs_root=args.logs_root,
        run_index=args.run_index,
    )

    rows: List[dict[str, str | float]] = []
    for run in runs:
        val_key, val_label = resolve_val_metric_key(run.tasks, args.val_metric)
        avg_f_final, bwt = final_average_forgetting_and_bwt(run.tasks, val_key)
        rows.append(
            {
                "algo": run.name,
                "val_metric": val_label,
                "val_key": val_key,
                "avg_forgetting_final": avg_f_final,
                "bwt": bwt,
                "metrics_dir": str(run.metrics_dir),
            }
        )

    if args.sort == "bwt":
        rows.sort(key=lambda row: float(row["bwt"]), reverse=True)
    elif args.sort == "forgetting":
        rows.sort(key=lambda row: float(row["avg_forgetting_final"]))
    else:
        rows.sort(key=lambda row: str(row["algo"]))

    header = f"{'algo':<12} {'val':<12} {'forget_final':>12} {'bwt':>12}"
    print(header)
    print("-" * len(header))
    for row in rows:
        print(
            f"{row['algo']:<12} {row['val_metric']:<12} "
            f"{row['avg_forgetting_final']:12.6f} {row['bwt']:12.6f}"
        )

    if args.csv is not None:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        with args.csv.open("w", newline="", encoding="utf-8") as csv_file:
            writer = csv.DictWriter(
                csv_file,
                fieldnames=[
                    "algo",
                    "val_metric",
                    "val_key",
                    "avg_forgetting_final",
                    "bwt",
                    "metrics_dir",
                ],
            )
            writer.writeheader()
            writer.writerows(rows)


if __name__ == "__main__":
    main()
