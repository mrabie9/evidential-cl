"""Plot metrics from multiple algorithms in a single figure.

Each row in the figure corresponds to one algorithm, while each column is a
different type of plot. Metrics are read from the usual ``metrics`` directories
containing ``task*.npz`` files, as produced by ``main.py``.

The script can either:

- search under ``logs/{algo}/`` for the latest run's metrics directory, or
- use metrics directories explicitly provided by the user.

Usage:
    # Auto-discover latest runs under logs/{algo}/
    python scripts/plot_multi_algorithms.py \\
        --algo cmaml --algo hat --algo ewc

    # Use a specific run by index (0 = latest, 1 = second latest, ...)
    python scripts/plot_multi_algorithms.py \\
        --algo cmaml --algo hat --run-index 1

    # Explicit metrics directories for each algorithm
    python scripts/plot_multi_algorithms.py \\
        --algo cmaml --metrics-dir logs/cmaml/runA/0/metrics \\
        --algo hat   --metrics-dir logs/hat/runB/0/metrics \\
        -o plots/

    # Auto-discover all algorithms from a full_experiments run folder
    python scripts/plot_multi_algorithms.py \\
        --run-dir logs/full_experiments/full-til_10epochs_w-zs/full-til_A_run_20260403_111257_lnx-elkk-2 \\
        --plot-layout separate -o plots/

    # Plot all discovered algorithms in one combined row
    python scripts/plot_multi_algorithms.py \\
        --run-dir logs/full_experiments/full-til_10epochs_w-zs/full-til_A_run_20260403_111257_lnx-elkk-2 \\
        --plot-layout together -o plots/

    # Auto-discover all algorithms from a saved_models directory
    python scripts/plot_multi_algorithms.py \\
        --runs-dir logs/00_sync/full-til_10epochs_w-zs/saved_models \\
        --plot-layout separate -o plots/
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

# Use non-interactive backend only when saving to file (-o); otherwise plt.show()
_pre_parser = argparse.ArgumentParser(add_help=False)
_pre_parser.add_argument("-o", "--output-dir", type=Path, default=None)
_pre_args, _ = _pre_parser.parse_known_args()
if _pre_args.output_dir is not None:
    import matplotlib

    matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.ticker import MultipleLocator  # noqa: E402
from matplotlib.transforms import Bbox  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils.metric_keys import (  # noqa: E402
    extract_metric,
    first_present_key,
    metric_aliases,
)

try:
    # Reuse the same plot styling config (til/cil sizing & legend layout)
    # to keep exported paper-ready subplots consistent across scripts.
    from scripts.plot_fwt_metrics import PLOT_STYLES_BY_EXPERIMENT  # type: ignore
except Exception:  # pragma: no cover - best-effort import
    PLOT_STYLES_BY_EXPERIMENT: dict[str, dict[str, Any]] = {
        "default": {"figsize": (6, 3), "ylim": None, "legend_kwargs": {}},
        "til": {"figsize": (7, 3.5), "ylim": (-0.08, 0.2), "legend_kwargs": {}},
    }

try:
    from scripts.plot_style_overrides import resolve_legend_kwargs  # type: ignore
except Exception:  # pragma: no cover - best-effort import

    def resolve_legend_kwargs(  # type: ignore[misc]
        *,
        style_key: str,
        panel_key: str,
        base_legend_kwargs: dict[str, Any] | None,
        run_count: int,
    ) -> dict[str, Any]:
        _ = style_key, panel_key, run_count
        return dict(base_legend_kwargs or {})


try:
    from scripts.plot_algorithm_group_styles import (  # type: ignore
        build_group_color_map,
        group_sort_key,
    )
except Exception:  # pragma: no cover - best-effort import

    def build_group_color_map(  # type: ignore[misc]
        algorithm_names: Sequence[str],
        fallback_colors: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        _ = algorithm_names
        return dict(fallback_colors or {})

    def group_sort_key(algorithm_name: str) -> tuple[int, str]:  # type: ignore[misc]
        return (0, algorithm_name.lower())


TaskMetrics = Dict[str, Any]
_SCRIPT_DIR = Path(__file__).resolve().parent


@dataclass
class AlgoRun:
    """Resolved information for a single algorithm run.

    Attributes:
        name: Short algorithm name to show in the plots.
        metrics_dir: Directory containing ``task*.npz`` metric files.
        tasks: Loaded per-task metrics dictionaries.
        task_names: Optional human-readable names for each task.
    """

    name: str
    metrics_dir: Path
    tasks: List[TaskMetrics]
    task_names: List[str] | None = None


def load_task_names(metrics_dir: Path) -> List[str] | None:
    """Load task names from a ``task_order.txt`` file if present.

    The file is expected to live inside ``metrics_dir`` and contain one task
    name per line in task index order.
    """
    order_file = metrics_dir / "task_order.txt"
    if not order_file.is_file():
        return None

    names: List[str] = []
    with order_file.open(encoding="utf-8") as f:
        for line in f:
            stripped = line.strip()
            if stripped:
                names.append(stripped)
    return names or None


def _group_for_task_name(task_name: str) -> str:
    """Return a semantic group label for a task name."""
    name = task_name.lower()
    if "rcn" in name:
        return "rcn"
    if "deeprad" in name:
        return "deeprad"
    if "uclresm" in name:
        return "uclresm"
    return "other"


def get_task_color(task_idx: int, task_names: Sequence[str] | None = None):
    """Assign a colour for a task, grouping by task name when available."""
    if task_names is None or task_idx >= len(task_names):
        group1 = [3, 4, 6, 9]
        group2 = [0, 1, 8]
        group3 = [2, 5, 7]

        if task_idx in group1:
            cmap = plt.get_cmap("Blues")
            idx = group1.index(task_idx)
            return cmap(0.4 + 0.6 * (idx / len(group1)))
        if task_idx in group2:
            cmap = plt.get_cmap("Oranges")
            idx = group2.index(task_idx)
            return cmap(0.4 + 0.6 * (idx / len(group2)))
        if task_idx in group3:
            cmap = plt.get_cmap("Greens")
            idx = group3.index(task_idx)
            return cmap(0.4 + 0.6 * (idx / len(group3)))
        return f"C{task_idx % 10}"

    groups: Dict[str, List[int]] = {}
    for idx, name in enumerate(task_names):
        group = _group_for_task_name(name)
        groups.setdefault(group, []).append(idx)

    task_name = task_names[task_idx]
    group = _group_for_task_name(task_name)
    group_indices = groups.get(group, [task_idx])
    position_in_group = group_indices.index(task_idx)
    group_size = len(group_indices)

    def _shade(index: int, size: int) -> float:
        if size <= 1:
            return 0.7
        return 0.4 + 0.6 * (index / float(size))

    cmap_name = {
        "rcn": "Oranges",
        "deeprad": "Blues",
        "uclresm": "Greens",
        "other": "Greys",
    }[group]
    cmap = plt.get_cmap(cmap_name)
    return cmap(_shade(position_in_group, group_size))


def _build_label_colors(
    labels: Sequence[str],
    grouping_keywords: Sequence[str] | None,
) -> Dict[str, Any]:
    """Build consistent colors for run labels with optional grouping.

    When grouping keywords are provided, labels containing the same keyword
    receive shades from the same colormap family.

    Args:
        labels: Display labels, one per run.
        grouping_keywords: Optional keyword list used to group labels by
            substring match.

    Returns:
        A mapping from label text to a matplotlib color value.

    Usage:
        >>> _build_label_colors(["iid2-a", "iid2-b"], ["a", "b"])
        {'iid2-a': (...), 'iid2-b': (...)}
    """
    if not labels:
        return {}

    qualitative_palette: List[Any] = []
    for cmap_name in ("tab20", "tab20b", "tab20c"):
        cmap = plt.get_cmap(cmap_name)
        cmap_colors = getattr(cmap, "colors", None)
        if cmap_colors is None:
            continue
        qualitative_palette.extend(list(cmap_colors))

    def _color_for_label_position(label_position: int, total_labels: int) -> Any:
        if label_position < len(qualitative_palette):
            return qualitative_palette[label_position]
        if total_labels <= 1:
            return plt.get_cmap("hsv")(0.0)
        return plt.get_cmap("hsv")(label_position / float(total_labels))

    if grouping_keywords is None or len(grouping_keywords) == 0:
        return {
            label: _color_for_label_position(idx, len(labels))
            for idx, label in enumerate(labels)
        }

    normalized_keywords = [keyword.strip().lower() for keyword in grouping_keywords]
    grouped_label_positions: Dict[int, List[int]] = {
        group_index: [] for group_index in range(len(normalized_keywords))
    }
    unmatched_label_positions: List[int] = []

    for label_position, label_text in enumerate(labels):
        lowered_label = label_text.lower()
        matched_group_index = None
        for group_index, group_keyword in enumerate(normalized_keywords):
            if group_keyword and group_keyword in lowered_label:
                matched_group_index = group_index
                break
        if matched_group_index is None:
            unmatched_label_positions.append(label_position)
        else:
            grouped_label_positions[matched_group_index].append(label_position)

    colormap_cycle = [
        "Blues",
        "Oranges",
        "Greens",
        "Purples",
        "Reds",
        "Greys",
        "YlOrBr",
        "PuBuGn",
    ]

    def _shade(index_in_group: int, group_size: int) -> float:
        if group_size <= 1:
            return 0.7
        return 0.35 + 0.6 * (index_in_group / float(max(group_size - 1, 1)))

    label_colors: Dict[str, Any] = {}

    for group_index, positions in grouped_label_positions.items():
        if not positions:
            continue
        cmap = plt.get_cmap(colormap_cycle[group_index % len(colormap_cycle)])
        for position_in_group, label_position in enumerate(positions):
            label_colors[labels[label_position]] = cmap(
                _shade(position_in_group, len(positions))
            )

    for label_position in unmatched_label_positions:
        label_colors[labels[label_position]] = _color_for_label_position(
            label_position, len(labels)
        )

    return label_colors


def _task_index(filename: str) -> int:
    """Extract task number from filename like task0.npz or task12.npz.

    Args:
        filename: Name of the metrics file.

    Returns:
        The parsed task index or -1 if it cannot be parsed.

    Usage:
        >>> _task_index("task3.npz")
        3
    """
    match = re.match(r"task(\d+)\.npz", filename, re.IGNORECASE)
    if not match:
        return -1
    return int(match.group(1))


def load_metrics(metrics_dir: Path) -> List[TaskMetrics]:
    """Load all ``task*.npz`` files from a metrics directory.

    Args:
        metrics_dir: Path to the metrics directory.

    Returns:
        List of dictionaries, one per task, with at least the keys
        ``losses``, ``cls_tr_rec`` and ``val_acc`` when present.

    Raises:
        FileNotFoundError: If no ``task*.npz`` files are found.

    Usage:
        >>> tasks = load_metrics(Path("logs/cmaml/run/0/metrics"))
        >>> len(tasks) > 0
        True
    """
    task_files = sorted(
        metrics_dir.glob("task*.npz"),
        key=lambda p: _task_index(p.name),
    )
    if not task_files:
        raise FileNotFoundError(f"No task*.npz files in {metrics_dir}")

    tasks: List[TaskMetrics] = []
    for path in task_files:
        data = np.load(path, allow_pickle=True)
        task_data: TaskMetrics = {key: np.asarray(data[key]) for key in data.files}
        # Truncate validation metrics (e.g. val_macro_rec, val_macro_f1) in the
        # same way as scripts/plot_metrics.py so that only the final per-task
        # values remain. val_macro_f1 in particular accumulates across the
        # whole run (never reset per task in life_experience), so without this
        # the same early entries get read back for every task and forgetting
        # comes out as 0. Walk both canonical and legacy key names so runs
        # produced before the cls_* -> macro_* rename still load correctly.
        task_idx = _task_index(path.name)
        num_tasks_seen = task_idx + 1
        for canonical_key in ("val_macro_rec", "val_macro_f1"):
            for key in metric_aliases(canonical_key):
                if key in task_data and len(task_data[key]) > num_tasks_seen:
                    task_data[key] = task_data[key][-num_tasks_seen:]
        tasks.append(task_data)
    return tasks


def compute_average_forgetting(
    tasks: Sequence[TaskMetrics],
    val_metric_key: str,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute average forgetting over previous tasks at each checkpoint.

    This uses a single validation metric key (e.g. ``val_f1`` or ``val_acc``)
    consistently for peak and current values.

    For each task ``t``, peak metric is taken as ``tasks[t][val_metric_key][t]``
    after training task ``t``. After training task ``K > t``, the metric on
    task ``t`` is ``tasks[K][val_metric_key][t]``. Forgetting for task ``t`` at
    ``K`` is ``peak[t] - current_metric`` (not clamped), and the average at
    ``K`` is the mean over all previous tasks. This allows negative values,
    which indicate backward transfer.

    Args:
        tasks: Sequence of per-task metric dictionaries.
        val_metric_key: Key of the validation metric to use (e.g. ``val_f1`` or ``val_acc``).

    Returns:
        A tuple ``(x, avg_forgetting)`` where ``x`` is an array of checkpoint
        indices and ``avg_forgetting`` is the corresponding forgetting values.

    Usage:
        >>> x, f = compute_average_forgetting(tasks, \"val_acc\")
        >>> x.shape == f.shape
        True
    """
    n_tasks = len(tasks)
    if n_tasks == 0:
        return np.array([]), np.array([])

    # Peak metric after learning each task t.
    peak_vals: List[float] = []
    for t in range(n_tasks):
        metrics_t = tasks[t]
        metric_vals = metrics_t.get(val_metric_key)
        if metric_vals is None or len(metric_vals) <= t:
            continue
        peak_vals.append(float(metric_vals[t]))
    peak_vals_arr = np.asarray(peak_vals, dtype=float)
    avg_forgetting = np.zeros(n_tasks, dtype=float)

    for k in range(n_tasks):
        if k == 0:
            avg_forgetting[k] = 0.0
            continue
        forgets: List[float] = []
        for t in range(k):
            metrics_k = tasks[k]
            metric_vals_k = metrics_k.get(val_metric_key)
            if metric_vals_k is None or len(metric_vals_k) <= t:
                continue
            current_metric = float(metric_vals_k[t])
            forgets.append(float(peak_vals_arr[t] - current_metric))
        avg_forgetting[k] = float(np.mean(forgets)) if forgets else 0.0

    return np.arange(n_tasks), avg_forgetting


def find_metrics_dir_for_algo(
    algo: str,
    logs_root: Path,
    run_index: int = 0,
) -> Path:
    """Find a metrics directory for an algorithm by run index (0 = latest).

    The function looks for directories named ``metrics`` under
    ``logs_root / algo`` that contain at least one ``task*.npz`` file, sorts
    them by parent run directory modification time (newest first), then
    returns the one at ``run_index`` (0 = latest, 1 = second latest, etc.).

    Args:
        algo: Algorithm name (subdirectory under ``logs_root``).
        logs_root: Root directory containing algorithm subdirectories.
        run_index: Which run to use: 0 for latest, 1 for second latest, etc.
            Default is 0 (latest run).

    Returns:
        Path to the metrics directory for the given algorithm and run index.

    Raises:
        FileNotFoundError: If no valid metrics directory is found, or
            ``run_index`` is out of range.

    Usage:
        >>> logs_root = Path("logs")
        >>> latest = find_metrics_dir_for_algo("cmaml", logs_root, run_index=0)
        >>> latest.name
        'metrics'
        >>> second = find_metrics_dir_for_algo("cmaml", logs_root, run_index=1)
    """
    algo_root = logs_root / algo
    if not algo_root.exists():
        raise FileNotFoundError(
            f"No logs found for algorithm '{algo}' under {logs_root}"
        )

    candidate_dirs: List[Tuple[float, Path]] = []
    for metrics_dir in algo_root.rglob("metrics"):
        if not metrics_dir.is_dir():
            continue
        if not any(metrics_dir.glob("task*.npz")):
            continue
        # Use the parent run directory modification time as a proxy for recency.
        run_dir = metrics_dir.parent
        mtime = run_dir.stat().st_mtime
        candidate_dirs.append((mtime, metrics_dir))

    if not candidate_dirs:
        raise FileNotFoundError(
            f"No metrics directories with task*.npz for '{algo}' under {algo_root}"
        )

    candidate_dirs.sort(key=lambda x: x[0], reverse=True)
    if run_index < 0 or run_index >= len(candidate_dirs):
        raise FileNotFoundError(
            f"Run index {run_index} out of range for '{algo}' "
            f"(found {len(candidate_dirs)} run(s); use 0 to {len(candidate_dirs) - 1})."
        )
    return candidate_dirs[run_index][1]


def _extract_algo_name_from_job_filename(log_path: Path) -> str | None:
    """Extract algorithm/config stem from a job log filename.

    Args:
        log_path: Path to a single ``job_*.log`` file.

    Returns:
        Algorithm/config stem, or ``None`` when the filename cannot be parsed.
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


def _discover_runs_from_run_dir(run_dir: Path) -> tuple[List[str], List[Path]]:
    """Discover model names and metrics dirs from a run folder.

    Args:
        run_dir: Full run directory containing ``job_logs``.

    Returns:
        Tuple ``(algorithm_names, metrics_dirs)`` sorted by algorithm name.
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

        normalized_output_dir = Path(str(logged_output_dir).replace("//", "/"))
        if normalized_output_dir.is_absolute():
            output_dir = normalized_output_dir
        else:
            output_dir = (_SCRIPT_DIR.parent / normalized_output_dir).resolve()
        metrics_dir = _resolve_output_dir_to_metrics_dir(output_dir).resolve()
        if not metrics_dir.is_dir():
            continue
        if not any(metrics_dir.glob("task*.npz")):
            continue
        discovered_by_model[algorithm_name] = metrics_dir

    if not discovered_by_model:
        raise SystemExit(
            "No metrics directories discovered from job logs under "
            f"{job_logs_dir}. Ensure the run has completed and wrote task*.npz files."
        )

    sorted_models = sorted(discovered_by_model.keys())
    algorithm_names = [model_name for model_name in sorted_models]
    metrics_dirs = [discovered_by_model[model_name] for model_name in sorted_models]
    return algorithm_names, metrics_dirs


def _discover_runs_from_algorithm_root(
    algorithms_root_dir: Path,
) -> tuple[List[str], List[Path]]:
    """Discover latest metrics directory for each algorithm under a root dir.

    Expected layout resembles:
    ``<algorithms_root_dir>/<algo>/<run...>/0/metrics/task*.npz``.
    """
    if not algorithms_root_dir.is_dir():
        raise SystemExit(f"--runs-dir is not a directory: {algorithms_root_dir}")

    discovered_by_algo: Dict[str, Tuple[float, Path]] = {}
    wrapper_dir_names = {"saved_models", "models"}
    skipped_root_dir_names = {"plots", "figures"}
    for metrics_dir in algorithms_root_dir.rglob("metrics"):
        if not metrics_dir.is_dir():
            continue
        if not any(metrics_dir.glob("task*.npz")):
            continue
        try:
            relative_parts = metrics_dir.relative_to(algorithms_root_dir).parts
        except ValueError:
            continue
        if not relative_parts:
            continue

        root_name = relative_parts[0]
        if root_name in skipped_root_dir_names:
            continue
        if root_name in wrapper_dir_names and len(relative_parts) >= 2:
            algorithm_name = relative_parts[1]
        else:
            algorithm_name = root_name

        metrics_mtime = metrics_dir.stat().st_mtime
        existing = discovered_by_algo.get(algorithm_name)
        if existing is None or metrics_mtime > existing[0]:
            discovered_by_algo[algorithm_name] = (metrics_mtime, metrics_dir.resolve())

    if not discovered_by_algo:
        raise SystemExit(
            "No algorithm metrics directories found under --runs-dir: "
            f"{algorithms_root_dir}"
        )

    sorted_algos = sorted(discovered_by_algo.keys())
    algorithm_names = [algo_name for algo_name in sorted_algos]
    metrics_dirs = [discovered_by_algo[algo_name][1] for algo_name in sorted_algos]
    return algorithm_names, metrics_dirs


def _discover_runs_from_saved_models_root(
    saved_models_root_dir: Path,
) -> tuple[List[str], List[Path]]:
    """Discover latest metrics directory per algorithm from a saved_models root.

    Expected layout:
    ``<saved_models_root>/<algorithm>/<experiment>/<seed>/metrics/task*.npz``.
    """
    if not saved_models_root_dir.is_dir():
        raise SystemExit(
            f"Saved-models root is not a directory: {saved_models_root_dir}"
        )

    discovered_by_algo: Dict[str, Tuple[float, Path]] = {}
    for algorithm_dir in sorted(saved_models_root_dir.iterdir()):
        if not algorithm_dir.is_dir():
            continue
        algorithm_name = algorithm_dir.name
        latest_candidate: Tuple[float, Path] | None = None

        for metrics_dir in algorithm_dir.rglob("metrics"):
            if not metrics_dir.is_dir():
                continue
            if not any(metrics_dir.glob("task*.npz")):
                continue
            metrics_mtime = metrics_dir.stat().st_mtime
            if latest_candidate is None or metrics_mtime > latest_candidate[0]:
                latest_candidate = (metrics_mtime, metrics_dir.resolve())

        if latest_candidate is not None:
            discovered_by_algo[algorithm_name] = latest_candidate

    if not discovered_by_algo:
        raise SystemExit(
            "No algorithm metrics directories found under saved_models root: "
            f"{saved_models_root_dir}"
        )

    sorted_algorithms = sorted(discovered_by_algo.keys())
    algorithm_names = [algorithm_name for algorithm_name in sorted_algorithms]
    metrics_dirs = [
        discovered_by_algo[algorithm_name][1] for algorithm_name in sorted_algorithms
    ]
    return algorithm_names, metrics_dirs


def _resolve_algo_name(metrics_dir: Path, explicit_name: str | None) -> str:
    """Resolve the algorithm name for a metrics directory.

    Args:
        metrics_dir: Directory containing metrics files.
        explicit_name: Optional name passed on the command line.

    Returns:
        A short name for labelling plots.
    """
    if explicit_name:
        return explicit_name
    # Fallback: use the algorithm directory name under logs/
    parts = metrics_dir.parts
    try:
        logs_index = parts.index("logs")
    except ValueError:
        return metrics_dir.parent.name
    if logs_index + 1 < len(parts):
        return parts[logs_index + 1]
    return metrics_dir.parent.name


def _prepare_algo_runs(
    algos: Sequence[str],
    metrics_dirs: Sequence[Path] | None,
    logs_root: Path,
    run_index: int = 0,
) -> List[AlgoRun]:
    """Resolve algorithms and metrics directories into a list of runs.

    Args:
        algos: Algorithm names from CLI.
        metrics_dirs: Optional list of metrics directories, same length as
            ``algos``. If empty or ``None``, metrics directories are discovered
            under ``logs_root/{algo}/`` using ``run_index``.
        logs_root: Root logs directory.
        run_index: When auto-discovering, which run to use: 0 = latest,
            1 = second latest, etc. Ignored when ``metrics_dirs`` is provided.

    Returns:
        List of ``AlgoRun`` instances ready for plotting.
    """
    runs: List[AlgoRun] = []

    if metrics_dirs and len(metrics_dirs) not in (0, len(algos)):
        raise SystemExit(
            "Number of --metrics-dir entries must be zero or match the number of --algo entries."
        )

    for idx, algo in enumerate(algos):
        if metrics_dirs and len(metrics_dirs) == len(algos):
            metrics_dir = metrics_dirs[idx]
        else:
            metrics_dir = find_metrics_dir_for_algo(
                algo, logs_root, run_index=run_index
            )

        metrics_dir = metrics_dir.resolve()
        if not metrics_dir.is_dir():
            raise SystemExit(f"Not a directory: {metrics_dir}")

        tasks = load_metrics(metrics_dir)
        task_names = load_task_names(metrics_dir)
        name = _resolve_algo_name(metrics_dir, explicit_name=algo)
        runs.append(
            AlgoRun(
                name=name,
                metrics_dir=metrics_dir,
                tasks=tasks,
                task_names=task_names,
            )
        )

    return runs


def _plot_loss_vs_steps(ax: plt.Axes, tasks: Sequence[TaskMetrics]) -> None:
    """Deprecated: kept for backwards compatibility, not used."""
    if not tasks:
        return


def _resolve_train_metric_for_task(
    task: TaskMetrics,
    train_metric_choice: str,
) -> Tuple[np.ndarray, str]:
    """Resolve train metric series and display label for one task.

    Args:
        task: Per-task metric dictionary loaded from ``task*.npz``.
        train_metric_choice: Selected training metric key semantic.

    Returns:
        Tuple ``(values, label)`` where ``values`` is the selected metric series
        and ``label`` is a human-readable metric name.

    Raises:
        KeyError: If required metric keys are missing for the selected choice.
    """
    if train_metric_choice == "macro_f1":
        f1_values = extract_metric(task, "train_macro_f1")
        if f1_values is None:
            raise KeyError("Requested train macro_f1 but no macro-F1 key is present.")
        return np.asarray(f1_values, dtype=float), "Train macro F1"

    # train_metric_choice == "macro_rec"
    recall_values = extract_metric(task, "tr_macro_rec")
    if recall_values is None:
        recall_values = task.get("tr_acc")
    if recall_values is None:
        raise KeyError("No training macro-recall key found in task metrics.")
    return np.asarray(recall_values, dtype=float), "Train macro recall"


def _plot_train_metric_vs_steps(
    ax: plt.Axes,
    tasks: Sequence[TaskMetrics],
    task_names: Sequence[str] | None,
    train_metric_choice: str,
) -> None:
    """Plot selected training metric vs step for each task in a run."""
    if not tasks:
        return
    y_label: str | None = None
    x_axis_label = "Step"
    for task_idx, task in enumerate(tasks):
        metric_values, metric_label = _resolve_train_metric_for_task(
            task, train_metric_choice
        )
        y_label = metric_label
        if train_metric_choice == "macro_f1":
            x_values = np.arange(1, len(metric_values) + 1)
            x_axis_label = "Epoch"
        else:
            x_values = np.arange(len(metric_values))
            x_axis_label = "Step"
        ax.plot(
            x_values,
            metric_values,
            color=get_task_color(task_idx, task_names),
            alpha=0.8,
            label=f"Task {task_idx}",
        )
    if y_label is not None:
        ax.set_ylabel(y_label)
    ax.set_xlabel(x_axis_label)
    ax.grid(True, alpha=0.3)


def _plot_val_acc_over_tasks(
    ax: plt.Axes,
    tasks: Sequence[TaskMetrics],
    task_names: Sequence[str] | None,
    val_metric_key: str,
    val_metric_label: str,
) -> None:
    """Plot a validation metric per task as more tasks are trained.

    This shows one curve per task (as in ``scripts/plot_metrics.py``), plus an
    additional aggregate curve where each point ``k`` is the mean metric value
    over tasks ``0..k`` at checkpoint ``k``.
    """
    n_tasks = len(tasks)
    if n_tasks == 0:
        return

    x = np.arange(1, n_tasks + 1)
    # Per-task curves.
    for task_idx in range(n_tasks):
        ys = []
        for k in range(n_tasks):
            metric_vals = tasks[k].get(val_metric_key)
            if metric_vals is None or task_idx >= len(metric_vals):
                ys.append(np.nan)
            else:
                ys.append(float(metric_vals[task_idx]))
        ax.plot(
            x,
            ys,
            "o-",
            label=f"Task {task_idx}",
            color=get_task_color(task_idx, task_names),
            alpha=0.8,
        )

    # Aggregate curve: at checkpoint k, mean over tasks 0..k using the same
    # validation metric key. Missing values and NaNs are skipped.
    agg_vals: List[float] = []
    for k in range(n_tasks):
        metrics_k = tasks[k]
        metric_vals_k = metrics_k.get(val_metric_key)
        if metric_vals_k is None:
            agg_vals.append(float("nan"))
            continue
        window: List[float] = []
        for t in range(k + 1):
            if t >= len(metric_vals_k):
                continue
            v = float(metric_vals_k[t])
            if np.isnan(v):
                continue
            window.append(v)
        if window:
            agg_vals.append(float(np.mean(window)))
        else:
            agg_vals.append(float("nan"))

    ax.plot(
        x,
        np.asarray(agg_vals, dtype=float),
        "k^-",
        linewidth=2.0,
        markersize=4.0,
        label=f"Mean {val_metric_label}",
    )

    ax.set_xlabel("Task")
    ax.set_ylabel(f"Val {val_metric_label}")
    ax.grid(True, alpha=0.3)


def _plot_average_forgetting(
    ax: plt.Axes,
    tasks: Sequence[TaskMetrics],
    val_metric_key: str,
    val_metric_label: str,
) -> None:
    """Plot average forgetting vs checkpoint index for an algorithm."""
    x, avg_forgetting = compute_average_forgetting(tasks, val_metric_key)
    if x.size == 0:
        return
    ax.plot(x, avg_forgetting, "o-", color="C3")
    ax.set_xlabel("Task")
    ax.set_ylabel(f"Avg forgetting ({val_metric_label})")
    ax.grid(True, alpha=0.3)


def _compute_mean_val_metric_over_tasks(
    tasks: Sequence[TaskMetrics],
    val_metric_key: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute mean validation metric over seen tasks at each checkpoint.

    Args:
        tasks: Sequence of per-task metric dictionaries.
        val_metric_key: Validation metric key to aggregate.

    Returns:
        Tuple ``(x, mean_values)`` where ``x`` is checkpoint index
        (1..num_tasks) and ``mean_values`` is the mean metric over tasks 0..k
        at each checkpoint ``k``.
    """
    n_tasks = len(tasks)
    if n_tasks == 0:
        return np.array([]), np.array([])

    x = np.arange(1, n_tasks + 1, dtype=float)
    mean_values: List[float] = []
    for checkpoint_index in range(n_tasks):
        metric_values = tasks[checkpoint_index].get(val_metric_key)
        if metric_values is None:
            mean_values.append(float("nan"))
            continue
        metric_array = np.asarray(metric_values, dtype=float).reshape(-1)
        valid_values: List[float] = []
        for task_index in range(checkpoint_index + 1):
            if task_index >= metric_array.size:
                continue
            value = float(metric_array[task_index])
            if np.isnan(value):
                continue
            valid_values.append(value)
        mean_values.append(
            float(np.mean(valid_values)) if valid_values else float("nan")
        )
    return x, np.asarray(mean_values, dtype=float)


def _plot_final_validation_metrics(
    ax: plt.Axes,
    tasks: Sequence[TaskMetrics],
    task_names: Sequence[str] | None,
) -> None:
    """Plot per-task final validation macro recall from the last task.

    Args:
        ax: Axis to draw on.
        tasks: Per-task metric dicts for one run, in task order.
        task_names: Optional task names used for consistent per-task colouring.

    Returns:
        None.

    Usage:
        _plot_final_validation_metrics(ax, run.tasks, task_names)
    """
    if not tasks:
        return

    val_recall = extract_metric(tasks[-1], "val_macro_rec")
    if val_recall is None:
        return

    recall_values = np.asarray(val_recall, dtype=float).reshape(-1)
    n_tasks = recall_values.size
    if n_tasks == 0:
        return
    task_indices = np.arange(n_tasks)
    colors = [get_task_color(i, task_names) for i in range(n_tasks)]

    ax.bar(
        task_indices,
        recall_values,
        width=0.6,
        label="Macro recall",
        color=colors,
    )

    ax.set_xlabel("Task")
    ax.set_ylabel("Metric value")
    ax.set_xticks(task_indices)
    ax.yaxis.set_major_locator(MultipleLocator(0.1))
    ax.grid(True, alpha=0.3, axis="y")


def _mean_final_metric_for_run(
    metrics: TaskMetrics,
    canonical_key: str,
    fallback_num_tasks: int,
) -> float | None:
    """Return a run-level final metric aligned with SUMMARY_TE semantics.

    Metrics may be stored either as a per-task final vector or as a flattened
    per-eval array. In the flattened case the last ``n_tasks`` entries are the
    final evaluation, where ``n_tasks`` is taken from the run's task count.

    Args:
        metrics: Stored metric arrays for the final task.
        canonical_key: Canonical metric name (legacy spellings are resolved).
        fallback_num_tasks: Number of tasks in the run.

    Returns:
        Mean of the final per-task values, or ``None`` when absent/empty.

    Usage:
        mean_rec = _mean_final_metric_for_run(last, "val_macro_rec", len(run.tasks))
    """
    values = extract_metric(metrics, canonical_key)
    if values is None:
        return None
    arr = np.asarray(values, dtype=float).reshape(-1)
    if arr.size == 0:
        return None

    num_tasks = max(int(fallback_num_tasks), 1)
    if arr.size > num_tasks:
        return float(np.mean(arr[-num_tasks:]))
    return float(np.mean(arr))


def _concat_train_metric_for_run(
    tasks: Sequence[TaskMetrics],
    train_metric_choice: str,
) -> Tuple[np.ndarray | None, str]:
    """Concatenate per-task train metric series into one run-level sequence.

    If there is only one task, this returns that task's train metric unchanged.
    If multiple tasks exist, task series are concatenated in task order.
    """
    if not tasks:
        return None, "Train macro recall"

    per_task_series: list[np.ndarray] = []
    metric_label = "Train macro recall"
    for task in tasks:
        try:
            metric_values, metric_label = _resolve_train_metric_for_task(
                task, train_metric_choice
            )
        except KeyError:
            continue
        arr = np.asarray(metric_values, dtype=float).reshape(-1)
        if arr.size > 0:
            per_task_series.append(arr)

    if not per_task_series:
        return None, metric_label
    if len(per_task_series) == 1:
        return per_task_series[0], metric_label
    return np.concatenate(per_task_series, axis=0), metric_label


def _resolve_train_x_axis_label(train_metric_choice: str) -> str:
    """Resolve x-axis label for the first panel train metric plot."""
    return "Epoch" if train_metric_choice == "macro_f1" else "Step"


def _build_axis_only_export_bbox(
    axis: plt.Axes,
    renderer: Any,
    pad_pixels: float,
) -> Bbox:
    """Build an export bbox using only artists that belong to one axis."""
    bbox_parts: List[Bbox] = [axis.get_window_extent(renderer)]

    axis_title = axis.title
    if axis_title.get_text():
        bbox_parts.append(axis_title.get_window_extent(renderer))
    x_label = axis.xaxis.label
    if x_label.get_text():
        bbox_parts.append(x_label.get_window_extent(renderer))
    y_label = axis.yaxis.label
    if y_label.get_text():
        bbox_parts.append(y_label.get_window_extent(renderer))

    for tick_label in axis.get_xticklabels() + axis.get_yticklabels():
        if tick_label.get_visible() and tick_label.get_text():
            bbox_parts.append(tick_label.get_window_extent(renderer))

    legend = axis.get_legend()
    if legend is not None and legend.get_visible():
        bbox_parts.append(legend.get_window_extent(renderer))

    combined_bbox = Bbox.union(bbox_parts)
    return Bbox.from_extents(
        combined_bbox.x0 - pad_pixels,
        combined_bbox.y0 - pad_pixels,
        combined_bbox.x1 + pad_pixels,
        combined_bbox.y1 + pad_pixels,
    )


def _resolve_safe_export_dpi(
    figure_width_inches: float,
    figure_height_inches: float,
    requested_dpi: int,
    max_dimension_pixels: int = 16000,
) -> int:
    """Clamp export DPI so no figure dimension becomes too large.

    Some image viewers fail to open very tall/high-resolution PNGs. This keeps
    the largest side under a practical pixel ceiling while preserving the
    requested DPI whenever possible.
    """
    largest_side_inches = max(figure_width_inches, figure_height_inches)
    if largest_side_inches <= 0:
        return requested_dpi
    max_safe_dpi = int(max_dimension_pixels / largest_side_inches)
    max_safe_dpi = max(72, max_safe_dpi)
    return min(requested_dpi, max_safe_dpi)


def plot_multi_algorithms(
    runs: Sequence[AlgoRun],
    output_dir: Path | None,
    same_y_limits: bool = False,
    val_metric_choice: str = "macro_f1",
    train_metric_choice: str = "macro_rec",
    labels: Sequence[str] | None = None,
    labels_grouping: Sequence[str] | None = None,
    plot_layout: str = "separate",
    save_subplots: bool = False,
) -> None:
    """Create the multi-row, multi-column figure for all algorithms.

    Args:
        runs: Sequence of resolved algorithm runs.
        output_dir: Optional directory in which to save the combined figure.

    Usage:
        >>> plot_multi_algorithms(runs, Path("plots"))
    """
    if not runs:
        raise ValueError("No algorithms provided for plotting.")

    # Special-case: IID oracle baseline. These runs each contain a single task
    # and it is more natural to compare multiple IID runs within a single row.
    iid2_mode = all(run.name.lower() == "iid2" for run in runs)
    combined_mode = plot_layout == "together" or iid2_mode

    n_algos = len(runs)
    if combined_mode:
        n_rows = 1
        n_cols = 4
    else:
        n_rows = n_algos
        n_cols = 4

    # Use a larger figure when saving to file to allow detailed zooming.
    if output_dir is not None:
        col_width, row_height = 9, 4.5
        requested_dpi = 550
    else:
        col_width, row_height = 4.0, 3.0
        requested_dpi = 200

    if combined_mode:
        # For combined layout use balanced widths across four panels.
        if output_dir is not None:
            fig_width = col_width * n_cols
        else:
            fig_width = col_width * n_cols
        fig, axes = plt.subplots(
            n_rows,
            n_cols,
            figsize=(fig_width, row_height * n_rows),
            squeeze=False,
            dpi=requested_dpi,
        )
    else:
        fig, axes = plt.subplots(
            n_rows,
            n_cols,
            figsize=(col_width * n_cols, row_height * n_rows),
            squeeze=False,
            dpi=requested_dpi,
        )

    figure_width_inches, figure_height_inches = fig.get_size_inches()
    dpi = _resolve_safe_export_dpi(
        figure_width_inches=float(figure_width_inches),
        figure_height_inches=float(figure_height_inches),
        requested_dpi=requested_dpi,
    )
    if dpi != requested_dpi and output_dir is not None:
        print(
            "[INFO] Reduced export DPI from "
            f"{requested_dpi} to {dpi} to keep image dimensions loadable."
        )

    def _resolve_val_metric_for_run(choice: str, run: AlgoRun) -> tuple[str, str]:
        """Resolve validation metric key and label for a single run."""
        recall_key = next(
            (
                k
                for task in run.tasks
                if (k := first_present_key(task, ["val_macro_rec"]))
            ),
            "val_macro_rec",
        )
        if choice == "macro_rec":
            return recall_key, "Macro recall"
        # choice == "macro_f1": prefer F1 when available, fall back to recall.
        f1_key = next(
            (
                k
                for task in run.tasks
                if (k := first_present_key(task, ["val_macro_f1"]))
            ),
            None,
        )
        if f1_key is not None:
            return f1_key, "Macro F1"
        print(
            f"[WARN] Requested val-metric=macro_f1 but run '{run.name}' at "
            f"{run.metrics_dir} has no macro-F1 key; falling back to macro recall."
        )
        return recall_key, "Macro recall"

    # Resolve label for column titles from the first run.
    first_key, first_label = _resolve_val_metric_for_run(val_metric_choice, runs[0])

    if combined_mode:
        # Optional custom labels for IID2 runs; fall back to run names.
        label_list: list[str] = []
        for idx, run in enumerate(runs):
            if labels is not None and idx < len(labels):
                label_list.append(str(labels[idx]))
            else:
                label_list.append(run.name)
        label_colors = _build_label_colors(label_list, labels_grouping)
        fallback_colors_by_algorithm = {
            run.name: label_colors.get(label_list[run_idx], f"C{run_idx % 10}")
            for run_idx, run in enumerate(runs)
        }
        group_colors_by_algorithm = build_group_color_map(
            [run.name for run in runs],
            fallback_colors=fallback_colors_by_algorithm,
        )

        row_axes = axes[0]

        # Train recall vs step: one line per run. If multiple tasks are present,
        # concatenate task series so x-axis reflects total iterations.
        train_metric_label = "Train recall"
        for run_idx, run in enumerate(runs):
            run_label = label_list[run_idx]
            run_color = group_colors_by_algorithm.get(
                run.name, label_colors.get(run_label, f"C{run_idx % 10}")
            )
            if not run.tasks:
                continue
            train_series, train_metric_label = _concat_train_metric_for_run(
                run.tasks, train_metric_choice
            )
            if train_series is None:
                continue
            if train_metric_choice == "macro_f1":
                x_values = np.arange(1, len(train_series) + 1)
            else:
                x_values = np.arange(len(train_series))
            row_axes[0].plot(
                x_values,
                train_series,
                ".-",
                label=run_label,
                color=run_color,
                alpha=0.9,
            )
        row_axes[0].set_ylabel(train_metric_label)
        row_axes[0].set_xlabel(_resolve_train_x_axis_label(train_metric_choice))
        row_axes[0].grid(True, alpha=0.3)

        # Final validation metrics: bars grouped by run, using the last task.
        for run_idx, run in enumerate(runs):
            run_color = group_colors_by_algorithm.get(
                run.name, label_colors.get(label_list[run_idx], f"C{run_idx % 10}")
            )
            if not run.tasks:
                continue
            last = run.tasks[-1]
            # Match SUMMARY_TE aggregation (mean over final per-task values).
            x_center = run_idx
            width = 0.3
            mean_rec = _mean_final_metric_for_run(last, "val_macro_rec", len(run.tasks))
            mean_f1 = _mean_final_metric_for_run(last, "val_macro_f1", len(run.tasks))
            if mean_rec is None and mean_f1 is None:
                continue

            if mean_rec is not None:
                row_axes[1].bar(
                    x_center - width / 2,
                    mean_rec,
                    width=width,
                    label="Macro recall" if run_idx == 0 else None,
                    color=run_color,
                    alpha=0.8,
                )
            if mean_f1 is not None:
                row_axes[1].bar(
                    x_center + width / 2,
                    mean_f1,
                    width=width,
                    label="Macro F1" if run_idx == 0 else None,
                    color=run_color,
                    hatch="..",
                    alpha=0.8,
                )
        row_axes[1].set_ylabel("Metric value")
        row_axes[1].set_xticks(np.arange(len(runs)))
        # The run names are already shown via the shared legend; avoid duplicating
        # them as x tick labels.
        row_axes[1].set_xticklabels([])
        row_axes[1].set_xlabel("")
        row_axes[1].yaxis.set_major_locator(MultipleLocator(0.1))
        row_axes[1].grid(True, alpha=0.3, axis="y")

        # Mean validation metric over tasks: one line per run.
        for run_idx, run in enumerate(runs):
            run_color = group_colors_by_algorithm.get(
                run.name, label_colors.get(label_list[run_idx], f"C{run_idx % 10}")
            )
            x_vals, y_vals = _compute_mean_val_metric_over_tasks(run.tasks, first_key)
            if x_vals.size == 0:
                continue
            row_axes[2].plot(
                x_vals,
                y_vals,
                ".-",
                label=label_list[run_idx],
                color=run_color,
                alpha=0.9,
            )
        row_axes[2].set_xlabel("Task")
        row_axes[2].set_ylabel(f"Val {first_label}")
        row_axes[2].grid(True, alpha=0.3)

        # Average forgetting over tasks: one line per run.
        for run_idx, run in enumerate(runs):
            run_color = group_colors_by_algorithm.get(
                run.name, label_colors.get(label_list[run_idx], f"C{run_idx % 10}")
            )
            val_metric_key, _ = _resolve_val_metric_for_run(val_metric_choice, run)
            x_vals, y_vals = compute_average_forgetting(run.tasks, val_metric_key)
            if x_vals.size == 0:
                continue
            row_axes[3].plot(
                x_vals + 1,
                y_vals,
                ".-",
                label=label_list[run_idx],
                color=run_color,
                alpha=0.9,
            )
        row_axes[3].set_xlabel("Task")
        row_axes[3].set_ylabel(f"Avg forgetting ({first_label})")
        row_axes[3].grid(True, alpha=0.3)

        # Column titles on the single combined row.
        axes[0][0].set_title(
            f"{train_metric_label} vs {_resolve_train_x_axis_label(train_metric_choice).lower()}"
        )
        axes[0][1].set_title("Final validation metrics")
        axes[0][2].set_title(f"Mean val {first_label} over tasks")
        axes[0][3].set_title(f"Average forgetting ({first_label})")

        # Legend for IID2 mode: keep only the "Final validation metrics"
        # legend in its panel. All other panels use the shared row legend.
        handles_fv, _ = axes[0][1].get_legend_handles_labels()
        if handles_fv:
            axes[0][1].legend(fontsize=8)

        # Put the run legend back onto the first column subplot.
        handles_train, labels_train = axes[0][0].get_legend_handles_labels()
        if handles_train and labels_train:
            legend_ncol = len(labels_train) if len(labels_train) <= 4 else 2
            axes[0][0].legend(
                ncol=legend_ncol,
                fontsize=8,
                loc="lower center",
                bbox_to_anchor=(0.5, 0.02),
                frameon=True,
                borderaxespad=0.2,
            )
            axes[0][2].legend(
                ncol=legend_ncol,
                fontsize=8,
                loc="lower center",
                bbox_to_anchor=(0.5, 0.02),
                frameon=True,
                borderaxespad=0.2,
            )
            axes[0][3].legend(
                ncol=legend_ncol,
                fontsize=8,
                loc="lower center",
                bbox_to_anchor=(0.5, 0.02),
                frameon=True,
                borderaxespad=0.2,
            )

    else:
        for row_idx, run in enumerate(runs):
            val_metric_key, val_metric_label = _resolve_val_metric_for_run(
                val_metric_choice, run
            )
            row_axes = axes[row_idx]
            _plot_train_metric_vs_steps(
                row_axes[0], run.tasks, run.task_names, train_metric_choice
            )
            _plot_final_validation_metrics(row_axes[1], run.tasks, run.task_names)
            # Avoid duplicating the same "task index" information as x tick
            # labels; keep the panel focused on the metric bars.
            row_axes[1].set_xticklabels([])
            row_axes[1].set_xlabel("")
            _plot_val_acc_over_tasks(
                row_axes[2],
                run.tasks,
                run.task_names,
                val_metric_key=val_metric_key,
                val_metric_label=val_metric_label,
            )
            _plot_average_forgetting(
                row_axes[3],
                run.tasks,
                val_metric_key=val_metric_key,
                val_metric_label=val_metric_label,
            )

            train_label_for_row = row_axes[0].get_ylabel() or "Train recall"
            row_axes[0].set_ylabel(f"{run.name}\n{train_label_for_row}")
            for subplot_ax in row_axes:
                subplot_ax.text(
                    0.99,
                    0.99,
                    run.name,
                    transform=subplot_ax.transAxes,
                    ha="right",
                    va="top",
                    fontsize=12,
                    bbox={
                        "boxstyle": "round,pad=0.15",
                        "facecolor": "white",
                        "alpha": 0.2,
                        "edgecolor": "none",
                    },
                )
            handles_train, labels_train = row_axes[0].get_legend_handles_labels()
            if handles_train and labels_train:
                legend_ncol = min(len(labels_train), 5)
                row_axes[0].legend(
                    ncol=legend_ncol,
                    fontsize=8,
                    loc="lower center",
                    bbox_to_anchor=(0.5, 0.02),
                    frameon=True,
                    borderaxespad=0.2,
                )

        # Column titles on the top row (non-IID2 mode).
        axes[0][0].set_title(
            f"{axes[0][0].get_ylabel().splitlines()[-1]} vs "
            f"{_resolve_train_x_axis_label(train_metric_choice).lower()}"
        )
        axes[0][1].set_title("Final validation metrics")
        axes[0][2].set_title(f"Val {first_label} over tasks")
        axes[0][3].set_title(f"Average forgetting ({first_label})")

        # Keep the "Final validation metrics" panel legend.
        handles_fv, _ = axes[0][1].get_legend_handles_labels()
        if handles_fv:
            axes[0][1].legend(ncol=len(handles_fv), fontsize=8)

    # Optionally enforce the same y-axis limits across rows for each column so
    # that comparisons between algorithms are visually consistent.
    if same_y_limits and not combined_mode:
        for col_idx in range(n_cols):
            y_mins: List[float] = []
            y_maxs: List[float] = []
            for row_idx in range(n_algos):
                y_min, y_max = axes[row_idx][col_idx].get_ylim()
                y_mins.append(y_min)
                y_maxs.append(y_max)
            if not y_mins:
                continue
            col_min = min(y_mins)
            col_max = max(y_maxs)
            for row_idx in range(n_algos):
                axes[row_idx][col_idx].set_ylim(col_min, col_max)

    plt.tight_layout()

    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        out_path = output_dir / "multi_algorithms_metrics.png"
        fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
        print(f"Saved figure to {out_path}")
        if save_subplots:
            subplot_dpi = max(300, dpi)
            subplot_pad_pixels = 8.0
            fixed_subplot_height_pixels = 1000.0
            subplot_left = 0.1
            subplot_right = 0.9
            subplot_bottom = 0.2
            subplot_top = 0.8
            fwt_plot_style: dict[str, Any] | None = None
            fwt_style_key = "default"
            legend_fontsize = 12.0
            subplot_figure_width_inches = 8.0
            subplot_figure_height_inches = 6.0

            # In combined mode with paper-ready exports, match the same plot
            # configuration used by plot_fwt_metrics.py.
            if plot_layout == "together":
                experiment_path_text = " ".join(
                    [str(run.metrics_dir) for run in runs] + [run.name for run in runs]
                ).lower()
                style_key = "til" if "til" in experiment_path_text else "default"
                fwt_style_key = style_key
                fwt_plot_style = (
                    PLOT_STYLES_BY_EXPERIMENT.get(style_key)
                    or PLOT_STYLES_BY_EXPERIMENT["default"]
                )
                subplot_figure_width_inches, subplot_figure_height_inches = map(
                    float,
                    fwt_plot_style.get(
                        "figsize",
                        (subplot_figure_width_inches, subplot_figure_height_inches),
                    ),
                )
                legend_kwargs = fwt_plot_style.get("legend_kwargs", {}) or {}
                legend_fontsize = float(legend_kwargs.get("fontsize", legend_fontsize))

            original_figure_size_inches = fig.get_size_inches().copy()
            all_axes: List[plt.Axes] = [ax for row in axes for ax in row]
            for axis in all_axes:
                axis.xaxis.label.set_size(16)
                axis.yaxis.label.set_size(16)
                axis.tick_params(axis="both", labelsize=13)
                legend = axis.get_legend()
                if legend is not None:
                    for legend_text in legend.get_texts():
                        legend_text.set_fontsize(legend_fontsize)

            original_titles: Dict[plt.Axes, str] = {
                axis: axis.get_title() for axis in all_axes
            }

            for row_idx in range(n_rows):
                for col_idx in range(n_cols):
                    axis = axes[row_idx][col_idx]
                    fig.set_size_inches(
                        subplot_figure_width_inches,
                        subplot_figure_height_inches,
                        forward=True,
                    )
                    original_position = axis.get_position().frozen()
                    axis.set_position(
                        [
                            subplot_left,
                            subplot_bottom,
                            max(subplot_right - subplot_left, 1e-3),
                            max(subplot_top - subplot_bottom, 1e-3),
                        ]
                    )
                    axis.set_anchor("W")
                    if axis.lines:
                        axis.margins(x=0.0)

                    if fwt_plot_style is not None:
                        fwt_ylim = fwt_plot_style.get("ylim")
                        if (
                            fwt_ylim is not None
                            and "Avg forgetting" in axis.get_ylabel()
                        ):
                            axis.set_ylim(*fwt_ylim)

                        fwt_legend_kwargs = (
                            fwt_plot_style.get("legend_kwargs", {}) or {}
                        )
                        handles, labels = axis.get_legend_handles_labels()
                        if handles and labels and fwt_legend_kwargs:
                            panel_key = "train"
                            if col_idx == 1:
                                panel_key = "final_validation"
                            elif "Avg forgetting" in axis.get_ylabel():
                                panel_key = "average_forgetting"
                            elif "Val " in axis.get_ylabel():
                                panel_key = "mean_val"
                            fwt_legend_kwargs = resolve_legend_kwargs(
                                style_key=fwt_style_key,
                                panel_key=panel_key,
                                base_legend_kwargs=fwt_legend_kwargs,
                                run_count=len(runs),
                            )
                            axis.legend(handles, labels, **fwt_legend_kwargs)

                    subplot_title = axis.get_title().strip()
                    if subplot_title:
                        title_stem = re.sub(r"[^A-Za-z0-9]+", "_", subplot_title).strip(
                            "_"
                        )
                        if not title_stem:
                            title_stem = "subplot"
                    else:
                        title_stem = "subplot"
                    axis.set_title("")

                    for other_axis in all_axes:
                        if other_axis is not axis:
                            other_axis.set_visible(False)
                    fig.canvas.draw()

                    renderer = fig.canvas.get_renderer()
                    axis_bbox_pixels = _build_axis_only_export_bbox(
                        axis=axis,
                        renderer=renderer,
                        pad_pixels=subplot_pad_pixels,
                    )
                    bbox = axis_bbox_pixels.transformed(fig.dpi_scale_trans.inverted())
                    bbox_height_inches = max(float(bbox.height), 1e-6)
                    height_fixed_dpi = fixed_subplot_height_pixels / bbox_height_inches
                    export_dpi = min(subplot_dpi, height_fixed_dpi)
                    export_dpi = max(72.0, export_dpi)
                    subplot_path = (
                        output_dir
                        / f"{title_stem.lower()}_r{row_idx + 1}_c{col_idx + 1}.png"
                    )
                    fig.savefig(
                        subplot_path,
                        dpi=export_dpi,
                        bbox_inches=bbox,
                    )
                    print(f"Saved subplot to {subplot_path}")
                    axis.set_position(original_position)
                    axis.set_anchor("C")

                    for other_axis in all_axes:
                        if other_axis is not axis:
                            other_axis.set_visible(True)

            for axis in all_axes:
                axis.set_title(original_titles[axis])
            fig.set_size_inches(
                float(original_figure_size_inches[0]),
                float(original_figure_size_inches[1]),
                forward=True,
            )
            fig.canvas.draw()
    else:
        backend_name = plt.get_backend().lower()
        if "agg" in backend_name:
            print(
                "Skipping interactive display because Matplotlib backend is "
                f"non-interactive ({plt.get_backend()})."
            )
        else:
            plt.show()

    plt.close(fig)


def _load_n_epochs(metrics_dir: Path) -> int | None:
    """Optionally load ``n_epochs`` from training_parameters.json if needed.

    This is provided for potential extensions that require per-epoch
    aggregation. It is currently unused in the default plots.

    Args:
        metrics_dir: Metrics directory for a single run.

    Returns:
        ``n_epochs`` if present, otherwise ``None``.
    """
    params_file = metrics_dir.parent / "training_parameters.json"
    if not params_file.is_file():
        return None
    try:
        with open(params_file) as f:
            params = json.load(f)
        value = params.get("n_epochs")
        return int(value) if value is not None else None
    except (json.JSONDecodeError, TypeError, ValueError):
        return None


def _parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the script."""
    parser = argparse.ArgumentParser(
        parents=[_pre_parser],
        description=(
            "Plot metrics from multiple algorithms in a single figure. "
            "Each row is an algorithm; each column is a plot type."
        ),
    )
    parser.add_argument(
        "--algo",
        type=str,
        action="append",
        dest="algos",
        required=False,
        help="Algorithm name to include (can be specified multiple times).",
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
        "--runs-dir",
        type=Path,
        default=None,
        help=(
            "Directory containing either run folders with job_logs/ or "
            "algorithm folders with per-algorithm runs (for example "
            "logs/00_sync/full-til_10epochs_w-zs or "
            "logs/00_sync/full-til_10epochs_w-zs/saved_models)."
        ),
    )
    parser.add_argument(
        "--metrics-dir",
        type=Path,
        action="append",
        dest="metrics_dirs",
        default=None,
        help=(
            "Metrics directory for an algorithm (can be specified multiple times). "
            "If omitted, the latest metrics dir under logs/{algo}/ is used."
        ),
    )
    parser.add_argument(
        "--logs-root",
        type=Path,
        default=Path("logs"),
        help="Root directory containing logs/{algo}/ subdirectories (default: logs).",
    )
    parser.add_argument(
        "--run-index",
        type=int,
        default=0,
        metavar="N",
        help=(
            "When auto-discovering runs (no --metrics-dir), use the Nth run by recency: "
            "0 = latest (default), 1 = second latest, etc."
        ),
    )
    parser.add_argument(
        "--same-y-limits",
        action="store_true",
        help=(
            "Use the same y-axis limits across all rows for each column to "
            "make cross-algorithm comparisons easier."
        ),
    )
    parser.add_argument(
        "--train-metric",
        type=str,
        choices=("macro_f1", "macro_rec"),
        default="macro_rec",
        help=(
            "Training metric to use for the 'vs step' panel. "
            "Choose total_f1 for total F1 or cls_recall for recall "
            "(default: cls_recall)."
        ),
    )
    parser.add_argument(
        "--val-metric",
        type=str,
        choices=("macro_f1", "macro_rec"),
        default="macro_f1",
        help=(
            "Validation metric to use for 'val over tasks' and average forgetting "
            "plots. Defaults to total_f1; falls back to cls_recall when F1 is "
            "not available for a run."
        ),
    )
    parser.add_argument(
        "--labels",
        type=str,
        default=None,
        help=(
            "Optional comma-separated labels for IID2 runs when they are "
            "combined into a single row (e.g. '0,1,2,3')."
        ),
    )
    parser.add_argument(
        "--labels-grouping",
        type=str,
        default=None,
        help=(
            "Optional comma-separated grouping keywords for labels. Labels "
            "containing the same keyword are colored with shades from the same "
            "color family (e.g. 'baseline,cross,pwr')."
        ),
    )
    parser.add_argument(
        "--plot-layout",
        type=str,
        choices=("separate", "together"),
        default="separate",
        help=(
            "Plot algorithms in separate rows (default) or together in a single "
            "combined row."
        ),
    )
    parser.add_argument(
        "--include-iid2",
        action="store_true",
        help=(
            "Include iid2 in plots. By default iid2 is ignored to reduce "
            "clutter in multi-algorithm comparisons."
        ),
    )
    parser.add_argument(
        "--save-subplots",
        action="store_true",
        help=(
            "When saving to --output-dir, also export each subplot panel as "
            "a separate PNG using paper-ready export settings (>=600 DPI)."
        ),
    )
    return parser.parse_args()


def main() -> None:
    """Entry point for multi-algorithm metrics plotting."""
    args = _parse_args()

    metrics_dirs: List[Path] | None
    algorithm_names: List[str]
    run_source_dir: Path | None = (
        args.runs_dir if args.runs_dir is not None else args.run_dir
    )
    if run_source_dir is not None:
        if args.algos is not None:
            raise SystemExit(
                "Use either --runs-dir/--run-dir or --algo/--metrics-dir, not both."
            )
        if args.metrics_dirs is not None:
            raise SystemExit(
                "Use either --runs-dir/--run-dir or --metrics-dir, not both."
            )
        if (run_source_dir / "job_logs").is_dir():
            algorithm_names, metrics_dirs = _discover_runs_from_run_dir(run_source_dir)
        elif run_source_dir.name == "saved_models":
            algorithm_names, metrics_dirs = _discover_runs_from_saved_models_root(
                run_source_dir
            )
        elif (run_source_dir / "saved_models").is_dir():
            algorithm_names, metrics_dirs = _discover_runs_from_saved_models_root(
                run_source_dir / "saved_models"
            )
        else:
            algorithm_names, metrics_dirs = _discover_runs_from_algorithm_root(
                run_source_dir
            )
    else:
        if args.algos is None:
            raise SystemExit(
                "Provide --algo (or use --run-dir/--runs-dir for auto-discovery)."
            )
        algorithm_names = list(args.algos)
        if args.metrics_dirs is None:
            metrics_dirs = None
        else:
            metrics_dirs = list(args.metrics_dirs)

    runs = _prepare_algo_runs(
        algos=algorithm_names,
        metrics_dirs=metrics_dirs,
        logs_root=args.logs_root,
        run_index=args.run_index,
    )
    if not args.include_iid2:
        runs = [run for run in runs if run.name.strip().lower() != "iid2"]
        if not runs:
            raise SystemExit(
                "No runs left after filtering iid2. Pass --include-iid2 to include it."
            )
    runs = sorted(runs, key=lambda run: group_sort_key(run.name))
    print("Algorithms and metrics directories:")
    for run in runs:
        print(f"  {run.name}: {run.metrics_dir}")

    if args.labels is not None:
        label_list = [part.strip() for part in args.labels.split(",") if part.strip()]
    else:
        label_list = None
    if args.labels_grouping is not None:
        labels_grouping_list = [
            part.strip() for part in args.labels_grouping.split(",") if part.strip()
        ]
    else:
        labels_grouping_list = None

    plot_multi_algorithms(
        runs=runs,
        output_dir=args.output_dir,
        same_y_limits=args.same_y_limits,
        val_metric_choice=args.val_metric,
        train_metric_choice=args.train_metric,
        labels=label_list,
        labels_grouping=labels_grouping_list,
        plot_layout=args.plot_layout,
        save_subplots=args.save_subplots,
    )


if __name__ == "__main__":
    main()
