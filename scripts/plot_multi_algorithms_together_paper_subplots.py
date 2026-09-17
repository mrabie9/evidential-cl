"""Export paper-ready subplot PNGs for combined ("together") multi-algorithm plots.

This script is a purpose-built alternative to `scripts/plot_multi_algorithms.py
--save-subplots` for the `--plot-layout together` case. It applies the same
legend/layout configuration used in `scripts/plot_fwt_metrics.py` so exported
subplots are consistent with the paper-ready forward-transfer figures.

Typical usage:
    python scripts/plot_multi_algorithms_together_paper_subplots.py \
        --runs-dir logs/00_sync/one-shot_CIL \
        -o logs/00_sync/one-shot_CIL/plots

    # Restrict to a subset of algorithms (still averaged across seeds)
    python scripts/plot_multi_algorithms_together_paper_subplots.py \
        --runs-dir logs/00_sync/one-shot_CIL \
        --algo lwf,eralg4,cmaml \
        -o plots/
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Any, Dict, Sequence

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils.metric_keys import first_present_key  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]

PlotStyle = Dict[str, Any]
LINESTYLES = ["-", "--", ":", "-.", (0, (5, 1))]
LINEWIDTHS = [2.4, 1.8, 1.8, 1.5, 1.5]

# Manual plot controls (set values to ``None`` to keep auto behavior).
# Keys: train, final_validation, mean_val, average_forgetting, val_per_task
PANEL_YLIM_OVERRIDES: Dict[str, tuple[float, float] | None] = {
    "train": None,
    "final_validation": None,
    "mean_val": (0, 0.9),
    "average_forgetting": (-0.7, 0.3),
    "val_per_task": None,
}

# Manual legend ncol controls (set values to ``None`` to keep auto behavior).
# Keys: train, final_validation, mean_val, average_forgetting, val_per_task
PANEL_LEGEND_NCOL_OVERRIDES: Dict[str, int | None] = {
    "train": None,
    "final_validation": None,
    "mean_val": 6,
    "average_forgetting": 6,
}

# Global x-axis label spacing (distance from axis to xlabel text).
X_LABEL_PAD: float = 4.0
# Global tick-label font size (applies to both x and y axes).
TICK_LABEL_FONT_SIZE: float = 12.0
# Optional legend placement: move legends above line plots for consistent axes.
LEGEND_ABOVE_PLOT: bool = True
# Vertical offset used when LEGEND_ABOVE_PLOT=True.
LEGEND_ABOVE_BBOX_Y: float = 1.02
# Fixed axes box (left, right, bottom, top) so graph areas stay consistent
# across exported panels while still trimming outer whitespace on save.
AXES_BOX_LEFT: float = 0.11
AXES_BOX_RIGHT: float = 0.99
AXES_BOX_BOTTOM: float = 0.16
AXES_BOX_TOP: float = 0.78


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument(
        "--run-dir",
        type=Path,
        default=None,
        help=(
            "Run directory containing job_logs/ (example: a full_experiments run folder)."
        ),
    )
    parser.add_argument(
        "--runs-dir",
        type=Path,
        default=None,
        help=(
            "Directory containing algorithm subfolders with metrics/ (example: logs/00_sync/full-til_10epochs_w-zs)."
        ),
    )
    parser.add_argument(
        "--logs-root",
        type=Path,
        default=Path("logs"),
        help="Root logs directory used by plot_multi_algorithms discovery helpers.",
    )
    parser.add_argument(
        "--run-index",
        type=int,
        default=0,
        help="When auto-discovering per-algorithm metrics, choose this run by recency.",
    )
    parser.add_argument(
        "--train-metric",
        type=str,
        choices=("macro_f1", "macro_rec"),
        default="macro_rec",
        help="Training metric used in the first panel (train vs step).",
    )
    parser.add_argument(
        "--val-metric",
        type=str,
        choices=("macro_f1", "macro_rec"),
        default="macro_f1",
        help="Validation metric used in mean-val and average-forgetting panels.",
    )
    parser.add_argument(
        "--include-iid2",
        action="store_true",
        help="Include iid2 runs (by default they are excluded for clarity).",
    )
    parser.add_argument(
        "--algo",
        action="append",
        default=None,
        help=(
            "Restrict plotting to these algorithms. Repeatable and/or "
            "comma-separated (example: --algo lwf,eralg4 --algo cmaml). "
            "Defaults to every algorithm discovered under the run source."
        ),
    )
    parser.add_argument(
        "--labels",
        type=str,
        default=None,
        help="Optional comma-separated labels for runs (legend entries).",
    )
    parser.add_argument(
        "--labels-grouping",
        type=str,
        default=None,
        help="Optional comma-separated keywords used to group labels by color family.",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        required=True,
        help="Output directory to save the combined PNG and individual subplot PNGs.",
    )
    parser.add_argument(
        "--fwt-json-path",
        type=Path,
        default=None,
        help=(
            "Optional path to fwt_metrics.json. When provided (or auto-discovered), "
            "an additional FWT figure is generated using scripts/plot_fwt_metrics.py."
        ),
    )
    parser.add_argument(
        "--shade-std",
        action="store_true",
        default=False,
        help="Shade ±1 std across seeds in multi-seed mode (off by default).",
    )
    return parser.parse_args()


def _resolve_val_metric_for_run(choice: str, run: Any) -> tuple[str, str]:
    """Resolve validation metric key and label for a single run.

    Mirrors the behavior in `plot_multi_algorithms.py`.
    """
    recall_key = next(
        (k for task in run.tasks if (k := first_present_key(task, ["val_macro_rec"]))),
        "val_macro_rec",
    )
    if choice == "macro_rec":
        return recall_key, "Macro recall"

    f1_key = next(
        (k for task in run.tasks if (k := first_present_key(task, ["val_macro_f1"]))),
        None,
    )
    if f1_key is not None:
        return f1_key, "Macro F1"

    print(
        f"[WARN] Requested val-metric=macro_f1 but run '{run.name}' at {run.metrics_dir} "
        "has no macro-F1 key; falling back to macro recall."
    )
    return recall_key, "Macro recall"


def _case_insensitive_detect_style(runs: Sequence[Any]) -> PlotStyle:
    """Detect til vs default style based on path/name substrings (case-insensitive)."""
    from scripts.plot_fwt_metrics import PLOT_STYLES_BY_EXPERIMENT

    style_key = _case_insensitive_detect_style_key(runs)
    return PLOT_STYLES_BY_EXPERIMENT[style_key]


def _case_insensitive_detect_style_key(runs: Sequence[Any]) -> str:
    """Detect style key based on path/name substrings (case-insensitive)."""
    experiment_path_text = " ".join(
        [str(run.metrics_dir) for run in runs] + [run.name for run in runs]
    ).lower()
    return "til" if "til" in experiment_path_text else "default"


def _build_export_legend_kwargs(
    base_legend_kwargs: Dict[str, Any], panel_key: str
) -> Dict[str, Any]:
    """Build legend kwargs for subplot export."""
    legend_kwargs = dict(base_legend_kwargs)
    manual_ncol = PANEL_LEGEND_NCOL_OVERRIDES.get(panel_key)
    if manual_ncol is not None:
        legend_kwargs["ncol"] = int(manual_ncol)
    if LEGEND_ABOVE_PLOT and panel_key in {"train", "mean_val", "average_forgetting"}:
        legend_kwargs["loc"] = "lower center"
        legend_kwargs["bbox_to_anchor"] = (0.5, LEGEND_ABOVE_BBOX_Y)
        legend_kwargs.setdefault("borderaxespad", 0.0)
    return legend_kwargs


def _format_task_dataset_label(task_name: str, fallback_task_number: int) -> str:
    """Format task dataset labels to mirror plot_fwt_metrics.py style."""
    task_name_parts = task_name.split("-", 1)
    if len(task_name_parts) == 2 and task_name_parts[1]:
        original_task_number = task_name_parts[0].lstrip("t")
        dataset_name = task_name_parts[1]
    else:
        original_task_number = str(fallback_task_number)
        dataset_name = task_name
    dataset_name_lower = dataset_name.lower()
    if "uclresm" in dataset_name_lower:
        return f"(RML{original_task_number})"
    if "deeprad" in dataset_name_lower:
        return f"(DR{original_task_number})"
    if "rcn" in dataset_name_lower:
        return f"(RCN{original_task_number})"
    return dataset_name


def _build_task_index_to_dataset_name(runs: Sequence[Any]) -> Dict[int, str]:
    """Build task-index label map for x-axis display."""
    task_index_to_dataset_name: Dict[int, str] = {}
    for run in runs:
        run_task_names = getattr(run, "task_names", None)
        for task_index, task in enumerate(run.tasks):
            if task_index in task_index_to_dataset_name:
                continue
            task_name = (
                str(run_task_names[task_index])
                if run_task_names is not None and task_index < len(run_task_names)
                else str(task.get("task_name", f"t{task_index}"))
            )
            task_index_to_dataset_name[task_index] = _format_task_dataset_label(
                task_name, task_index
            )
    return task_index_to_dataset_name


def _set_task_axis_like_fwt(
    axis: plt.Axes,
    task_positions: Sequence[int],
    task_index_to_dataset_name: Dict[int, str],
) -> None:
    """Set x ticks every 1 and labels like plot_fwt_metrics.py."""
    if not task_positions:
        return
    min_position = int(min(task_positions))
    max_position = int(max(task_positions))
    tick_positions = list(range(min_position, max_position + 1))
    axis.set_xticks(tick_positions)
    axis.set_xticklabels(
        [
            f"{task_position}\n"
            f"{task_index_to_dataset_name.get(task_position, 'unknown')}"
            for task_position in tick_positions
        ]
    )


def _hide_top_right_spines(axis: plt.Axes) -> None:
    """Hide top/right spines for consistent plot framing."""
    axis.spines[["top", "right"]].set_visible(False)


def _save_independent_figure(fig: plt.Figure, output_path: Path, dpi: int) -> None:
    """Save one independent panel figure as both PDF and PNG."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.subplots_adjust(
        left=AXES_BOX_LEFT,
        right=AXES_BOX_RIGHT,
        bottom=AXES_BOX_BOTTOM,
        top=AXES_BOX_TOP,
    )
    base_output_path = output_path.with_suffix("")
    pdf_output_path = base_output_path.with_suffix(".pdf")
    png_output_path = base_output_path.with_suffix(".png")
    fig.savefig(pdf_output_path, dpi=dpi, bbox_inches="tight", pad_inches=0.03)
    fig.savefig(png_output_path, dpi=dpi, bbox_inches="tight", pad_inches=0.03)
    print(f"Saved subplot to {pdf_output_path} and {png_output_path}")
    plt.close(fig)


def _discover_all_metrics_dirs_per_algo(
    saved_models_root: Path,
) -> Dict[str, list[Path]]:
    """Return all metrics directories grouped by algorithm name.

    Expected layout: <saved_models_root>/<algo>/.../<seed>/metrics/task*.npz.
    All qualifying metrics dirs are returned (not just the latest), enabling
    per-algorithm seed aggregation.
    """
    result: Dict[str, list[Path]] = {}
    if not saved_models_root.is_dir():
        return result
    for algo_dir in sorted(saved_models_root.iterdir()):
        if not algo_dir.is_dir():
            continue
        dirs: list[Path] = []
        for metrics_dir in sorted(algo_dir.rglob("metrics")):
            if metrics_dir.is_dir() and any(metrics_dir.glob("task*.npz")):
                dirs.append(metrics_dir.resolve())
        if dirs:
            result[algo_dir.name] = dirs
    return result


def _compute_val_metric_for_single_task(
    tasks: Sequence[Any],
    val_metric_key: str,
    task_index: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Track one task's validation metric as later tasks are trained.

    Args:
        tasks: Per-checkpoint metric dictionaries for a single run, ordered by
            the task that had just finished training.
        val_metric_key: Validation metric key holding a per-task vector.
        task_index: Zero-based index of the task to follow.

    Returns:
        Tuple ``(checkpoint_indices, values)`` covering checkpoints from
        ``task_index`` onwards. Checkpoints missing the metric yield ``NaN``.

    Usage:
        >>> checkpoints, values = _compute_val_metric_for_single_task(
        ...     tasks, "val_f1", task_index=0
        ... )
    """
    checkpoint_indices: list[int] = []
    values: list[float] = []
    for checkpoint_index in range(task_index, len(tasks)):
        metric_vector = tasks[checkpoint_index].get(val_metric_key)
        checkpoint_indices.append(checkpoint_index)
        if metric_vector is None or task_index >= len(metric_vector):
            values.append(float("nan"))
        else:
            values.append(float(metric_vector[task_index]))
    return (
        np.asarray(checkpoint_indices, dtype=float),
        np.asarray(values, dtype=float),
    )


def _metric_label_to_filename_token(metric_label: str) -> str:
    """Convert a metric label into a hyphenated filename token.

    Usage:
        >>> _metric_label_to_filename_token("Total F1")
        'Total-F1'
    """
    return re.sub(r"[^A-Za-z0-9]+", "-", metric_label).strip("-")


def _parse_requested_algorithm_names(
    algorithm_arguments: Sequence[str] | None,
) -> list[str] | None:
    """Normalise repeatable/comma-separated ``--algo`` values.

    Args:
        algorithm_arguments: Raw ``--algo`` values, or ``None`` when the flag
            was not supplied.

    Returns:
        Lower-cased algorithm names in the order first requested, or ``None``
        when no filtering was requested.

    Usage:
        >>> _parse_requested_algorithm_names(["lwf,eralg4", "cmaml"])
        ['lwf', 'eralg4', 'cmaml']
        >>> _parse_requested_algorithm_names(None) is None
        True
    """
    if algorithm_arguments is None:
        return None
    requested_names: list[str] = []
    for argument_value in algorithm_arguments:
        for name in argument_value.split(","):
            normalised_name = name.strip().lower()
            if normalised_name and normalised_name not in requested_names:
                requested_names.append(normalised_name)
    return requested_names or None


def _filter_runs_by_algorithm(
    runs: Sequence[Any],
    requested_algorithm_names: Sequence[str] | None,
) -> list[Any]:
    """Keep only the runs whose algorithm name was requested.

    Args:
        runs: Discovered runs, each exposing a ``name`` attribute.
        requested_algorithm_names: Lower-cased names to keep, or ``None`` to
            keep every run.

    Returns:
        The retained runs, in their original order.

    Raises:
        SystemExit: If a requested name matches no discovered run.

    Usage:
        >>> from types import SimpleNamespace
        >>> discovered = [SimpleNamespace(name="lwf"), SimpleNamespace(name="gem")]
        >>> [run.name for run in _filter_runs_by_algorithm(discovered, ["lwf"])]
        ['lwf']
    """
    if requested_algorithm_names is None:
        return list(runs)
    discovered_names = {run.name.strip().lower() for run in runs}
    missing_names = [
        name for name in requested_algorithm_names if name not in discovered_names
    ]
    if missing_names:
        raise SystemExit(
            f"--algo requested {', '.join(missing_names)} but only "
            f"{', '.join(sorted(discovered_names))} were discovered."
        )
    requested_name_set = set(requested_algorithm_names)
    return [run for run in runs if run.name.strip().lower() in requested_name_set]


def _aggregate_across_seeds(
    seed_tasks_list: list[Any],
    compute_fn: Any,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute mean and std of a per-task series across multiple seeds.

    Returns (x, mean_y, std_y). Seeds with empty results are skipped.
    x is taken from the first valid seed; all seeds are clipped to the
    minimum length so that shapes align.
    """
    all_x: list[np.ndarray] = []
    all_y: list[np.ndarray] = []
    for tasks in seed_tasks_list:
        x, y = compute_fn(tasks)
        if x is None or (hasattr(x, "size") and x.size == 0):
            continue
        all_x.append(np.asarray(x, dtype=float))
        all_y.append(np.asarray(y, dtype=float))
    if not all_y:
        return np.array([]), np.array([]), np.array([])
    min_len = min(len(y) for y in all_y)
    y_matrix = np.array([y[:min_len] for y in all_y], dtype=float)
    return all_x[0][:min_len], np.nanmean(y_matrix, axis=0), np.nanstd(y_matrix, axis=0)


def _build_experiment_prefix(run_source_dir: Path, runs: Sequence[Any]) -> str:
    """Build '<Multi/Single>-Epoch_<TIL/CIL>' prefix from run context."""
    experiment_path_parts = [str(run_source_dir)] + [
        str(run.metrics_dir) for run in runs
    ]
    experiment_path_text = " ".join(experiment_path_parts).lower()
    epoch_mode = "Single" if "one-shot" in experiment_path_text else "Multi"

    def has_mode_token(text: str, mode: str) -> bool:
        return bool(re.search(rf"(^|[-_/]){mode}($|[-_/])", text.lower()))

    # Prefer explicit mode from the user-provided run root.
    run_root_text = str(run_source_dir).lower()
    if has_mode_token(run_root_text, "til"):
        learning_setup = "TIL"
    elif has_mode_token(run_root_text, "cil"):
        learning_setup = "CIL"
    else:
        # Fallback: infer from discovered metrics directories.
        metrics_paths_text = " ".join(str(run.metrics_dir).lower() for run in runs)
        if has_mode_token(metrics_paths_text, "til"):
            learning_setup = "TIL"
        elif has_mode_token(metrics_paths_text, "cil"):
            learning_setup = "CIL"
        else:
            learning_setup = "TIL"
    return f"{epoch_mode}-Epoch_{learning_setup}"


FWT_JSON_FILENAMES: tuple[str, ...] = ("fwt_metrics.json", "fwt_metrics_A.json")
FWT_SEARCH_PARENT_LEVELS: int = 3


def _mirrored_full_experiments_dir(search_dir: Path) -> Path | None:
    """Map a ``logs/00_sync/...`` directory onto its ``full_experiments`` twin.

    Args:
        search_dir: Directory that may live under a ``00_sync`` path segment.

    Returns:
        The equivalent directory with ``00_sync`` replaced by
        ``full_experiments``, or ``None`` when the segment is absent.
    """
    search_parts = list(search_dir.parts)
    if "00_sync" not in search_parts:
        return None
    search_parts[search_parts.index("00_sync")] = "full_experiments"
    return Path(*search_parts)


def _candidate_fwt_json_paths(run_source_dir: Path) -> list[Path]:
    """Build candidate fwt_metrics.json paths from the run source directory.

    The JSON is usually written at the experiment root (for example
    ``logs/00_sync/one-shot_CIL/``) while ``--runs`` commonly points at the
    nested ``saved_models`` directory, so parent directories are searched too.

    Args:
        run_source_dir: Directory the runs were discovered from.

    Returns:
        Candidate paths in priority order, nearest directory first.

    Usage:
        >>> paths = _candidate_fwt_json_paths(Path("logs/00_sync/cil/saved_models"))
        >>> paths[0].name
        'fwt_metrics.json'
    """
    search_dirs: list[Path] = [run_source_dir]
    for parent_dir in list(run_source_dir.parents)[:FWT_SEARCH_PARENT_LEVELS]:
        search_dirs.append(parent_dir)

    candidates: list[Path] = []
    for search_dir in search_dirs:
        mirrored_dir = _mirrored_full_experiments_dir(search_dir)
        for directory in (search_dir, mirrored_dir):
            if directory is None:
                continue
            candidates.extend(directory / filename for filename in FWT_JSON_FILENAMES)

    return candidates


def _resolve_fwt_json_path(
    explicit_fwt_json_path: Path | None, run_source_dir: Path
) -> Path | None:
    """Resolve FWT metrics JSON path from explicit input or known defaults."""
    if explicit_fwt_json_path is not None:
        return explicit_fwt_json_path

    for candidate_path in _candidate_fwt_json_paths(run_source_dir):
        if candidate_path.is_file():
            return candidate_path
    return None


def _line_style_for_index(index: int) -> Dict[str, Any]:
    """Return line width/style settings aligned with plot_fwt_metrics.py."""
    return {
        "linestyle": LINESTYLES[index % len(LINESTYLES)],
        "linewidth": LINEWIDTHS[index % len(LINEWIDTHS)],
        "alpha": 0.9,
    }


def _build_algorithm_to_fwt_panel_line_member_index(
    algorithm_names: Sequence[str],
) -> Dict[str, int]:
    """Map each algorithm to its linestyle slot within its FWT color group.

    ``plot_fwt_metrics.plot_series`` assigns ``LINESTYLES[member_index]`` per
    algorithm using this grouping. The BWT/CL-F1/train panels previously used
    the global run index instead, so e.g. A-GEM could be solid on FWT but
    dash-dot on BWT. Using the same member index keeps legend line styles
    aligned across figures.

    Args:
        algorithm_names: Algorithm ids (e.g. run names) included in the export.

    Returns:
        Mapping from each name in ``algorithm_names`` to a non-negative member
        index within its ordered group.

    Usage:
        >>> idx = _build_algorithm_to_fwt_panel_line_member_index(["agem", "gem"])
        >>> idx["agem"]
        0
    """
    from scripts.plot_algorithm_group_styles import group_sort_key
    from scripts.plot_fwt_metrics import _build_ordered_groups

    sorted_names = sorted(algorithm_names, key=group_sort_key)
    ordered_groups = _build_ordered_groups(sorted_names)
    algorithm_to_member_index: Dict[str, int] = {}
    for _, member_names in ordered_groups:
        for member_index, name in enumerate(member_names):
            algorithm_to_member_index[name] = member_index
    return algorithm_to_member_index


def _build_group_color_lookup_for_runs(
    run_names: Sequence[str],
    group_colors: Sequence[str],
) -> Dict[str, str]:
    """Build run->color mapping using FWT group ordering/color rules."""
    from scripts.plot_fwt_metrics import _build_ordered_groups
    from scripts.plot_algorithm_group_styles import group_sort_key

    sorted_run_names = sorted(run_names, key=group_sort_key)
    ordered_groups = _build_ordered_groups(sorted_run_names)
    run_to_color: Dict[str, str] = {}
    for group_index, (_, member_names) in enumerate(ordered_groups):
        group_color = group_colors[group_index % len(group_colors)]
        for member_name in member_names:
            run_to_color[member_name] = group_color
    return run_to_color


def main() -> None:
    """Run the plotting pipeline and export paper-ready subplot PNGs."""
    # Ensure `scripts/` can be imported as a namespace package when this file is
    # executed via `python scripts/<this_script>.py`.
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))

    from scripts.plot_multi_algorithms import (  # pylint: disable=import-error
        _compute_mean_val_metric_over_tasks,
        _concat_train_metric_for_run,
        _discover_runs_from_algorithm_root,
        _discover_runs_from_run_dir,
        _mean_final_metric_for_run,
        _prepare_algo_runs,
        _resolve_train_x_axis_label,
        compute_average_forgetting,
        get_task_color,
        load_metrics as _load_task_metrics,
    )
    from scripts.plot_fwt_metrics import (
        ALGORITHM_DISPLAY_NAMES,
        GROUP_COLORS as fwt_group_colors,
        build_series_by_algo,
        load_metrics,
        plot_series,
    )
    from scripts.plot_algorithm_group_styles import group_sort_key
    from scripts.plot_style_overrides import resolve_legend_kwargs

    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.labelsize": 14,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
            "legend.fontsize": 10,
            "axes.linewidth": 1,
            "figure.dpi": 600,
            "savefig.dpi": 600,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.02,
        }
    )

    args = parse_args()
    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.run_dir is not None and args.runs_dir is not None:
        raise SystemExit("Use either --run-dir or --runs-dir, not both.")

    run_source_dir: Path | None = (
        args.runs_dir if args.runs_dir is not None else args.run_dir
    )
    if run_source_dir is None:
        raise SystemExit("Provide --run-dir or --runs-dir.")

    if (run_source_dir / "job_logs").is_dir():
        algorithm_names, metrics_dirs = _discover_runs_from_run_dir(run_source_dir)
    else:
        algorithm_names, metrics_dirs = _discover_runs_from_algorithm_root(
            run_source_dir
        )

    runs = _prepare_algo_runs(
        algos=algorithm_names,
        metrics_dirs=metrics_dirs,
        logs_root=args.logs_root,
        run_index=args.run_index,
    )

    requested_algorithm_names = _parse_requested_algorithm_names(args.algo)
    excluded_run_name_set = {"saved_models", "models", "plots", "figures"}
    iid2_requested_explicitly = (
        requested_algorithm_names is not None and "iid2" in requested_algorithm_names
    )
    if not args.include_iid2 and not iid2_requested_explicitly:
        excluded_run_name_set.add("iid2")
    runs = [
        run for run in runs if run.name.strip().lower() not in excluded_run_name_set
    ]
    if not runs:
        raise SystemExit(
            "No runs left after filtering wrapper directories/iid2. "
            "Pass --include-iid2 to include iid2."
        )
    runs = _filter_runs_by_algorithm(runs, requested_algorithm_names)
    runs = sorted(runs, key=lambda run: group_sort_key(run.name))

    print("Algorithms and metrics directories:")
    for run in runs:
        print(f"  {run.name}: {run.metrics_dir}")

    # Multi-seed: auto-detect if saved_models layout has >1 metrics dir per algo.
    _saved_models_candidate = (
        run_source_dir
        if run_source_dir.name == "saved_models"
        else run_source_dir / "saved_models"
    )
    _all_metrics_dirs = _discover_all_metrics_dirs_per_algo(_saved_models_candidate)
    _active_algo_names = {run.name for run in runs}
    _all_metrics_dirs = {
        algo: dirs
        for algo, dirs in _all_metrics_dirs.items()
        if algo in _active_algo_names
    }
    is_multi_seed = any(len(dirs) > 1 for dirs in _all_metrics_dirs.values())
    algo_seed_tasks: Dict[str, list[list[Any]]] = {}
    if is_multi_seed:
        for algo_name, dirs in _all_metrics_dirs.items():
            algo_seed_tasks[algo_name] = [_load_task_metrics(d) for d in dirs]
        print(
            f"Multi-seed mode: found multiple seeds for "
            f"{sum(1 for v in _all_metrics_dirs.values() if len(v) > 1)} algorithm(s)"
        )
        for algo, seed_tasks in algo_seed_tasks.items():
            if len(seed_tasks) > 1:
                print(f"  {algo}: {len(seed_tasks)} seeds")

    label_list: list[str] | None = None
    if args.labels is not None:
        label_list = [part.strip() for part in args.labels.split(",") if part.strip()]

    labels_grouping: list[str] | None = None
    if args.labels_grouping is not None:
        labels_grouping = [
            part.strip().lower()
            for part in args.labels_grouping.split(",")
            if part.strip()
        ]

    # Styling config derived from plot_fwt_metrics.py.
    plot_style = _case_insensitive_detect_style(runs)
    legend_kwargs: Dict[str, Any] = plot_style.get("legend_kwargs", {}) or {}
    dpi = 550
    figure_size = tuple(map(float, plot_style.get("figsize", (7.0, 3.5))))
    first_key, first_label = _resolve_val_metric_for_run(args.val_metric, runs[0])

    # Legend labels/colors for algorithm lines.
    if label_list is not None:
        if len(label_list) < len(runs):
            raise SystemExit(
                "Too few --labels entries for the number of runs discovered."
            )
        run_labels = label_list
    else:
        run_labels = [ALGORITHM_DISPLAY_NAMES.get(run.name, run.name) for run in runs]

    if labels_grouping:
        print(
            "[WARN] --labels-grouping is ignored in this script to preserve "
            "per-algorithm colors from plot_fwt_metrics.py."
        )

    algorithm_colors = _build_group_color_lookup_for_runs(
        [run.name for run in runs],
        fwt_group_colors,
    )
    algorithm_line_member_index = _build_algorithm_to_fwt_panel_line_member_index(
        [run.name for run in runs]
    )
    task_index_to_dataset_name = _build_task_index_to_dataset_name(runs)
    style_key = _case_insensitive_detect_style_key(runs)
    experiment_prefix = _build_experiment_prefix(run_source_dir, runs)

    # Figure 1: train recall vs step.
    fig_train, axis_train = plt.subplots(figsize=figure_size, dpi=dpi)
    train_metric_label = "Train recall"
    for run_idx, run in enumerate(runs):
        run_label = run_labels[run_idx]
        if not run.tasks:
            continue
        _, train_metric_label = _concat_train_metric_for_run(
            run.tasks, args.train_metric
        )
        run_color = algorithm_colors.get(run.name, f"C{run_idx % 10}")
        line_style = _line_style_for_index(
            algorithm_line_member_index.get(run.name, run_idx)
        )
        seed_tasks_list = algo_seed_tasks.get(run.name, [run.tasks])
        if len(seed_tasks_list) > 1:
            _, mean_train, std_train = _aggregate_across_seeds(
                seed_tasks_list,
                lambda tasks: (
                    (
                        (
                            np.arange(1, len(s) + 1)
                            if args.train_metric == "macro_f1"
                            else np.arange(len(s))
                        ),
                        s,
                    )
                    if (s := _concat_train_metric_for_run(tasks, args.train_metric)[0])
                    is not None
                    else (np.array([]), np.array([]))
                ),
            )
            if mean_train.size == 0:
                continue
            x_values = (
                np.arange(1, len(mean_train) + 1)
                if args.train_metric == "macro_f1"
                else np.arange(len(mean_train))
            )
            axis_train.plot(
                x_values, mean_train, label=run_label, color=run_color, **line_style
            )
            if args.shade_std:
                axis_train.fill_between(
                    x_values,
                    mean_train - std_train,
                    mean_train + std_train,
                    color=run_color,
                    alpha=0.15,
                )
        else:
            train_series, train_metric_label = _concat_train_metric_for_run(
                seed_tasks_list[0], args.train_metric
            )
            if train_series is None:
                continue
            x_values = (
                np.arange(1, len(train_series) + 1)
                if args.train_metric == "macro_f1"
                else np.arange(len(train_series))
            )
            axis_train.plot(
                x_values, train_series, label=run_label, color=run_color, **line_style
            )
    axis_train.set_ylabel(train_metric_label, fontsize=16)
    axis_train.set_xlabel(
        _resolve_train_x_axis_label(args.train_metric),
        fontsize=16,
        labelpad=X_LABEL_PAD,
    )
    _hide_top_right_spines(axis_train)
    axis_train.tick_params(axis="both", labelsize=TICK_LABEL_FONT_SIZE)
    axis_train.grid(True, alpha=0.3)
    handles_train, labels_train = axis_train.get_legend_handles_labels()
    if handles_train and labels_train:
        export_legend_kwargs = _build_export_legend_kwargs(
            resolve_legend_kwargs(
                style_key=style_key,
                panel_key="train",
                base_legend_kwargs=legend_kwargs,
                run_count=len(runs),
            ),
            "train",
        )
        axis_train.legend(handles_train, labels_train, **export_legend_kwargs)
    manual_ylim_train = PANEL_YLIM_OVERRIDES.get("train")
    if manual_ylim_train is not None:
        axis_train.set_ylim(*manual_ylim_train)
    _save_independent_figure(
        fig_train,
        output_dir / f"{experiment_prefix}_TR-F1",
        dpi,
    )

    # Figure 2: final validation metrics (bars for macro recall / macro F1).
    fig_final, axis_final = plt.subplots(figsize=figure_size, dpi=dpi)
    for run_idx, run in enumerate(runs):
        if not run.tasks:
            continue
        seed_tasks_list = algo_seed_tasks.get(run.name, [run.tasks])

        def _seed_final_metric(tasks: list[Any], key: str) -> float | None:
            last = tasks[-1]
            return _mean_final_metric_for_run(last, key, len(tasks))

        def _mean_across_seeds(key: str) -> float | None:
            vals = [_seed_final_metric(t, key) for t in seed_tasks_list]
            vals = [v for v in vals if v is not None]
            return float(np.mean(vals)) if vals else None

        mean_rec = _mean_across_seeds("val_macro_rec")
        mean_f1 = _mean_across_seeds("val_macro_f1")

        x_center = run_idx
        width = 0.3
        if mean_rec is not None:
            axis_final.bar(
                x_center - width / 2,
                mean_rec,
                width=width,
                label="Macro recall" if run_idx == 0 else None,
                color=algorithm_colors.get(run.name, f"C{run_idx % 10}"),
                alpha=0.8,
            )
        if mean_f1 is not None:
            axis_final.bar(
                x_center + width / 2,
                mean_f1,
                width=width,
                label="Macro F1" if run_idx == 0 else None,
                color=algorithm_colors.get(run.name, f"C{run_idx % 10}"),
                hatch="..",
                alpha=0.8,
            )
    axis_final.set_ylabel("Metric value", fontsize=16)
    axis_final.set_xticks(np.arange(len(runs)))
    axis_final.set_xticklabels([])
    axis_final.set_xlabel("")
    _hide_top_right_spines(axis_final)
    axis_final.tick_params(axis="both", labelsize=TICK_LABEL_FONT_SIZE)
    axis_final.grid(True, alpha=0.3, axis="y")
    handles_final, labels_final = axis_final.get_legend_handles_labels()
    if handles_final and labels_final:
        export_legend_kwargs = resolve_legend_kwargs(
            style_key=style_key,
            panel_key="final_validation",
            base_legend_kwargs=legend_kwargs,
            run_count=len(runs),
        )
        manual_ncol_final = PANEL_LEGEND_NCOL_OVERRIDES.get("final_validation")
        if manual_ncol_final is not None:
            export_legend_kwargs["ncol"] = int(manual_ncol_final)
        axis_final.legend(handles_final, labels_final, **export_legend_kwargs)
    manual_ylim_final = PANEL_YLIM_OVERRIDES.get("final_validation")
    if manual_ylim_final is not None:
        axis_final.set_ylim(*manual_ylim_final)
    _save_independent_figure(
        fig_final, output_dir / f"{experiment_prefix}_Final-VAL", dpi
    )

    # Figure 3: mean validation metric over tasks (one line per run).
    fig_mean, axis_mean = plt.subplots(figsize=figure_size, dpi=dpi)
    mean_task_positions: set[int] = set()
    for run_idx, run in enumerate(runs):
        run_color = algorithm_colors.get(run.name, f"C{run_idx % 10}")
        line_style = _line_style_for_index(
            algorithm_line_member_index.get(run.name, run_idx)
        )
        seed_tasks_list = algo_seed_tasks.get(run.name, [run.tasks])
        if len(seed_tasks_list) > 1:
            x_vals, mean_y, std_y = _aggregate_across_seeds(
                seed_tasks_list,
                lambda tasks: _compute_mean_val_metric_over_tasks(tasks, first_key),
            )
            if x_vals.size == 0:
                continue
            zero_based_x = x_vals - 1
            mean_task_positions.update(int(v) for v in zero_based_x.tolist())
            axis_mean.plot(
                zero_based_x,
                mean_y,
                label=run_labels[run_idx],
                color=run_color,
                **line_style,
            )
            if args.shade_std:
                axis_mean.fill_between(
                    zero_based_x,
                    mean_y - std_y,
                    mean_y + std_y,
                    color=run_color,
                    alpha=0.15,
                )
        else:
            x_vals, y_vals = _compute_mean_val_metric_over_tasks(
                seed_tasks_list[0], first_key
            )
            if x_vals.size == 0:
                continue
            zero_based_x = x_vals - 1
            mean_task_positions.update(int(v) for v in zero_based_x.tolist())
            axis_mean.plot(
                zero_based_x,
                y_vals,
                label=run_labels[run_idx],
                color=run_color,
                **line_style,
            )
    axis_mean.set_xlabel("Task", fontsize=16, labelpad=X_LABEL_PAD)
    axis_mean.set_ylabel("F1 Score", fontsize=16)
    _set_task_axis_like_fwt(
        axis_mean,
        sorted(mean_task_positions),
        task_index_to_dataset_name,
    )
    _hide_top_right_spines(axis_mean)
    axis_mean.tick_params(axis="both", labelsize=TICK_LABEL_FONT_SIZE)
    axis_mean.grid(True, alpha=0.3)
    handles_mean, labels_mean = axis_mean.get_legend_handles_labels()
    if handles_mean and labels_mean:
        export_legend_kwargs = _build_export_legend_kwargs(
            resolve_legend_kwargs(
                style_key=style_key,
                panel_key="mean_val",
                base_legend_kwargs=legend_kwargs,
                run_count=len(runs),
            ),
            "mean_val",
        )
        axis_mean.legend(handles_mean, labels_mean, **export_legend_kwargs)
    manual_ylim_mean = PANEL_YLIM_OVERRIDES.get("mean_val")
    if manual_ylim_mean is not None:
        axis_mean.set_ylim(*manual_ylim_mean)
    _save_independent_figure(
        fig_mean,
        output_dir / f"{experiment_prefix}_CL-F1",
        dpi,
    )

    # Figure 4: backward transfer over tasks (one line per run).
    fig_forgetting, axis_forgetting = plt.subplots(figsize=figure_size, dpi=dpi)
    forgetting_task_positions: set[int] = set()
    for run_idx, run in enumerate(runs):
        val_metric_key, _ = _resolve_val_metric_for_run(args.val_metric, run)
        run_color = algorithm_colors.get(run.name, f"C{run_idx % 10}")
        line_style = _line_style_for_index(
            algorithm_line_member_index.get(run.name, run_idx)
        )
        seed_tasks_list = algo_seed_tasks.get(run.name, [run.tasks])
        if len(seed_tasks_list) > 1:
            x_vals, mean_fgt, std_fgt = _aggregate_across_seeds(
                seed_tasks_list,
                lambda tasks: compute_average_forgetting(tasks, val_metric_key),
            )
            if x_vals.size == 0:
                continue
            forgetting_task_positions.update(int(v) for v in x_vals.tolist())
            bwt_mean = -mean_fgt
            axis_forgetting.plot(
                x_vals,
                bwt_mean,
                label=run_labels[run_idx],
                color=run_color,
                **line_style,
            )
            if args.shade_std:
                axis_forgetting.fill_between(
                    x_vals,
                    bwt_mean - std_fgt,
                    bwt_mean + std_fgt,
                    color=run_color,
                    alpha=0.15,
                )
        else:
            x_vals, y_vals = compute_average_forgetting(
                seed_tasks_list[0], val_metric_key
            )
            if x_vals.size == 0:
                continue
            forgetting_task_positions.update(int(v) for v in x_vals.tolist())
            axis_forgetting.plot(
                x_vals,
                -y_vals,
                label=run_labels[run_idx],
                color=run_color,
                **line_style,
            )
    axis_forgetting.set_xlabel("Task", fontsize=16, labelpad=X_LABEL_PAD)
    axis_forgetting.set_ylabel("Backward Transfer", fontsize=16)
    _set_task_axis_like_fwt(
        axis_forgetting,
        sorted(forgetting_task_positions),
        task_index_to_dataset_name,
    )
    _hide_top_right_spines(axis_forgetting)
    axis_forgetting.tick_params(axis="both", labelsize=TICK_LABEL_FONT_SIZE)
    axis_forgetting.grid(True, alpha=0.3)
    manual_ylim_forgetting = PANEL_YLIM_OVERRIDES.get("average_forgetting")
    if manual_ylim_forgetting is not None:
        axis_forgetting.set_ylim(*manual_ylim_forgetting)
    elif plot_style.get("ylim") is not None:
        axis_forgetting.set_ylim(*plot_style["ylim"])
    handles_forgetting, labels_forgetting = axis_forgetting.get_legend_handles_labels()
    if handles_forgetting and labels_forgetting:
        export_legend_kwargs = _build_export_legend_kwargs(
            resolve_legend_kwargs(
                style_key=style_key,
                panel_key="average_forgetting",
                base_legend_kwargs=legend_kwargs,
                run_count=len(runs),
            ),
            "average_forgetting",
        )
        axis_forgetting.legend(
            handles_forgetting, labels_forgetting, **export_legend_kwargs
        )
    _save_independent_figure(
        fig_forgetting,
        output_dir / f"{experiment_prefix}_BWT",
        dpi,
    )

    # Figure 5: per-task validation metric as later tasks arrive. Colour encodes
    # the task being evaluated; line style encodes the algorithm.
    fig_per_task, axis_per_task = plt.subplots(figsize=figure_size, dpi=dpi)
    per_task_positions: set[int] = set()
    task_names_for_colors = getattr(runs[0], "task_names", None)
    for run_idx, run in enumerate(runs):
        val_metric_key, _ = _resolve_val_metric_for_run(args.val_metric, run)
        run_linestyle = LINESTYLES[run_idx % len(LINESTYLES)]
        seed_tasks_list = algo_seed_tasks.get(run.name, [run.tasks])
        task_count = min(len(seed_tasks) for seed_tasks in seed_tasks_list)
        for task_index in range(task_count):
            if len(seed_tasks_list) > 1:
                x_vals, y_vals, std_vals = _aggregate_across_seeds(
                    seed_tasks_list,
                    lambda tasks, task_index=task_index: (
                        _compute_val_metric_for_single_task(
                            tasks, val_metric_key, task_index
                        )
                    ),
                )
            else:
                x_vals, y_vals = _compute_val_metric_for_single_task(
                    seed_tasks_list[0], val_metric_key, task_index
                )
                std_vals = np.zeros_like(y_vals)
            if x_vals.size == 0:
                continue
            per_task_positions.update(int(position) for position in x_vals.tolist())
            task_color = get_task_color(task_index, task_names_for_colors)
            axis_per_task.plot(
                x_vals,
                y_vals,
                color=task_color,
                linestyle=run_linestyle,
                linewidth=1.6,
            )
            axis_per_task.plot(
                x_vals[:1],
                y_vals[:1],
                marker="o",
                markersize=4.0,
                color=task_color,
                linestyle="none",
            )
            if args.shade_std and len(seed_tasks_list) > 1:
                axis_per_task.fill_between(
                    x_vals,
                    y_vals - std_vals,
                    y_vals + std_vals,
                    color=task_color,
                    alpha=0.12,
                )
    axis_per_task.set_xlabel("Task", fontsize=16, labelpad=X_LABEL_PAD)
    axis_per_task.set_ylabel("F1 Score", fontsize=16)
    _set_task_axis_like_fwt(
        axis_per_task,
        sorted(per_task_positions),
        task_index_to_dataset_name,
    )
    _hide_top_right_spines(axis_per_task)
    axis_per_task.tick_params(axis="both", labelsize=TICK_LABEL_FONT_SIZE)
    axis_per_task.grid(True, alpha=0.3)
    algorithm_style_handles = [
        Line2D(
            [],
            [],
            color="black",
            linestyle=LINESTYLES[run_idx % len(LINESTYLES)],
            linewidth=1.6,
            label=run_labels[run_idx],
        )
        for run_idx in range(len(runs))
    ]
    axis_per_task.legend(
        handles=algorithm_style_handles,
        **_build_export_legend_kwargs(
            resolve_legend_kwargs(
                style_key=style_key,
                panel_key="mean_val",
                base_legend_kwargs=legend_kwargs,
                run_count=len(runs),
            ),
            "mean_val",
        ),
    )
    manual_ylim_per_task = PANEL_YLIM_OVERRIDES.get("val_per_task")
    if manual_ylim_per_task is not None:
        axis_per_task.set_ylim(*manual_ylim_per_task)
    _save_independent_figure(
        fig_per_task,
        output_dir
        / (
            f"{experiment_prefix}_Val-"
            f"{_metric_label_to_filename_token(first_label)}-per-task"
        ),
        dpi,
    )

    # Figure 6: forward transfer over tasks (generated by plot_fwt_metrics.py).
    fwt_json_path = _resolve_fwt_json_path(args.fwt_json_path, run_source_dir)
    if fwt_json_path is None:
        print(
            "[WARN] Could not locate fwt_metrics.json automatically; "
            "skipping FWT figure. Pass --fwt-json-path to enable it."
        )
    else:
        try:
            fwt_records = load_metrics(fwt_json_path)
            fwt_series_by_algorithm = build_series_by_algo(
                fwt_records, "forward_transfer_total_macro_f1_zs"
            )
            if not args.include_iid2:
                fwt_series_by_algorithm.pop("iid2", None)

            # Keep only discovered algorithms so FWT uses the same run set.
            discovered_algorithm_names = {run.name for run in runs}
            fwt_series_by_algorithm = {
                algorithm_name: series
                for algorithm_name, series in fwt_series_by_algorithm.items()
                if algorithm_name in discovered_algorithm_names
            }
            if not fwt_series_by_algorithm:
                print(
                    f"[WARN] FWT metrics found at {fwt_json_path}, but no algorithms "
                    "overlap with discovered runs; skipping FWT figure."
                )
            else:
                fwt_output_path = output_dir / f"{experiment_prefix}_FWT"
                fwt_plot_style = _case_insensitive_detect_style(runs)
                fwt_plot_style = dict(fwt_plot_style)
                fwt_plot_style["figsize"] = figure_size
                plot_series(
                    series_by_algorithm=fwt_series_by_algorithm,
                    metric_name="forward_transfer_total_macro_f1_zs",
                    output_path=fwt_output_path,
                    plot_style=fwt_plot_style,
                    task_index_to_dataset_name_override=task_index_to_dataset_name,
                )
                print(
                    "Saved FWT subplot to "
                    f"{fwt_output_path.with_suffix('.pdf')} and "
                    f"{fwt_output_path.with_suffix('.png')}"
                )
        except (FileNotFoundError, ValueError) as error:
            print(f"[WARN] Unable to generate FWT figure from {fwt_json_path}: {error}")


if __name__ == "__main__":
    main()
