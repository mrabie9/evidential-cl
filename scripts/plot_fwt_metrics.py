"""Plot forward-transfer metrics from a metrics JSON file.

This script reads a JSON file containing per-task metrics for multiple
algorithms and generates a line plot for one selected metric.

Usage:
    python scripts/plot_fwt_metrics.py \
        --json-path logs/full_experiments/one-shot_cil/fwt_metrics.json

    python scripts/plot_fwt_metrics.py \
        --json-path logs/full_experiments/one-shot_cil/fwt_metrics.json \
        --metric forward_transfer_total_macro_f1_zs \
        --output-path logs/full_experiments/one-shot_cil/fwt_metrics_plot.png
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils.metric_keys import extract_metric  # noqa: E402

import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    from scripts import plot_algorithm_group_styles as group_style_config  # type: ignore
    from scripts.plot_algorithm_group_styles import (  # type: ignore
        build_group_color_map,
        group_sort_key,
    )
except Exception:  # pragma: no cover - optional styling helper
    group_style_config = None

    def build_group_color_map(  # type: ignore[misc]
        algorithm_names: List[str],
        fallback_colors: Dict[str, Any] | None = None,
    ) -> Dict[str, Any]:
        _ = algorithm_names
        return dict(fallback_colors or {})

    def group_sort_key(algorithm_name: str) -> tuple[int, str]:  # type: ignore[misc]
        """Fallback sort key when group-style helper is unavailable."""
        return (0, algorithm_name.lower())


MetricRecord = Dict[str, Any]
SeriesPoint = Tuple[int, Optional[float], str]
PlotStyle = Dict[str, Any]
# GROUP_COLORS = ["#4477AA", "#EE6677", "#228833", "#66CCEE", "#AA3377"]
# Cycled by the *filtered* position among groups actually present (empty
# groups, e.g. "ungrouped" when nothing is unclassified, are skipped), so a
# group's index here isn't stable across runs; index 5 covers any leftover
# ungrouped algorithm.
GROUP_COLORS = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#999999"]
# GROUP_COLORS = ['#4E79A7', '#F28E2B', '#E15759', '#76B7B2', '#59A14F']
# Explicit per-group-name color overrides, applied before the GROUP_COLORS
# cycling above. Use this for a group whose color must stay fixed regardless
# of which/how-many other groups are present, e.g. reference baselines.
GROUP_COLOR_OVERRIDES: Dict[str, str] = {"baseline": "#000000"}
LINESTYLES = ["-", "--", ":", "-.", (0, (5, 1))]
LINEWIDTHS = [2.4, 1.8, 1.8, 1.5, 1.5]
MAX_GROUPS_SINGLE_AXIS = 5
MAX_MEMBERS_PER_GROUP = 5
MAX_LINES_SINGLE_AXIS = 18
IEEE_SINGLE_COLUMN_FIGSIZE = (3.5, 2.6)
IEEE_DOUBLE_COLUMN_FIGSIZE = (7, 3.5)
DEFAULT_ALGORITHM_COLOR_ORDER = [
    "agem",
    "bcl_dual",
    "eralg4",
    "ewc",
    "iid2",
    "la-er",
    "si",
    "ucl",
    "cmaml",
    "er_ring",
    "gem",
    "icarl",
    "lamaml",
    "lwf",
    "rwalk",
    "smaml",
]
PLOT_STYLES_BY_EXPERIMENT: Dict[str, PlotStyle] = {
    "default": {
        "figsize": IEEE_DOUBLE_COLUMN_FIGSIZE,
        "ylim": None,
    },
    "til": {
        "figsize": IEEE_DOUBLE_COLUMN_FIGSIZE,
        "ylim": (-0.06, 0.15),
    },
}
ALGORITHM_DISPLAY_NAMES: Dict[str, str] = {
    "agem": "A-GEM",
    "bcl_dual": "BCL-Dual",
    "cmaml": "C-MAML",
    "ctn": "CTN",
    "eralg4": "Res-ER",
    "er_ring": "Ring-ER",
    "ewc": "EWC",
    "ft": "FT",
    "gem": "GEM",
    "hat": "HAT",
    "icarl": "iCaRL",
    "iid2": "JT",
    "la-er": "La-ER",
    "lamaml": "La-MAML",
    "lwf": "LwF",
    "packnet": "PackNet",
    "rwalk": "RWalk",
    "si": "SI",
    "smaml": "S-MAML",
    "ucl": "UCL",
}


def build_label_colors(labels: List[str]) -> Dict[str, Any]:
    """Build distinct colors for labels using the combined-layout palette.

    This mirrors the color-selection approach from
    ``scripts/plot_multi_algorithms.py`` used by ``--plot-layout together``.
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

    def color_for_position(label_position: int, total_labels: int) -> Any:
        if label_position < len(qualitative_palette):
            return qualitative_palette[label_position]
        if total_labels <= 1:
            return plt.get_cmap("hsv")(0.0)
        return plt.get_cmap("hsv")(label_position / float(total_labels))

    colors_by_label: Dict[str, Any] = {}
    known_algorithm_to_palette_index = {
        algorithm_name: index
        for index, algorithm_name in enumerate(DEFAULT_ALGORITHM_COLOR_ORDER)
    }

    # Use fixed palette positions per known algorithm so color remains stable
    # across plots even when some algorithms are absent.
    for label in labels:
        if label in known_algorithm_to_palette_index:
            colors_by_label[label] = color_for_position(
                known_algorithm_to_palette_index[label],
                len(DEFAULT_ALGORITHM_COLOR_ORDER),
            )

    # Assign deterministic colors to unknown labels after the known block.
    next_palette_index = len(DEFAULT_ALGORITHM_COLOR_ORDER)
    for label in sorted(labels):
        if label in colors_by_label:
            continue
        colors_by_label[label] = color_for_position(
            next_palette_index, next_palette_index + 1
        )
        next_palette_index += 1

    return colors_by_label


def _normalize_algorithm_name(algorithm_name: str) -> str:
    """Normalize algorithm naming variants for stable matching."""
    return (
        algorithm_name.strip()
        .lower()
        .replace("-", "")
        .replace("_", "")
        .replace(" ", "")
    )


def _build_ordered_groups(
    sorted_algorithm_names: List[str],
) -> List[tuple[str, List[str]]]:
    """Build ordered algorithm groups using shared group config when available."""
    group_to_algorithms: Dict[str, List[str]] = {}
    algorithm_to_group: Dict[str, str] = {}
    configured_group_order: List[str] = []
    configured_group_members: Dict[str, List[str]] = {}

    if group_style_config is not None:
        configured_group_order = list(
            getattr(
                group_style_config,
                "GROUP_ORDER",
                [],
            )
        )
        configured_group_members = dict(
            getattr(
                group_style_config,
                "ALGORITHM_GROUPS",
                {},
            )
        )
        for (
            configured_group_name,
            configured_member_names,
        ) in configured_group_members.items():
            for configured_member_name in configured_member_names:
                algorithm_to_group[
                    _normalize_algorithm_name(configured_member_name)
                ] = configured_group_name

    for algorithm_name in sorted_algorithm_names:
        normalized_name = _normalize_algorithm_name(algorithm_name)
        group_name = algorithm_to_group.get(normalized_name, "ungrouped")
        group_to_algorithms.setdefault(group_name, []).append(algorithm_name)

    ordered_groups: List[tuple[str, List[str]]] = []
    for configured_group_name in configured_group_order:
        if configured_group_name not in group_to_algorithms:
            continue
        present_algorithms = group_to_algorithms[configured_group_name]
        configured_members = configured_group_members.get(configured_group_name, [])
        configured_member_positions = {
            _normalize_algorithm_name(member_name): position
            for position, member_name in enumerate(configured_members)
        }
        sorted_present_algorithms = sorted(
            present_algorithms,
            key=lambda algorithm_name: (
                configured_member_positions.get(
                    _normalize_algorithm_name(algorithm_name), 10_000
                ),
                _normalize_algorithm_name(algorithm_name),
            ),
        )
        ordered_groups.append((configured_group_name, sorted_present_algorithms))

    for group_name, member_names in group_to_algorithms.items():
        if group_name in configured_group_order:
            continue
        ordered_groups.append(
            (
                group_name,
                sorted(
                    member_names,
                    key=lambda algorithm_name: _normalize_algorithm_name(
                        algorithm_name
                    ),
                ),
            )
        )
    return ordered_groups


def _extract_task_axis_metadata(
    series_by_algorithm: Dict[str, List[SeriesPoint]],
) -> tuple[List[int], Dict[int, str]]:
    """Collect task ordering and printable dataset names for x-axis labels."""
    all_task_indices = set()
    task_index_to_dataset_name: Dict[int, str] = {}
    for series in series_by_algorithm.values():
        for task_index, _, task_name in series:
            all_task_indices.add(task_index)
            if task_index in task_index_to_dataset_name:
                continue
            task_name_parts = task_name.split("-", 1)
            if len(task_name_parts) == 2 and task_name_parts[1]:
                original_task_number = task_name_parts[0].lstrip("t")
                dataset_name = task_name_parts[1]
            else:
                original_task_number = "?"
                dataset_name = task_name
            dataset_name_lower = dataset_name.lower()
            if "uclresm" in dataset_name_lower:
                dataset_name = f"(RML-{original_task_number})"
            elif "deeprad" in dataset_name_lower:
                dataset_name = f"(DR-{original_task_number})"
            elif "rcn" in dataset_name_lower:
                dataset_name = f"(RCN-{original_task_number})"
            task_index_to_dataset_name[task_index] = dataset_name
    return sorted(all_task_indices), task_index_to_dataset_name


def _style_axis(
    axis: Any, sorted_task_indices: List[int], task_labels: Dict[int, str]
) -> None:
    """Apply shared axis formatting for publication-ready plots."""
    axis.spines[["top", "right"]].set_visible(False)
    axis.grid(True, linestyle=":", linewidth=0.5, alpha=0.5)
    axis.axhline(0.0, color="black", linewidth=1.0, linestyle="--", alpha=0.7)
    axis.set_xticks(sorted_task_indices)
    axis.set_xticklabels(
        [
            f"{task_index}\n{task_labels.get(task_index, 'unknown')}"
            for task_index in sorted_task_indices
        ]
    )


def parse_args() -> argparse.Namespace:
    """Parse command line arguments.

    Returns:
        Parsed CLI namespace.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--json-path",
        type=Path,
        required=True,
        help="Path to the metrics JSON file.",
    )
    parser.add_argument(
        "--metric",
        type=str,
        default="forward_transfer_total_macro_f1_zs",
        help="Metric key to plot from each record.",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=None,
        help=(
            "Path to save the figure. "
            "Defaults to '<repo>/logs/00_sync/<experiment>/plots_grouped/"
            "<Multi/Single>-Epoch_<TIL/CIL>_FWT.png'."
        ),
    )
    parser.add_argument(
        "--title",
        type=str,
        default=None,
        help="Optional custom plot title.",
    )
    return parser.parse_args()


def load_metrics(json_path: Path) -> List[MetricRecord]:
    """Load metrics records from JSON.

    Args:
        json_path: Path to JSON containing a list of metric records.

    Returns:
        List of metric dictionaries.

    Raises:
        FileNotFoundError: If the JSON path does not exist.
        ValueError: If JSON is not a list of records.
    """
    if not json_path.is_file():
        raise FileNotFoundError(f"Metrics JSON not found: {json_path}")

    with json_path.open("r", encoding="utf-8") as handle:
        loaded = json.load(handle)

    if not isinstance(loaded, list):
        raise ValueError("Expected JSON root to be a list of metric records.")
    return loaded


def resolve_default_output_path(json_path: Path, metric_name: str) -> Path:
    """Resolve the default PNG output path under logs/00_sync.

    Args:
        json_path: Input metrics JSON path.
        metric_name: Metric key used in the generated filename.

    Returns:
        Default destination path for the output plot.
    """
    _ = metric_name  # Filename now follows a fixed naming convention.

    experiment_path_text = str(json_path).lower()
    epoch_mode = "Single" if "one-shot" in experiment_path_text else "Multi"
    learning_setup = "CIL" if "cil" in experiment_path_text else "TIL"
    default_filename = f"{epoch_mode}-Epoch_{learning_setup}_FWT.png"

    json_parts = json_path.parts
    if "logs" in json_parts:
        logs_index = json_parts.index("logs")
        suffix_parts = list(json_parts[logs_index + 1 : -1])
        if suffix_parts:
            suffix_parts[0] = "00_sync"

            def resolve_child_dir_case_insensitive(
                parent_dir: Path, desired_child_dir_name: str
            ) -> Path:
                """Resolve a child dir name with case-insensitive matching.

                This helps when input paths use `one-shot_TIL` vs `one-shot_til`
                (or similar for `cil`) on a case-sensitive filesystem.
                """
                direct_child = parent_dir / desired_child_dir_name
                if direct_child.exists():
                    return direct_child

                if not parent_dir.exists():
                    return direct_child

                desired_lower = desired_child_dir_name.lower()
                for child in parent_dir.iterdir():
                    if child.is_dir() and child.name.lower() == desired_lower:
                        return child

                return direct_child

            current_dir = Path(*json_parts[:logs_index]) / "logs" / suffix_parts[0]
            for part in suffix_parts[1:]:
                if "til" in part.lower() or "cil" in part.lower():
                    current_dir = resolve_child_dir_case_insensitive(current_dir, part)
                else:
                    current_dir = current_dir / part

            return current_dir / "plots_grouped" / default_filename

    return json_path.parent / "plots_grouped" / default_filename


def build_series_by_algo(
    records: List[MetricRecord], metric_name: str
) -> Dict[str, List[SeriesPoint]]:
    """Group metric points by algorithm and sort by task index.

    Args:
        records: Flat list of metric records.
        metric_name: Metric key to extract from each record.

    Returns:
        Mapping from algorithm name to ordered points:
        ``(task_index, metric_value, task_name)``.
    """
    series_by_algorithm: Dict[str, List[SeriesPoint]] = {}
    for record in records:
        algorithm_name = str(record.get("algo", "unknown"))
        task_index = int(record.get("task_index", -1))
        task_name = str(record.get("task_name", f"task_{task_index}"))
        raw_metric_value = extract_metric(record, metric_name)
        metric_value = float(raw_metric_value) if raw_metric_value is not None else None

        series_by_algorithm.setdefault(algorithm_name, []).append(
            (task_index, metric_value, task_name)
        )

    for algorithm_name, series in series_by_algorithm.items():
        series_by_algorithm[algorithm_name] = sorted(series, key=lambda point: point[0])
    return series_by_algorithm


def plot_series(
    series_by_algorithm: Dict[str, List[SeriesPoint]],
    metric_name: str,
    output_path: Path,
    plot_style: PlotStyle,
    title: Optional[str] = None,
    task_index_to_dataset_name_override: Optional[Dict[int, str]] = None,
) -> None:
    """Create and save the metric line plot.

    Args:
        series_by_algorithm: Algorithm-to-series mapping.
        metric_name: Metric key being plotted.
        output_path: Where to save the PNG.
        title: Optional custom chart title.
        task_index_to_dataset_name_override: Ready-made x-axis dataset labels.
            Callers that discovered the runs themselves (and therefore read
            ``metrics/task_order.txt``) should pass them so the labels do not
            depend on the metrics JSON carrying ``task_name``.

    Usage:
        >>> plot_series({"algo": [(0, 0.1, "t0")]}, "metric", Path("out.png"))
    """
    sorted_algorithm_names = sorted(series_by_algorithm.keys(), key=group_sort_key)
    ordered_groups = _build_ordered_groups(sorted_algorithm_names)
    sorted_task_indices, task_index_to_dataset_name = _extract_task_axis_metadata(
        series_by_algorithm
    )
    if task_index_to_dataset_name_override:
        task_index_to_dataset_name = dict(task_index_to_dataset_name_override)
    total_lines = sum(len(member_names) for _, member_names in ordered_groups)
    maximum_group_size = max(
        (len(member_names) for _, member_names in ordered_groups), default=0
    )
    use_small_multiples = (
        total_lines > MAX_LINES_SINGLE_AXIS
        or len(ordered_groups) > MAX_GROUPS_SINGLE_AXIS
        or maximum_group_size > MAX_MEMBERS_PER_GROUP
    )

    if use_small_multiples:
        figure, axes = plt.subplots(
            nrows=len(ordered_groups),
            ncols=1,
            sharex=True,
            sharey=True,
            figsize=(
                IEEE_DOUBLE_COLUMN_FIGSIZE[0],
                max(2.2 * len(ordered_groups), IEEE_DOUBLE_COLUMN_FIGSIZE[1]),
            ),
        )
        if len(ordered_groups) == 1:
            axes = [axes]
        for group_index, (group_name, group_algorithms) in enumerate(ordered_groups):
            axis = axes[group_index]
            group_color = GROUP_COLOR_OVERRIDES.get(
                group_name, GROUP_COLORS[group_index % len(GROUP_COLORS)]
            )
            for member_index, algorithm_name in enumerate(
                group_algorithms[:MAX_MEMBERS_PER_GROUP]
            ):
                series = series_by_algorithm[algorithm_name]
                x_positions = [point[0] for point in series]
                y_values = [point[1] for point in series]
                axis.plot(
                    x_positions,
                    y_values,
                    linewidth=LINEWIDTHS[member_index % len(LINEWIDTHS)],
                    linestyle=LINESTYLES[member_index % len(LINESTYLES)],
                    alpha=0.9,
                    color=group_color,
                    label=ALGORITHM_DISPLAY_NAMES.get(algorithm_name, algorithm_name),
                )
            _style_axis(axis, sorted_task_indices, task_index_to_dataset_name)
            axis.set_ylabel("Forward Transfer")
            axis.legend(
                loc="upper right",
                fontsize=8,
                framealpha=0.9,
                edgecolor="0.8",
                borderpad=0.5,
                labelspacing=0.3,
                ncol=6,
            )
            axis.text(
                0.01,
                0.97,
                group_name.replace("_", " ").title(),
                transform=axis.transAxes,
                va="top",
                ha="left",
                fontsize=8,
            )
        axes[-1].set_xlabel("Task")
    else:
        figure, axis = plt.subplots(figsize=plot_style["figsize"])
        for group_index, (group_name, group_algorithms) in enumerate(ordered_groups):
            group_color = GROUP_COLOR_OVERRIDES.get(
                group_name, GROUP_COLORS[group_index % len(GROUP_COLORS)]
            )
            for member_index, algorithm_name in enumerate(group_algorithms):
                series = series_by_algorithm[algorithm_name]
                x_positions = [point[0] for point in series]
                y_values = [point[1] for point in series]
                axis.plot(
                    x_positions,
                    y_values,
                    linewidth=LINEWIDTHS[member_index % len(LINEWIDTHS)],
                    linestyle=LINESTYLES[member_index % len(LINESTYLES)],
                    alpha=0.9,
                    color=group_color,
                    label=ALGORITHM_DISPLAY_NAMES.get(algorithm_name, algorithm_name),
                )
        _style_axis(axis, sorted_task_indices, task_index_to_dataset_name)
        axis.set_xlabel("Task")
        axis.set_ylabel("Forward Transfer")
        axis.legend(
            loc="upper right",
            fontsize=10,
            edgecolor="0.8",
            ncol=6,
            columnspacing=0.5,
            labelspacing=0.2,
            framealpha=0.5,
            borderaxespad=0.1,
            borderpad=0.2,
        )

    y_limits = plot_style.get("ylim")
    if y_limits is not None:
        if use_small_multiples:
            for axis in axes:
                axis.set_ylim(*y_limits)
        else:
            axis.set_ylim(*y_limits)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout(pad=0.1)
    base_output_path = output_path.with_suffix("")
    figure.savefig(base_output_path.with_suffix(".pdf"), dpi=600)
    figure.savefig(base_output_path.with_suffix(".png"), dpi=600)
    plt.close(figure)


def main() -> None:
    """Run the plotting pipeline."""
    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.labelsize": 14,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
            "legend.fontsize": 10,
            "axes.linewidth": 1,
            # "lines.linewidth": 1.2,
            "figure.dpi": 600,
            "savefig.dpi": 600,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.02,
        }
    )
    arguments = parse_args()
    if group_style_config is not None:
        group_style_config.ENABLE_GROUP_STYLING = True
    default_output_path = resolve_default_output_path(
        json_path=arguments.json_path,
        metric_name=arguments.metric,
    )
    output_path = arguments.output_path or default_output_path

    records = load_metrics(arguments.json_path)
    series_by_algorithm = build_series_by_algo(records, arguments.metric)
    experiment_path_text = str(arguments.json_path).lower()
    style_key = "til" if "til" in experiment_path_text else "default"
    selected_plot_style = PLOT_STYLES_BY_EXPERIMENT[style_key]
    plot_series(
        series_by_algorithm=series_by_algorithm,
        metric_name=arguments.metric,
        output_path=output_path,
        plot_style=selected_plot_style,
        title=arguments.title,
    )
    print(
        "Saved plot to: "
        f"{output_path.with_suffix('.pdf')} and {output_path.with_suffix('.png')}"
    )


if __name__ == "__main__":
    main()
