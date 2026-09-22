"""Plot final macro-F1 and backward transfer as a function of SNR.

Reads the SNR sweep logs produced under ``logs/00_sync/snr_TIL`` and
``logs/00_sync/snr_CIL``, aggregates each model's per-seed metrics, and renders
a two-panel figure per mode: final validation performance on the left, backward
transfer on the right, both against SNR in dB.

Expected log layout::

    <logs-root>/snr_<MODE>/<snr-point>/saved_models/<model>/<run>/<seed>/seed_metrics.json

Usage:
    python scripts/plot_snr_sweep.py
    python scripts/plot_snr_sweep.py --mode til --metric f1
    python scripts/plot_snr_sweep.py --logs-root logs/00_sync --output-dir analysis/snr_sweep
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

SCRIPTS_DIRECTORY = Path(__file__).resolve().parent
if str(SCRIPTS_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIRECTORY))

from plot_algorithm_group_styles import (  # noqa: E402
    build_group_color_map,
    group_sort_key,
    resolve_group_linestyle,
    resolve_group_marker,
)
from plot_fwt_metrics import ALGORITHM_DISPLAY_NAMES  # noqa: E402

SNR_POINT_PATTERN = re.compile(r"^(m?)(\d+)db$", re.IGNORECASE)
SUPPORTED_MODES: tuple[str, ...] = ("til", "cil")
SUPPORTED_METRICS: tuple[str, ...] = ("f1", "rec", "prec")
METRIC_DISPLAY_NAMES: Dict[str, str] = {
    "f1": "macro F1",
    "rec": "macro recall",
    "prec": "macro precision",
}
SPLIT_DISPLAY_NAMES: Dict[str, str] = {
    "val": "Validation",
    "tr": "Training",
}
FIGURE_SIZE_INCHES: tuple[float, float] = (7.2, 3.6)
ERROR_BAND_ALPHA: float = 0.15


@dataclass(frozen=True)
class AggregatedMetricPoint:
    """Seed-aggregated value of one metric at one SNR point.

    Attributes:
        snr_db: Signal-to-noise ratio in dB.
        mean_value: Mean of the metric across seeds.
        standard_deviation: Sample standard deviation across seeds.
        seed_count: Number of seeds contributing to the aggregate.
    """

    snr_db: int
    mean_value: float
    standard_deviation: float
    seed_count: int


def parse_snr_label_to_db(snr_label: str) -> Optional[int]:
    """Convert an SNR directory name into a signed dB value.

    Args:
        snr_label: Directory name such as ``"0db"``, ``"10db"`` or ``"m4db"``.

    Returns:
        The SNR in dB, or ``None`` when the name is not an SNR point.

    Usage:
        >>> parse_snr_label_to_db("m4db")
        -4
        >>> parse_snr_label_to_db("10db")
        10
    """
    match = SNR_POINT_PATTERN.match(snr_label.strip())
    if match is None:
        return None
    sign = -1 if match.group(1).lower() == "m" else 1
    return sign * int(match.group(2))


def discover_snr_point_directories(mode_root: Path) -> Dict[int, Path]:
    """Find every SNR point directory beneath a mode root.

    Args:
        mode_root: Directory such as ``logs/00_sync/snr_TIL``.

    Returns:
        Mapping from SNR in dB to the corresponding directory.

    Usage:
        >>> discover_snr_point_directories(Path("logs/00_sync/snr_TIL"))
        {-10: PosixPath('logs/00_sync/snr_TIL/m10db'), ...}
    """
    snr_point_directories: Dict[int, Path] = {}
    if not mode_root.is_dir():
        return snr_point_directories
    for candidate_directory in sorted(mode_root.iterdir()):
        if not candidate_directory.is_dir():
            continue
        snr_db = parse_snr_label_to_db(candidate_directory.name)
        if snr_db is None:
            continue
        snr_point_directories[snr_db] = candidate_directory
    return snr_point_directories


def load_seed_metric_values(model_directory: Path, metric_key: str) -> List[float]:
    """Read one metric from every ``seed_metrics.json`` under a model directory.

    Args:
        model_directory: Directory such as ``<snr point>/saved_models/gem``.
        metric_key: Key inside ``seed_metrics.json``, e.g. ``"val_macro_f1"``.

    Returns:
        The metric value for each seed that provides it.

    Usage:
        >>> load_seed_metric_values(Path("logs/.../saved_models/gem"), "val_macro_f1")
        [0.711, 0.735, ...]
    """
    seed_metric_values: List[float] = []
    for seed_metrics_path in sorted(model_directory.glob("*/*/seed_metrics.json")):
        try:
            seed_metrics = json.loads(seed_metrics_path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        metric_value = seed_metrics.get(metric_key)
        if isinstance(metric_value, (int, float)):
            seed_metric_values.append(float(metric_value))
    return seed_metric_values


def aggregate_seed_values(
    snr_db: int, seed_metric_values: Sequence[float]
) -> Optional[AggregatedMetricPoint]:
    """Summarise per-seed metric values into a single plot point.

    Args:
        snr_db: SNR in dB the values belong to.
        seed_metric_values: Metric value for each seed.

    Returns:
        The aggregated point, or ``None`` when no seed provided a value.

    Usage:
        >>> aggregate_seed_values(0, [0.70, 0.72])
        AggregatedMetricPoint(snr_db=0, mean_value=0.71, ...)
    """
    if not seed_metric_values:
        return None
    standard_deviation = (
        statistics.stdev(seed_metric_values) if len(seed_metric_values) > 1 else 0.0
    )
    return AggregatedMetricPoint(
        snr_db=snr_db,
        mean_value=statistics.fmean(seed_metric_values),
        standard_deviation=standard_deviation,
        seed_count=len(seed_metric_values),
    )


def collect_series_by_model(
    mode_root: Path, metric_key: str
) -> Dict[str, List[AggregatedMetricPoint]]:
    """Build the SNR series of one metric for every model in a sweep.

    Args:
        mode_root: Directory such as ``logs/00_sync/snr_TIL``.
        metric_key: Key inside ``seed_metrics.json``, e.g. ``"val_bwt_f1"``.

    Returns:
        Mapping from model name to its SNR-ordered aggregated points.

    Usage:
        >>> collect_series_by_model(Path("logs/00_sync/snr_TIL"), "val_macro_f1")
        {'gem': [AggregatedMetricPoint(snr_db=-10, ...), ...], ...}
    """
    series_by_model: Dict[str, List[AggregatedMetricPoint]] = {}
    snr_point_directories = discover_snr_point_directories(mode_root)
    for snr_db in sorted(snr_point_directories):
        saved_models_directory = snr_point_directories[snr_db] / "saved_models"
        if not saved_models_directory.is_dir():
            continue
        for model_directory in sorted(saved_models_directory.iterdir()):
            if not model_directory.is_dir():
                continue
            aggregated_point = aggregate_seed_values(
                snr_db, load_seed_metric_values(model_directory, metric_key)
            )
            if aggregated_point is None:
                continue
            series_by_model.setdefault(model_directory.name, []).append(
                aggregated_point
            )
    return {
        model_name: sorted(points, key=lambda point: point.snr_db)
        for model_name, points in series_by_model.items()
    }


def resolve_display_name(model_name: str) -> str:
    """Map a model directory name to its published label.

    Args:
        model_name: Model identifier such as ``"eralg4"``.

    Returns:
        A human-readable label.

    Usage:
        >>> resolve_display_name("eralg4")
        'Res-ER'
    """
    return ALGORITHM_DISPLAY_NAMES.get(model_name, model_name)


def draw_metric_panel(
    axis: plt.Axes,
    series_by_model: Dict[str, List[AggregatedMetricPoint]],
    model_names: Sequence[str],
    colors_by_model: Dict[str, Any],
    y_axis_label: str,
    panel_title: str,
    show_error_bands: bool,
) -> None:
    """Draw one model-per-line panel of a metric against SNR.

    Args:
        axis: Target matplotlib axes.
        series_by_model: SNR series for each model.
        model_names: Models to draw, in legend order.
        colors_by_model: Per-model line colors.
        y_axis_label: Label for the y axis.
        panel_title: Title above the panel.
        show_error_bands: Whether to shade the across-seed standard deviation.
    """
    for model_name in model_names:
        aggregated_points = series_by_model.get(model_name, [])
        if not aggregated_points:
            continue
        snr_values = [point.snr_db for point in aggregated_points]
        mean_values = [point.mean_value for point in aggregated_points]
        axis.plot(
            snr_values,
            mean_values,
            label=resolve_display_name(model_name),
            color=colors_by_model[model_name],
            linestyle=resolve_group_linestyle(model_name),
            marker=resolve_group_marker(model_name, fallback_marker="o"),
            linewidth=1.8,
            markersize=4.5,
        )
        if not show_error_bands:
            continue
        lower_bounds = [
            point.mean_value - point.standard_deviation for point in aggregated_points
        ]
        upper_bounds = [
            point.mean_value + point.standard_deviation for point in aggregated_points
        ]
        axis.fill_between(
            snr_values,
            lower_bounds,
            upper_bounds,
            color=colors_by_model[model_name],
            alpha=ERROR_BAND_ALPHA,
            linewidth=0.0,
        )

    axis.set_xlabel("SNR (dB)")
    axis.set_ylabel(y_axis_label)
    axis.set_title(panel_title, fontsize=10)
    axis.grid(True, linestyle=":", linewidth=0.6, alpha=0.6)
    axis.set_axisbelow(True)


def collect_all_snr_values(
    series_by_model: Dict[str, List[AggregatedMetricPoint]],
) -> List[int]:
    """List every SNR value present in a set of series.

    Args:
        series_by_model: SNR series for each model.

    Returns:
        Sorted unique SNR values in dB.

    Usage:
        >>> collect_all_snr_values({"gem": [AggregatedMetricPoint(0, 0.7, 0.0, 6)]})
        [0]
    """
    snr_values = {
        point.snr_db for points in series_by_model.values() for point in points
    }
    return sorted(snr_values)


def build_mode_figure(
    performance_series: Dict[str, List[AggregatedMetricPoint]],
    backward_transfer_series: Dict[str, List[AggregatedMetricPoint]],
    mode: str,
    metric: str,
    split: str,
    show_error_bands: bool,
) -> plt.Figure:
    """Render the two-panel SNR figure for one continual-learning mode.

    Args:
        performance_series: Final-performance series for each model.
        backward_transfer_series: Backward-transfer series for each model.
        mode: Either ``"til"`` or ``"cil"``.
        metric: One of ``"f1"``, ``"rec"``, ``"prec"``.
        split: Either ``"val"`` or ``"tr"`` for the performance panel.
        show_error_bands: Whether to shade the across-seed standard deviation.

    Returns:
        The completed matplotlib figure.
    """
    model_names = sorted(
        set(performance_series) | set(backward_transfer_series), key=group_sort_key
    )
    colors_by_model = build_group_color_map(model_names)
    metric_display_name = METRIC_DISPLAY_NAMES[metric]

    figure, (performance_axis, backward_transfer_axis) = plt.subplots(
        1, 2, figsize=FIGURE_SIZE_INCHES
    )
    draw_metric_panel(
        axis=performance_axis,
        series_by_model=performance_series,
        model_names=model_names,
        colors_by_model=colors_by_model,
        y_axis_label=f"{SPLIT_DISPLAY_NAMES[split]} {metric_display_name}",
        panel_title=f"Final {metric_display_name}",
        show_error_bands=show_error_bands,
    )
    draw_metric_panel(
        axis=backward_transfer_axis,
        series_by_model=backward_transfer_series,
        model_names=model_names,
        colors_by_model=colors_by_model,
        y_axis_label=f"BWT ({metric_display_name})",
        panel_title="Backward transfer",
        show_error_bands=show_error_bands,
    )
    backward_transfer_axis.axhline(
        0.0, color="0.35", linewidth=0.8, linestyle="-", zorder=1
    )

    snr_values = collect_all_snr_values(performance_series) or collect_all_snr_values(
        backward_transfer_series
    )
    for axis in (performance_axis, backward_transfer_axis):
        axis.set_xticks(snr_values)
        axis.tick_params(axis="both", labelsize=8)

    legend_handles, legend_labels = performance_axis.get_legend_handles_labels()
    legend_column_count = min(len(legend_labels), 5) or 1
    figure.legend(
        legend_handles,
        legend_labels,
        loc="lower center",
        ncol=legend_column_count,
        fontsize=8,
        frameon=False,
        bbox_to_anchor=(0.5, -0.02),
    )
    figure.suptitle(f"SNR sweep ({mode.upper()})", fontsize=11, y=1.0)
    figure.tight_layout(rect=(0.0, 0.08, 1.0, 0.97))
    return figure


def write_series_csv(
    output_path: Path,
    performance_series: Dict[str, List[AggregatedMetricPoint]],
    backward_transfer_series: Dict[str, List[AggregatedMetricPoint]],
) -> None:
    """Write the aggregated plot data alongside the figure.

    Args:
        output_path: Destination ``.csv`` path.
        performance_series: Final-performance series for each model.
        backward_transfer_series: Backward-transfer series for each model.
    """
    labelled_series = (
        ("performance", performance_series),
        ("bwt", backward_transfer_series),
    )
    with output_path.open("w", newline="") as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(["panel", "model", "snr_db", "mean", "std", "n_seeds"])
        for panel_name, series_by_model in labelled_series:
            for model_name in sorted(series_by_model, key=group_sort_key):
                for point in series_by_model[model_name]:
                    csv_writer.writerow(
                        [
                            panel_name,
                            model_name,
                            point.snr_db,
                            f"{point.mean_value:.6f}",
                            f"{point.standard_deviation:.6f}",
                            point.seed_count,
                        ]
                    )


def plot_mode(
    logs_root: Path,
    output_directory: Path,
    mode: str,
    metric: str,
    split: str,
    output_formats: Sequence[str],
    show_error_bands: bool,
) -> List[Path]:
    """Build and save the SNR figure and data table for one mode.

    Args:
        logs_root: Directory containing ``snr_TIL`` and ``snr_CIL``.
        output_directory: Directory the figure and csv are written to.
        mode: Either ``"til"`` or ``"cil"``.
        metric: One of ``"f1"``, ``"rec"``, ``"prec"``.
        split: Either ``"val"`` or ``"tr"`` for the performance panel.
        output_formats: Image extensions to save, e.g. ``["png", "pdf"]``.
        show_error_bands: Whether to shade the across-seed standard deviation.

    Returns:
        Paths of the files written.

    Usage:
        >>> plot_mode(Path("logs/00_sync"), Path("analysis/snr_sweep"), "til",
        ...           "f1", "val", ["png"], True)
        [PosixPath('analysis/snr_sweep/snr_sweep_til_f1.png'), ...]
    """
    mode_root = logs_root / f"snr_{mode.upper()}"
    performance_series = collect_series_by_model(mode_root, f"{split}_macro_{metric}")
    backward_transfer_series = collect_series_by_model(mode_root, f"val_bwt_{metric}")
    if not performance_series and not backward_transfer_series:
        print(f"No seed metrics found under {mode_root}", file=sys.stderr)
        return []

    figure = build_mode_figure(
        performance_series=performance_series,
        backward_transfer_series=backward_transfer_series,
        mode=mode,
        metric=metric,
        split=split,
        show_error_bands=show_error_bands,
    )
    output_directory.mkdir(parents=True, exist_ok=True)
    output_stem = output_directory / f"snr_sweep_{mode}_{metric}"

    written_paths: List[Path] = []
    for output_format in output_formats:
        figure_path = output_stem.with_suffix(f".{output_format}")
        figure.savefig(figure_path, dpi=300, bbox_inches="tight")
        written_paths.append(figure_path)
    plt.close(figure)

    csv_path = output_stem.with_suffix(".csv")
    write_series_csv(csv_path, performance_series, backward_transfer_series)
    written_paths.append(csv_path)
    return written_paths


def parse_command_line_arguments(
    argument_values: Optional[Sequence[str]] = None,
) -> argparse.Namespace:
    """Parse command-line arguments for the SNR sweep plot.

    Args:
        argument_values: Argument list, defaulting to ``sys.argv[1:]``.

    Returns:
        The parsed arguments.
    """
    argument_parser = argparse.ArgumentParser(
        description="Plot final performance and BWT against SNR for each model."
    )
    argument_parser.add_argument(
        "--logs-root",
        type=Path,
        default=Path("logs/00_sync"),
        help="Directory containing snr_TIL and snr_CIL.",
    )
    argument_parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("analysis/snr_sweep"),
        help="Directory the figures and csv tables are written to.",
    )
    argument_parser.add_argument(
        "--mode",
        choices=(*SUPPORTED_MODES, "both"),
        default="both",
        help="Which sweep to plot.",
    )
    argument_parser.add_argument(
        "--metric",
        choices=SUPPORTED_METRICS,
        default="f1",
        help="Macro metric shown in both panels.",
    )
    argument_parser.add_argument(
        "--split",
        choices=("val", "tr"),
        default="val",
        help="Split used for the final-performance panel; BWT is always validation.",
    )
    argument_parser.add_argument(
        "--format",
        dest="output_formats",
        default="png",
        help="Comma-separated image formats, for example 'png,pdf'.",
    )
    argument_parser.add_argument(
        "--no-error-bands",
        dest="show_error_bands",
        action="store_false",
        help="Hide the across-seed standard deviation bands.",
    )
    return argument_parser.parse_args(argument_values)


def main(argument_values: Optional[Sequence[str]] = None) -> int:
    """Entry point for the SNR sweep plotting script.

    Args:
        argument_values: Argument list, defaulting to ``sys.argv[1:]``.

    Returns:
        Process exit code.

    Usage:
        >>> main(["--mode", "til"])
        0
    """
    parsed_arguments = parse_command_line_arguments(argument_values)
    output_formats = [
        output_format.strip().lstrip(".")
        for output_format in parsed_arguments.output_formats.split(",")
        if output_format.strip()
    ]
    requested_modes = (
        list(SUPPORTED_MODES)
        if parsed_arguments.mode == "both"
        else [parsed_arguments.mode]
    )

    written_paths: List[Path] = []
    for mode in requested_modes:
        written_paths.extend(
            plot_mode(
                logs_root=parsed_arguments.logs_root,
                output_directory=parsed_arguments.output_dir,
                mode=mode,
                metric=parsed_arguments.metric,
                split=parsed_arguments.split,
                output_formats=output_formats,
                show_error_bands=parsed_arguments.show_error_bands,
            )
        )

    if not written_paths:
        return 1
    for written_path in written_paths:
        print(f"Wrote {written_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
