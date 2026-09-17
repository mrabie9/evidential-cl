#!/usr/bin/env python
"""Summarise per-model storage footprint from a run directory.

Scans the ``job_<model>_*.log`` files under one or more run directories and
reports, for each model (algorithm):

- model size (trainable parameters), in GB
- memory/replay buffer size, in GB
- total size (model + buffer), in GB

Sizes are read from the lines that ``save_results`` in ``main.py`` writes to
each job log::

    Model size: 0.0289 GB
    Memory buffer size: 0.0196 GB

and, as a fallback, from the results one-liner suffix
``# sizes: model_gb=0.0289 mem_gb=0.0196``.

When runs include the richer per-category breakdown (written by newer
``save_results`` runs)::

    Persistent state sizes (GB): model_params=0.0289 replay=0.0196 \
        regularization=0.0000 arch_mask=0.0000 other=0.0000 total=0.0486

the ``--breakdown`` flag prints those categories instead of the single buffer
column. The breakdown is also parsed from the one-liner suffix
``# state_gb: model_params=... total=...``. Note that the legacy
``Memory buffer size`` column only counts tensor attributes named ``*mem*``, so
it under-reports replay lists (La-MAML family), Fisher/SI regularization
(EWC, RWalk, SI), masks/snapshots (PackNet, LWF, UCL); use ``--breakdown`` for a
fair comparison.

A "run directory" may be any folder under ``logs/`` that contains job logs at
any depth, e.g. a per-seed folder such as
``logs/full_experiments/one-shot_til/seed-0`` (which holds several host-sharded
coordinator runs), or a single coordinator run directory.

Usage:
    python scripts/summarise_model_sizes.py logs/full_experiments/one-shot_til/seed-0
    python scripts/summarise_model_sizes.py --breakdown RUN_DIR
    python scripts/summarise_model_sizes.py --output-format markdown RUN_DIR
    python scripts/summarise_model_sizes.py RUN_DIR_A RUN_DIR_B
"""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

MODEL_SIZE_RE = re.compile(r"Model size:\s+(?P<gb>[0-9.]+)\s+GB")
MEMORY_BUFFER_SIZE_RE = re.compile(r"Memory buffer size:\s+(?P<gb>[0-9.]+)\s+GB")
SIZES_ONE_LINER_RE = re.compile(
    r"#\s+sizes:\s+model_gb=(?P<model_gb>[0-9.]+)\s+mem_gb=(?P<mem_gb>[0-9.]+)"
)
PERSISTENT_STATE_LINE_RE = re.compile(r"Persistent state sizes \(GB\):\s*(?P<pairs>.+)")
PERSISTENT_STATE_ONE_LINER_RE = re.compile(r"#\s+state_gb:\s*(?P<pairs>[^#]+)")
STATE_PAIR_RE = re.compile(r"(?P<key>[A-Za-z_]+)=(?P<value>[0-9.]+)")
JOB_LOG_NAME_RE = re.compile(r"job_(?P<model>[\w\-]+)_\d{8}_\d{6}_\d+\.log")

BYTES_PER_GB = 1024**3

# Persistent-state breakdown categories, in display order (must match
# ``PERSISTENT_STATE_CATEGORIES`` in ``main.py``). ``total`` is derived.
BREAKDOWN_CATEGORIES = (
    "model_params",
    "replay",
    "regularization",
    "arch_mask",
    "other",
)
BREAKDOWN_COLUMN_LABELS = {
    "model_params": "Model",
    "replay": "Replay",
    "regularization": "Reg",
    "arch_mask": "Arch",
    "other": "Other",
}


@dataclass
class ModelSizeSummary:
    """Storage footprint parsed for a single model.

    Attributes:
        name: Model/algorithm name (e.g. ``gem``).
        model_size_gb: Trainable-parameter size in GB, if found.
        memory_buffer_size_gb: Replay/memory buffer size in GB, if found.
        breakdown_gb: Per-category persistent-state sizes in GB, when the run
            logged the richer ``Persistent state sizes (GB)`` line. Keys are a
            subset of :data:`BREAKDOWN_CATEGORIES`.
        source_job_log: Path to the job log the sizes were read from.
    """

    name: str
    model_size_gb: Optional[float] = None
    memory_buffer_size_gb: Optional[float] = None
    breakdown_gb: Optional[dict[str, float]] = None
    source_job_log: Optional[str] = None

    @property
    def total_size_gb(self) -> Optional[float]:
        """Return total persistent size in GB, or None if nothing was found.

        Prefers the sum of the per-category breakdown when available; otherwise
        falls back to model + memory-buffer size (treating a missing component
        as zero).

        Returns:
            Combined size in GB, or None when no size information was found.
        """
        if self.breakdown_gb:
            return sum(self.breakdown_gb.values())
        if self.model_size_gb is None and self.memory_buffer_size_gb is None:
            return None
        return (self.model_size_gb or 0.0) + (self.memory_buffer_size_gb or 0.0)


def _parse_state_pairs(pairs_text: str) -> dict[str, float]:
    """Parse ``key=value`` size pairs from a breakdown fragment.

    Args:
        pairs_text: Text such as ``"model_params=0.0289 replay=0.0196 total=0.0486"``.

    Returns:
        Mapping of each recognised :data:`BREAKDOWN_CATEGORIES` key to its GB
        value. The derived ``total`` key is ignored.
    """
    parsed_breakdown: dict[str, float] = {}
    for pair_match in STATE_PAIR_RE.finditer(pairs_text):
        key = pair_match.group("key")
        if key in BREAKDOWN_CATEGORIES:
            parsed_breakdown[key] = float(pair_match.group("value"))
    return parsed_breakdown


def parse_job_log_sizes(
    job_log_path: Path,
) -> tuple[Optional[float], Optional[float], Optional[dict[str, float]]]:
    """Parse model size, memory-buffer size, and the category breakdown.

    Prefers the explicit ``Model size`` / ``Memory buffer size`` /
    ``Persistent state sizes (GB)`` lines, falling back to the results one-liner
    suffixes (``# sizes: ...`` and ``# state_gb: ...``) when those lines are
    absent.

    Args:
        job_log_path: Path to a ``job_<model>_*.log`` file.

    Returns:
        Tuple of ``(model_size_gb, memory_buffer_size_gb, breakdown_gb)``; any
        element may be None when not present in the log.

    Usage:
        model_gb, buffer_gb, breakdown = parse_job_log_sizes(Path("job_gem_....log"))
    """
    model_size_gb: Optional[float] = None
    memory_buffer_size_gb: Optional[float] = None
    breakdown_gb: Optional[dict[str, float]] = None

    with open(job_log_path, "r", encoding="utf-8", errors="replace") as file_handle:
        for line in file_handle:
            model_match = MODEL_SIZE_RE.search(line)
            if model_match:
                model_size_gb = float(model_match.group("gb"))
                continue

            buffer_match = MEMORY_BUFFER_SIZE_RE.search(line)
            if buffer_match:
                memory_buffer_size_gb = float(buffer_match.group("gb"))
                continue

            state_match = PERSISTENT_STATE_LINE_RE.search(line)
            if state_match:
                parsed_breakdown = _parse_state_pairs(state_match.group("pairs"))
                if parsed_breakdown:
                    breakdown_gb = parsed_breakdown
                continue

            if breakdown_gb is None:
                state_one_liner_match = PERSISTENT_STATE_ONE_LINER_RE.search(line)
                if state_one_liner_match:
                    parsed_breakdown = _parse_state_pairs(
                        state_one_liner_match.group("pairs")
                    )
                    if parsed_breakdown:
                        breakdown_gb = parsed_breakdown

            if model_size_gb is None or memory_buffer_size_gb is None:
                one_liner_match = SIZES_ONE_LINER_RE.search(line)
                if one_liner_match:
                    if model_size_gb is None:
                        model_size_gb = float(one_liner_match.group("model_gb"))
                    if memory_buffer_size_gb is None:
                        memory_buffer_size_gb = float(one_liner_match.group("mem_gb"))

    return model_size_gb, memory_buffer_size_gb, breakdown_gb


def _model_name_from_job_log(job_log_path: Path) -> Optional[str]:
    """Extract the model name from a job log file name.

    Args:
        job_log_path: Path to a ``job_<model>_*.log`` file.

    Returns:
        The parsed model name, or None when the file name does not match.
    """
    match = JOB_LOG_NAME_RE.match(job_log_path.name)
    if match:
        return match.group("model")
    return None


def collect_model_sizes(run_directories: List[Path]) -> Dict[str, ModelSizeSummary]:
    """Collect per-model sizes from job logs under the given run directories.

    Job logs are discovered recursively. When the same model appears in more
    than one job log (e.g. host-sharded coordinator runs), the first log that
    reports a model size wins; remaining logs only fill in missing fields.

    Args:
        run_directories: Run directories (or any parent folders) to scan.

    Returns:
        Mapping from model name to its :class:`ModelSizeSummary`.
    """
    summaries: Dict[str, ModelSizeSummary] = {}

    for run_directory in run_directories:
        for job_log_path in sorted(run_directory.rglob("job_*.log")):
            model_name = _model_name_from_job_log(job_log_path)
            if model_name is None:
                continue

            model_size_gb, memory_buffer_size_gb, breakdown_gb = parse_job_log_sizes(
                job_log_path
            )
            if (
                model_size_gb is None
                and memory_buffer_size_gb is None
                and breakdown_gb is None
            ):
                continue

            summary = summaries.setdefault(
                model_name, ModelSizeSummary(name=model_name)
            )
            if summary.model_size_gb is None and model_size_gb is not None:
                summary.model_size_gb = model_size_gb
                summary.source_job_log = str(job_log_path)
            if (
                summary.memory_buffer_size_gb is None
                and memory_buffer_size_gb is not None
            ):
                summary.memory_buffer_size_gb = memory_buffer_size_gb
                if summary.source_job_log is None:
                    summary.source_job_log = str(job_log_path)
            if summary.breakdown_gb is None and breakdown_gb is not None:
                summary.breakdown_gb = breakdown_gb
                if summary.source_job_log is None:
                    summary.source_job_log = str(job_log_path)

    return summaries


def _format_size(size_gb: Optional[float], unit: str) -> str:
    """Format a size value in the requested unit.

    Args:
        size_gb: Size in GB, or None when unknown.
        unit: Either ``"GB"`` or ``"MB"``.

    Returns:
        Formatted size string, or ``"-"`` when the size is None.
    """
    if size_gb is None:
        return "-"
    if unit == "MB":
        return f"{size_gb * 1024:.2f}"
    return f"{size_gb:.4f}"


def _legacy_total_size_gb(summary: ModelSizeSummary) -> Optional[float]:
    """Return model + memory-buffer size in GB for the legacy two-column view.

    This keeps the default table self-consistent (``Total = Model + Buffer``),
    independent of any richer per-category breakdown.

    Args:
        summary: Per-model size summary.

    Returns:
        Combined model + buffer size in GB, or None when both are missing.
    """
    if summary.model_size_gb is None and summary.memory_buffer_size_gb is None:
        return None
    return (summary.model_size_gb or 0.0) + (summary.memory_buffer_size_gb or 0.0)


def _category_size_gb(summary: ModelSizeSummary, category: str) -> Optional[float]:
    """Return a model's size for one breakdown category in GB.

    When the run logged a full breakdown, the stored category value is used.
    Otherwise the legacy two-number format is mapped on a best-effort basis:
    ``model_params`` from the model size and ``replay`` from the memory buffer
    size; remaining categories are unknown.

    Args:
        summary: Per-model size summary.
        category: One of :data:`BREAKDOWN_CATEGORIES`.

    Returns:
        Size in GB, or None when the value is unavailable for this model.
    """
    if summary.breakdown_gb is not None:
        return summary.breakdown_gb.get(category)
    if category == "model_params":
        return summary.model_size_gb
    if category == "replay":
        return summary.memory_buffer_size_gb
    return None


def _print_breakdown_summary(
    sorted_summaries: List[ModelSizeSummary],
    unit: str,
    output_format: str,
) -> None:
    """Print a per-category persistent-state table.

    Args:
        sorted_summaries: Summaries already sorted for display.
        unit: Display unit, ``"GB"`` or ``"MB"``.
        output_format: ``"readable"`` or ``"markdown"`` table style.

    Returns:
        None.
    """
    category_headers = [
        f"{BREAKDOWN_COLUMN_LABELS[category]} ({unit})"
        for category in BREAKDOWN_CATEGORIES
    ]
    total_header = f"Total ({unit})"

    if output_format == "markdown":
        header_columns = ["Model"] + category_headers + [total_header]
        print("| " + " | ".join(header_columns) + " |")
        print("| " + " | ".join(["---"] * len(header_columns)) + " |")
        for summary in sorted_summaries:
            row_values = [summary.name]
            row_values += [
                _format_size(_category_size_gb(summary, category), unit)
                for category in BREAKDOWN_CATEGORIES
            ]
            row_values.append(_format_size(summary.total_size_gb, unit))
            print("| " + " | ".join(row_values) + " |")
        return

    width_name = max(len(summary.name) for summary in sorted_summaries)
    width_name = max(width_name, len("Model"))
    width_num = max(max(len(header) for header in category_headers), 10)

    header_cells = [f"{'Model':<{width_name}}"]
    header_cells += [f"{header:>{width_num}}" for header in category_headers]
    header_cells.append(f"{total_header:>{width_num}}")
    header = " ".join(header_cells)
    print(header)
    print("-" * len(header))
    for summary in sorted_summaries:
        row_cells = [f"{summary.name:<{width_name}}"]
        row_cells += [
            f"{_format_size(_category_size_gb(summary, category), unit):>{width_num}}"
            for category in BREAKDOWN_CATEGORIES
        ]
        row_cells.append(f"{_format_size(summary.total_size_gb, unit):>{width_num}}")
        print(" ".join(row_cells))


def print_summary(
    summaries: Dict[str, ModelSizeSummary],
    unit: str = "GB",
    output_format: str = "readable",
    show_breakdown: bool = False,
) -> None:
    """Print a per-model size table sorted by total size (descending).

    Args:
        summaries: Per-model size summaries to display.
        unit: Display unit, ``"GB"`` or ``"MB"``.
        output_format: ``"readable"`` or ``"markdown"`` table style.
        show_breakdown: When True, print one column per persistent-state
            category (model_params, replay, regularization, arch_mask, other)
            instead of the single memory-buffer column.

    Returns:
        None.
    """
    if not summaries:
        print("No model sizes found in the provided run directories.")
        return

    sorted_summaries = sorted(
        summaries.values(),
        key=lambda summary: (
            summary.total_size_gb is None,
            -(summary.total_size_gb or 0.0),
            summary.name,
        ),
    )

    if show_breakdown:
        if not any(summary.breakdown_gb for summary in sorted_summaries):
            print(
                "No per-category breakdown found in these runs "
                "(they predate the 'Persistent state sizes' logging). "
                "Showing model/replay from the legacy size lines; other "
                "categories are unknown."
            )
        _print_breakdown_summary(sorted_summaries, unit, output_format)
        return

    model_header = f"Model ({unit})"
    buffer_header = f"Buffer ({unit})"
    total_header = f"Total ({unit})"

    if output_format == "markdown":
        header_columns = ["Model", model_header, buffer_header, total_header]
        print("| " + " | ".join(header_columns) + " |")
        print("| " + " | ".join(["---"] * len(header_columns)) + " |")
        for summary in sorted_summaries:
            row_values = [
                summary.name,
                _format_size(summary.model_size_gb, unit),
                _format_size(summary.memory_buffer_size_gb, unit),
                _format_size(_legacy_total_size_gb(summary), unit),
            ]
            print("| " + " | ".join(row_values) + " |")
        return

    width_name = max(len(summary.name) for summary in sorted_summaries)
    width_name = max(width_name, len("Model"))
    width_num = max(len(model_header), len(buffer_header), len(total_header), 10)

    header = (
        f"{'Model':<{width_name}} "
        f"{model_header:>{width_num}} "
        f"{buffer_header:>{width_num}} "
        f"{total_header:>{width_num}}"
    )
    print(header)
    print("-" * len(header))
    for summary in sorted_summaries:
        print(
            f"{summary.name:<{width_name}} "
            f"{_format_size(summary.model_size_gb, unit):>{width_num}} "
            f"{_format_size(summary.memory_buffer_size_gb, unit):>{width_num}} "
            f"{_format_size(_legacy_total_size_gb(summary), unit):>{width_num}}"
        )


def _parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Parsed arguments namespace.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Summarise per-model model size, memory buffer size, and total size "
            "(model + buffer) from the job logs in a run directory."
        )
    )
    parser.add_argument(
        "run_directories",
        type=Path,
        nargs="+",
        metavar="RUN_DIR",
        help=(
            "One or more run directories to scan for job_<model>_*.log files "
            "(e.g. logs/full_experiments/one-shot_til/seed-0). Searched recursively."
        ),
    )
    parser.add_argument(
        "--unit",
        choices=("GB", "MB"),
        default="GB",
        help="Display unit for sizes (default: GB).",
    )
    parser.add_argument(
        "--output-format",
        choices=("readable", "markdown"),
        default="readable",
        help="Output format for the size table (default: readable).",
    )
    parser.add_argument(
        "--breakdown",
        action="store_true",
        help=(
            "Show one column per persistent-state category (model_params, "
            "replay, regularization, arch_mask, other) instead of the single "
            "memory-buffer column. Requires runs that logged 'Persistent state "
            "sizes (GB)'; older runs fall back to model/replay only."
        ),
    )
    return parser.parse_args()


def main() -> int:
    """Entry point for the size summary CLI.

    Returns:
        Process exit code (0 on success, 1 when no sizes were found).
    """
    arguments = _parse_arguments()

    missing_directories = [
        run_directory
        for run_directory in arguments.run_directories
        if not run_directory.is_dir()
    ]
    if missing_directories:
        joined = ", ".join(str(path) for path in missing_directories)
        raise SystemExit(f"Not a directory: {joined}")

    summaries = collect_model_sizes(arguments.run_directories)
    print_summary(
        summaries,
        unit=arguments.unit,
        output_format=arguments.output_format,
        show_breakdown=arguments.breakdown,
    )
    return 0 if summaries else 1


if __name__ == "__main__":
    raise SystemExit(main())
