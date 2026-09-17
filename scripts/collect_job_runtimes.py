#!/usr/bin/env python3
"""Collect per-job runtimes from a full-experiments run directory.

Scans ``job_logs/job_*.log`` under a run directory, extracts each job's total
runtime, and prints jobs sorted slowest to fastest.

Runtime is resolved in this order (same precedence as ``summarise_full_experiments``):
1. ``Total runtime: <hours> hours`` at the end of a completed job log
2. Seconds embedded in the final results dict line (``# val: ... # <seconds>``)
3. Sum of ``Epoch Time <seconds>s`` lines for incomplete or missing final totals

Usage:
    python scripts/collect_job_runtimes.py \\
        logs/full_experiments/run_20260602_164556_lnx-elkk-2_seed_55

    python scripts/collect_job_runtimes.py
"""

from __future__ import annotations

import argparse
import glob
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

TOTAL_RUNTIME_HOURS_RE = re.compile(r"Total runtime:\s+(?P<hours>[0-9.]+)\s+hours")
STATE_TOTAL_RUNTIME_HOURS_RE = re.compile(r"total runtime\s+(?P<hours>[0-9.]+)h")
RESULTS_TIME_RE = re.compile(r"#\s+val:\s+[^#]+#\s+(?P<sec>[0-9.]+)\s*$")
EPOCH_TIME_RE = re.compile(r"Epoch Time\s+(?P<sec>[0-9.]+)s")
JOB_FILENAME_RE = re.compile(r"job_(?P<algo>[\w\-]+)_\d{8}_\d{6}_\d+\.log$")
SECONDS_PER_HOUR = 3600.0


@dataclass(frozen=True)
class JobRuntime:
    """Runtime metadata for a single algorithm job."""

    algorithm_name: str
    runtime_seconds: Optional[float]
    job_log_path: Path


def _find_default_run_dir() -> Path:
    """Return the newest ``run_*`` directory under ``logs/full_experiments/``.

    Returns:
        Path to the newest run directory.

    Raises:
        SystemExit: When no run directories exist.
    """
    run_directories = sorted(glob.glob("logs/full_experiments/run_*"))
    if not run_directories:
        raise SystemExit("No run directories found under logs/full_experiments/.")
    return Path(run_directories[-1])


def _extract_algorithm_name(job_log_path: Path) -> str:
    """Extract the algorithm name from a ``job_*.log`` filename.

    Args:
        job_log_path: Path to a job log file.

    Returns:
        Algorithm/config stem parsed from the filename.
    """
    filename_match = JOB_FILENAME_RE.match(job_log_path.name)
    if filename_match:
        return filename_match.group("algo")
    stem_without_prefix = job_log_path.stem.removeprefix("job_")
    return stem_without_prefix or job_log_path.stem


def _format_runtime_hours(seconds: Optional[float]) -> str:
    """Format a runtime in seconds as hours for display.

    Args:
        seconds: Runtime in seconds, or ``None`` when unavailable.

    Returns:
        Duration in hours, or ``-`` when unavailable.
    """
    if seconds is None:
        return "-"
    return f"{seconds / SECONDS_PER_HOUR:.2f}"


def _parse_job_runtime(job_log_path: Path) -> JobRuntime:
    """Parse runtime information from a single job log.

    Args:
        job_log_path: Path to a ``job_*.log`` file.

    Returns:
        Parsed runtime metadata for that job.
    """
    algorithm_name = _extract_algorithm_name(job_log_path)
    runtime_seconds: Optional[float] = None
    epoch_time_sum = 0.0

    with job_log_path.open("r", encoding="utf-8", errors="replace") as log_file:
        for line in log_file:
            epoch_match = EPOCH_TIME_RE.search(line)
            if epoch_match:
                epoch_time_sum += float(epoch_match.group("sec"))

            results_match = RESULTS_TIME_RE.search(line)
            if results_match:
                runtime_seconds = float(results_match.group("sec"))

            state_match = STATE_TOTAL_RUNTIME_HOURS_RE.search(line)
            if state_match:
                runtime_seconds = float(state_match.group("hours")) * SECONDS_PER_HOUR

            total_match = TOTAL_RUNTIME_HOURS_RE.search(line)
            if total_match:
                runtime_seconds = float(total_match.group("hours")) * SECONDS_PER_HOUR

    if runtime_seconds is None and epoch_time_sum > 0.0:
        runtime_seconds = epoch_time_sum

    return JobRuntime(
        algorithm_name=algorithm_name,
        runtime_seconds=runtime_seconds,
        job_log_path=job_log_path,
    )


def _discover_job_logs(run_dir: Path) -> Sequence[Path]:
    """Discover job logs under a run directory.

    Args:
        run_dir: Full-experiments run directory.

    Returns:
        Sorted list of ``job_*.log`` paths.

    Raises:
        SystemExit: When ``job_logs/`` is missing or empty.
    """
    job_logs_dir = run_dir / "job_logs"
    if not job_logs_dir.is_dir():
        raise SystemExit(f"Missing job_logs/ under run directory: {run_dir}")

    job_logs = sorted(job_logs_dir.glob("job_*.log"))
    if not job_logs:
        raise SystemExit(f"No job_*.log files found in: {job_logs_dir}")
    return job_logs


def collect_job_runtimes(run_dir: Path) -> list[JobRuntime]:
    """Collect runtime metadata for all jobs in a run directory.

    Args:
        run_dir: Full-experiments run directory.

    Returns:
        Job runtime records, unsorted.
    """
    return [
        _parse_job_runtime(job_log_path) for job_log_path in _discover_job_logs(run_dir)
    ]


def _print_runtime_table(job_runtimes: Sequence[JobRuntime], run_dir: Path) -> None:
    """Print runtimes sorted slowest to fastest.

    Args:
        job_runtimes: Parsed job runtime records.
        run_dir: Run directory being summarised.
    """
    sorted_runtimes = sorted(
        job_runtimes,
        key=lambda job: (
            job.runtime_seconds is None,
            -(job.runtime_seconds or 0.0),
            job.algorithm_name,
        ),
    )

    print(f"Run directory: {run_dir}")
    print(f"Jobs found: {len(sorted_runtimes)}")
    print()
    print(f"{'Algorithm':<12}  {'Hours':>8}")
    print("-" * 24)

    known_total_seconds = 0.0
    for job in sorted_runtimes:
        if job.runtime_seconds is not None:
            known_total_seconds += job.runtime_seconds

        print(
            f"{job.algorithm_name:<12}  "
            f"{_format_runtime_hours(job.runtime_seconds):>8}"
        )

    print("-" * 24)
    print(
        "Total serial runtime (known jobs): "
        f"{_format_runtime_hours(known_total_seconds)} hours"
    )


def _parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Collect and rank job runtimes for a full-experiments run."
    )
    parser.add_argument(
        "run_dir",
        nargs="?",
        type=Path,
        default=None,
        help=(
            "Run directory containing job_logs/ "
            "(default: newest logs/full_experiments/run_*)"
        ),
    )
    return parser.parse_args()


def main() -> None:
    """Entry point."""
    args = _parse_args()
    run_dir = args.run_dir if args.run_dir is not None else _find_default_run_dir()
    run_dir = run_dir.resolve()

    if not run_dir.is_dir():
        raise SystemExit(f"Run directory does not exist: {run_dir}")

    job_runtimes = collect_job_runtimes(run_dir)
    _print_runtime_table(job_runtimes, run_dir)


if __name__ == "__main__":
    main()
