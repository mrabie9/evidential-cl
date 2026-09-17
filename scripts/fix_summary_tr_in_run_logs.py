#!/usr/bin/env python3
"""Fix SUMMARY_TR lines in all job logs under a run directory.

LEGACY TOOL. This repairs run logs written *before* detection metrics and the
noise class were removed, so every pattern below deliberately matches the old
``cls_rec=/cls_prec=/cls_f1=/det=/fa=`` summary line and the old ``Det Rec`` /
``Det FA`` epoch line. Current runs emit ``macro_rec=/macro_prec=/macro_f1=``
and are not matched (or modified) by this script.

The script reads each ``job_*.log`` in ``<run_dir>/job_logs`` and replaces the
``SUMMARY_TR ...`` line using per-task *final epoch* training lines:

``T{task} Ep {epoch}/{total} | ... | Train Acc <rec> | Prec <p> | F1 <f> | Det Rec <d> | Det FA <a> | ...``

For each task, the last observed epoch line is kept. Then SUMMARY_TR ``cls_*``
is rebuilt as the mean over tasks, while ``det/fa`` are preserved from the
existing SUMMARY_TR line in the log:
- ``cls_rec = mean(task_final_train_acc)``
- ``cls_prec = mean(task_final_prec)``
- ``cls_f1 = mean(task_final_f1)``
- ``det = existing SUMMARY_TR det`` (unchanged)
- ``fa = existing SUMMARY_TR fa`` (unchanged)

Usage:
    python scripts/fix_summary_tr_in_run_logs.py --run-dir logs/full_experiments/run_20260430_130830_lnx-elkk-2
    python scripts/fix_summary_tr_in_run_logs.py --run-dir logs/full_experiments/full-til_10epochs_w-zs/full-til_A_run_20260403_111257_lnx-elkk-2 --dry-run
"""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path
from typing import List

TASK_EPOCH_METRIC_LINE_RE = re.compile(
    r"T(?P<task>\d+)\s+Ep\s+(?P<epoch>\d+)/(?P<total>\d+)\s+\|\s+"
    r"L\s+[0-9.]+\s+\|\s+Train Acc\s+(?P<cls_rec>[0-9.]+)\s+\|\s+"
    r"Prec\s+(?P<prec>[0-9.]+)\s+\|\s+F1\s+(?P<f1>[0-9.]+)\s+\|\s+"
    r"Det Rec\s+(?P<det>[0-9.]+)\s+\|\s+Det FA\s+(?P<fa>[0-9.]+)\s+\|"
)
SUMMARY_TR_RE = re.compile(
    r"^SUMMARY_TR\s+cls_rec=(?P<cls_rec>[0-9.]+)\s+cls_prec=(?P<cls_prec>[0-9.]+)\s+"
    r"cls_f1=(?P<cls_f1>[0-9.]+)\s+det=(?P<det>[0-9.]+)\s+fa=(?P<fa>[0-9.]+)\s*$"
)


@dataclass
class CorrectionResult:
    """Outcome for one file fix operation.

    Usage:
        >>> isinstance(CorrectionResult, type)
        True
    """

    log_path: Path
    status: str
    old_line: str | None
    new_line: str | None


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fix SUMMARY_TR lines in all job logs under a run directory."
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        required=True,
        help="Run directory containing job_logs/job_*.log files.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be changed without writing files.",
    )
    return parser.parse_args()


def _build_cls_summary_from_task_final_averages(
    log_text: str,
) -> tuple[float, float, float] | None:
    """Build a corrected SUMMARY_TR line from task-final metric averages.

    Args:
        log_text: Full text content of a job log.

    Returns:
        Tuple ``(cls_rec, cls_prec, cls_f1)``, or ``None`` if no per-task metric lines exist.

    Usage:
        >>> isinstance(_build_cls_summary_from_task_final_averages(""), (tuple, type(None)))
        True
    """
    last_by_task: dict[int, re.Match[str]] = {}
    for match in TASK_EPOCH_METRIC_LINE_RE.finditer(log_text):
        task_index = int(match.group("task"))
        last_by_task[task_index] = match
    if not last_by_task:
        return None

    cls_rec_values: List[float] = []
    cls_prec_values: List[float] = []
    cls_f1_values: List[float] = []
    for task_index in sorted(last_by_task):
        task_match = last_by_task[task_index]
        cls_rec_values.append(float(task_match.group("cls_rec")))
        cls_prec_values.append(float(task_match.group("prec")))
        cls_f1_values.append(float(task_match.group("f1")))

    cls_rec = sum(cls_rec_values) / len(cls_rec_values)
    cls_prec = sum(cls_prec_values) / len(cls_prec_values)
    cls_f1 = sum(cls_f1_values) / len(cls_f1_values)
    return cls_rec, cls_prec, cls_f1


def _fix_one_log(log_path: Path, dry_run: bool) -> CorrectionResult:
    """Fix the SUMMARY_TR line in one job log file.

    Args:
        log_path: Path to one ``job_*.log`` file.
        dry_run: If True, do not write; only report potential change.

    Returns:
        CorrectionResult with status in {updated, unchanged, skipped_no_epoch, skipped_no_summary}.

    Usage:
        >>> isinstance(_fix_one_log, object)
        True
    """

    log_text = log_path.read_text(encoding="utf-8", errors="replace")
    cls_summary = _build_cls_summary_from_task_final_averages(log_text)
    if cls_summary is None:
        return CorrectionResult(
            log_path=log_path,
            status="skipped_no_epoch",
            old_line=None,
            new_line=None,
        )

    lines = log_text.splitlines()
    summary_line_index = None
    existing_summary_match = None
    for line_index, line_text in enumerate(lines):
        summary_match = SUMMARY_TR_RE.match(line_text)
        if summary_match:
            summary_line_index = line_index
            existing_summary_match = summary_match
    if summary_line_index is None:
        return CorrectionResult(
            log_path=log_path,
            status="skipped_no_summary",
            old_line=None,
            new_line=None,
        )
    if existing_summary_match is None:
        return CorrectionResult(
            log_path=log_path,
            status="skipped_no_summary",
            old_line=None,
            new_line=None,
        )

    cls_rec, cls_prec, cls_f1 = cls_summary
    det_existing = float(existing_summary_match.group("det"))
    fa_existing = float(existing_summary_match.group("fa"))
    new_summary_line = (
        f"SUMMARY_TR cls_rec={cls_rec:.4f} cls_prec={cls_prec:.4f} "
        f"cls_f1={cls_f1:.4f} det={det_existing:.4f} fa={fa_existing:.4f}"
    )

    old_line = lines[summary_line_index]
    if old_line == new_summary_line:
        return CorrectionResult(
            log_path=log_path,
            status="unchanged",
            old_line=old_line,
            new_line=new_summary_line,
        )

    lines[summary_line_index] = new_summary_line
    if not dry_run:
        log_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return CorrectionResult(
        log_path=log_path,
        status="updated",
        old_line=old_line,
        new_line=new_summary_line,
    )


def _collect_job_logs(run_dir: Path) -> List[Path]:
    """Collect all job logs inside a run directory.

    Args:
        run_dir: Directory containing a ``job_logs`` folder.

    Returns:
        Sorted list of ``job_*.log`` paths.

    Raises:
        SystemExit: If directory structure is invalid.

    Usage:
        >>> isinstance(_collect_job_logs, object)
        True
    """

    resolved_run_dir = run_dir.resolve()
    job_logs_dir = resolved_run_dir / "job_logs"
    if not job_logs_dir.is_dir():
        raise SystemExit(f"--run-dir missing job_logs/: {resolved_run_dir}")
    job_logs = sorted(job_logs_dir.glob("job_*.log"))
    if not job_logs:
        raise SystemExit(f"No job_*.log files found under: {job_logs_dir}")
    return job_logs


def main() -> None:
    args = _parse_args()
    job_logs = _collect_job_logs(args.run_dir)

    print(f"Processing {len(job_logs)} logs under {args.run_dir.resolve()}")
    if args.dry_run:
        print("Mode: dry-run (no files will be modified)")

    results: List[CorrectionResult] = []
    for log_path in job_logs:
        results.append(_fix_one_log(log_path, dry_run=args.dry_run))

    updated = [result for result in results if result.status == "updated"]
    unchanged = [result for result in results if result.status == "unchanged"]
    skipped_no_epoch = [
        result for result in results if result.status == "skipped_no_epoch"
    ]
    skipped_no_summary = [
        result for result in results if result.status == "skipped_no_summary"
    ]

    print(f"Updated: {len(updated)}")
    print(f"Unchanged: {len(unchanged)}")
    print(f"Skipped (no epoch-complete metrics): {len(skipped_no_epoch)}")
    print(f"Skipped (no SUMMARY_TR line): {len(skipped_no_summary)}")

    for result in updated:
        print(f"\n[{result.log_path.name}]")
        print(f"- old: {result.old_line}")
        print(f"- new: {result.new_line}")

    if skipped_no_summary:
        print("\nLogs skipped due to missing SUMMARY_TR:")
        for result in skipped_no_summary:
            print(f"- {result.log_path}")


if __name__ == "__main__":
    main()
