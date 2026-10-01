#!/usr/bin/env python3
"""Rescale stored BWT so it averages over every task but the last.

``metrics.transfer_stats`` used to average backward transfer over all T tasks,
including the last one, whose BWT is always exactly 0 (nothing was trained
after it). BWT now averages over the first T-1 tasks only, so every stored
value is the old one times ``T / (T - 1)``.

For each seed dir under ``--root`` without the ``bwt_excludes_last_task``
marker, this rewrites:

* ``seed_metrics.json``: ``val_bwt_{rec,prec,f1}``, then sets the marker;
* the seed's ``results.txt``: the ``Backward`` lines, the ``bwt`` column of
  the validation summary table and its footer;
* the run-level ``results.txt``: the three ``Validation BWT`` sweep lines.

Every rescaled F1 BWT is cross-checked against the F1 task matrix printed in
the seed's ``results.txt``; a seed that disagrees is skipped. Any directory
named in ``--exclude`` (default ``00_old`` and ``01_old``) is left alone.
Without ``--apply`` nothing is written.

Usage:
    python scripts/backfill_bwt_excl_last_task.py
    python scripts/backfill_bwt_excl_last_task.py --apply
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from collections import defaultdict
from pathlib import Path

MARKER_KEY = "bwt_excludes_last_task"
METRIC_SUFFIXES = ("rec", "prec", "f1")
SEED_BACKWARD_PREFIXES = {
    "Backward:": "rec",
    "Backward Precision:": "prec",
    "Backward F1:": "f1",
}
SUMMARY_ROW_LABELS = {"recall": "rec", "precision": "prec", "f1": "f1"}
RUN_SUMMARY_LABELS = {
    "Validation BWT rec": "rec",
    "Validation BWT prec": "prec",
    "Validation BWT f1": "f1",
}
OLD_FOOTER_FRAGMENT = "final = mean of the last row (bwt = final - diagonal); "
NEW_FOOTER_FRAGMENT = (
    "final = mean of the last row; bwt = mean of (last row - diagonal) over all "
    "tasks but the last; "
)
CROSS_CHECK_TOLERANCE = 5e-4


def find_seed_metrics_files(root: Path, excluded_names: set[str]) -> list[Path]:
    """Return every ``seed_metrics.json`` under ``root`` outside excluded dirs."""
    return sorted(
        path
        for path in root.rglob("seed_metrics.json")
        if not excluded_names.intersection(path.relative_to(root).parts)
    )


def parse_matrix_block(lines: list[str], header_index: int) -> list[list[float]]:
    """Parse the task-matrix rows following the ``|`` separator after ``header_index``."""
    separator_index = lines.index("|", header_index)
    rows: list[list[float]] = []
    for line in lines[separator_index + 1 :]:
        if not line.strip() or not re.match(r"^-?\d", line):
            break
        rows.append([float(value) for value in line.split()])
    return rows


def f1_matrix_from_results(results_text: str) -> list[list[float]] | None:
    """Return the per-task F1 matrix printed in a seed's ``results.txt``, if any."""
    lines = results_text.splitlines()
    header_index = next(
        (index for index, line in enumerate(lines) if line.startswith("F1 (")),
        None,
    )
    if header_index is None:
        return None
    return parse_matrix_block(lines, header_index)


def bwt_excluding_last_task(matrix: list[list[float]]) -> float:
    """Mean of (last row - diagonal) over every task but the last."""
    task_count = len(matrix)
    last_row = matrix[-1]
    gaps = [last_row[task] - matrix[task][task] for task in range(task_count - 1)]
    return sum(gaps) / len(gaps)


def count_tasks(seed_dir: Path, results_text: str | None) -> int | None:
    """Task count from the recall matrix in ``results.txt``, else the metrics npz files.

    Returns None when both sources exist and disagree.
    """
    npz_count = len(list((seed_dir / "metrics").glob("task*.npz")))
    if results_text is None or "|" not in results_text.splitlines():
        return npz_count or None
    matrix_count = len(parse_matrix_block(results_text.splitlines(), 0))
    if npz_count and npz_count != matrix_count:
        return None
    return matrix_count


def rescale_payload(payload: dict, scale: float) -> dict:
    """Return a copy of ``payload`` with numeric ``val_bwt_*`` scaled and the marker set."""
    updated = dict(payload)
    for suffix in METRIC_SUFFIXES:
        value = updated.get("val_bwt_" + suffix)
        if isinstance(value, (int, float)):
            updated["val_bwt_" + suffix] = value * scale
    updated[MARKER_KEY] = True
    return updated


def rewrite_seed_results(results_text: str, payload: dict) -> str:
    """Swap the BWT values in a seed's ``results.txt`` for the rescaled ones."""
    rewritten: list[str] = []
    in_summary = False
    for line in results_text.splitlines():
        rewritten.append(_rewrite_seed_line(line, payload, in_summary))
        if line.startswith("Summary (validation):"):
            in_summary = True
    return "\n".join(rewritten) + ("\n" if results_text.endswith("\n") else "")


def _rewrite_seed_line(line: str, payload: dict, in_summary: bool) -> str:
    """Rewrite one ``results.txt`` line if it carries a BWT value."""
    for prefix, suffix in SEED_BACKWARD_PREFIXES.items():
        value = payload.get("val_bwt_" + suffix)
        if line.startswith(prefix) and isinstance(value, (int, float)):
            return "%s %.4f" % (prefix, value)
    if line.startswith(OLD_FOOTER_FRAGMENT):
        return NEW_FOOTER_FRAGMENT + line[len(OLD_FOOTER_FRAGMENT) :]
    tokens = line.split()
    if not in_summary or len(tokens) != 6 or tokens[0] not in SUMMARY_ROW_LABELS:
        return line
    value = payload.get("val_bwt_" + SUMMARY_ROW_LABELS[tokens[0]])
    if not isinstance(value, (int, float)):
        return line
    return "{:<10} {:>8} {:>8} {:>8.4f} {:>8} {:>8}".format(
        tokens[0], tokens[1], tokens[2], value, tokens[4], tokens[5]
    )


def rewrite_run_summary(summary_text: str, payload_by_seed: dict[str, dict]) -> str:
    """Recompute the ``Validation BWT`` mean/std lines of a run-level ``results.txt``."""
    lines = summary_text.splitlines()
    seeds_line = next((line for line in lines if line.startswith("Seeds:")), None)
    if seeds_line is None:
        return summary_text
    seeds = [seed.strip() for seed in seeds_line.split(":", 1)[1].split(",")]
    rewritten = []
    for line in lines:
        label = line.split("(n=")[0].strip()
        suffix = RUN_SUMMARY_LABELS.get(label)
        if suffix is None:
            rewritten.append(line)
            continue
        values = [
            payload_by_seed.get(seed, {}).get("val_bwt_" + suffix) for seed in seeds
        ]
        rewritten.append(
            format_summary_line(label, [v for v in values if _is_number(v)]) or line
        )
    return "\n".join(rewritten) + ("\n" if summary_text.endswith("\n") else "")


def _is_number(value: object) -> bool:
    """True for ints/floats, matching ``main._write_sweep_summary``'s filter."""
    return isinstance(value, (int, float))


def format_summary_line(label: str, values: list[float]) -> str | None:
    """Format a sweep line exactly as ``main._write_sweep_summary`` does."""
    if not values:
        return None
    mean = sum(values) / len(values)
    std = 0.0
    if len(values) > 1:
        std = math.sqrt(sum((v - mean) ** 2 for v in values) / (len(values) - 1))
    per_seed = ", ".join("{:.4f}".format(v) for v in values)
    return "  {:<22} (n={}): {:.4f} +/- {:.4f}   [{}]".format(
        label, len(values), mean, std, per_seed
    )


def plan_seed_update(seed_metrics_path: Path) -> tuple[dict, str | None] | str:
    """Work out one seed's new payload and results text, or return a skip reason."""
    payload = json.loads(seed_metrics_path.read_text(encoding="utf-8"))
    if payload.get(MARKER_KEY):
        return "already migrated"
    results_path = seed_metrics_path.parent / "results.txt"
    results_text = (
        results_path.read_text(encoding="utf-8") if results_path.is_file() else None
    )
    task_count = count_tasks(seed_metrics_path.parent, results_text)
    if task_count is None or task_count < 2:
        return f"unusable task count ({task_count})"
    updated = rescale_payload(payload, task_count / (task_count - 1))
    if results_text is None:
        return updated, None
    f1_matrix = f1_matrix_from_results(results_text)
    new_f1 = updated.get("val_bwt_f1")
    if f1_matrix and _is_number(new_f1):
        expected = bwt_excluding_last_task(f1_matrix)
        if abs(expected - new_f1) > CROSS_CHECK_TOLERANCE:
            return f"F1 cross-check failed ({new_f1:.4f} vs matrix {expected:.4f})"
    return updated, rewrite_seed_results(results_text, updated)


def migrate(root: Path, excluded_names: set[str], apply: bool) -> int:
    """Rescale every unmigrated seed under ``root``; return the number of skips."""
    seeds_by_run: dict[Path, dict[str, dict]] = defaultdict(dict)
    updated_count = skipped_count = 0
    for seed_metrics_path in find_seed_metrics_files(root, excluded_names):
        seed_dir = seed_metrics_path.parent
        plan = plan_seed_update(seed_metrics_path)
        if isinstance(plan, str):
            if plan != "already migrated":
                skipped_count += 1
                print(f"[SKIP] {seed_dir}: {plan}")
            continue
        payload, results_text = plan
        seeds_by_run[seed_dir.parent][seed_dir.name] = payload
        updated_count += 1
        if apply:
            seed_metrics_path.write_text(json.dumps(payload, indent=2))
            if results_text is not None:
                (seed_dir / "results.txt").write_text(results_text, encoding="utf-8")
    run_count = rewrite_run_summaries(seeds_by_run, apply)
    mode = "Updated" if apply else "Would update"
    print(f"{mode} {updated_count} seeds and {run_count} run summaries.")
    print(f"Skipped {skipped_count} seeds.")
    return skipped_count


def rewrite_run_summaries(
    seeds_by_run: dict[Path, dict[str, dict]], apply: bool
) -> int:
    """Rewrite run-level summaries for runs with updated seeds; return the count."""
    run_count = 0
    for run_dir, updated_payloads in seeds_by_run.items():
        summary_path = run_dir / "results.txt"
        if not summary_path.is_file():
            continue
        payload_by_seed = {
            path.parent.name: json.loads(path.read_text(encoding="utf-8"))
            for path in run_dir.glob("*/seed_metrics.json")
        }
        payload_by_seed.update(updated_payloads)
        old_text = summary_path.read_text(encoding="utf-8")
        new_text = rewrite_run_summary(old_text, payload_by_seed)
        if new_text != old_text:
            run_count += 1
            if apply:
                summary_path.write_text(new_text, encoding="utf-8")
    return run_count


def main(argv: list[str] | None = None) -> int:
    """Parse arguments and run the migration (dry run unless ``--apply``)."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--root", type=Path, default=Path("logs/00_sync"))
    parser.add_argument(
        "--exclude",
        default="00_old,01_old",
        help="Comma-separated directory names to leave untouched.",
    )
    parser.add_argument("--apply", action="store_true", help="Write the changes.")
    args = parser.parse_args(argv)
    excluded_names = {name.strip() for name in args.exclude.split(",") if name}
    migrate(args.root, excluded_names, args.apply)
    return 0


if __name__ == "__main__":
    sys.exit(main())
