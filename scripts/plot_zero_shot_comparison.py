#!/usr/bin/env python3
"""Plot baseline and CL zero-shot metrics on the same figure.

This script compares:
- a baseline zero-shot JSON file (for example ``zs_baseline.json``), and
- a CL zero-shot JSON file (for example ``zs_val_metrics.json``).

Both files are expected to be JSON arrays of row dictionaries. The script is
robust to common key variants used in this repository:
- task index: ``task`` or ``task_index``
- algorithm: ``algo``
- metrics: ``macro_f1`` / ``macro_rec`` / ``macro_prec`` (pre-removal spellings
  such as ``f1_cls`` / ``zero_shot_f1_cls`` are still accepted)

Usage:
    python scripts/plot_zero_shot_comparison.py \
        --baseline zs_baseline.json \
        --cl logs/full_experiments/run_20260403_111437_lnx-elkk-1/zs_val_metrics.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import matplotlib.pyplot as plt

Row = Dict[str, Any]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Overlay baseline and CL zero-shot metrics by task."
    )
    parser.add_argument(
        "--baseline",
        type=Path,
        default=Path("zs_baseline.json"),
        help="Path to baseline JSON metrics file.",
    )
    parser.add_argument(
        "--cl",
        type=Path,
        default=Path(
            "logs/full_experiments/run_20260403_111437_lnx-elkk-1/zs_val_metrics.json"
        ),
        help="Path to CL zero-shot JSON metrics file.",
    )
    parser.add_argument(
        "--algo",
        type=str,
        default=None,
        help="Optional single CL algorithm filter. Baseline is still plotted.",
    )
    parser.add_argument(
        "--metric",
        choices=("macro_f1", "macro_rec", "macro_prec", "total_macro_f1_zs"),
        default="macro_f1",
        help="Metric to plot.",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("plots/zero_shot_baseline_vs_cl.png"),
        help="Output image path. Parent directory is created automatically.",
    )
    return parser.parse_args()


def _read_json_rows(path: Path) -> List[Row]:
    if not path.is_file():
        raise FileNotFoundError(f"File not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, list):
        raise ValueError(f"Expected JSON array in {path}")
    return [row for row in payload if isinstance(row, dict)]


def _task_index(row: Row) -> int | None:
    raw = row.get("task", row.get("task_index"))
    if raw is None:
        return None
    try:
        return int(raw)
    except (TypeError, ValueError):
        return None


def _metric_value(row: Row, metric_name: str) -> float:
    # Each entry lists the current key first, then pre-removal spellings.
    key_map = {
        "macro_f1": ("macro_f1", "zero_shot_macro_f1", "f1_cls", "zero_shot_f1_cls"),
        "macro_rec": (
            "macro_rec",
            "zero_shot_macro_rec",
            "rec_cls",
            "zero_shot_rec_cls",
        ),
        "macro_prec": (
            "macro_prec",
            "zero_shot_macro_prec",
            "prec_cls",
            "zero_shot_prec_cls",
        ),
        "total_macro_f1_zs": (
            "total_macro_f1_zs",
            "zero_shot_total_macro_f1_zs",
            "total_f1_zs",
            "zero_shot_total_f1_zs",
        ),
    }
    for key in key_map[metric_name]:
        if key in row:
            try:
                return float(row[key])
            except (TypeError, ValueError):
                return float("nan")
    return float("nan")


def _algo_rows(rows: Sequence[Row], algo: str) -> List[Tuple[int, Row]]:
    selected: List[Tuple[int, Row]] = []
    for row in rows:
        row_algo = str(row.get("algo", "")).strip()
        if row_algo != algo:
            continue
        task_idx = _task_index(row)
        if task_idx is None:
            continue
        selected.append((task_idx, row))
    selected.sort(key=lambda item: item[0])
    return selected


def _shared_algorithms(
    baseline_rows: Sequence[Row], cl_rows: Sequence[Row]
) -> List[str]:
    baseline_algos = {str(row.get("algo", "")).strip() for row in baseline_rows}
    cl_algos = {str(row.get("algo", "")).strip() for row in cl_rows}
    shared = sorted(algo for algo in baseline_algos.intersection(cl_algos) if algo)
    return shared


def _all_algorithms(baseline_rows: Sequence[Row], cl_rows: Sequence[Row]) -> List[str]:
    baseline_algos = {str(row.get("algo", "")).strip() for row in baseline_rows}
    cl_algos = {str(row.get("algo", "")).strip() for row in cl_rows}
    return sorted(algo for algo in baseline_algos.union(cl_algos) if algo)


def _series_from_rows(
    rows: Sequence[Row], metric_name: str
) -> Tuple[List[int], List[float]]:
    indexed_rows: List[Tuple[int, Row]] = []
    for row in rows:
        task_idx = _task_index(row)
        if task_idx is None:
            continue
        indexed_rows.append((task_idx, row))
    indexed_rows.sort(key=lambda item: item[0])
    x_values = [task for task, _ in indexed_rows]
    y_values = [_metric_value(row, metric_name) for _, row in indexed_rows]
    return x_values, y_values


def _select_baseline_rows(baseline_rows: Sequence[Row]) -> Tuple[str, List[Row]]:
    baseline_algorithms = sorted(
        {
            str(row.get("algo", "")).strip()
            for row in baseline_rows
            if str(row.get("algo", "")).strip()
        }
    )
    if not baseline_algorithms:
        raise SystemExit("Baseline JSON has no valid 'algo' values.")
    if "iid2" in baseline_algorithms:
        chosen_algo = "iid2"
    else:
        chosen_algo = baseline_algorithms[0]
    selected = [
        row for row in baseline_rows if str(row.get("algo", "")).strip() == chosen_algo
    ]
    return chosen_algo, selected


def _ensure_output_parent(path: Path) -> None:
    parent = path.parent
    if str(parent) and not parent.exists():
        parent.mkdir(parents=True, exist_ok=True)


def main() -> None:
    args = _parse_args()
    baseline_rows = _read_json_rows(args.baseline)
    cl_rows = _read_json_rows(args.cl)

    baseline_algo, baseline_selected_rows = _select_baseline_rows(baseline_rows)
    baseline_x, baseline_y = _series_from_rows(baseline_selected_rows, args.metric)

    cl_algorithms = sorted(
        {
            str(row.get("algo", "")).strip()
            for row in cl_rows
            if str(row.get("algo", "")).strip()
        }
    )
    if args.algo is not None:
        cl_algorithms = [args.algo]

    if not cl_algorithms:
        raise SystemExit("CL JSON has no valid algorithms to plot.")

    figure, axis = plt.subplots(1, 1, figsize=(11, 6), dpi=180)
    axis.plot(
        baseline_x,
        baseline_y,
        "k^-",
        linewidth=2.2,
        markersize=5,
        label=f"baseline ({baseline_algo})",
    )

    for algo in cl_algorithms:
        cl_pairs = _algo_rows(cl_rows, algo)
        if not cl_pairs:
            continue
        cl_x = [task for task, _ in cl_pairs]
        cl_y = [_metric_value(row, args.metric) for _, row in cl_pairs]
        axis.plot(cl_x, cl_y, "o-", alpha=0.9, label=algo)

    axis.set_title(f"Zero-shot comparison ({args.metric})")
    axis.set_xlabel("Task index")
    axis.set_ylabel(args.metric)
    axis.grid(True, alpha=0.3)
    axis.legend(fontsize=8, ncol=2)
    figure.tight_layout()
    _ensure_output_parent(args.output)
    figure.savefig(args.output, bbox_inches="tight")
    plt.close(figure)
    print(f"Saved plot to {args.output}")


if __name__ == "__main__":
    main()
