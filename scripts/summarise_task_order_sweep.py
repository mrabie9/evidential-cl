#!/usr/bin/env python3
"""Summarise a task-order-sweep group under ``logs/00_sync`` into one results table.

Given a task-order sweep's root dir (e.g. ``logs/00_sync/taskorder_TIL``, holding
one subdir per randomly drawn task order such as ``to39/saved_models/...``,
``to55/saved_models/...``), this reads each model's single run at each task-order
point (one training seed, one task-order seed per point -- see each run's
``training_parameters.json``: ``seed`` is fixed, ``task_order_seed`` varies) and
aggregates validation macro F1 *across task-order points* into one value per
model.

Unlike ``summarise_group_runs.py`` (which aggregates across seeds within one
run), this aggregates across the points themselves, since each point run is a
single seed. The result is a single-row table: one ``F1_CL`` value per model,
mean +/- sample std over the task-order points.

Only models present under every discovered point are included, since a partial
model can't be given a fair mean over the same set of task orders as the rest.

The ``--format latex`` table is also written to
``../cl-radar-benchmark/05_Results/Tables/task-order_sweep_<mode>.tex``,
matching the paper's existing table filenames.

Usage:
    python scripts/summarise_task_order_sweep.py logs/00_sync/taskorder_TIL
    python scripts/summarise_task_order_sweep.py logs/00_sync/taskorder_CIL --format latex
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

_SCRIPTS_DIR = Path(__file__).resolve().parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

from summarise_group_runs import (  # noqa: E402
    ALGORITHM_DISPLAY_NAMES,
    DEFAULT_OUTPUT_DIR,
    _fmt_pmv,
    find_seed_dirs,
    mean_std,
    render_aligned_table,
)


def discover_point_dirs(sweep_dir: Path) -> list[Path]:
    """Return this sweep's task-order point dirs (e.g. to39, to55), sorted by name."""
    return sorted(
        p for p in sweep_dir.iterdir() if p.is_dir() and (p / "saved_models").is_dir()
    )


def discover_common_models(point_dirs: list[Path]) -> list[str]:
    """Return the models present under every point's saved_models dir."""
    if not point_dirs:
        return []
    model_sets = [
        {p.name for p in (point / "saved_models").iterdir() if p.is_dir()}
        for point in point_dirs
    ]
    return sorted(set.intersection(*model_sets))


def find_any_latest_run_dir(model_dir: Path) -> Path:
    """Return the lexicographically last run subdir that has at least one seed_metrics.json.

    Task-order points are single-seed runs, so unlike ``summarise_group_runs``'s
    seed-sweep groups, no top-level ``results.txt`` cross-seed summary is ever
    written here -- run dirs are identified by their seed data directly instead.
    """
    candidates = sorted(
        p for p in model_dir.iterdir() if p.is_dir() and find_seed_dirs(str(p))
    )
    if not candidates:
        raise SystemExit(f"No seed_metrics.json found under any run dir in {model_dir}")
    return candidates[-1]


def collect_f1_across_points(point_dirs: list[Path], model: str) -> list[float]:
    """Return one macro-F1 value per task-order point for one model.

    Each point is expected to hold a single seed; if more than one is present
    they're averaged first so every point contributes one value.
    """
    values = []
    for point in point_dirs:
        model_dir = point / "saved_models" / model
        if not model_dir.is_dir():
            continue
        run_dir = find_any_latest_run_dir(model_dir)
        f1_values = []
        for seed_dir in find_seed_dirs(str(run_dir)):
            with open(
                os.path.join(seed_dir, "seed_metrics.json"), encoding="utf-8"
            ) as fh:
                metrics = json.load(fh)
            f1 = metrics.get("val_macro_f1")
            if isinstance(f1, (int, float)):
                f1_values.append(float(f1))
        if f1_values:
            values.append(sum(f1_values) / len(f1_values))
    return values


def infer_mode(sweep_dir: Path) -> str:
    """Infer "til"/"cil" from a sweep dir name like "taskorder_TIL"."""
    name = sweep_dir.name.upper()
    if name.endswith("_TIL"):
        return "til"
    if name.endswith("_CIL"):
        return "cil"
    raise SystemExit(
        f"cannot infer TIL/CIL mode from sweep dir name {sweep_dir.name!r}"
    )


def build_model_stats(
    sweep_dir: Path,
) -> tuple[dict[str, tuple[float, float, int]], int]:
    """Return {model: (mean, std, n_points)} and the number of task-order points found."""
    point_dirs = discover_point_dirs(sweep_dir)
    if not point_dirs:
        raise SystemExit(f"No task-order point dirs found under {sweep_dir}")
    models = discover_common_models(point_dirs)
    if not models:
        raise SystemExit(f"No model is present under every point in {sweep_dir}")

    stats = {}
    for model in models:
        values = collect_f1_across_points(point_dirs, model)
        mean, std = mean_std(values)
        stats[model] = (mean, std, len(values))
    return stats, len(point_dirs)


def order_models(model_stats: dict[str, tuple[float, float, int]]) -> list[str]:
    """Sort models by mean F1 ascending."""

    def key(model: str) -> float:
        mean = model_stats[model][0]
        return math.inf if math.isnan(mean) else mean

    return sorted(model_stats, key=key)


def render_terminal(model_stats: dict[str, tuple[float, float, int]], mode: str) -> str:
    """Render as a column-aligned Markdown table: one row of F1_CL, models as columns."""
    ordered = order_models(model_stats)
    headers = ["Metric"] + [ALGORITHM_DISPLAY_NAMES.get(m, m) for m in ordered]
    row = ["F1_CL"] + [
        (
            f"{model_stats[m][0] * 100:.2f} +/- {model_stats[m][1] * 100:.2f}"
            if not math.isnan(model_stats[m][0])
            else "--"
        )
        for m in ordered
    ]
    return f"## Task-order sweep summary: {mode.upper()}\n\n" + render_aligned_table(
        headers, [row]
    )


def render_latex(
    model_stats: dict[str, tuple[float, float, int]], mode: str, n_points: int
) -> str:
    """Render as the paper's single-row task-order-impact table."""
    ordered = order_models(model_stats)
    caption = (
        r"Impact of task order on $\bar{F1}_\text{CL}$ in the single-epoch "
        rf"\gls{{{mode}}} setting. {n_points} randomly selected task orders are "
        r"used (same training seed), shown as mean$\pm$sample std."
    )
    label = f"tab:results_task-order_{mode}"
    col_spec = "l|" + "c" * len(ordered)
    headers = [r"\textbf{Method}"] + [
        ALGORITHM_DISPLAY_NAMES.get(m, m) for m in ordered
    ]
    row_label = r"$\mathbf{\overline{F1}_\text{CL}}$"
    cells = [row_label] + [
        _fmt_pmv(model_stats[m][0], model_stats[m][1]) for m in ordered
    ]

    lines = [
        r"\begin{table}[t]",
        r"\centering",
        rf"\caption{{{caption}}}",
        rf"\label{{{label}}}",
        r"\resizebox{\columnwidth}{!}{%",
        rf"\begin{{tabular}}{{{col_spec}}}",
        r"\toprule",
        " & ".join(headers) + r" \\",
        r"\midrule",
        " & ".join(cells) + r" \\",
        r"\bottomrule",
        r"\end{tabular}%",
        "}",
        r"\end{table}",
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "sweep_dir",
        help="A task-order sweep root under logs/00_sync, e.g. "
        "logs/00_sync/taskorder_TIL",
    )
    ap.add_argument(
        "--format",
        choices=["terminal", "latex"],
        default="terminal",
        help="Output format (default: terminal).",
    )
    args = ap.parse_args(argv)

    sweep_dir = Path(args.sweep_dir)
    if not sweep_dir.is_dir():
        ap.error(f"not a directory: {sweep_dir}")

    mode = infer_mode(sweep_dir)
    model_stats, n_points = build_model_stats(sweep_dir)

    if args.format == "latex":
        output = render_latex(model_stats, mode, n_points)
        print(output)
        output_path = DEFAULT_OUTPUT_DIR / f"task-order_sweep_{mode}.tex"
        output_path.write_text(output + "\n")
        print(f"Wrote {output_path}", file=sys.stderr)
    else:
        print(render_terminal(model_stats, mode))
    return 0


if __name__ == "__main__":
    sys.exit(main())
