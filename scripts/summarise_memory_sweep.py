#!/usr/bin/env python3
"""Summarise a memory-buffer-size-sweep group under ``logs/00_sync`` into one table.

Given a memory sweep's root dir (e.g. ``logs/00_sync/memsweep_TIL``, holding one
subdir per swept buffer size such as ``m1024/saved_models/...``,
``m2048/saved_models/...``), this reads each model's seed-sweep run at each
memory point (same cross-seed aggregation as ``summarise_group_runs.py``) and
reports mean +/- sample std macro F1 per (model, memory size).

The sweep typically skips the *default* memory size (5120), since that's
already covered by the corresponding un-swept group. By default this script
pulls that column from the sibling ``logs/00_sync/1e_<MODE>/saved_models``
group (these ablations are run single-epoch); pass ``--base-group`` to point
at a different one, or ``--no-base-group`` to omit that column entirely.

The ``--format latex`` table is also written to
``../cl-radar-benchmark/05_Results/Tables/mem-sweep_<mode>.tex``, matching
the paper's existing table filenames.

Usage:
    python scripts/summarise_memory_sweep.py logs/00_sync/memsweep_TIL
    python scripts/summarise_memory_sweep.py logs/00_sync/memsweep_CIL --format latex
    python scripts/summarise_memory_sweep.py logs/00_sync/memsweep_TIL \\
        --base-group logs/00_sync/5e_TIL/saved_models --default-memories 5120
"""

from __future__ import annotations

import argparse
import math
import re
import sys
from pathlib import Path

_SCRIPTS_DIR = Path(__file__).resolve().parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

from summarise_group_runs import (  # noqa: E402
    ALGORITHM_DISPLAY_NAMES,
    DEFAULT_OUTPUT_DIR,
    _fmt_pmv,
    render_aligned_table,
    summarise_model,
)

MEMORY_POINT_PATTERN = re.compile(r"^m(\d+)$")


def discover_memory_points(sweep_dir: Path) -> list[tuple[int, Path]]:
    """Return (memory_size, saved_models_dir) for each swept point, sorted ascending."""
    points = []
    for p in sweep_dir.iterdir():
        match = MEMORY_POINT_PATTERN.match(p.name)
        if match and (p / "saved_models").is_dir():
            points.append((int(match.group(1)), p / "saved_models"))
    return sorted(points)


def infer_mode(sweep_dir: Path) -> str:
    """Infer "til"/"cil" from a sweep dir name like "memsweep_TIL"."""
    name = sweep_dir.name.upper()
    if name.endswith("_TIL"):
        return "til"
    if name.endswith("_CIL"):
        return "cil"
    raise SystemExit(
        f"cannot infer TIL/CIL mode from sweep dir name {sweep_dir.name!r}"
    )


def default_base_group_dir(sweep_dir: Path, mode: str) -> Path:
    """The sibling un-swept group's saved_models dir supplying the default-memory column."""
    return sweep_dir.resolve().parent / f"1e_{mode.upper()}" / "saved_models"


def collect_model_points(
    memory_points: list[tuple[int, Path]],
    base_group_dir: Path | None,
    default_memories: int,
) -> dict[str, dict[int, tuple[float, float]]]:
    """Return {model: {memory_size: (mean, std)}} across every swept point plus the base.

    Models come only from the swept points themselves (methods the sweep was
    actually run for, i.e. the ones that use a memory buffer) -- the base
    group only ever supplies an extra column for those same models, never new
    rows, since most of it wasn't part of this ablation at all.
    """
    models: set[str] = set()
    for _, saved_models_dir in memory_points:
        models.update(p.name for p in saved_models_dir.iterdir() if p.is_dir())

    result: dict[str, dict[int, tuple[float, float]]] = {}
    for model in sorted(models):
        per_point: dict[int, tuple[float, float]] = {}
        for memory_size, saved_models_dir in memory_points:
            model_dir = saved_models_dir / model
            if model_dir.is_dir():
                per_point[memory_size] = summarise_model(str(model_dir))["stats"]["f1"]
        if base_group_dir is not None:
            base_model_dir = base_group_dir / model
            if base_model_dir.is_dir():
                per_point[default_memories] = summarise_model(str(base_model_dir))[
                    "stats"
                ]["f1"]
        if per_point:
            result[model] = per_point
    return result


def order_models(
    model_points: dict[str, dict[int, tuple[float, float]]], memory_sizes: list[int]
) -> list[str]:
    """Sort models by mean F1 at the largest memory point ascending."""
    largest = memory_sizes[-1]

    def key(model: str) -> float:
        mean, _ = model_points[model].get(largest, (float("nan"), float("nan")))
        return math.inf if math.isnan(mean) else mean

    return sorted(model_points, key=key)


def best_model_per_column(
    ordered_models: list[str],
    model_points: dict[str, dict[int, tuple[float, float]]],
    memory_sizes: list[int],
) -> dict[int, str]:
    """Return {memory_size: model} for the highest mean F1 in that column."""
    best = {}
    for size in memory_sizes:
        candidates = [
            (model, model_points[model][size][0])
            for model in ordered_models
            if size in model_points[model]
            and not math.isnan(model_points[model][size][0])
        ]
        if candidates:
            best[size] = max(candidates, key=lambda pair: pair[1])[0]
    return best


def render_terminal(
    model_points: dict[str, dict[int, tuple[float, float]]],
    memory_sizes: list[int],
    mode: str,
) -> str:
    """Render as a column-aligned Markdown table: models as rows, memory sizes as columns."""
    ordered = order_models(model_points, memory_sizes)
    headers = ["Model"] + [str(size) for size in memory_sizes]
    rows = []
    for model in ordered:
        cells = [model]
        for size in memory_sizes:
            mean, std = model_points[model].get(size, (float("nan"), float("nan")))
            cells.append(
                f"{mean * 100:.2f} +/- {std * 100:.2f}"
                if not math.isnan(mean)
                else "--"
            )
        rows.append(cells)
    return f"## Memory sweep summary: {mode.upper()}\n\n" + render_aligned_table(
        headers, rows
    )


def render_latex(
    model_points: dict[str, dict[int, tuple[float, float]]],
    memory_sizes: list[int],
    mode: str,
) -> str:
    """Render as the paper's memory-buffer-size-vs-F1_CL table, bolding each column's best."""
    ordered = order_models(model_points, memory_sizes)
    best = best_model_per_column(ordered, model_points, memory_sizes)
    caption = (
        r"Memory buffer size vs $\bar{F1}_\text{CL}$ for methods utilising a memory "
        rf"buffer, in the single-epoch \gls{{{mode}}} setting."
    )
    label = f"tab:results_mem-sweep_{mode}"
    col_spec = "l" + "c" * len(memory_sizes)
    headers = [r"\textbf{Method}"] + [str(size) for size in memory_sizes]

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
    ]
    for model in ordered:
        display_name = ALGORITHM_DISPLAY_NAMES.get(model, model)
        cells = [display_name.replace("_", r"\_")]
        for size in memory_sizes:
            mean, std = model_points[model].get(size, (float("nan"), float("nan")))
            cells.append(_fmt_pmv(mean, std, bold=best.get(size) == model))
        lines.append(" & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}%", "}", r"\end{table}"]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "sweep_dir",
        help="A memory sweep root under logs/00_sync, e.g. logs/00_sync/memsweep_TIL",
    )
    ap.add_argument(
        "--format",
        choices=["terminal", "latex"],
        default="terminal",
        help="Output format (default: terminal).",
    )
    ap.add_argument(
        "--base-group",
        help="saved_models dir supplying the default-memory column (default: the "
        "sibling logs/00_sync/1e_<MODE>/saved_models group).",
    )
    ap.add_argument(
        "--no-base-group",
        action="store_true",
        help="Don't add a default-memory column at all.",
    )
    ap.add_argument(
        "--default-memories",
        type=int,
        default=5120,
        help="Memory size to label the base group's column with (default: 5120).",
    )
    args = ap.parse_args(argv)

    sweep_dir = Path(args.sweep_dir)
    if not sweep_dir.is_dir():
        ap.error(f"not a directory: {sweep_dir}")

    mode = infer_mode(sweep_dir)
    memory_points = discover_memory_points(sweep_dir)
    if not memory_points:
        ap.error(f"no memory-point dirs (m<size>) found under {sweep_dir}")

    if args.no_base_group:
        base_group_dir = None
    elif args.base_group:
        base_group_dir = Path(args.base_group)
        if not base_group_dir.is_dir():
            ap.error(f"not a directory: {base_group_dir}")
    else:
        base_group_dir = default_base_group_dir(sweep_dir, mode)
        if not base_group_dir.is_dir():
            base_group_dir = None

    model_points = collect_model_points(
        memory_points, base_group_dir, args.default_memories
    )
    if not model_points:
        ap.error(f"no models found under any memory point in {sweep_dir}")

    memory_sizes = {size for size, _ in memory_points}
    if base_group_dir is not None:
        memory_sizes.add(args.default_memories)
    memory_sizes = sorted(memory_sizes)

    if args.format == "latex":
        output = render_latex(model_points, memory_sizes, mode)
        print(output)
        output_path = DEFAULT_OUTPUT_DIR / f"mem-sweep_{mode}.tex"
        output_path.write_text(output + "\n")
        print(f"Wrote {output_path}", file=sys.stderr)
    else:
        print(render_terminal(model_points, memory_sizes, mode))
    return 0


if __name__ == "__main__":
    sys.exit(main())
