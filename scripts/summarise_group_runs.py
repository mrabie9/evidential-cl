#!/usr/bin/env python3
"""Summarise every model's seed-sweep run under a ``logs/00_sync`` group directory.

Given a group's ``saved_models`` directory (e.g.
``logs/00_sync/5e_CIL/saved_models``), this reads each model's seed subdirs
(``<model>/<run>/<seed>/seed_metrics.json``) and reports validation macro
recall/precision/F1 and backward transfer, meaned over seeds.

Rows are ordered by macro F1, lowest to highest, with two baselines pinned to
the ends regardless of their score: ``ft`` (lower-bound, no continual learning
strategy) always leads the table and ``iid2`` (upper-bound, joint/IID
training) always closes it.

When a model directory holds more than one run (e.g. mid seed-sweep
migration), the lexicographically last run directory name is used, which is
also the most recent since run directories are timestamp-prefixed.

Both the terminal and ``--format latex`` tables include FWT (mean/std over
the group's 6 training seeds), alongside Rec/Prec/F1/BWT. FWT is read from
``analysis/bwt_fwt/<group>/fwt_mean_std.csv`` when that's been computed for
the group; models it doesn't cover show "--".

The ``--format latex`` table follows the paper's results-table convention:
Method, macro-averaged CL recall/precision/F1 (headed R_CL/P_CL/F1_CL), then
BWT and FWT, with value cells as ``\\pmv{mean}{std}``. Model names use the
same display names as ``scripts/plot_fwt_metrics.py`` (e.g. "agem" ->
"A-GEM"). The rendered table is also written to
``../cl-radar-benchmark/05_Results/Tables/<group>_table.tex`` (lowercased),
matching the paper's existing table filenames.

Usage:
    python scripts/summarise_group_runs.py logs/00_sync/5e_CIL/saved_models
    python scripts/summarise_group_runs.py logs/00_sync/5e_CIL/saved_models --format latex
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SCRIPTS_DIR = Path(__file__).resolve().parent
for _path in (_REPO_ROOT, _SCRIPTS_DIR):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from plot_fwt_metrics import ALGORITHM_DISPLAY_NAMES  # noqa: E402

PINNED_TOP = "ft"
PINNED_BOTTOM = "iid2"

DEFAULT_OUTPUT_DIR = _REPO_ROOT.parent / "cl-radar-benchmark" / "05_Results" / "Tables"

METRIC_KEYS = {
    "rec": "val_macro_rec",
    "prec": "val_macro_prec",
    "f1": "val_macro_f1",
    "bwt": "val_bwt_f1",
}


def mean_std(values: list[float]) -> tuple[float, float]:
    """Mean and *sample* std (ddof=1) of a list of floats; std is nan for n<2."""
    clean = [v for v in values if not math.isnan(v)]
    n = len(clean)
    if n == 0:
        return float("nan"), float("nan")
    mean = sum(clean) / n
    if n < 2:
        return mean, float("nan")
    var = sum((v - mean) ** 2 for v in clean) / (n - 1)
    return mean, math.sqrt(var)


def find_latest_run_dir(model_dir: str) -> str:
    """Return the lexicographically last (most recent) completed run subdir.

    A run subdir only counts if it has a top-level ``results.txt``; smoke
    tests and abandoned/in-progress runs are left without one and are
    skipped, per the ``logs/00_sync`` promotion convention.
    """
    candidates = sorted(
        name
        for name in os.listdir(model_dir)
        if os.path.isdir(os.path.join(model_dir, name))
        and os.path.isfile(os.path.join(model_dir, name, "results.txt"))
    )
    if not candidates:
        raise SystemExit(
            f"No completed run directories (with results.txt) found under {model_dir!r}"
        )
    return os.path.join(model_dir, candidates[-1])


def find_seed_dirs(run_dir: str) -> list[str]:
    """Return every seed subdir of a run dir that has a ``seed_metrics.json``."""
    seed_dirs = []
    for name in sorted(os.listdir(run_dir)):
        sub = os.path.join(run_dir, name)
        if os.path.isdir(sub) and os.path.isfile(
            os.path.join(sub, "seed_metrics.json")
        ):
            seed_dirs.append(sub)
    return seed_dirs


def summarise_model(model_dir: str) -> dict:
    """Aggregate one model's seeded runs into mean/std stats per metric."""
    run_dir = find_latest_run_dir(model_dir)
    seed_dirs = find_seed_dirs(run_dir)
    if not seed_dirs:
        raise SystemExit(f"No seed_metrics.json found under {run_dir!r}")

    per_seed_metrics = []
    for seed_dir in seed_dirs:
        with open(os.path.join(seed_dir, "seed_metrics.json")) as fh:
            per_seed_metrics.append(json.load(fh))

    stats = {}
    for column_name, json_key in METRIC_KEYS.items():
        values = [
            float(metrics.get(json_key, float("nan"))) for metrics in per_seed_metrics
        ]
        stats[column_name] = mean_std(values)

    return {
        "model": os.path.basename(model_dir),
        "run_dir": run_dir,
        "n": len(seed_dirs),
        "stats": stats,
    }


def load_fwt_stats(group_label: str) -> dict[str, tuple[float, float]]:
    """Read per-model mean/std forward transfer for a group, if it's been computed.

    Sourced from ``analysis/bwt_fwt/<group_label>/fwt_mean_std.csv``, written by
    ``analysis/bwt_fwt/fwt_00sync.py`` for the ``{1e,5e}_{CIL,TIL}`` sweeps. Models
    the FWT pipeline hasn't reached yet, or groups it doesn't cover, are simply
    absent from the returned mapping.
    """
    path = _REPO_ROOT / "analysis" / "bwt_fwt" / group_label / "fwt_mean_std.csv"
    if not path.is_file():
        return {}
    stats = {}
    with open(path, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            try:
                stats[row["algo"]] = (
                    float(row["mean_fwt_tasks_1_9_mean"]),
                    float(row["mean_fwt_tasks_1_9_std"]),
                )
            except (KeyError, ValueError):
                continue
    return stats


def order_rows(rows: list[dict]) -> list[dict]:
    """Sort by mean F1 ascending, pinning ft first and iid2 last regardless of score."""
    pinned_top = [row for row in rows if row["model"] == PINNED_TOP]
    pinned_bottom = [row for row in rows if row["model"] == PINNED_BOTTOM]
    middle = [row for row in rows if row["model"] not in (PINNED_TOP, PINNED_BOTTOM)]

    def f1_key(row: dict) -> float:
        mean, _ = row["stats"]["f1"]
        return math.inf if math.isnan(mean) else mean

    middle.sort(key=f1_key)
    return pinned_top + middle + pinned_bottom


def discover_model_rows(group_dir: str) -> list[dict]:
    """Summarise every model subdir found directly under a saved_models group dir."""
    model_names = sorted(
        name
        for name in os.listdir(group_dir)
        if os.path.isdir(os.path.join(group_dir, name))
    )
    if not model_names:
        raise SystemExit(f"No model directories found under {group_dir!r}")
    return [summarise_model(os.path.join(group_dir, name)) for name in model_names]


def _fmt_pct(mean: float, std: float) -> str:
    """Format a "mean ± std" percentage cell, falling back to "--" when undefined."""
    if math.isnan(mean):
        return "--"
    std_s = "--" if math.isnan(std) else f"{std * 100:.2f}"
    return f"{mean * 100:.2f} +/- {std_s}"


def _fmt_pmv(mean: float, std: float, bold: bool = False) -> str:
    """Format a value as \\pmv{mean}{std} (percentage, 1dp), or "--" when undefined.

    Pass ``bold=True`` to use \\pmvbf instead, for highlighting a column's best value.
    """
    if math.isnan(mean):
        return "--"
    std_s = "0.0" if math.isnan(std) else f"{std * 100:.1f}"
    macro = "pmvbf" if bold else "pmv"
    return f"\\{macro}{{{mean * 100:.1f}}}{{{std_s}}}"


def render_aligned_table(headers: list[str], rows: list[list[str]]) -> str:
    """Render a header + rows of pre-formatted strings as a column-aligned Markdown table."""
    widths = [
        max(len(header), *(len(row[i]) for row in rows)) if rows else len(header)
        for i, header in enumerate(headers)
    ]

    def render_row(cells: list[str]) -> str:
        return "| " + " | ".join(cell.ljust(w) for cell, w in zip(cells, widths)) + " |"

    lines = [
        render_row(headers),
        "|" + "|".join("-" * (w + 2) for w in widths) + "|",
    ]
    lines.extend(render_row(row) for row in rows)
    return "\n".join(lines)


COLUMNS = [
    ("Rec", "rec"),
    ("Prec", "prec"),
    ("F1", "f1"),
    ("BWT", "bwt"),
    ("FWT", "fwt"),
]

# Column set/headers for the LaTeX table, matching the paper's results-table
# convention: macro-averaged CL performance, then transfer metrics. All five
# are "higher is better", so every header carries an up-arrow.
LATEX_COLUMNS = [
    (r"$\uparrow\bar{R}_\text{CL}$", "rec"),
    (r"$\uparrow\bar{P}_\text{CL}$", "prec"),
    (r"$\uparrow\bar{F1}_\text{CL}$", "f1"),
    (r"$\uparrow$ BWT", "bwt"),
    (r"$\uparrow$ FWT", "fwt"),
]


def render_terminal(rows: list[dict], group_label: str) -> str:
    """Render rows as a column-aligned Markdown table, readable straight in a terminal."""
    headers = ["Model"] + [header for header, _ in COLUMNS] + ["n"]
    table_rows = []
    for row in rows:
        cells = [row["model"]]
        for _, key in COLUMNS:
            mean, std = row["stats"][key]
            cells.append(_fmt_pct(mean, std))
        cells.append(str(row["n"]))
        table_rows.append(cells)

    return f"## Seed-sweep summary: {group_label}\n\n" + render_aligned_table(
        headers, table_rows
    )


def render_latex(rows: list[dict], group_label: str) -> str:
    """Render rows as a LaTeX results table: booktabs, \\pmv value cells, resizebox-wrapped.

    Matches the paper's results-table convention: Method, macro-averaged
    CL recall/precision/F1, then BWT/FWT, with double vertical rules
    separating the CL-performance block from the transfer-metric block.
    Value cells use ``\\pmv{mean}{std}``, assumed defined in the document
    preamble.
    """
    caption = (
        f"Continual learning performance for the {group_label} setting. "
        r"Results show the mean and sample standard deviation (in $\%$) "
        "over training seeds."
    )
    label = "tab:results_" + group_label.lower().replace(" ", "-").replace("/", "-")
    col_spec = "l||c|c|c||c|c"
    headers = ["Method"] + [header for header, _ in LATEX_COLUMNS]

    lines = [
        r"\begin{table}[t]",
        r"\centering",
        rf"\caption{{{caption}}}",
        rf"\label{{{label}}}",
        r"\setlength{\tabcolsep}{6pt}",
        r"\resizebox{\columnwidth}{!}{%",
        rf"\begin{{tabular}}{{{col_spec}}}",
        r"\toprule",
        " & ".join(headers) + r" \\",
        r"\midrule",
    ]
    for row in rows:
        display_name = ALGORITHM_DISPLAY_NAMES.get(row["model"], row["model"])
        cells = [display_name.replace("_", r"\_")]
        for _, key in LATEX_COLUMNS:
            mean, std = row["stats"][key]
            cells.append(_fmt_pmv(mean, std))
        lines.append(" & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}%", "}", r"\end{table}"]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "group_dir",
        help="A saved_models directory under logs/00_sync, e.g. "
        "logs/00_sync/5e_CIL/saved_models",
    )
    ap.add_argument(
        "--format",
        choices=["terminal", "latex"],
        default="terminal",
        help="Output format (default: terminal).",
    )
    args = ap.parse_args(argv)

    if not os.path.isdir(args.group_dir):
        ap.error(f"not a directory: {args.group_dir}")

    rows = order_rows(discover_model_rows(args.group_dir))
    group_label = os.path.basename(os.path.dirname(os.path.abspath(args.group_dir)))

    fwt_stats = load_fwt_stats(group_label)
    for row in rows:
        row["stats"]["fwt"] = fwt_stats.get(row["model"], (float("nan"), float("nan")))

    if args.format == "latex":
        output = render_latex(rows, group_label)
        print(output)
        output_path = DEFAULT_OUTPUT_DIR / f"{group_label.lower()}_table.tex"
        output_path.write_text(output + "\n")
        print(f"Wrote {output_path}", file=sys.stderr)
    else:
        print(render_terminal(rows, group_label))
    return 0


if __name__ == "__main__":
    sys.exit(main())
