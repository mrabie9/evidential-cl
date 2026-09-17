#!/usr/bin/env python
"""Per-task view of a finished run: worst task, dispersion, per-task forgetting.

Every number A7 was decided on is an *average* over tasks, and averaging is
exactly the assumption under dispute. Dempster's rule (sum) accumulates the
total damage a parameter can do across all past tasks; the cautious rule (max)
is bounded by the single most demanding one. If that difference is meaningful,
it should show up as a **minimax** effect -- max protecting the worst-served
task, or narrowing the spread across tasks -- even where it loses on the mean,
which is the only functional the benchmark reports.

Reads the accuracy matrix in ``results.txt``: row t is the evaluation after
training task t, column k is task k. The diagonal is plasticity, the last row is
the final state.

Usage:
    python scripts/per_task_dispersion.py logs/woe_si_lc/caut_*_s0-*/0
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import List, Optional

import numpy as np


def read_matrix(results: Path) -> Optional[np.ndarray]:
    """Parse the lower-triangular accuracy matrix, dropping the '|' separator."""
    rows: List[List[float]] = []
    for line in results.read_text().splitlines():
        line = line.strip()
        if not line or line == "|" or line[0].isalpha():
            continue
        try:
            rows.append([float(piece) for piece in line.split()])
        except ValueError:
            break
    if not rows:
        return None
    width = max(len(row) for row in rows)
    square = [row for row in rows if len(row) == width]
    # The first line is the task-0 row printed twice (once before the '|').
    return np.asarray(square[1:] if len(square) > width else square, dtype=float)


def main(paths: List[str]) -> int:
    header = (
        f"{'run':<30}{'mean':>8}{'worst':>8}{'best':>8}{'std':>8}"
        f"{'p10':>8}{'wforg':>8}{'mforg':>8}"
    )
    print(header)
    print("-" * len(header))
    for path in sorted(paths):
        results = Path(path)
        if results.is_dir():
            results = results / "results.txt"
        if not results.exists():
            continue
        matrix = read_matrix(results)
        if matrix is None or matrix.shape[0] < 2:
            continue
        final = matrix[-1]
        diag = np.array([matrix[t, t] for t in range(matrix.shape[1])])
        forget = diag - final
        name = Path(path).parent.name.split("-2026")[0]
        print(
            f"{name:<30}{final.mean():>8.4f}{final.min():>8.4f}{final.max():>8.4f}"
            f"{final.std():>8.4f}{np.percentile(final, 10):>8.4f}"
            f"{forget.max():>8.4f}{forget.mean():>8.4f}"
        )
    print()
    print("worst/best/std/p10 are over the 10 final per-task scores.")
    print("wforg = largest single-task forgetting (diag_t - final_t); mforg = mean.")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
