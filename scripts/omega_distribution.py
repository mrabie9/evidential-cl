#!/usr/bin/env python
"""Distribution of Omega under Dempster (sum) vs cautious (max) accumulation.

A7 was decided on *totals* -- one scalar per task -- which cannot distinguish
"max discards real structure" from "max and sum are nearly the same vector here".
``WOE_OMEGA_DUMP`` writes each task's path integral before it is combined, so a
single run gives both rules on an **identical trajectory** and the comparison is
per parameter.

The statistic that carries the argument is the per-parameter ratio

    r_i = sum_t omega_i^t / max_t omega_i^t   in [1, n_tasks]

i.e. the *effective number of tasks* whose evidence depends on parameter i. It is
exactly what the two rules disagree about: sum keeps it, max sets it to 1. If r
is near 1 everywhere the two rules are numerically almost the same and the
benchmark is a weak test; if r is broadly spread, they differ substantively and
the measured outcome gap is informative.

Also reported is the anchor's operating point. The proximal update is
``theta <- (theta+ + b theta*) / (1 + b)`` with ``b = 2 lr lambda Omega``, so
``b >> 1`` is a frozen parameter, ``b << 1`` a free one, and the interesting
question is how much of the parameter vector each rule pushes into each regime.

Usage:
    python scripts/omega_distribution.py scripts/logs/omega_dumps/sum_s0
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import numpy as np
import torch


def gini(values: np.ndarray) -> float:
    """Gini coefficient of a non-negative vector (0 = uniform, 1 = one atom)."""
    ordered = np.sort(values)
    n = ordered.size
    total = ordered.sum()
    if total <= 0:
        return 0.0
    index = np.arange(1, n + 1)
    return float((2.0 * index - n - 1.0).dot(ordered) / (n * total))


def top_share(values: np.ndarray, frac: float) -> float:
    """Share of total mass held by the largest ``frac`` of entries."""
    k = max(1, int(round(values.size * frac)))
    return float(np.sort(values)[-k:].sum() / max(values.sum(), 1e-12))


def load_tasks(dump_dir: Path) -> List[np.ndarray]:
    files = sorted(dump_dir.glob("omega_task*.pt"))
    if not files:
        raise SystemExit(f"no omega dumps in {dump_dir}")
    return [torch.load(f, map_location="cpu")["task_omega"].numpy() for f in files]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("dump_dir")
    ap.add_argument("--lr", type=float, default=0.003)
    ap.add_argument("--lam", type=float, default=240000.0)
    args = ap.parse_args()

    per_task = load_tasks(Path(args.dump_dir))
    stack = np.stack(per_task)  # (n_tasks, n_params)
    omega_sum = stack.sum(axis=0)
    omega_max = stack.max(axis=0)
    n_tasks, n_params = stack.shape

    print(f"dump={args.dump_dir}  tasks={n_tasks}  params={n_params:,}")
    print()
    print("per-task path integral")
    print(f"{'task':>5}{'total':>12}{'gini':>8}{'top1%':>8}{'top10%':>8}{'nonzero':>9}")
    for t, vec in enumerate(per_task):
        print(
            f"{t:>5}{vec.sum():>12.4f}{gini(vec):>8.4f}{top_share(vec, 0.01):>8.4f}"
            f"{top_share(vec, 0.10):>8.4f}{(vec > 0).mean():>9.4f}"
        )

    print()
    print("cumulative Omega, same trajectory, both rules")
    print(f"{'rule':>6}{'total':>12}{'mean':>12}{'median':>12}{'p99':>12}{'gini':>8}")
    for label, vec in (("sum", omega_sum), ("max", omega_max)):
        print(
            f"{label:>6}{vec.sum():>12.4f}{vec.mean():>12.3e}"
            f"{np.median(vec):>12.3e}{np.percentile(vec, 99):>12.3e}{gini(vec):>8.4f}"
        )
    print(f"{'ratio':>6}{omega_sum.sum() / max(omega_max.sum(), 1e-12):>12.4f}")

    ratio = omega_sum / np.maximum(omega_max, 1e-30)
    ratio = ratio[omega_max > 0]
    print()
    print(f"effective dependent tasks r_i = sum/max, over {ratio.size:,} live params")
    print(f"  mean {ratio.mean():.3f}   median {np.median(ratio):.3f}")
    qs = [1, 5, 25, 50, 75, 95, 99]
    print("  " + "  ".join(f"p{q}={np.percentile(ratio, q):.3f}" for q in qs))
    for threshold in (1.1, 1.5, 2.0, 3.0, 5.0):
        print(f"  frac r > {threshold:<4} : {(ratio > threshold).mean():.4f}")

    # Where the *mass* lives, which is what the anchor actually feels: weight the
    # r distribution by each parameter's own Omega rather than counting params.
    weights = omega_sum[omega_max > 0]
    order = np.argsort(ratio)
    cumulative = np.cumsum(weights[order]) / max(weights.sum(), 1e-12)
    for target in (0.5, 0.9):
        idx = int(np.searchsorted(cumulative, target))
        idx = min(idx, ratio.size - 1)
        print(f"  r at the {target:.0%} mass quantile: {ratio[order][idx]:.3f}")

    # The anchor is inert wherever b << 1, and Omega is extremely heavy-tailed,
    # so the population that actually feels it is a small subset. r over *that*
    # subset is the number that decides whether the two rules can differ in
    # practice, and it need not resemble r over all parameters.
    live_sum = omega_sum[omega_max > 0]
    b_live = 2.0 * args.lr * args.lam * live_sum
    print()
    for label, mask in (
        ("b >= 1 (stiff)", b_live >= 1.0),
        ("b >= 0.1", b_live >= 0.1),
        ("top 1% of mass-carrying params", live_sum >= np.percentile(live_sum, 99)),
    ):
        subset = ratio[mask]
        if subset.size == 0:
            continue
        share = float(live_sum[mask].sum() / max(live_sum.sum(), 1e-12))
        print(
            f"r over {label:<32} n={subset.size:>9,}  mass={share:.4f}  "
            f"mean={subset.mean():.3f}  median={np.median(subset):.3f}  "
            f"p90={np.percentile(subset, 90):.3f}"
        )

    print()
    print("anchor operating point  b = 2 lr lambda Omega")
    print(f"{'rule':>6}{'b<0.1':>9}{'0.1-1':>9}{'1-10':>9}{'b>10':>9}{'median b':>12}")
    for label, vec in (("sum", omega_sum), ("max", omega_max)):
        b = 2.0 * args.lr * args.lam * vec
        bands = [
            float((b < 0.1).mean()),
            float(((b >= 0.1) & (b < 1)).mean()),
            float(((b >= 1) & (b < 10)).mean()),
            float((b >= 10).mean()),
        ]
        print(
            f"{label:>6}" + "".join(f"{v:>9.4f}" for v in bands)
            + f"{np.median(b):>12.3e}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
