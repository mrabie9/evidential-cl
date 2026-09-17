#!/usr/bin/env python
"""How much do the Omega fields of different scalars agree on one trajectory?

B6's null (``i2`` ties ``ce``) has an appealing mechanistic explanation: the
DS-specific content of ``I_2`` is near-collinear with the plain confidence half,
so whichever is tracked, ``Omega`` extracts almost the same parameter ranking.
Collinearity is measurable, but not from the ``omega_dumps`` B6 left behind --
those are one run per scalar, so any disagreement between two fields mixes the
scalar with the trajectory (and at the shared ``lambda`` of 2.4e5 three of the
four arms were anchored far past their own peaks). ``WOE_OMEGA_SHADOW`` removes
that confound by differentiating every scalar at the same importance windows of
a single run; this reads the resulting dump.

Reported per pair, because "correlated" is not one number here:

* **Spearman** over all parameters -- the rank agreement the mechanism claims.
* **Pearson / cosine** -- agreement weighted by magnitude, which is what the
  anchor actually charges (``lambda * Omega * delta^2``).
* **Jaccard of the top-k sets** -- whether the *stiff* parameters coincide. A6
  established that the useful ``lambda`` tracks the count of anchored
  parameters, so this is the statistic closest to what the anchor does.
* **Mass overlap** -- the share of one field's top-k mass that sits on
  parameters the other also ranks top-k. A low Jaccard with a high mass overlap
  means the two disagree only about parameters neither of them anchors.

Cross-run mode (``--extra``) reproduces the same table over separate dump
directories, for comparison with the confounded version.

Usage:
    python scripts/omega_shadow_correlate.py scripts/logs/omega_dumps/shadow_s0
    python scripts/omega_shadow_correlate.py DIR --task 4 --top-frac 0.001
"""

from __future__ import annotations

import argparse
import itertools
from pathlib import Path
from typing import Dict

import numpy as np
import torch
from scipy.stats import spearmanr


def _load(dump_dir: Path, task: int | None) -> Dict[str, np.ndarray]:
    files = sorted(dump_dir.glob("omega_task*.pt"))
    if not files:
        raise SystemExit(f"no dumps in {dump_dir}")
    blob = torch.load(files[-1] if task is None else files[task], map_location="cpu")
    fields = {"live": blob["omega"].numpy().astype(np.float64)}
    for name, tensor in blob.get("shadow_omega", {}).items():
        fields[name] = tensor.numpy().astype(np.float64)
    return fields


def _pair_stats(x: np.ndarray, y: np.ndarray, idx: np.ndarray, k: int) -> tuple:
    rho = float(spearmanr(x[idx], y[idx]).statistic)
    pear = float(np.corrcoef(x[idx], y[idx])[0, 1])
    cos = float(x @ y / max(np.linalg.norm(x) * np.linalg.norm(y), 1e-30))
    top_x = np.argpartition(-x, k)[:k]
    top_y = np.argpartition(-y, k)[:k]
    inter = np.intersect1d(top_x, top_y)
    jac = inter.size / (2 * k - inter.size)
    mass = float(x[inter].sum() / max(x[top_x].sum(), 1e-30))
    return rho, pear, cos, jac, mass


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("dump_dir", type=Path)
    ap.add_argument(
        "--extra",
        type=Path,
        nargs="*",
        default=[],
        help="further dump dirs; their live Omega joins the table, labelled by "
        "directory name. Cross-run, so read with the trajectory confound in mind.",
    )
    ap.add_argument("--task", type=int, default=None, help="index into the dumps")
    ap.add_argument("--top-frac", type=float, default=0.01)
    ap.add_argument("--sample", type=int, default=400000)
    args = ap.parse_args()

    fields = _load(args.dump_dir, args.task)
    for extra in args.extra:
        fields[extra.name] = _load(extra, args.task)["live"]

    sizes = {v.size for v in fields.values()}
    if len(sizes) != 1:
        raise SystemExit(f"dumps disagree on parameter count: {sizes}")
    n = sizes.pop()
    k = max(1, int(args.top_frac * n))

    print(f"params={n:,}  top-k={k:,} ({args.top_frac:.3%})")
    for name, vec in fields.items():
        print(f"  {name:9s} total={vec.sum():.4e}  nonzero={np.mean(vec > 0):.4f}")

    idx = np.random.default_rng(0).choice(n, min(args.sample, n), replace=False)
    print()
    head = f"{'pair':22s} {'spearman':>9s} {'pearson':>8s} {'cos':>7s}"
    print(f"{head} {'jaccard':>8s} {'massovl':>8s}")
    for a, b in itertools.combinations(fields, 2):
        rho, pear, cos, jac, mass = _pair_stats(fields[a], fields[b], idx, k)
        print(
            f"{a + '/' + b:22s} {rho:9.3f} {pear:8.3f} {cos:7.3f} "
            f"{jac:8.3f} {mass:8.3f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
