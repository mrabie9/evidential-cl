#!/usr/bin/env python
"""Is SI's path-length denominator doing anything, at this xi, on this network?

``Omega_i^t = |omega_i^t| / ((Delta_i^t)^2 + xi)``. The correction only divides by
path length while ``Delta^2 >> xi``. This rebuilds ``Omega`` from the dumped
numerator and ``Delta^2`` at any xi and asks three things:

* ``Spearman(Omega, numerator)`` -- if ~1.0 the denominator is inert and ``Omega``
  is the raw path integral up to a constant. This replaces the between-scalar
  statistic PR-2a first registered: it is computable from a *single* dump, is not
  compressed near the top of its range, and tests the floor hypothesis directly
  rather than through its consequences.
* ``Spearman(Omega@xi, Omega@xi_small)`` -- how much the floor reshapes the
  profile.
* The floored **mass** share, split by the anchor's stiff set, because a floored
  inert bulk is benign where a floored stiff set is fatal, and a count cannot
  tell them apart (``nonzero_frac`` is 0.999 here).

Usage:
    python scripts/xi_floor_check.py scripts/logs/omega_dumps/xiparts_s0
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from scipy.stats import spearmanr


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("dump_dir")
    ap.add_argument("--xi-small", type=float, default=1e-6)
    ap.add_argument("--lr", type=float, default=0.003)
    ap.add_argument("--lam", type=float, default=240000.0)
    ap.add_argument("--sample", type=int, default=400000)
    args = ap.parse_args()

    files = sorted(Path(args.dump_dir).glob("omega_task*.pt"))
    if not files:
        raise SystemExit(f"no dumps in {args.dump_dir}")
    blobs = [torch.load(f, map_location="cpu") for f in files]
    if "numerator" not in blobs[0]:
        raise SystemExit("dump predates the numerator/delta_sq split; rerun needed")
    xi = float(blobs[0].get("xi", 1e-3))

    num = np.stack([b["numerator"].numpy() for b in blobs])
    dsq = np.stack([b["delta_sq"].numpy() for b in blobs])
    om_big = (num / (dsq + xi)).sum(axis=0)
    om_small = (num / (dsq + args.xi_small)).sum(axis=0)
    num_total = num.sum(axis=0)

    rng = np.random.default_rng(0)
    idx = rng.choice(om_big.size, min(args.sample, om_big.size), replace=False)
    print(f"dump={args.dump_dir}  tasks={len(files)}  params={om_big.size:,}  xi={xi:g}")
    print(f"sqrt(xi) = {xi ** 0.5:.4e}   RMS |Delta| = {np.sqrt(dsq.mean()):.4e}")
    print()
    print(f"Spearman(Omega@xi, numerator)      = {spearmanr(om_big[idx], num_total[idx]).statistic:.6f}")
    print(f"Spearman(Omega@xi, Omega@{args.xi_small:g}) = {spearmanr(om_big[idx], om_small[idx]).statistic:.6f}")
    print(f"total mass ratio Omega@xi_small / Omega@xi = {om_small.sum() / om_big.sum():.2f}x")
    # The floor hypothesis is not just "the denominator is inert" but "Omega is
    # then a displacement measure". Those are different: the tracked scalar still
    # enters through h in the numerator omega = sum_steps h.Delta. This asks how
    # much of Omega total displacement alone accounts for.
    travel = np.sqrt(dsq).sum(axis=0)
    print(f"Spearman(Omega@xi, total |Delta|)  = {spearmanr(om_big[idx], travel[idx]).statistic:.6f}")

    floored = dsq < xi
    print()
    print(f"{'population':<26}{'n':>12}{'floored frac':>14}{'floored mass':>14}")
    b = 2.0 * args.lr * args.lam * om_big
    for label, mask in (
        ("all parameters", np.ones_like(b, dtype=bool)),
        ("stiff (b >= 1)", b >= 1.0),
        ("inert bulk (b < 0.1)", b < 0.1),
    ):
        if mask.sum() == 0:
            continue
        # Mass-weighted over the per-task integrals, summed over tasks.
        per_task_floored = float(
            (num[:, mask] / (dsq[:, mask] + xi) * floored[:, mask]).sum()
        )
        per_task_total = float((num[:, mask] / (dsq[:, mask] + xi)).sum())
        print(
            f"{label:<26}{int(mask.sum()):>12,}"
            f"{float(floored[:, mask].mean()):>14.4f}"
            f"{per_task_floored / max(per_task_total, 1e-30):>14.4f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
