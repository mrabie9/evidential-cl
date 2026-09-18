#!/usr/bin/env python
"""Does the anchor *penalty* agree across scalars more than ``Omega`` does?

``Omega`` is not what the trajectory feels. The proximal anchor charges
``lambda * Omega_i * (theta_i - theta_i*)^2``, so two very different ``Omega``
rankings can still produce nearly the same penalty field if the displacement
they multiply is concentrated -- and that, not collinearity of the scalars,
would be a mechanism for scalars tying on accuracy.

This mediates the comparison through the acting quantity. For every task ``t``
it forms the penalty actually charged over that task,

    p_i^t = lambda * Omega_i^(t-1) * (Delta_i^t)^2

-- the cumulative anchor mass standing at the start of the task times the
displacement taken during it -- and sums over tasks. It then reports the same
pairwise agreement table for ``p`` as ``omega_shadow_correlate.py`` reports for
``Omega``, so the two can be read against each other, plus the concentration of
each field.

``lambda`` cancels out of every agreement statistic and every concentration
share (they are scale-free), so the per-scalar tuned values are applied only to
the reported totals, where they are the only thing that makes the magnitudes
comparable at all. The columns that carry the argument do not depend on them.

Usage:
    python scripts/omega_penalty_mediation.py scripts/logs/omega_dumps/shadow_s0
"""

from __future__ import annotations

import argparse
import itertools
from pathlib import Path
from typing import Dict

import numpy as np
import torch
from scipy.stats import spearmanr

# Each scalar's own tuned peak on the abs/proximal anchor-only host (README B6,
# "Re-run on the anchor-only host"). Only used for the reported totals.
TUNED_LAMBDA: Dict[str, float] = {
    "live": 2.4e5,
    "ce": 75.0,
    "z2": 9.0,
    "phi2": 55.0,
}


def _concentration(v: np.ndarray) -> str:
    total = max(v.sum(), 1e-30)
    out = []
    for frac in (0.001, 0.01, 0.1):
        k = max(1, int(frac * v.size))
        out.append(np.partition(v, -k)[-k:].sum() / total)
    return f"{out[0]:.3f} {out[1]:.3f} {out[2]:.3f}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("dump_dir", type=Path)
    ap.add_argument("--top-frac", type=float, default=0.01)
    ap.add_argument("--sample", type=int, default=400000)
    args = ap.parse_args()

    files = sorted(args.dump_dir.glob("omega_task*.pt"))
    if len(files) < 2:
        raise SystemExit(f"need >=2 task dumps in {args.dump_dir}")
    blobs = [torch.load(f, map_location="cpu") for f in files]
    names = ["live"] + list(blobs[0].get("shadow_omega", {}))

    penalty = {n: np.zeros(blobs[0]["omega"].numel(), dtype=np.float64) for n in names}
    delta_sq_total = np.zeros_like(penalty["live"])
    for prev, cur in zip(blobs[:-1], blobs[1:]):
        dsq = cur["delta_sq"].numpy().astype(np.float64)
        delta_sq_total += dsq
        for name in names:
            field = (
                (prev["omega"] if name == "live" else prev["shadow_omega"][name])
                .numpy()
                .astype(np.float64)
            )
            penalty[name] += TUNED_LAMBDA.get(name, 1.0) * field * dsq

    omega = {
        n: (blobs[-1]["omega"] if n == "live" else blobs[-1]["shadow_omega"][n])
        .numpy()
        .astype(np.float64)
        for n in names
    }

    n = penalty["live"].size
    k = max(1, int(args.top_frac * n))
    idx = np.random.default_rng(0).choice(n, min(args.sample, n), replace=False)

    print(f"params={n:,}  tasks={len(files)}  top-k={k:,}")
    print("\nconcentration (share of mass in top 0.1% / 1% / 10%)")
    print(f"  {'delta^2':10s} {_concentration(delta_sq_total)}")
    for name in names:
        print(
            f"  {'Omega ' + name:16s} {_concentration(omega[name])}"
            f"   {'penalty ' + name:16s} {_concentration(penalty[name])}"
            f"   total={penalty[name].sum():.4e}"
        )

    print(
        f"\n{'pair':22s} {'rho(Omega)':>11s} {'rho(penalty)':>13s} "
        f"{'cos(Omega)':>11s} {'cos(penalty)':>13s} {'massovl(pen)':>13s}"
    )
    for a, b in itertools.combinations(names, 2):
        ro = float(spearmanr(omega[a][idx], omega[b][idx]).statistic)
        rp = float(spearmanr(penalty[a][idx], penalty[b][idx]).statistic)
        co = float(
            omega[a]
            @ omega[b]
            / max(np.linalg.norm(omega[a]) * np.linalg.norm(omega[b]), 1e-30)
        )
        cp = float(
            penalty[a]
            @ penalty[b]
            / max(np.linalg.norm(penalty[a]) * np.linalg.norm(penalty[b]), 1e-30)
        )
        ta = np.argpartition(-penalty[a], k)[:k]
        tb = np.argpartition(-penalty[b], k)[:k]
        inter = np.intersect1d(ta, tb)
        mass = float(penalty[a][inter].sum() / max(penalty[a][ta].sum(), 1e-30))
        print(
            f"{a + '/' + b:22s} {ro:11.3f} {rp:13.3f} {co:11.3f} {cp:13.3f} "
            f"{mass:13.3f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
