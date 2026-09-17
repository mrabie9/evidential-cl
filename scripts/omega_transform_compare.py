#!/usr/bin/env python
"""Does ``abs`` hide a disagreement that ``relu`` would expose?

The per-window gradient geometry says the tracked scalars do not merely differ
in magnitude: ``cos(h_i2, h_ce)`` is **negative at every task** (mean -0.28,
range -0.59..-0.11, with ``ce`` carrying the module's sign convention of
*negated* loss). Two anti-aligned gradient fields should build different
importance -- yet ``Omega(i2)`` and ``Omega(ce)`` agree at Spearman 0.71.

``woe_omega_transform`` is the candidate reason. The path integral is signed;
``abs`` keeps ``|sum_w h.delta|`` and throws the direction away, so a scalar
that *fell* while a parameter moved is scored exactly like one that rose. Under
``relu`` the sign is load-bearing. If ``abs`` is what makes the scalars
interchangeable, between-scalar agreement should drop sharply when ``Omega`` is
rebuilt under ``relu`` on the identical trajectory.

Rebuilds both from the signed per-task path integrals in the dump, so no
retraining and no lambda retuning is involved -- the trajectory is held fixed
and only the consolidation transform changes. This tests the *field* claim; it
cannot test the accuracy claim, which needs each transform to drive its own run.

Usage:
    python scripts/omega_transform_compare.py scripts/logs/omega_dumps/shadow_s0
"""

from __future__ import annotations

import argparse
import itertools
from pathlib import Path
from typing import Dict

import numpy as np
import torch
from scipy.stats import spearmanr


def _fields(blobs, transform: str, xi: float) -> Dict[str, np.ndarray]:
    names = ["live"] + list(blobs[0].get("shadow_signed", {}))
    out = {n: None for n in names}
    for blob in blobs:
        denom = blob["delta_sq"].numpy().astype(np.float64) + xi
        for name in names:
            raw = (
                (blob["signed"] if name == "live" else blob["shadow_signed"][name])
                .numpy()
                .astype(np.float64)
            )
            projected = np.abs(raw) if transform == "abs" else np.maximum(raw, 0.0)
            contribution = projected / denom
            out[name] = contribution if out[name] is None else out[name] + contribution
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("dump_dir", type=Path)
    ap.add_argument("--sample", type=int, default=400000)
    args = ap.parse_args()

    files = sorted(args.dump_dir.glob("omega_task*.pt"))
    blobs = [torch.load(f, map_location="cpu") for f in files]
    if "signed" not in blobs[0]:
        raise SystemExit("dump predates the signed path integral; rerun needed")
    xi = float(blobs[0].get("xi", 1e-3))

    built = {t: _fields(blobs, t, xi) for t in ("abs", "relu")}
    names = list(built["abs"])
    n = built["abs"]["live"].size
    idx = np.random.default_rng(0).choice(n, min(args.sample, n), replace=False)

    print(f"params={n:,}  tasks={len(files)}  xi={xi:g}")
    print("\npositive share of the signed path integral (per scalar, all tasks)")
    for name in names:
        raw = np.concatenate(
            [
                (b["signed"] if name == "live" else b["shadow_signed"][name])
                .numpy()
                .astype(np.float64)
                for b in blobs
            ]
        )
        print(
            f"  {name:9s} frac>0={np.mean(raw > 0):.4f}  "
            f"mass+={np.maximum(raw, 0).sum() / np.abs(raw).sum():.4f}"
        )

    print(
        f"\n{'pair':22s} {'rho(abs)':>9s} {'rho(relu)':>10s} {'delta':>8s} "
        f"{'cos(abs)':>9s} {'cos(relu)':>10s}"
    )
    for a, b in itertools.combinations(names, 2):
        row = []
        for t in ("abs", "relu"):
            x, y = built[t][a], built[t][b]
            row.append(
                (
                    float(spearmanr(x[idx], y[idx]).statistic),
                    float(x @ y / max(np.linalg.norm(x) * np.linalg.norm(y), 1e-30)),
                )
            )
        print(
            f"{a + '/' + b:22s} {row[0][0]:9.3f} {row[1][0]:10.3f} "
            f"{row[1][0] - row[0][0]:+8.3f} {row[0][1]:9.3f} {row[1][1]:10.3f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
