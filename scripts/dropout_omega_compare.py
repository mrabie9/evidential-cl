#!/usr/bin/env python
"""Does trunk dropout change the *shape* of the importance distribution?

Reads the `WOE_OMEGA_DUMP` directories written by
``scripts/run_dropout_omega_dumps.sh`` (one per dropout setting, seed 0) and
separates three questions the accuracy arms could not:

1. **Scale** -- total Omega, which the `[LC]` log line already showed moving
   2.5x. Scale alone rescales the anchor's operating point `b = 2 lr lambda
   Omega` and is fixable with lambda.
2. **Shape** -- Gini, top-1% / top-10% mass share, and the participation ratio
   `(sum w)^2 / sum w^2`, i.e. the effective number of parameters carrying the
   importance. These are scale-free, so they are the ones that answer the
   original hypothesis: does dropout stop importance concentrating on a few
   parameters?
3. **Support** -- Spearman rank agreement between arms on the cumulative Omega,
   and the overlap of their top-1% sets. Two arms can share a shape yet rank
   different parameters, which would matter as much as concentration.

Also reported per layer, because "a few parameters" and "a few layers" are
different claims and dropout is applied between stages.

Usage:
    python scripts/dropout_omega_compare.py
    python scripts/dropout_omega_compare.py --dumps <dir> --lam 240000
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ARMS = [
    ("none(p=0) s0", "drop_none_s0"),
    ("none(p=0) s39", "drop_none_s39"),
    ("flat(p=0.2) s0", "drop_flat02_s0"),
    ("flat(p=0.2) s39", "drop_flat02_s39"),
    ("sched(0.5-0.2) s0", "drop_sched_s0"),
]

# Seed 39 exists for two arms so the SUPPORT block has a null: within-arm,
# across-seed agreement is the yardstick the between-arm figure must beat. It
# does not -- see the memory note. Read the SHAPE block *paired within a seed*;
# the absolute participation ratio swings 4x across seeds even though the
# no-dropout/flat ratio is ~3x on both.


def gini(values: np.ndarray) -> float:
    """Gini of a non-negative vector (0 = uniform, 1 = all mass on one atom)."""
    ordered = np.sort(values)
    n = ordered.size
    total = ordered.sum()
    if total <= 0:
        return 0.0
    index = np.arange(1, n + 1)
    return float((2.0 * index - n - 1.0).dot(ordered) / (n * total))


def top_share(values: np.ndarray, frac: float) -> float:
    k = max(1, int(round(values.size * frac)))
    return float(np.sort(values)[-k:].sum() / max(values.sum(), 1e-12))


def participation(values: np.ndarray) -> float:
    """Effective number of parameters: (sum w)^2 / sum w^2.

    Equals n for a flat vector and 1 when a single parameter holds everything,
    so `participation / n` is a scale-free "what fraction of the network is
    actually carrying importance".
    """
    total = float(values.sum())
    sq = float(np.square(values.astype(np.float64)).sum())
    return total * total / max(sq, 1e-30)


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    ra = np.argsort(np.argsort(a)).astype(np.float64)
    rb = np.argsort(np.argsort(b)).astype(np.float64)
    ra -= ra.mean()
    rb -= rb.mean()
    denom = np.sqrt((ra * ra).sum() * (rb * rb).sum())
    return float((ra * rb).sum() / max(denom, 1e-30))


def load(
    dump_dir: Path,
) -> Tuple[List[np.ndarray], np.ndarray, List[Tuple[str, tuple]]]:
    files = sorted(dump_dir.glob("omega_task*.pt"))
    if not files:
        raise SystemExit(f"no omega dumps in {dump_dir}")
    per_task = []
    for path in files:
        blob = torch.load(path, map_location="cpu", weights_only=False)
        per_task.append(blob["task_omega"].numpy())
    final = torch.load(files[-1], map_location="cpu", weights_only=False)
    return per_task, final["omega"].numpy(), final["shapes"]


def layer_slices(shapes) -> Dict[str, slice]:
    out, start = {}, 0
    for name, shape in shapes:
        size = int(np.prod(shape)) if len(shape) else 1
        out[name] = slice(start, start + size)
        start += size
    return out


def stage_of(name: str) -> str:
    """Group parameters by the trunk stage the dropout sits after."""
    for stage in ("layer1", "layer2", "layer3", "layer4"):
        if name.startswith(stage + ".") or f".{stage}." in name:
            return stage
    if "fc" in name:
        return "fc"
    return "stem"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dumps", default=str(HERE / "logs" / "omega_dumps"))
    ap.add_argument("--lr", type=float, default=0.003)
    ap.add_argument("--lam", type=float, default=240000.0)
    args = ap.parse_args()

    root = Path(args.dumps)
    loaded = {}
    for label, sub in ARMS:
        path = root / sub
        if not path.exists() or not list(path.glob("omega_task*.pt")):
            print(f"[missing] {label}: {path}")
            continue
        loaded[label] = load(path)
    if not loaded:
        return 1

    print("SCALE  (cumulative Omega after the last task)")
    print(f"  {'arm':<19}{'total':>12}{'mean':>12}{'median':>12}{'p99':>12}")
    for label, (_, omega, _) in loaded.items():
        print(
            f"  {label:<19}{omega.sum():>12.4f}{omega.mean():>12.3e}"
            f"{np.median(omega):>12.3e}{np.percentile(omega, 99):>12.3e}"
        )

    print()
    print("SHAPE  (scale-free -- this is the question)")
    print(
        f"  {'arm':<19}{'gini':>8}{'top1%':>9}{'top10%':>9}"
        f"{'part.ratio':>12}{'part/n':>9}{'nonzero':>9}"
    )
    for label, (_, omega, _) in loaded.items():
        pr = participation(omega)
        print(
            f"  {label:<19}{gini(omega):>8.4f}{top_share(omega, 0.01):>9.4f}"
            f"{top_share(omega, 0.10):>9.4f}{pr:>12,.0f}{pr / omega.size:>9.4f}"
            f"{(omega > 0).mean():>9.4f}"
        )

    print()
    print("SHAPE per task (task path integral, not the cumulative Omega)")
    print(f"  {'arm':<19}{'stat':<8}" + "".join(f"{t:>8}" for t in range(10)))
    for label, (per_task, _, _) in loaded.items():
        for stat, fn in (("gini", gini), ("top1%", lambda v: top_share(v, 0.01))):
            row = "".join(f"{fn(v):>8.4f}" for v in per_task)
            print(f"  {label:<19}{stat:<8}{row}")

    print()
    print("SUPPORT  (do the arms rank the same parameters?)")
    labels = list(loaded)
    print(f"  {'pair':<34}{'spearman':>10}{'top1% overlap':>15}")
    for i, a in enumerate(labels):
        for b in labels[i + 1 :]:
            oa, ob = loaded[a][1], loaded[b][1]
            k = max(1, int(round(oa.size * 0.01)))
            sa = set(np.argpartition(oa, -k)[-k:].tolist())
            sb = set(np.argpartition(ob, -k)[-k:].tolist())
            overlap = len(sa & sb) / k
            print(f"  {a + ' vs ' + b:<34}{spearman(oa, ob):>10.4f}{overlap:>15.4f}")

    print()
    print(
        f"ANCHOR OPERATING POINT  b = 2 lr lambda Omega  (lr={args.lr}, lam={args.lam:g})"
    )
    print(
        f"  {'arm':<19}{'b<0.1':>9}{'0.1-1':>9}{'1-10':>9}{'b>10':>9}{'median b':>12}"
    )
    for label, (_, omega, _) in loaded.items():
        b = 2.0 * args.lr * args.lam * omega
        print(
            f"  {label:<19}{(b < 0.1).mean():>9.4f}{((b >= 0.1) & (b < 1)).mean():>9.4f}"
            f"{((b >= 1) & (b < 10)).mean():>9.4f}{(b >= 10).mean():>9.4f}"
            f"{np.median(b):>12.3e}"
        )

    print()
    print("MASS BY TRUNK STAGE  (share of cumulative Omega)")
    stages = ["stem", "layer1", "layer2", "layer3", "layer4", "fc"]
    print(f"  {'arm':<19}" + "".join(f"{s:>9}" for s in stages))
    for label, (_, omega, shapes) in loaded.items():
        slices = layer_slices(shapes)
        share = {s: 0.0 for s in stages}
        for name, sl in slices.items():
            share[stage_of(name)] += float(omega[sl].sum())
        total = max(sum(share.values()), 1e-12)
        print(f"  {label:<19}" + "".join(f"{share[s] / total:>9.4f}" for s in stages))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
