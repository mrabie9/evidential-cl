#!/usr/bin/env python
"""Would a shorter task sequence have produced the same conclusions?

The accuracy matrix is causal: row ``t`` is the evaluation after training task
``t``, and nothing about tasks ``> t`` influences it. So the top-left ``k x k``
block of a finished 10-task run is **exactly** what a ``k``-task run with the
same seed would have reported -- not an approximation. That makes this question
answerable from runs already on disk, with no new compute.

For every run it recomputes diagonal / final / BWT at each prefix length, then
for each registered comparison reports the gap at each ``k`` against the gap at
``k=10``. A comparison "survives" truncation if the sign is preserved and the
magnitude is not badly distorted.

Also reports the actual speed-up, which is *not* ``k/10``: the per-task step
counts are wildly uneven, so truncating to four tasks removes far less than
60% of the work.

Usage:
    python scripts/truncated_sequence_check.py
    python scripts/truncated_sequence_check.py --ks 3 4 5 6 8 10
"""

from __future__ import annotations

import argparse
import glob
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np


def read_matrix(run_dir: Path) -> Optional[np.ndarray]:
    res = run_dir / "results.txt"
    if not res.exists():
        return None
    rows: List[List[float]] = []
    for line in res.read_text().splitlines():
        line = line.strip()
        if not line or line == "|" or line[0].isalpha():
            continue
        try:
            rows.append([float(v) for v in line.split()])
        except ValueError:
            break
    if not rows:
        return None
    width = max(len(r) for r in rows)
    square = [r for r in rows if len(r) == width]
    return np.asarray(square[1:] if len(square) > width else square, dtype=float)


def metrics(matrix: np.ndarray, k: int) -> Optional[Tuple[float, float, float]]:
    """diag / final / BWT as a k-task run would have reported them."""
    if matrix.shape[0] < k:
        return None
    block = matrix[:k, :k]
    diag = np.array([block[t, t] for t in range(k)])
    final = block[k - 1]
    return float(diag.mean()), float(final.mean()), float((final - diag).mean())


def find(pattern: str, seed: str = "0") -> Optional[np.ndarray]:
    hits = sorted(glob.glob(f"logs/*/{pattern}-*/{seed}"))
    if not hits:
        return None
    return read_matrix(Path(hits[-1]))


def steps_by_task() -> Dict[int, int]:
    hits = sorted(glob.glob("logs/woe_si_lc/caut_sum_lam240000_s0-*/0/terminal.log"))
    if not hits:
        return {}
    txt = Path(hits[-1]).read_text()
    return {
        int(t): int(s)
        for t, s in re.findall(r"\[LC\] split task=(\d+) steps=(\d+)", txt)
    }


COMPARISONS = [
    # label, arm A pattern, arm B pattern, seed
    ("A7  sum vs max  (lam 2.4e5)", "caut_sum_lam240000_s0", "caut_max_lam240000_s0", "0"),
    ("A7  sum vs max  seed 39", "caut_sum_lam240000_s39", "caut_max_lam240000_s39", "39"),
    ("A7  sum vs max  seed 55", "caut_sum_lam240000_s55", "caut_max_lam240000_s55", "55"),
    ("A7  sum_norm vs max_norm (peaks)", "cn_sumnorm_lam480000", "cn_maxnorm_lam240000", "0"),
    ("xi  1e-3 vs 1e-6 (peaks)", "cn_sum_regression", "xifix_xi1e-6_lam1350", "0"),
    ("xi  1e-3 vs 1e-8 (peaks)", "cn_sum_regression", "xifix_xi1e-8_lam135", "0"),
    ("path integral vs uniform s0", "caut_sum_lam240000_s0", "unif_lam3", "0"),
    ("path integral vs uniform s39", "caut_sum_lam240000_s39", "unif_lam3_s39", "39"),
    ("path integral vs uniform s55", "caut_sum_lam240000_s55", "unif_lam3_s55", "55"),
    ("B6ao i2 vs ce (peaks)", "caut_sum_lam240000_s0", "b6ao_ce_lam75", "0"),
    ("B6ao i2 vs z2 (peaks)", "caut_sum_lam240000_s0", "b6ao_z2_lam9", "0"),
    ("B6ao i2 vs phi2 (peaks)", "caut_sum_lam240000_s0", "b6ao_phi2_lam55", "0"),
]

LAMBDA_CURVES = {
    "A7 max": [
        ("1.5e4", "caut_max_lam15000_s0"), ("3e4", "caut_max_lam30000_s0"),
        ("6e4", "caut_max_lam60000_s0"), ("1.2e5", "caut_max_lam120000_s0"),
        ("2.4e5", "caut_max_lam240000_s0"), ("3.6e5", "caut_max_lam360000_s0"),
        ("7.2e5", "caut_max_lam720000_s0"),
    ],
    "xi=1e-8": [
        ("1.5", "xifix_xi1e-8_lam1.5"), ("5", "xifix_xi1e-8_lam5"),
        ("15", "xifix_xi1e-8_lam15"), ("45", "xifix_xi1e-8_lam45"),
        ("135", "xifix_xi1e-8_lam135"), ("405", "xifix_xi1e-8_lam405"),
        ("1215", "xifix_xi1e-8_lam1215"),
    ],
    "uniform": [
        ("0.3", "unif_lam0.3"), ("3", "unif_lam3"), ("30", "unif_lam30"),
        ("100", "unif_lam100"), ("300", "unif_lam300"),
    ],
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ks", type=int, nargs="+", default=[3, 4, 5, 6, 8, 10])
    args = ap.parse_args()
    ks = args.ks

    steps = steps_by_task()
    if steps:
        total = sum(steps.values())
        print("cost of a prefix (from the per-task step counts):")
        header = "  " + "".join(f"{f'k={k}':>9}" for k in ks)
        print(header)
        frac = [
            sum(v for t, v in steps.items() if t < k) / total for k in ks
        ]
        print("  " + "".join(f"{f:>9.2f}" for f in frac))
        print(f"  per-task steps: {[steps.get(t) for t in sorted(steps)]}")
        print()

    print("gap = arm A final - arm B final, at each prefix length")
    print("  " + f"{'comparison':<36}" + "".join(f"{f'k={k}':>9}" for k in ks) + f"{'sign ok':>9}")
    print("  " + "-" * (36 + 9 * len(ks) + 9))
    for label, a_pat, b_pat, seed in COMPARISONS:
        ma, mb = find(a_pat, seed), find(b_pat, seed)
        if ma is None or mb is None:
            print(f"  {label:<36}  (missing)")
            continue
        gaps = []
        for k in ks:
            xa, xb = metrics(ma, k), metrics(mb, k)
            gaps.append(None if xa is None or xb is None else xa[1] - xb[1])
        full = gaps[-1]
        ok = all(
            g is not None and full is not None and np.sign(g) == np.sign(full)
            for g in gaps
        )
        cells = "".join(f"{g:>+9.4f}" if g is not None else f"{'--':>9}" for g in gaps)
        print(f"  {label:<36}{cells}{('yes' if ok else 'NO'):>9}")

    print()
    print("does the optimal lambda move with sequence length?")
    for name, cells in LAMBDA_CURVES.items():
        mats = [(lab, find(pat)) for lab, pat in cells]
        mats = [(lab, m) for lab, m in mats if m is not None]
        if not mats:
            continue
        print(f"  {name}:")
        for k in ks:
            vals = [(lab, metrics(m, k)) for lab, m in mats]
            vals = [(lab, v[1]) for lab, v in vals if v is not None]
            if not vals:
                continue
            best = max(vals, key=lambda x: x[1])
            print(
                f"    k={k:<3} argmax lambda = {best[0]:<7} "
                f"(final {best[1]:.4f})   curve "
                + " ".join(f"{v:.3f}" for _, v in vals)
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
