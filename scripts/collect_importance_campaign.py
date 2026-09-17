"""Collect the measured-vs-uniform Omega comparison across methods and regimes.

The question: is per-parameter importance carrying anything on this benchmark, or
is every anchor here L2-SP with a decorative Omega? The statistic that answers it
is the **gap** between a method's best measured-Omega cell and its best
uniform-Omega cell, at matched anchor form (proximal) and matched schedule. A gap
near zero means the ranking is doing nothing.

Two regimes, because a competing reading of a small gap is that importance is
merely under-estimated at one epoch: SI's path integral and EWC's Fisher both get
~3x more steps per task at --n_epochs 3. If the gap widens there, the estimate was
noisy; if it does not, importance is uninformative however well it is estimated.

Reference already on record: WoE-SI one-shot, abs Omega 0.5008 vs uniform 0.4345,
a gap of +0.066.

Usage:
    python scripts/collect_importance_campaign.py
"""

from __future__ import annotations

import glob
import os
import re
from typing import Dict, List, Optional, Tuple

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DIAGONAL_RE = re.compile(r"^Diagonal\s+\S+:\s+(-?[\d.]+)", re.MULTILINE)
FINAL_RE = re.compile(r"^Final\s+\S+:\s+(-?[\d.]+)", re.MULTILINE)
BACKWARD_RE = re.compile(r"^Backward:\s+(-?[\d.]+)", re.MULTILINE)

# (label, experiment-name glob). One entry per (method, Omega type, regime) cell.
CELLS: List[Tuple[str, str, str, str]] = [
    # regime,    method,   omega,      glob
    ("1 epoch", "si", "measured", "anchorbench-oneshot_si_proximal_*"),
    ("1 epoch", "si", "uniform", "omguni_si_lam*"),
    ("1 epoch", "ewc", "measured", "anchorbench-oneshot_ewc_proximal_*"),
    ("1 epoch", "ewc", "uniform", "omguni_ewc_lam*"),
    ("3 epoch", "si", "measured", "me3_si_meas_lam*"),
    ("3 epoch", "si", "uniform", "me3_si_unif_lam*"),
    ("3 epoch", "ewc", "measured", "me3_ewc_meas_lam*"),
    ("3 epoch", "ewc", "uniform", "me3_ewc_unif_lam*"),
    ("3 epoch", "woe_si", "measured", "me3_woe_meas_lam*"),
    ("3 epoch", "woe_si", "uniform", "me3_woe_unif_lam*"),
]

# Measured elsewhere, on the same one-shot schedule and the same proximal anchor.
# Kept here so the 2x2 reads in one place rather than across three documents.
RECORDED: Dict[Tuple[str, str, str], float] = {
    ("1 epoch", "woe_si", "measured"): 0.5008,  # abs Omega, lambda 2.4e5
    # The A9 write-up quotes 0.4345 for the uniform control, but that is not the
    # best cell of its own sweep: lambda=3 scores 0.4441 (0.3/3/30/100/300/1000 =
    # 0.3792/0.4441/0.3378/0.2543/0.2029/0.1777, an interior peak). The gap must
    # be read against the best uniform cell or it is inflated by 0.0096.
    ("1 epoch", "woe_si", "uniform"): 0.4441,
}


def parse(path: str) -> Dict[str, Optional[float]]:
    """Pull the summary metrics out of one ``results.txt``."""
    with open(path, encoding="utf-8") as handle:
        text = handle.read()

    def first(pattern: re.Pattern[str]) -> Optional[float]:
        match = pattern.search(text)
        return float(match.group(1)) if match else None

    return {
        "diagonal": first(DIAGONAL_RE),
        "final": first(FINAL_RE),
        "bwt": first(BACKWARD_RE),
    }


def cells_for(pattern: str) -> List[Tuple[str, Dict[str, Optional[float]]]]:
    """Every finished run matching one experiment-name glob, newest per name."""
    found: Dict[str, Tuple[float, str]] = {}
    for path in glob.glob(os.path.join(ROOT, "logs", "*", pattern, "*", "results.txt")):
        name = os.path.basename(os.path.dirname(os.path.dirname(path)))
        stamp = os.path.getmtime(path)
        if name not in found or stamp > found[name][0]:
            found[name] = (stamp, path)
    return [(name, parse(path)) for name, (_, path) in sorted(found.items())]


def main() -> int:
    best: Dict[Tuple[str, str, str], Tuple[str, float]] = {}
    print("=== every finished cell ===")
    for regime, method, omega, pattern in CELLS:
        rows = cells_for(pattern)
        if not rows:
            print(f"\n{regime:>8} {method:>7} {omega:>9}: (no runs yet)")
            continue
        print(f"\n{regime:>8} {method:>7} {omega:>9}:")
        for name, metrics in rows:
            final = metrics["final"]
            if final is None:
                print(f"    {name:<44} (incomplete)")
                continue
            print(
                f"    {name:<44} diag {metrics['diagonal']:.4f}  "
                f"final {final:.4f}  bwt {metrics['bwt']:+.4f}"
            )
            key = (regime, method, omega)
            if key not in best or final > best[key][1]:
                best[key] = (name, final)

    print("\n\n=== the deciding statistic: best measured - best uniform ===")
    print(f"{'regime':>8} {'method':>7} {'measured':>10} {'uniform':>10} {'gap':>9}")
    for regime in ("1 epoch", "3 epoch"):
        for method in ("si", "ewc", "woe_si"):
            meas = best.get((regime, method, "measured"), (None, None))[1]
            unif = best.get((regime, method, "uniform"), (None, None))[1]
            meas = (
                meas if meas is not None else RECORDED.get((regime, method, "measured"))
            )
            unif = (
                unif if unif is not None else RECORDED.get((regime, method, "uniform"))
            )
            if meas is None or unif is None:
                have = (
                    "measured"
                    if meas is not None
                    else "uniform" if unif is not None else "neither"
                )
                print(
                    f"{regime:>8} {method:>7} {'-':>10} {'-':>10} {'pending':>9}  ({have} only)"
                )
                continue
            print(
                f"{regime:>8} {method:>7} {meas:>10.4f} {unif:>10.4f} {meas - unif:>+9.4f}"
            )
    print(
        "\nA gap near zero means the Omega ranking is inert and the anchor is "
        "L2-SP.\nA gap that grows from 1 to 3 epochs means the importance was "
        "merely under-estimated."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
