"""TIL test of the claim "B5a reduces to Ring-ER + distillation".

B5a is BCL-Dual with the bilevel round collapsed to a single loop at B0's inner-loop budget
(inner_steps 2). What survives is replay + KL distillation on frozen soft targets, which is
what er_ring --er_distill computes: model/er_ring.py:473 scales self.reg * KL against frozen
buffer targets, and self.reg = memory_strength in both er_ring.py:61 and bcl_dual.py:89.

The no-distill control is what makes this decisive. A null result against the distill arm
alone would be ambiguous -- it could mean the two are equivalent, or merely that neither
term matters at this operating point. The claim holds only if B5a is indistinguishable from
the distill arm AND separated from the control.

Residual known difference: B5a retains BCL's second (meta) memory buffer, which er_ring has
no equivalent of. TIL row B3 measures that buffer at -0.4 fcl (inconclusive).

Usage:
    la-maml_env/bin/python scripts/analyse_b5a_reduces_to_ering.py
"""

from __future__ import annotations

import glob
import itertools
import math
import os
import statistics as st
from results_txt import f1_stats

REPO = "/home/lunet/wsmr11/repos/evidential-cl"
METRICS = ["Final F1", "Diagonal F1", "Backward"]

POOLS = {
    "B0  BCL-Dual (full)": ["logs/ablations/til/bcl-dual/B0"],
    # B5a pools the original ablation dirs plus the 2026-07-30 n=9 top-up (seeds 1,2,3),
    # which is an identical bcl_singlelevel/is2 config under a separate expt name.
    "B5a BCL-Dual single-loop": [
        "logs/ablations/til/bcl-dual/B5a",
        "logs/bcl_dual/b5a_topup_se_til-*",
    ],
    "Ring-ER + distill": ["logs/er_ring/ering_distill_b5amatch_se_til-*"],
    "Ring-ER no distill": ["logs/er_ring/ering_nodistill_b5amatch_se_til-*"],
}

CONTRASTS = [
    ("B5a - (Ring-ER + distill)   [equivalence test]", "B5a BCL-Dual single-loop", "Ring-ER + distill"),
    ("B5a - (Ring-ER no distill)  [control]", "B5a BCL-Dual single-loop", "Ring-ER no distill"),
    ("distill - no distill        [is distill load-bearing?]", "Ring-ER + distill", "Ring-ER no distill"),
    ("B0  - (Ring-ER + distill)   [full BCL vs the reduction]", "B0  BCL-Dual (full)", "Ring-ER + distill"),
]


def parse(path: str) -> dict[str, float]:
    out: dict[str, float] = {}
    for key, value in f1_stats(open(path).read()).items():
        if key in ('Diagonal F1', 'Final F1', 'Backward'):
            out[key] = value * 100
    return out


def collect(patterns: list[str]) -> dict[int, dict[str, float]]:
    """Pool seed dirs across every pattern. First occurrence of a seed wins, so an
    original pool is never overwritten by a top-up that happens to repeat a seed."""
    seeds: dict[int, dict[str, float]] = {}
    for pattern in patterns:
        for base in sorted(glob.glob(os.path.join(REPO, pattern))):
            for dp, _d, files in os.walk(base):
                leaf = os.path.basename(dp)
                if "results.txt" in files and leaf.isdigit():
                    rec = parse(os.path.join(dp, "results.txt"))
                    if "Final F1" in rec:
                        seeds.setdefault(int(leaf), rec)
    return seeds


def perm_p(d: list[float]) -> float:
    n = len(d)
    if n > 20:
        return float("nan")
    obs = abs(sum(d) / n)
    return sum(
        1
        for s in itertools.product((1, -1), repeat=n)
        if abs(sum(a * b for a, b in zip(s, d)) / n) >= obs - 1e-12
    ) / 2**n


def main() -> None:
    data = {k: collect(v) for k, v in POOLS.items()}
    print("TIL, single-epoch, lr 0.001, inner_steps 2, bf16. Percent, mean +- sample sd.\n")
    for k, s in data.items():
        f1 = [s[x]["Final F1"] for x in sorted(s)]
        bwt = [s[x]["Backward"] for x in sorted(s)]
        dia = [s[x]["Diagonal F1"] for x in sorted(s)]
        print(
            f"  {k:26s} F1 {st.mean(f1):5.2f}+-{st.stdev(f1):4.2f}({len(f1)})  "
            f"Diag {st.mean(dia):5.2f}  BWT {st.mean(bwt):6.2f}"
        )

    for label, ka, kb in CONTRASTS:
        a, b = data[ka], data[kb]
        common = sorted(set(a) & set(b))
        print(f"\n=== {label} ===")
        if len(common) < 2:
            print(f"  insufficient common seeds (n={len(common)})")
            continue
        for key in METRICS:
            d = [a[s][key] - b[s][key] for s in common]
            m = st.mean(d)
            half = 1.96 * st.stdev(d) / math.sqrt(len(d))
            flag = ""
            if key == "Final F1":
                flag = (
                    "  <-- indistinguishable"
                    if (m - half) <= 0 <= (m + half)
                    else "  <-- SEPARATED"
                )
            print(
                f"  {key:12s} D {m:+5.2f} +-{half:4.2f}  p={perm_p(d):.3f}  "
                f"{sum(1 for v in d if v > 0)}/{len(d)}{flag}"
            )
        print(f"  n={len(common)}: {common}")

    print("\np is the exact two-sided sign-flip test (floor 0.031 at n=6).")


if __name__ == "__main__":
    main()
