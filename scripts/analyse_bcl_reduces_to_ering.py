"""CIL test of the claim "BCL-Dual reduces to Ring-ER".

Neither CIL Ring-ER grid row can serve as the comparator: both run at lr 0.01 against BCL's
0.001, and E2 also swaps the buffer (dynamic vs static ring). These runs match B0 exactly on
both axes, on B0's own seed set.

Two Ring-ER arms, because BCL's bilevel round performs 2 SGD steps per inner step, so B0 at
inner_steps 2 spends a 4-step budget (why the B5b flattening row matches at is4):
  is2 -- config-matched  (same inner_steps, half the SGD steps)
  is4 -- budget-matched  (same total SGD steps)

Usage:
    la-maml_env/bin/python scripts/analyse_bcl_reduces_to_ering.py
"""

from __future__ import annotations

import glob
import itertools
import math
import os
import re
import statistics as st

REPO = "/home/lunet/wsmr11/repos/evidential-cl"

B0 = ("B0 BCL-Dual", "logs/ablations/cil/bcl-dual/B0")
ARMS = [
    ("Ring-ER is2 (config-matched)", "logs/er_ring/ering_static_lr001_is2_bclmatch_se_cil-*"),
    ("Ring-ER is4 (budget-matched)", "logs/er_ring/ering_static_lr001_is4_bclmatch_se_cil-*"),
    ("E1 Ring-ER static lr.01", "logs/ablations/cil/res-er/E1"),
    ("E2 Ring-ER dynring lr.01", "logs/ablations/cil/res-er/E2"),
]
METRICS = ["Final F1", "Diagonal F1", "Backward"]


def parse(path: str) -> dict[str, float]:
    out: dict[str, float] = {}
    for line in open(path):
        m = re.match(r"(Diagonal F1|Final F1|Backward):\s+(-?[\d.]+)", line)
        if m:
            out[m.group(1)] = float(m.group(2)) * 100
    return out


def collect(pattern: str) -> dict[int, dict[str, float]]:
    seeds: dict[int, dict[str, float]] = {}
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
    base = collect(B0[1])
    f1 = [base[s]["Final F1"] for s in sorted(base)]
    bwt = [base[s]["Backward"] for s in sorted(base)]
    print("CIL, single-epoch, bf16, lr 0.001 throughout. Percent, mean +- sample sd.\n")
    print(f"  {B0[0]:32s} F1 {st.mean(f1):5.2f}+-{st.stdev(f1):4.2f}({len(f1)})  BWT {st.mean(bwt):6.2f}")
    for label, pat in ARMS:
        s = collect(pat)
        if not s:
            print(f"  {label:32s} -- no runs --")
            continue
        a = [s[k]["Final F1"] for k in sorted(s)]
        b = [s[k]["Backward"] for k in sorted(s)]
        print(f"  {label:32s} F1 {st.mean(a):5.2f}+-{st.stdev(a):4.2f}({len(a)})  BWT {st.mean(b):6.2f}")

    print("\n  paired vs B0 (positive => Ring-ER better)")
    for label, pat in ARMS:
        s = collect(pat)
        common = sorted(set(base) & set(s))
        if len(common) < 2:
            continue
        print(f"\n  --- {label}, n={len(common)} ---")
        for key in METRICS:
            d = [s[k][key] - base[k][key] for k in common]
            m = st.mean(d)
            half = 1.96 * st.stdev(d) / math.sqrt(len(d))
            flag = "  <-- indistinguishable" if (m - half) <= 0 <= (m + half) and key == "Final F1" else ""
            print(
                f"    {key:12s} D {m:+5.2f} +-{half:4.2f}  p={perm_p(d):.3f}  "
                f"{sum(1 for v in d if v > 0)}/{len(d)}{flag}"
            )

    print("\n  p is the exact two-sided sign-flip test (floor 0.004 at n=9).")


if __name__ == "__main__":
    main()
