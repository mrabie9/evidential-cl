"""Cross-family test of the claim "C-MAML reduces to Res-ER", in both TIL and CIL.

The ablation tables show every C-MAML mechanism inert (TIL M1 -0.1, M3 -0.4, M4 -0.4;
CIL M1 -0.4, M3 +0.3, M4 +0.2), which says nothing inside C-MAML earns its score. This
tests the complementary claim directly: is C-MAML distinguishable from Res-ER at all?

TIL is tested three ways because the answer depends on what is matched:
  1. raw, bf16       -- both families post split-CE fix, both with the two-forward loop
  2. raw, fp32       -- same, without AMP (M0's fp32 pool is thin, n=3)
  3. matched averaging -- Res-ER given --eralg4_grad_avg 3, i.e. the same three stochastic
     forwards per step that C-MAML's meta_batches=3 loop performs incidentally. This is
     the comparison that isolates algorithm from variance reduction.

CIL pools are pinned to the same rerun directories as scripts/analyse_cil_ablations.py.

Usage:
    la-maml_env/bin/python scripts/analyse_cmaml_reduces_to_reser.py
"""

from __future__ import annotations

import glob
import itertools
import math
import os
import statistics as st
from results_txt import f1_stats

REPO = "/home/lunet/wsmr11/repos/evidential-cl"

CONTRASTS = [
    (
        "TIL bf16   M0 - E0",
        "logs/ablations/til/cmaml/M0",
        "logs/eralg4/e0_resER_joint_splitfix_se_til-*",
    ),
    (
        "TIL fp32   M0 - E0",
        "logs/cmaml/m0_noamp_joint_se_til-*",
        "logs/ablations/til/res-er/E0",
    ),
    (
        "TIL mixed  M0(bf16) - E0(fp32)",
        "logs/ablations/til/cmaml/M0",
        "logs/ablations/til/res-er/E0",
    ),
    (
        "TIL bf16   M0 - E0+K3",
        "logs/ablations/til/cmaml/M0",
        "logs/eralg4/e0_gradavg3_joint_se_til-*",
    ),
    (
        "TIL bf16   E0+K3 - E0   (does averaging help Res-ER?)",
        "logs/eralg4/e0_gradavg3_joint_se_til-*",
        "logs/eralg4/e0_resER_joint_splitfix_se_til-*",
    ),
    (
        "TIL bf16   E0+K5 - E0   (does it saturate at 3?)",
        "logs/eralg4/e0_gradavg5_joint_se_til-*",
        "logs/eralg4/e0_resER_joint_splitfix_se_til-*",
    ),
    (
        "TIL bf16   E0+K5 - E0+K3",
        "logs/eralg4/e0_gradavg5_joint_se_til-*",
        "logs/eralg4/e0_gradavg3_joint_se_til-*",
    ),
    (
        "CIL bf16   M0 - E0",
        "logs/ablations/cil/cmaml/M0",
        "logs/ablations/cil/res-er/E0",
    ),
]

METRICS = ["Final F1", "Diagonal F1", "Backward"]


def parse(path: str) -> dict[str, float]:
    out: dict[str, float] = {}
    for key, value in f1_stats(open(path).read()).items():
        if key in ('Diagonal F1', 'Final F1', 'Backward'):
            out[key] = value * 100
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
    for label, pa, pb in CONTRASTS:
        a, b = collect(pa), collect(pb)
        common = sorted(set(a) & set(b))
        print(f"=== {label} ===")
        if len(common) < 2:
            print(f"  insufficient common seeds (n={len(common)})\n")
            continue
        for key in METRICS:
            va = [a[s][key] for s in common]
            vb = [b[s][key] for s in common]
            d = [x - y for x, y in zip(va, vb)]
            m = st.mean(d)
            half = 1.96 * st.stdev(d) / math.sqrt(len(d))
            wins = sum(1 for v in d if v > 0)
            flag = ""
            if key == "Final F1":
                flag = "  <-- indistinguishable" if (m - half) <= 0 <= (m + half) else ""
            print(
                f"  {key:12s} {st.mean(va):6.2f} vs {st.mean(vb):6.2f}   "
                f"D {m:+5.2f} +-{half:4.2f}  p={perm_p(d):.3f}  {wins}/{len(d)}{flag}"
            )
        print(f"  n={len(common)} common seeds: {common}\n")

    print("p is the exact two-sided sign-flip test (floor 0.031 at n=6, 0.004 at n=9).")


if __name__ == "__main__":
    main()
