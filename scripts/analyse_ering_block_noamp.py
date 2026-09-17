"""Res-ER ablation block (E0/E1/E2) recomputed without AMP, for tab:ablation_optim.

E0 = Res-ER (eralg4 --eralg4_joint_er), E1 = static ring (er_ring), E2 = dynamic ring
(er_ring --er_dynamic_ring). All at lr 0.01, inner_steps 2, memory_loss_lambda 1,
single-epoch TIL, --no-amp.

The three rows form a self-consistent block: er_ring computes replay and current losses
from separate forwards (model/er_ring.py:443-491), matching eralg4's --eralg4_joint_er
two-forward loop, so neither the BN-mixing nor the pooled-CE defect distinguishes them.

Holm correction is applied across the 2 ablation rows in the family, matching the protocol
in docs/ablation_studies.tex sec:stat_protocol.

Usage:
    la-maml_env/bin/python scripts/analyse_ering_block_noamp.py
"""

from __future__ import annotations

import glob
import itertools
import math
import os
import re
import statistics as st

REPO = "/home/lunet/wsmr11/repos/evidential-cl"
DELTA = 1.0  # smallest effect of interest, F1 points

ROWS = {
    "E0": ("Res-ER, full method", "logs/ablations/til/res-er/E0"),
    "E1": ("reservoir -> static ring", "logs/ablations/til/res-er/E1"),
    "E2": ("reservoir -> dynamic ring", "logs/ablations/til/res-er/E2"),
}


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
    obs = abs(sum(d) / n)
    hits = sum(
        1
        for s in itertools.product((1, -1), repeat=n)
        if abs(sum(a * b for a, b in zip(s, d)) / n) >= obs - 1e-12
    )
    return hits / 2**n


def t_p(d: list[float]) -> float:
    """Two-sided paired t-test p-value via the normal approximation's exact t CDF."""
    n = len(d)
    m, sd = st.mean(d), st.stdev(d)
    if sd == 0:
        return 0.0
    t = abs(m) / (sd / math.sqrt(n))
    df = n - 1
    # regularised incomplete beta for the t distribution
    x = df / (df + t * t)
    return _betainc(df / 2, 0.5, x)


def _betainc(a: float, b: float, x: float) -> float:
    if x <= 0:
        return 0.0
    if x >= 1:
        return 1.0
    lbeta = math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b)
    front = math.exp(a * math.log(x) + b * math.log(1 - x) - lbeta) / a
    f, c, d = 1.0, 1.0, 0.0
    for i in range(0, 300):
        m2 = i // 2
        if i == 0:
            num = 1.0
        elif i % 2 == 0:
            num = (m2 * (b - m2) * x) / ((a + 2 * m2 - 1) * (a + 2 * m2))
        else:
            num = -((a + m2) * (a + b + m2) * x) / ((a + 2 * m2) * (a + 2 * m2 + 1))
        d = 1.0 + num * d
        d = 1e-30 if abs(d) < 1e-30 else d
        d = 1.0 / d
        c = 1.0 + num / c
        c = 1e-30 if abs(c) < 1e-30 else c
        f *= c * d
        if abs(1.0 - c * d) < 1e-12:
            break
    return front * (f - 1.0)


def verdict(lo: float, hi: float, ph: float, pp: float) -> str:
    if ph < 0.05 and pp < 0.05 and not (lo <= 0 <= hi):
        return "load-bearing"
    if not (lo <= 0 <= hi):
        return "borderline"
    if lo >= -DELTA and hi <= DELTA:
        return "inert"
    return "inconclusive"


def main() -> None:
    data = {k: collect(v[1]) for k, v in ROWS.items()}
    base = data["E0"]

    print("Res-ER block, --no-amp, lr 0.01, inner_steps 2, single-epoch TIL.")
    print("Percent, mean +- sample sd (n).\n")
    for rid, (desc, _) in ROWS.items():
        s = data[rid]
        f1 = [s[k]["Final F1"] for k in sorted(s)]
        bwt = [s[k]["Backward"] for k in sorted(s)]
        print(
            f"  {rid}  F1 {st.mean(f1):5.2f}+-{st.stdev(f1):4.2f}({len(f1)})   "
            f"BWT {st.mean(bwt):5.2f}+-{st.stdev(bwt):4.2f}   {desc}"
        )

    # Holm across the 2 ablation rows
    raw = {}
    for rid in ("E1", "E2"):
        common = sorted(set(base) & set(data[rid]))
        d = [data[rid][s]["Final F1"] - base[s]["Final F1"] for s in common]
        raw[rid] = (d, common)

    order = sorted(raw, key=lambda r: t_p(raw[r][0]))
    holm = {}
    prev = 0.0
    for i, rid in enumerate(order):
        p = t_p(raw[rid][0]) * (len(order) - i)
        prev = max(prev, min(p, 1.0))
        holm[rid] = prev

    print("\n  row  dF1     95% CI            p_Holm  p_perm  verdict        n")
    for rid in ("E1", "E2"):
        d, common = raw[rid]
        m = st.mean(d)
        half = 1.96 * st.stdev(d) / math.sqrt(len(d))
        lo, hi = m - half, m + half
        pp = perm_p(d)
        print(
            f"  {rid}   {m:+5.2f}  [{lo:+5.2f}, {hi:+5.2f}]   {holm[rid]:6.3f}  "
            f"{pp:6.3f}  {verdict(lo, hi, holm[rid], pp):14s} {len(common)}"
        )

    print("\n  Paired on common seeds; E0 and E2 have 9 seeds, E1 has 6.")
    print("  CI is the 95% normal interval on the paired difference; delta = 1.0 F1 pt.")


if __name__ == "__main__":
    main()
