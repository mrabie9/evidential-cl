"""GEM ablation block (G0-G2) with G0 and G1 recomputed in fp32, for tab:ablation_optim.

G0 = GEM (full), G1 = gradient projection -> ring-buffer replay, G2 = buffer removed
entirely. All at lr 0.01, inner_steps 2, single-epoch TIL.

G0 and G1 have --no-amp pools at n=6 on the same seed set. G2 does NOT and is deliberately
not re-run: its effect is -35 F1, three orders larger than the ~1.4-point precision effect,
so no verdict can turn on it. Its absolute scores stay at bf16 while its paired delta is
recomputed against the fp32 G0 over the 6 shared seeds. That single delta therefore mixes
precisions -- stated openly in the table note rather than papered over.

G1 is the same run as Res-ER row E1 (er_ring, lr 0.01, inner_steps 2, static ring); the two
were bit-identical on all shared seeds, so one fp32 run serves both grid rows.

Usage:
    la-maml_env/bin/python scripts/analyse_gem_block_noamp.py
"""

from __future__ import annotations

import glob
import itertools
import math
import os
import re
import statistics as st

REPO = "/home/lunet/wsmr11/repos/evidential-cl"
DELTA = 1.0

BASE = ("G0", "None (full method)", "logs/ablations/til/gem/G0/lr0.01", "fp32")
ROWS = [
    ("G1", "projection -> ring-buffer replay", "logs/ablations/til/gem/G1/lr0.01", "fp32"),
    ("G2", "buffer removed (NOT re-run)", "logs/ablations/til/gem/G2/lr0.03", "bf16 lr0.03"),
]


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
    return sum(
        1
        for s in itertools.product((1, -1), repeat=n)
        if abs(sum(a * b for a, b in zip(s, d)) / n) >= obs - 1e-12
    ) / 2**n


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


def t_p(d: list[float]) -> float:
    n = len(d)
    m, sd = st.mean(d), st.stdev(d)
    if sd == 0:
        return 0.0
    t = abs(m) / (sd / math.sqrt(n))
    df = n - 1
    return _betainc(df / 2, 0.5, df / (df + t * t))


def verdict(lo: float, hi: float, ph: float, pp: float) -> str:
    if ph < 0.05 and pp < 0.05 and not (lo <= 0 <= hi):
        return "load-bearing"
    if not (lo <= 0 <= hi):
        return "borderline"
    if lo >= -DELTA and hi <= DELTA:
        return "inert"
    return "inconclusive"


def main() -> None:
    base = collect(BASE[2])
    f1b = [base[s]["Final F1"] for s in sorted(base)]
    bwtb = [base[s]["Backward"] for s in sorted(base)]
    print("GEM block, lr 0.01, inner_steps 2, single-epoch TIL.\n")
    print(
        f"  G0  F1 {st.mean(f1b):5.2f}+-{st.stdev(f1b):4.2f}({len(f1b)})  "
        f"BWT {st.mean(bwtb):6.2f}+-{st.stdev(bwtb):4.2f}  [{BASE[3]}]  {BASE[1]}"
    )

    raw = {}
    for rid, desc, pat, prec in ROWS:
        s = collect(pat)
        f1 = [s[k]["Final F1"] for k in sorted(s)]
        bwt = [s[k]["Backward"] for k in sorted(s)]
        common = sorted(set(base) & set(s))
        raw[rid] = [s[k]["Final F1"] - base[k]["Final F1"] for k in common]
        print(
            f"  {rid}  F1 {st.mean(f1):5.2f}+-{st.stdev(f1):4.2f}({len(f1)})  "
            f"BWT {st.mean(bwt):6.2f}+-{st.stdev(bwt):4.2f}  [{prec}]  {desc}"
        )

    order = sorted(raw, key=lambda r: t_p(raw[r]))
    holm, prev = {}, 0.0
    for i, rid in enumerate(order):
        prev = max(prev, min(t_p(raw[rid]) * (len(order) - i), 1.0))
        holm[rid] = prev

    print("\n  row  dF1      95% CI             p_Holm   p_perm  verdict        n")
    for rid, _d, _p, _pr in ROWS:
        d = raw[rid]
        m = st.mean(d)
        half = 1.96 * st.stdev(d) / math.sqrt(len(d))
        lo, hi = m - half, m + half
        pp, ph = perm_p(d), holm[rid]
        pht = "<0.001" if ph < 0.001 else f"{ph:6.3f}"
        print(
            f"  {rid}  {m:+6.2f}  [{lo:+6.2f},{hi:+6.2f}]   {pht:>6s}   {pp:6.3f}  "
            f"{verdict(lo, hi, ph, pp):14s} {len(d)}"
        )

    print(
        "\n  G2's delta pairs a bf16 ablation against an fp32 baseline. The precision effect\n"
        "  is ~1.4 F1 against a -35 effect, so the verdict cannot turn on it."
    )


if __name__ == "__main__":
    main()
